#!/usr/bin/env python3
"""Measure loudspeaker leakage in simultaneous raw and software-AEC mic recordings.

Run with an isolated HARK_DIR. Playback is opt-in, limited to eight seconds,
and refused unless Shuttle's meeting API confirms no recording is active.
The script never changes system volume, routing or someone else's recording.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import shlex
import time
import urllib.request

import numpy as np
from scipy.signal import correlate

from hark.aec import EchoCanceller
from hark.capture import MicSource, SystemSource, WavRecorder, load_audio

MIC = "MacBook Pro Microphone"
SPEAKER = "MacBook Pro Speakers"


def live_hark_command(command):
    """Recognize ordinary local live CLI invocations, not shell scripts or file replays."""
    try:
        argv = shlex.split(command)
    except ValueError:
        return False
    if not argv:
        return False
    if Path(argv[0]).name == "hark":
        arguments = argv[1:]
    elif Path(argv[0]).name.startswith("python") and len(argv) > 1:
        if Path(argv[1]).name == "hark":
            arguments = argv[2:]
        elif argv[1:3] in (["-m", "hark"], ["-m", "hark.cli"]):
            arguments = argv[3:]
        else:
            return False
    else:
        return False
    return not ("--file" in arguments or any(arg.startswith("--file=") for arg in arguments)
                or arguments[:1] in [[name] for name in ("name", "enroll", "mirror", "pause", "resume", "processes")])


def assert_no_local_recording():
    commands = subprocess.check_output(["ps", "-axo", "command="], text=True)
    if any(live_hark_command(line) for line in commands.splitlines()):
        raise RuntimeError("A local hark may be recording; refusing playback (do not stop it)")


def playback_allowed(url):
    assert_no_local_recording()
    with urllib.request.urlopen(url, timeout=5) as response:
        state = json.load(response)
    if "meeting" not in state:
        raise RuntimeError("Meeting API has no meeting field; refusing playback")
    meeting = state["meeting"]
    if meeting is not None and meeting.get("state") not in {"ended", "failed"}:
        raise RuntimeError("A meeting may be recording; refusing playback (do not stop it)")
    return state


def metrics(audio, reference, offset=None):
    if len(audio) < len(reference) or not np.any(reference):
        raise ValueError("Capture is shorter than the reference, or reference is silent")
    if not np.any(audio):
        raise ValueError("Capture is all-zero PCM; suppression is inconclusive, not successful AEC")
    if offset is None:
        correlation = correlate(audio, reference, mode="valid", method="fft")
        offset = int(np.argmax(np.abs(correlation)))
    span = audio[offset:offset + len(reference)]
    rms = float(np.sqrt(np.mean(span ** 2)))
    norm = float(np.linalg.norm(span) * np.linalg.norm(reference))
    return {
        "rms": rms,
        "absolute_peak_correlation": abs(float(np.dot(span, reference))) / norm if norm else 0,
        "reference_projection_gain": abs(float(np.dot(span, reference) / np.dot(reference, reference))),
        "best_offset_seconds": offset / 16000,
        "captured_seconds": len(audio) / 16000,
    }


def play_reference(reference, gain):
    """Target laptop speakers explicitly, never a connected headphone default."""
    import sounddevice as sd

    sd.play(reference * gain, samplerate=16000, device=SPEAKER)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="known speech clip, at most eight seconds")
    parser.add_argument("out", type=Path, help="new output directory for recordings and metrics")
    parser.add_argument("--play", action="store_true", help="allow short speaker playback")
    parser.add_argument("--volume", type=float, default=0.35, help="playback gain, 0–1; system volume unchanged")
    parser.add_argument("--meeting-api", default="http://localhost:4000/api/v1/meeting")
    args = parser.parse_args()
    if not args.play:
        parser.error("Playback requires explicit --play")
    if not os.environ.get("HARK_DIR"):
        parser.error("Set an isolated HARK_DIR")
    if not 0 < args.volume <= 1:
        parser.error("--volume must be in (0, 1]")
    reference = load_audio(args.audio)
    if not 0 < len(reference) <= 8 * 16000:
        parser.error("Use a non-empty speech clip no longer than eight seconds")
    playback_allowed(args.meeting_api)
    args.out.mkdir(parents=True, exist_ok=False)
    import sounddevice as sd

    report = {"clip": str(args.audio.resolve()), "playback_gain": args.volume,
              "volume_settings": subprocess.check_output(["osascript", "-e", "get volume settings"], text=True).strip(),
              "input_device": sd.query_devices(MIC, kind="input"),
              "output_device": sd.query_devices(SPEAKER, kind="output"),
              "macos": subprocess.check_output(["sw_vers", "-productVersion"], text=True).strip(),
              "near_end_control_tested": False,
              "note": "Same raw mic before/after software AEC, measured in the same window. RMS includes ambient noise; near-end speech preservation needs a separate control."}
    mic, system = MicSource(MIC), SystemSource()
    recorders = {name: WavRecorder(args.out / f"{name}.wav") for name in ("raw", "aec", "system")}
    started = []
    aec = None
    try:
        for source in (mic, system):
            source.start()
            started.append(source)
        aec = EchoCanceller(round((system.anchor - mic.anchor) * 16000), report=print)
        start = time.monotonic()
        played = False
        while time.monotonic() - start < 2 + len(reference) / 16000 + 2:
            if not played and time.monotonic() - start >= 2:
                if not mic.stats["device"] or not system.stats["device"]:
                    raise RuntimeError("Capture unavailable; check microphone/system-audio permissions")
                playback_allowed(args.meeting_api)
                play_reference(reference, args.volume)
                played = True
            raw, far = mic.drain(), system.drain()
            recorders["raw"].write(raw)
            recorders["system"].write(far)
            recorders["aec"].write(aec.feed(raw, far))
            time.sleep(0.02)
        sd.wait()
    finally:
        sd.stop()
        for source in started:
            source.stop()
        if aec is not None:
            raw, far = mic.drain(), system.drain()
            recorders["raw"].write(raw)
            recorders["system"].write(far)
            recorders["aec"].write(aec.feed(raw, far, final=True))
        for recorder in recorders.values():
            recorder.close()
    report["raw"] = metrics(load_audio(args.out / "raw.wav"), reference)
    offset = round(report["raw"]["best_offset_seconds"] * 16000)
    report["aec"] = metrics(load_audio(args.out / "aec.wav"), reference, offset=offset)
    report["final_alignment_seconds"] = aec.delay / 16000
    raw, aec = report["raw"], report["aec"]
    report["baseline_detected"] = raw["absolute_peak_correlation"] >= 0.1
    report["rms_reduction_db"] = 20 * math.log10(raw["rms"] / aec["rms"]) if aec["rms"] else None
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["baseline_detected"]:
        raise SystemExit("Baseline playback not established; do not interpret attenuation as AEC performance")


if __name__ == "__main__":
    main()
