#!/usr/bin/env python3
"""Measure loudspeaker leakage in raw and VoiceProcessingIO mic recordings.

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
import sys
import time
import urllib.request

import numpy as np
from scipy.signal import correlate

from hark.capture import VOICE_MIC, WavRecorder, load_audio


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


def metrics(audio, reference):
    if len(audio) < len(reference) or not np.any(reference):
        raise ValueError("Capture is shorter than the reference, or reference is silent")
    if not np.any(audio):
        raise ValueError("Capture is all-zero PCM; suppression is inconclusive, not successful AEC")
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


def raw_worker():
    import sounddevice as sd

    def callback(data, *_):
        sys.stdout.buffer.write(data.tobytes())
        sys.stdout.buffer.flush()

    with sd.InputStream(samplerate=16000, channels=1, dtype="int16", callback=callback):
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="known speech clip, at most eight seconds")
    parser.add_argument("out", type=Path, help="new output directory for recordings and metrics")
    parser.add_argument("--play", action="store_true", help="allow short speaker playback")
    parser.add_argument("--volume", type=float, default=0.35, help="afplay gain, 0–1; system volume unchanged")
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
    if not VOICE_MIC.is_file():
        parser.error("Build bin/voice-mic with scripts/build-voice-mic.sh")
    playback_allowed(args.meeting_api)
    args.out.mkdir(parents=True, exist_ok=False)
    import sounddevice as sd

    report = {"clip": str(args.audio.resolve()), "afplay_gain": args.volume,
              "volume_settings": subprocess.check_output(["osascript", "-e", "get volume settings"], text=True).strip(),
              "input_device": sd.query_devices(kind="input"),
              "output_device": sd.query_devices(kind="output"),
              "macos": subprocess.check_output(["sw_vers", "-productVersion"], text=True).strip(),
              "near_end_control_tested": False,
              "note": "Sequential raw/AEC trials; RMS includes ambient noise. Correlation measures waveform similarity, not word accuracy. Near-end speech preservation is not established."}
    for mode in ("raw", "aec"):
        playback_allowed(args.meeting_api)
        pcm = args.out / f"{mode}.pcm"
        command = ([str(VOICE_MIC)] if mode == "aec" else
                   [sys.executable, str(Path(__file__).resolve()), "--raw-worker"])
        with pcm.open("wb") as output, (args.out / f"{mode}.log").open("wb") as log:
            process = subprocess.Popen(command, stdout=output, stderr=log, start_new_session=True)
            try:
                deadline = time.monotonic() + 8
                while pcm.stat().st_size == 0 and process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.05)
                if process.poll() is not None or pcm.stat().st_size == 0:
                    raise RuntimeError(f"{mode} capture failed; see {mode}.log")
                time.sleep(1)
                playback_allowed(args.meeting_api)
                subprocess.run(["afplay", "-v", str(args.volume), str(args.audio)], check=True, timeout=12)
                time.sleep(1)
            finally:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        samples = np.fromfile(pcm, dtype="<i2").astype(np.float32) / 32768
        recorder = WavRecorder(args.out / f"{mode}.wav")
        recorder.write(samples)
        recorder.close()
        report[mode] = metrics(samples, reference)
    raw, aec = report["raw"], report["aec"]
    report["baseline_detected"] = raw["absolute_peak_correlation"] >= 0.1
    report["rms_reduction_db"] = 20 * math.log10(raw["rms"] / aec["rms"]) if aec["rms"] else None
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["baseline_detected"]:
        raise SystemExit("Baseline playback not established; do not interpret attenuation as AEC performance")


if __name__ == "__main__":
    if sys.argv[1:] == ["--raw-worker"]:
        raw_worker()
    else:
        main()
