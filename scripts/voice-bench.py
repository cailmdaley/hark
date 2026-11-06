#!/usr/bin/env python3
"""Replay local audio to measure voice latency, compute and retained state.

Run with uv run scripts/voice-bench.py --file speech.wav --seconds 3600.
Mixed input repeats the clip followed by equally long digital silence, or the
specified --silence-seconds. Voice runs use a constant external-ear stub: they
exercise run buffering/alignment without uploading audio or measuring cloud cost.
Use --realtime --gpu for whole-device GPU sampling while listening; those
samples include other apps and are not Hark-attributable power measurements.
No microphone opens, no audio is saved, and transcript text is not printed.
"""

import argparse
import json
import os
import platform
import re
import subprocess
import threading
import time

import mlx.core as mx
import numpy as np

from hark.capture import SAMPLE_RATE, load_audio
from hark.live_voice import VOICE_GAP, VOICE_MAX_LEN, VoiceTrack
from hark.masking import MaskPolicy
from hark.transcript import Track, flush_tracks, load_models, numbered


def rss():
    return int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())])) * 1024


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--file", help="local speech clip; required except for silence")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--pattern", choices=["mixed", "speech", "silence"], default="mixed")
    ap.add_argument("--mode", choices=["meeting", "tuned", "voice"], default="voice")
    ap.add_argument("--silence-seconds", type=float, help="quiet time after each mixed speech clip")
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--gpu", action="store_true", help="sample aggregate IOAccelerator utilization")
    args = ap.parse_args()
    if args.pattern != "silence" and not args.file:
        ap.error("--file is required for speech or mixed input")
    if args.seconds <= 0:
        ap.error("--seconds must be positive")
    if args.pattern == "silence":
        cycle = np.zeros(SAMPLE_RATE, np.float32)
    else:
        cycle = load_audio(args.file)
        if args.pattern == "mixed":
            quiet = np.zeros_like(cycle) if args.silence_seconds is None else np.zeros(round(args.silence_seconds * SAMPLE_RATE), np.float32)
            cycle = np.concatenate((cycle, quiet))
    if not cycle.size:
        ap.error("input clip is empty")
    preset = "low" if args.mode == "meeting" else "ultra_low"
    asr, diar = load_models(preset)
    cls = VoiceTrack if args.mode == "voice" else Track
    options = {"refiner": lambda _: "replayed speech"} if args.mode == "voice" else {}
    track = cls("mic", asr, diar, speaker_label=numbered, mask=MaskPolicy.parse("exclusive"),
                gap=3 if args.mode == "meeting" else VOICE_GAP if args.mode == "voice" else .5,
                max_len=30 if args.mode == "meeting" else VOICE_MAX_LEN if args.mode == "voice" else 6,
                **options)
    # Keep the normal rolling voice-match audio buffer enabled.
    print(json.dumps({"configuration": vars(args), "device": mx.device_info(),
                      "platform": platform.platform(), "pid": os.getpid(),
                      "gpu_scope": "whole-device, includes other applications"}), flush=True)
    gpu = []
    stop = threading.Event()

    def sample_gpu():
        while not stop.is_set():
            result = subprocess.run(["ioreg", "-r", "-c", "IOAccelerator", "-l"],
                                    capture_output=True, text=True, check=True)
            match = re.search(r'"Device Utilization %"=(\d+)', result.stdout)
            if match:
                gpu.append(int(match[1]))
            stop.wait(1)

    sampler = threading.Thread(target=sample_gpu, daemon=True) if args.gpu else None
    if sampler:
        sampler.start()
    began, cpu = time.monotonic(), time.process_time()
    interval_wall, interval_cpu = began, cpu
    compute = 0
    lines = []
    total = round(args.seconds * SAMPLE_RATE)
    next_report = min(60 * SAMPLE_RATE, total)
    try:
        for start in range(0, total, SAMPLE_RATE // 2):
            count = min(SAMPLE_RATE // 2, total - start)
            # Wrap without retaining a session-length waveform.
            indices = (np.arange(count) + start) % cycle.size
            samples = cycle[indices]
            horizon = (start + count) / SAMPLE_RATE
            if args.realtime:
                time.sleep(max(0, began + horizon - time.monotonic()))
            before = time.monotonic()
            track.feed(samples)
            for u in flush_tracks([track]):
                lines.append({"at": horizon, "wall_at": time.monotonic() - began,
                              "start": u.start, "end": u.end})
            compute += time.monotonic() - before
            if start + count >= next_report:
                wall, used_cpu = time.monotonic(), time.process_time()
                print(json.dumps({"audio_seconds": horizon, "interval_wall": wall - interval_wall,
                                  "interval_cpu": used_cpu - interval_cpu,
                                  "compute_seconds": compute, "rss": rss(),
                                  "mlx_active": mx.get_active_memory(),
                                  "mlx_cache": mx.get_cache_memory(),
                                  "audio_samples": len(track.audio.samples),
                                  "pending_tokens": sum(len(p.tokens) for p in track.pending.values()),
                                  "clock_entries": len(track._clock) if args.mode == "voice" else None,
                                  "lines": lines,
                                  "gpu_util_percent": list(gpu)}), flush=True)
                gpu.clear()
                lines = []
                compute = 0
                interval_wall, interval_cpu = wall, used_cpu
                next_report = min(next_report + 60 * SAMPLE_RATE, total)
        track.feed(np.zeros(0, np.float32), final=True)
        final_lines = [{"start": u.start, "end": u.end} for u in flush_tracks([track], force=True)]
        print(json.dumps({"done": True, "audio_seconds": args.seconds,
                          "wall_seconds": time.monotonic() - began,
                          "cpu_seconds": time.process_time() - cpu,
                          "final_lines": final_lines}), flush=True)
    finally:
        stop.set()
        if sampler:
            sampler.join()


if __name__ == "__main__":
    main()
