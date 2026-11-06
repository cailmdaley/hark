#!/usr/bin/env python3
"""Reproducible technical-English ASR evaluation without opening a microphone.

Build synthetic fixtures with macOS say, or supply a manifest of recorded files:
  uv run scripts/voice-accuracy.py --build /tmp/hark-accuracy
  uv run scripts/voice-accuracy.py /tmp/hark-accuracy/manifest.json --backend streaming
  uv run scripts/voice-accuracy.py /tmp/hark-accuracy/manifest.json --backend parakeet

Each manifest case has file, reference, group and optional segments (start/end/reference).
Batch models decode each segment independently. --lines instead decodes the streaming
utterance windows saved by a streaming run, including their original speaker/timing.
JSON output includes normalized word edit counts, text, wall/CPU cost and peak RSS.
--backend external explicitly calls the configured voice ear (which may upload audio).
Whisper experiments require uv run --with mlx-whisper; no local batch model runs in voice mode.
Synthetic speech measures these fixtures, not the user's accent or real-room accuracy.
"""

import argparse
import json
from pathlib import Path
import platform
import resource
import subprocess
import time
import unicodedata
import re

import numpy as np

from hark.capture import SAMPLE_RATE, load_audio

PARAKEET = "mlx-community/parakeet-tdt-0.6b-v2"
WHISPER = "mlx-community/whisper-large-v3-turbo-q4"
JARGON = "Cosmology, CMB lensing, cosmic shear, kinematic lensing, Euclid, NaMaster, SLURM, Apptainer, Singularity, HTCondor, gfal, GitHub, JSONL, MLX, Nemotron, Parakeet, Shuttle, Hark."
CASES = [
    ("cosmology", "single", [
        ("Daniel", "Before we interpret the cosmic shear cross correlation, compare the measured power spectrum with the fiducial model. The CMB lensing estimator runs on projected maps. Check whether the smooth droop comes from the pixel window, the survey mask, or the transfer function."),
    ]),
    ("pipeline", "single", [
        ("Daniel", "Launch one SLURM allocation and run everything inside the Apptainer container. The HTCondor jobs should stage the lightcone files before starting the pipeline. Fetch the remote branch before you commit, then push the tests to GitHub. Keep the JSONL transcript beside the plain text file."),
    ]),
    ("dictation", "single", [
        ("Daniel", "Okay, so I think the covariance is wrong, or rather the area in the covariance is wrong. No, wait, compare the analytic shape noise against the jackknife first. Claude, can you make that plot and show the residuals? Leave the actual science cuts alone until we understand the difference."),
    ]),
    ("names", "single", [
        ("Samantha", "Shuttle routes the live voice to the constitution in view. Hark should use Nemotron for speaker timing and compare Parakeet with Whisper for the finished words. The Euclid cross correlations and kinematic lensing results need careful labels. Don't replace a technical name merely because a common word sounds similar."),
    ]),
    ("control", "single", [
        ("Samantha", "Please move the meeting to tomorrow afternoon. I can finish the report after lunch, but I need to check the numbers before sending it. There are two people in this room. When the other person starts speaking, keep their words separate from mine and do not repeat the same sentence twice."),
    ]),
    ("conversation", "two-speaker", [
        ("Daniel", "The shear response changes when we widen the galaxy sample. Can you check whether that is a weighting effect or a change in the population?"),
        ("Samantha", "Yes, but the covariance also needs the right survey area and effective number density. I would compare the raw coupling before interpreting the calibrated amplitude."),
        ("Daniel", "Right, and please run the CMB lensing estimator on the projected maps. Keep the mean field subtraction independent of the simulation used for calibration."),
        ("Samantha", "I will check the power spectrum and the mask first. The pipeline should produce the figure and the tests in the same container."),
    ]),
]


def words(text):
    text = unicodedata.normalize("NFKC", text).lower().replace("’", "'")
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text)


def score(reference, hypothesis):
    ref, hyp = words(reference), words(hypothesis)
    # (total edits, substitutions, deletions, insertions); deterministic tie breaks.
    row = [(j, 0, 0, j) for j in range(len(hyp) + 1)]
    for i, a in enumerate(ref, 1):
        new = [(i, 0, i, 0)]
        for j, b in enumerate(hyp, 1):
            if a == b:
                new.append(row[j - 1])
            else:
                e, s, d, ins = row[j - 1]
                substitution = (e + 1, s + 1, d, ins)
                e, s, d, ins = row[j]
                deletion = (e + 1, s, d + 1, ins)
                e, s, d, ins = new[-1]
                insertion = (e + 1, s, d, ins + 1)
                new.append(min(substitution, deletion, insertion))
        row = new
    edits, substitutions, deletions, insertions = row[-1]
    return dict(words=len(ref), edits=edits, substitutions=substitutions,
                deletions=deletions, insertions=insertions,
                wer=edits / len(ref) if ref else None)


def quiet_boundary(samples, at):
    """Find a quiet 40 ms frame near a streaming token boundary."""
    frame = SAMPLE_RATE // 25
    lo = max(0, round((at - .6) * SAMPLE_RATE))
    hi = min(len(samples), round((at + .2) * SAMPLE_RATE))
    frames = [samples[i:i + frame] for i in range(lo, hi - frame + 1, frame)]
    if not frames:
        return min(at, len(samples) / SAMPLE_RATE)
    energy = np.array([np.mean(f * f) for f in frames])
    quiet = np.flatnonzero(energy <= max(energy.min() * 2, 1e-7))
    centers = (lo + (quiet + .5) * frame) / SAMPLE_RATE
    return float(centers[np.argmin(abs(centers - at))])


def build(directory):
    import soundfile as sf

    directory.mkdir(parents=True, exist_ok=True)
    cases = []
    for name, group, turns in CASES:
        audio, segments, at = [], [], 0.0
        for i, (voice, text) in enumerate(turns):
            path = directory / f"{name}-{i}.aiff"
            subprocess.run(["say", "-v", voice, "-r", "180", "-o", str(path), text], check=True)
            samples = load_audio(path)
            gap = np.zeros(SAMPLE_RATE, np.float32)
            audio.extend((samples, gap))
            end = at + len(samples) / SAMPLE_RATE
            segments.append(dict(start=at, end=end, reference=text, voice=voice))
            at = end + 1
        path = directory / f"{name}.wav"
        sf.write(path, np.concatenate(audio), SAMPLE_RATE)
        cases.append(dict(id=name, group=group, file=str(path.resolve()),
                          reference=" ".join(t[1] for t in turns), segments=segments))
    manifest = dict(source="macOS say at 180 words/min; no microphone", cases=cases)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?")
    parser.add_argument("--build", type=Path)
    parser.add_argument("--backend", choices=["streaming", "external", "parakeet", "whisper"], default="streaming")
    parser.add_argument("--preset", choices=["ultra_low", "very_low", "low", "offline"], default="ultra_low")
    parser.add_argument("--lang", default=None)
    parser.add_argument("--gap", type=float, default=.5)
    parser.add_argument("--max-len", type=float, default=6)
    parser.add_argument("--margin", type=float, default=.2, help="batch window edge context in seconds")
    parser.add_argument("--partition", action="store_true", help="use consecutive nonoverlapping windows, split at quiet audio near each boundary")
    parser.add_argument("--prompt", action="store_true", help="Whisper vocabulary hint")
    parser.add_argument("--lines", type=Path, help="decode windows from a streaming JSONL result")
    args = parser.parse_args()
    if args.build:
        build(args.build)
        return
    if not args.manifest:
        parser.error("manifest required unless --build")
    import mlx.core as mx
    from mlx_audio.stt import load
    from hark.live_voice import VoiceTrack, load_refiner
    from hark.masking import MaskPolicy
    from hark.transcript import flush_tracks, load_models, numbered

    manifest = json.loads(Path(args.manifest).read_text())
    windows = {}
    if args.lines:
        windows = {r["id"]: r["lines"] for r in map(json.loads, args.lines.read_text().splitlines()) if r.get("kind") == "case"}
    mx.set_cache_limit(512 * 1024**2)
    began = time.perf_counter()
    refiner = None
    if args.backend in {"streaming", "external"}:
        asr, diar = load_models(args.preset)
        if args.backend == "external":
            refiner = load_refiner()
    elif args.backend == "parakeet":
        asr = load(PARAKEET)
    else:
        from mlx_whisper.transcribe import ModelHolder
        import mlx_whisper

        ModelHolder.get_model(WHISPER, mx.float16)
    print(json.dumps(dict(kind="configuration", backend=args.backend, preset=args.preset,
                         language=args.lang, gap=args.gap, max_len=args.max_len,
                         prompt=args.prompt, lines=str(args.lines),
                         device=mx.device_info(), platform=platform.platform(),
                         load_seconds=time.perf_counter() - began)), flush=True)
    results = []
    for case in manifest["cases"]:
        samples = load_audio(case["file"])
        duration = len(samples) / SAMPLE_RATE
        begin, cpu = time.perf_counter(), time.process_time()
        lines, calls = [], []
        if args.backend in {"streaming", "external"}:
            def recognize(audio):
                before = time.perf_counter()
                text = refiner(audio)
                calls.append(time.perf_counter() - before)
                return text

            track = VoiceTrack("mic", asr, diar, speaker_label=numbered,
                               language=args.lang, gap=args.gap, max_len=args.max_len,
                               refiner=recognize if refiner else None,
                               mask=MaskPolicy.parse("exclusive"))
            padded = np.concatenate((samples, np.zeros(4 * SAMPLE_RATE, np.float32)))
            for start in range(0, len(padded), SAMPLE_RATE // 2):
                track.feed(padded[start:start + SAMPLE_RATE // 2])
                for u in flush_tracks([track]):
                    lines.append(dict(start=u.start, end=u.end, speaker=u.speaker, text=u.text,
                                      emitted_at=(start + SAMPLE_RATE // 2) / SAMPLE_RATE))
            track.feed(np.zeros(0, np.float32), final=True)
            for u in flush_tracks([track], force=True):
                lines.append(dict(start=u.start, end=u.end, speaker=u.speaker, text=u.text,
                                  emitted_at=duration + 4))
            del track
        else:
            previous_end = 0
            for segment in windows.get(case["id"], case["segments"]):
                # A small edge margin includes word onsets/tails outside RNNT token times.
                start, end = max(0, segment["start"] - args.margin), min(duration, segment["end"] + args.margin)
                if args.partition:
                    start = previous_end
                    end = quiet_boundary(samples, segment["end"])
                    previous_end = end
                audio = mx.array(samples[round(start * SAMPLE_RATE):round(end * SAMPLE_RATE)])
                before = time.perf_counter()
                if args.backend == "parakeet":
                    result = asr.generate(audio)
                else:
                    result = mlx_whisper.transcribe(audio, path_or_hf_repo=WHISPER,
                                                    language="en" if args.lang else None,
                                                    initial_prompt=JARGON if args.prompt else None,
                                                    temperature=0, verbose=None)
                text = result.text if args.backend == "parakeet" else result["text"]
                mx.synchronize()
                calls.append(time.perf_counter() - before)
                lines.append(dict(start=segment["start"], end=segment["end"],
                                  speaker=segment.get("speaker", segment.get("voice")), text=text))
        mx.synchronize()
        text = " ".join(line["text"] for line in sorted(lines, key=lambda line: line["start"]))
        result = dict(kind="case", id=case["id"], group=case["group"], duration=duration,
                      wall_seconds=time.perf_counter() - begin, cpu_seconds=time.process_time() - cpu,
                      reference=case["reference"], text=text, lines=lines, calls=calls,
                      **score(case["reference"], text))
        results.append(result)
        print(json.dumps(result), flush=True)
        mx.clear_cache()
    for group in ["single", "two-speaker", "all"]:
        selected = [r for r in results if group == "all" or r["group"] == group]
        if not selected:
            continue
        counts = {key: sum(r[key] for r in selected) for key in ["words", "edits", "substitutions", "deletions", "insertions"]}
        duration = sum(r["duration"] for r in selected)
        wall = sum(r["wall_seconds"] for r in selected)
        print(json.dumps(dict(kind="summary", group=group, **counts, wer=counts["edits"] / counts["words"],
                              duration=duration, wall_seconds=wall, wall_per_audio_second=wall / duration,
                              peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)), flush=True)


if __name__ == "__main__":
    main()
