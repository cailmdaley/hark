"""Replay an audio file through hark's live pipeline and dump what the models saw.

    uv run python -m hark.replay AUDIO OUTDIR [--start S] [--duration S] [--latency low]

Feeds the file to a `Track` in 0.5 s pieces and flushes turns exactly as the live
loop does, while recording the diarizer's per-frame speaker probabilities (10 ms)
as they are handed to the ASR mask. Writes to OUTDIR:

    probs.npy         float32 (frames, 8): committed diarizer probabilities
    tokens.jsonl      every ASR token per speaker stream: speaker, start, end, text
    utterances.jsonl  the turns hark would have written
    meta.json         audio span, latency preset, session knobs, chunk sizes
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .capture import SAMPLE_RATE
from .transcript import Track, flush_tracks, load_models, numbered


def replay(audio, asr, diar, *, gap=3.0, policy=None, session_kwargs=None, log_every=60.0):
    """Run `audio` through a Track; return probs, raw tokens and utterances."""
    track = Track("replay", asr, diar, speaker_label=numbered, gap=gap, mask=policy)
    if session_kwargs:
        from .session import GatedSession

        track.session = GatedSession(asr, diar, policy=track.session.policy, **session_kwargs)
    track.audio = None
    session = track.session
    probs, tokens = [], []
    push = session._push_features

    def recording_push(mel, p, *, final=False):
        probs.append(np.array(p, dtype=np.float32))
        updates = push(mel, p, final=final)
        tokens.extend((u.speaker, t.start, t.end, t.text) for u in updates for t in u.tokens)
        return updates

    session._push_features = recording_push
    utterances = []
    step = SAMPLE_RATE // 2
    next_log, t0 = log_every, time.monotonic()
    for i in range(0, audio.size, step):
        track.feed(audio[i : i + step])
        utterances.extend(flush_tracks([track]))
        if (i + step) / SAMPLE_RATE >= next_log:
            print(f"[replay] {next_log:.0f} s in {time.monotonic() - t0:.0f} s", flush=True)
            next_log += log_every
    track.feed(audio[:0], final=True)
    utterances.extend(flush_tracks([track], force=True))
    return {
        "probs": np.concatenate(probs) if probs else np.zeros((0, 8), np.float32),
        "tokens": [dict(speaker=s, start=round(a, 3), end=round(b, 3), text=x) for s, a, b, x in tokens],
        "utterances": [u.record() for u in sorted(utterances, key=lambda u: u.start)],
        "session": {"threshold": session.threshold, "factor": session.factor,
                    "chunk_mel": session.chunk_mel, "cache_gating": session.cache_gating,
                    "cache_gating_buffer_size": session.cache_gating_buffer_size},
    }


def load_span(path, start=0.0, duration=None):
    from mlx_audio.stt.utils import load_audio

    audio = np.array(load_audio(str(path), sr=SAMPLE_RATE), dtype=np.float32)
    a = round(start * SAMPLE_RATE)
    return audio[a : None if duration is None else a + round(duration * SAMPLE_RATE)]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="hark.replay", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float)
    ap.add_argument("--latency", default="low")
    ap.add_argument("--gap", type=float, default=3.0)
    ap.add_argument("--mask", default="exclusive",
                    help="speaker mask policy (hark.masking.MaskPolicy.parse); default: exclusive, as live")
    ap.add_argument("--session", default="{}", help="JSON kwargs for the speaker-streaming session")
    args = ap.parse_args(argv)

    audio = load_span(args.audio, args.start, args.duration)
    asr, diar = load_models(args.latency)
    t = time.monotonic()
    from .masking import MaskPolicy

    policy = MaskPolicy.parse(args.mask)
    result = replay(audio, asr, diar, gap=args.gap, policy=policy, session_kwargs=json.loads(args.session))
    wall = time.monotonic() - t
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "probs.npy", result["probs"])
    for name in ("tokens", "utterances"):
        (args.out / f"{name}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in result[name]))
    meta = {"audio": str(args.audio.resolve()), "start": args.start, "duration": audio.size / SAMPLE_RATE,
            "latency": args.latency, "session": result["session"], "session_kwargs": json.loads(args.session),
            "mask": vars(policy),
            "wall": round(wall, 1)}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[replay] {meta['duration']:.0f} s of audio in {wall:.0f} s → {args.out}")


if __name__ == "__main__":
    main()
