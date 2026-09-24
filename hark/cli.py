"""hark — capture a conversation, append one speaker-labelled line per utterance.

    hark                 call: mic is "me", system audio (Zoom…) diarized as S1…S8
    hark --room          in person: the mic alone, diarized
    hark --system        system audio only (a talk, a recording playing)
    hark --file x.wav    transcribe a file through the same streaming path

The live transcript is ~/.hark/current.txt (a symlink to the session file);
follow it with `tail -F`. A JSONL sidecar sits beside it.
"""

import argparse
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

from .capture import FileSource, MicSource, SystemSource, log
from .transcript import Sink, Track, load_models, numbered

HOME = Path(os.environ.get("HARK_DIR", Path.home() / ".hark"))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="hark", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--room", action="store_true", help="mic only, diarized")
    mode.add_argument("--system", action="store_true", help="system audio only, diarized")
    mode.add_argument("--file", help="transcribe an audio file instead of live input")
    ap.add_argument("--realtime", action="store_true", help="with --file: replay at real-time pace")
    ap.add_argument("--mic", help="input device name or index (default: system default)")
    ap.add_argument("--lang", default=None, help="ASR language, e.g. en-US, fr-FR (default: auto)")
    ap.add_argument("--latency", default="low", choices=["low", "very_low", "ultra_low"],
                    help="diarizer buffer: low=1.04 s (default), very_low=0.64 s, ultra_low=0.32 s")
    ap.add_argument("--gap", type=float, default=1.0, help="seconds of quiet that end an utterance")
    ap.add_argument("--title", help="appended to the session filename")
    ap.add_argument("-o", "--out", type=Path, help="write the transcript here instead of ~/.hark/sessions/")
    args = ap.parse_args(argv)

    if args.file:
        sources = [(FileSource(args.file, realtime=args.realtime), numbered)]
        what = f"file {args.file}"
    elif args.room:
        sources = [(MicSource(_device(args.mic)), numbered)]
        what = "room: mic diarized as S1…"
    elif args.system:
        sources = [(SystemSource(), numbered)]
        what = "system audio diarized as S1…"
    else:
        sources = [(MicSource(_device(args.mic)), lambda _: "me"), (SystemSource(), numbered)]
        what = "call: me = mic, S1… = system audio"

    now = datetime.now()
    out = args.out or _session_path(now, args.title)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not args.out:
        _point_current(out)

    log("loading models…")
    asr, diar = load_models(args.latency)
    t0 = datetime.combine(now.date(), datetime.min.time()) if args.file else None
    tracks = [(src, Track(src.name, asr, diar, speaker_label=label, language=args.lang,
                          gap=args.gap, t0=t0))
              for src, label in sources]
    sink = Sink(out, f"hark {now:%Y-%m-%d %H:%M} — {what}")
    log(f"transcript → {out}")

    stop = False

    def on_signal(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    for src, _ in sources:
        src.start()
    log("listening (Ctrl-C to stop)")

    started = time.monotonic()
    while not stop:
        busy = False
        for src, track in tracks:
            samples = src.drain()
            if samples.size:
                busy = True
                for u in track.feed(samples):
                    sink.write(u)
                    print(u.line(), flush=True)
        if args.file and sources[0][0].done.is_set() and not busy and sources[0][0].queue.empty():
            break
        if not busy:
            time.sleep(0.05)

    for src, _ in sources:
        src.stop()
    for src, track in tracks:
        for u in track.feed(src.drain(limit=float("inf")), final=True):
            sink.write(u)
            print(u.line(), flush=True)
    audio = max(t.processed for _, t in tracks)
    wall = time.monotonic() - started
    sink.close(f"ended {datetime.now():%H:%M:%S}")
    log(f"done: {audio:.0f} s of audio in {wall:.0f} s (real-time factor {wall / max(audio, 1e-9):.2f})")


def _device(spec):
    if spec is None:
        return None
    return int(spec) if spec.isdigit() else spec


def _session_path(now, title):
    slug = f"{now:%Y-%m-%d_%H%M}" + (f"_{_slug(title)}" if title else "")
    return HOME / "sessions" / f"{slug}.txt"


def _slug(s):
    return "".join(c if c.isalnum() else "-" for c in s.lower()).strip("-")


def _point_current(out):
    for suffix in (".txt", ".jsonl"):
        link = HOME / f"current{suffix}"
        target = out.with_suffix(suffix)
        target.touch()
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target)


if __name__ == "__main__":
    sys.exit(main())
