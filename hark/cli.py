"""hark — capture a conversation, append one speaker-labelled line per utterance.

    hark                 call: mic is "me", system audio (Zoom…) diarized as S1…S8
    hark --room          in person: the mic alone, diarized
    hark --file x.wav    transcribe a file through the same streaming path

The live transcript is ~/.hark/current.txt (a symlink to the session file);
follow it with `tail -F`. A JSONL sidecar sits beside it.
"""

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

from .capture import FileSource, MicSource, SystemSource, log
from .transcript import EchoGate, Sink, Track, flush_tracks, load_models, numbered

HOME = Path(os.environ.get("HARK_DIR", Path.home() / ".hark")).expanduser().resolve()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "enroll":
        _enroll(argv[1:])
        return 0
    if argv and argv[0] == "name":
        if len(argv) != 3:
            raise SystemExit('usage: hark name S1 "Speaker Name"')
        _name_current(argv[1], argv[2])
        return 0
    ap = argparse.ArgumentParser(prog="hark", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--room", action="store_true", help="mic only, diarized")
    mode.add_argument("--file", help="transcribe an audio file instead of live input")
    ap.add_argument("--realtime", action="store_true", help="with --file: replay at real-time pace")
    ap.add_argument("--mic", help="input device name or index (default: system default)")
    ap.add_argument("--lang", default=None, help="ASR language, e.g. en-US, fr-FR (default: auto)")
    ap.add_argument("--latency", default="low", choices=["low", "very_low", "ultra_low"],
                    help="diarizer buffer: low=1.04 s (default), very_low=0.64 s, ultra_low=0.32 s")
    ap.add_argument("--gap", type=float, default=3.0,
                    help="seconds of silence that end a turn when nobody else takes over")
    ap.add_argument("--title", help="appended to the session filename")
    ap.add_argument("-o", "--out", type=Path, help="write the transcript here instead of ~/.hark/sessions/")
    args = ap.parse_args(argv)
    if args.realtime and not args.file:
        ap.error("--realtime only applies to --file")

    if args.file:
        sources = [(FileSource(args.file, realtime=args.realtime), numbered)]
        what = f"file {args.file}"
    elif args.room:
        sources = [(MicSource(_device(args.mic)), numbered)]
        what = "room: mic diarized as S1…"
    else:
        sources = [(MicSource(_device(args.mic)), lambda _: "me"), (SystemSource(), numbered)]
        what = "call: me = mic, S1… = system audio"

    log("loading models…")
    try:
        asr, diar = load_models(args.latency)
    except KeyboardInterrupt:
        return 130
    tracks = [(src, Track(src.name, asr, diar, speaker_label=label, language=args.lang,
                          gap=args.gap))
              for src, label in sources]

    all_tracks = [track for _, track in tracks]
    now = datetime.now()
    out = args.out.with_suffix(".txt") if args.out else _session_path(now, args.title)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not (args.out or args.file):  # a replay never hijacks the live session's link
        _point_current(out)
    sink = Sink(out, f"hark {now:%Y-%m-%d %H:%M} — {what}")
    import numpy as np

    voices = {path.stem: np.load(path) for path in (HOME / "voices").glob("*.npy")} if (HOME / "voices").exists() else {}
    from .voice import VoiceMatcher

    matcher = VoiceMatcher(voices, sink) if voices else None
    tracks_by_name = {track.name: track for _, track in tracks}
    log(f"transcript → {out}")

    stop = False

    def on_signal(*_):
        nonlocal stop
        stop = True
        signal.signal(signal.SIGINT, signal.SIG_DFL)  # a second Ctrl-C quits hard

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, on_signal)

    def emit(u):
        sink.poll_names()
        if matcher:
            matcher.finished(tracks_by_name[u.track], u)
        sink.write(u)
        try:
            print(u.line(), flush=True)
        except BrokenPipeError:
            sys.stdout = open(os.devnull, "w")

    gate = EchoGate() if not (args.room or args.file) else None

    def dispatch(u):
        for ready in ([u] if gate is None else gate.push(u)):
            emit(ready)

    started = time.monotonic()
    try:
        for src, track in tracks:
            src.start()
            track.t0 = (datetime.combine(now.date(), datetime.min.time()) if args.file
                        else datetime.fromtimestamp(src.anchor))
        log("listening (Ctrl-C to stop)")
        while not stop:
            sink.poll_names()
            busy = False
            for src, track in tracks:
                samples = src.drain()
                if samples.size:
                    busy = True
                    track.feed(samples)
            for u in flush_tracks(all_tracks):
                dispatch(u)
            if gate:
                system_track = tracks[1][1]
                watermark = (system_track.t0.timestamp() + system_track.processed
                             if system_track.t0 else float("-inf"))
                for u in gate.release(watermark):
                    emit(u)
            if args.file and sources[0][0].done.is_set() and not busy and sources[0][0].queue.empty():
                break
            if not busy:
                time.sleep(0.05)
    finally:
        for src, _ in sources:
            src.stop()
        for src, track in tracks:
            track.feed(src.drain(limit=float("inf")), final=True)
        for u in flush_tracks(all_tracks, force=True):
            dispatch(u)
        if gate:
            for u in gate.release(float("inf"), final=True):
                emit(u)
        sink.close(f"ended {datetime.now():%H:%M:%S}")
    audio = max(t.processed for _, t in tracks)
    wall = time.monotonic() - started
    log(f"done: {audio:.0f} s of audio in {wall:.0f} s (real-time factor {wall / max(audio, 1e-9):.2f})")


def _enroll(argv):
    ap = argparse.ArgumentParser(prog="hark enroll")
    ap.add_argument("name")
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--file")
    ap.add_argument("--mic")
    args = ap.parse_args(argv)
    if args.seconds <= 0:
        ap.error("--seconds must be positive")
    if not args.name or Path(args.name).name != args.name or args.name in {".", ".."}:
        ap.error("name must be a non-empty filename component")
    import numpy as np
    from .voice import Embedder

    if args.file:
        from mlx_audio.stt.utils import load_audio

        samples = np.asarray(load_audio(args.file, sr=16000), dtype=np.float32).reshape(-1)
    else:
        import sounddevice as sd

        samples = sd.rec(round(args.seconds * 16000), samplerate=16000, channels=1,
                         dtype="float32", device=_device(args.mic), blocking=True).reshape(-1)
    voice_dir = HOME / "voices"
    voice_dir.mkdir(parents=True, exist_ok=True)
    samples = samples[:round(args.seconds * 16000)]
    if samples.size < 5 * 16000:
        ap.error("enrollment audio must contain at least 5 seconds")
    vector = Embedder()(samples)
    np.save(voice_dir / f"{args.name}.npy", vector)
    import wave

    with wave.open(str(voice_dir / f"{args.name}.wav"), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    log(f"enrolled {args.name}: {len(samples) / 16000:.1f} s → {voice_dir}")


def _name_current(speaker, name):
    if not speaker.startswith("S") or not speaker[1:].isdigit():
        raise SystemExit("speaker must be a label such as S1")
    txt = HOME / "current.txt"
    if not txt.exists():
        raise SystemExit(f"no live session at {txt}")
    txt = txt.resolve()
    with txt.open("a") as transcript:
        transcript.write(f"# {speaker} = {name}\n")
    with txt.with_suffix(".jsonl").open("a") as records:
        record = {"wall": datetime.now().isoformat(timespec="seconds"),
                  "name": {"speaker": speaker, "as": name}}
        records.write(json.dumps(record, ensure_ascii=False) + "\n")


def _device(spec):
    if spec is None:
        return None
    return int(spec) if spec.isdigit() else spec


def _session_path(now, title):
    slug = f"{now:%Y-%m-%d_%H%M%S}" + (f"_{_slug(title)}" if title else "")
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
