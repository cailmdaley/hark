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
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from .capture import SAMPLE_RATE, FileSource, MicSource, SystemSource, log
from .meeting import prepare_local_meeting, prepare_remote_meeting, render_meeting_fiber
from .mirror import TranscriptMirror
from .transcript import Sink, Track, flush_tracks, load_models, numbered

HOME = Path(os.environ.get("HARK_DIR", Path.home() / ".hark")).expanduser().resolve()


class MeetingLifecycle:
    def __init__(self, *, title, host, project, under, store, fiber, started):
        self.path = HOME / "meeting.json"
        self.data = {
            "pid": os.getpid(), "phase": "loading", "title": title, "host": host,
            "project": project, "under": under, "store": store, "fiber": fiber,
            "started": started, "transcript": None, "mirror": None, "error": None,
        }
        self.write()

    def update(self, phase=None, **values):
        if phase and (self.data["phase"] != "stopping" or phase in {"ended", "failed"}):
            self.data["phase"] = phase
        self.data.update(values)
        self.write()

    def stopping(self):
        if self.data["phase"] in {"loading", "live", "local"}:
            self.update("stopping")

    def write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=self.path.parent, prefix=".meeting-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as state:
                json.dump(self.data, state, indent=2)
                state.write("\n")
            os.replace(temporary, self.path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _emit(u, sink, matcher, tracks_by_name, failed_slots):
    sink.poll_names()
    slot = (u.track, u.speaker)
    if matcher and slot not in failed_slots:
        try:
            matcher.finished(tracks_by_name[u.track], u)
        except Exception as error:
            failed_slots.add(slot)
            log(f"voice: disabled matching for {u.track}/{u.speaker} after error: {error}")
    sink.write(u)
    try:
        print(u.line(), flush=True)
    except BrokenPipeError:
        sys.stdout = open(os.devnull, "w")


def main(argv=None, *, session_now=None, session_path=None, after_sources=None,
         fallback_target=None, meeting_lifecycle=None, setup_before_sink=False):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "enroll":
        _enroll(argv[1:])
        return 0
    if argv and argv[0] == "name":
        ap = argparse.ArgumentParser(prog="hark name")
        ap.add_argument("speaker")
        ap.add_argument("name")
        ap.add_argument("--session", type=Path)
        args = ap.parse_args(argv[1:])
        _name_current(args.speaker, args.name, args.session)
        return 0
    if argv and argv[0] == "meeting":
        return _meeting(argv[1:])
    if argv and argv[0] == "mirror":
        return _resume_mirror(argv[1:])
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
    ap.add_argument("--mirror", help="append the transcript to HOST:PATH over SSH")
    ap.add_argument("-o", "--out", type=Path, help="write the transcript here instead of ~/.hark/sessions/")
    args = ap.parse_args(argv)
    if args.realtime and not args.file:
        ap.error("--realtime only applies to --file")
    mirror_target = None
    if args.mirror:
        host, separator, remote_path = args.mirror.partition(":")
        if not separator or not host or not remote_path:
            ap.error("--mirror must be HOST:PATH")
        mirror_target = host, remote_path

    stop = False

    def on_signal(*_):
        nonlocal stop
        stop = True
        if meeting_lifecycle:
            meeting_lifecycle.stopping()
        signal.signal(signal.SIGINT, signal.SIG_DFL)  # a second Ctrl-C quits hard

    def install_signal_handlers():
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, on_signal)

    if meeting_lifecycle:
        install_signal_handlers()

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
        if meeting_lifecycle:
            raise
        return 130
    tracks = [(src, Track(src.name, asr, diar, speaker_label=label, language=args.lang,
                          gap=args.gap))
              for src, label in sources]

    all_tracks = [track for _, track in tracks]
    now = session_now or datetime.now()
    out = (Path(session_path) if session_path else
           args.out.with_suffix(".txt") if args.out else _session_path(now, args.title))
    out.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np

    voices = {path.stem: np.load(path) for path in (HOME / "voices").glob("*.npy")} if (HOME / "voices").exists() else {}
    from .voice import VoiceMatcher

    if not voices:
        for track in all_tracks:
            track.audio = None
    tracks_by_name = {track.name: track for _, track in tracks}
    sink = None
    matcher = None
    mirror = None
    host_setup_failed = False
    failed_slots = set()

    def open_sink():
        nonlocal sink, matcher
        if not args.file and (not args.out or session_path):
            _point_current(out)
        sink = Sink(out, f"hark {now:%Y-%m-%d %H:%M} — {what}")
        if voices:
            matcher = VoiceMatcher(voices, sink)
        log(f"transcript → {out}")

    if not setup_before_sink:
        open_sink()
    if not meeting_lifecycle:
        install_signal_handlers()

    started = time.monotonic()
    try:
        for src, track in tracks:
            src.start()
            track.t0 = (datetime.combine(now.date(), datetime.min.time()) if args.file
                        else datetime.fromtimestamp(src.anchor))
        if meeting_lifecycle and sink:
            meeting_lifecycle.update(transcript=str(out))
        if after_sources and not stop:
            try:
                meeting_target = after_sources(out)
            except Exception as error:
                log(f"meeting: host setup failed ({error}); continuing with the local transcript at {out}")
                meeting_target = None
            if meeting_target is None:
                host_setup_failed = True
            elif meeting_target:
                host, _, remote_path = meeting_target.partition(":")
                mirror_target = (host, remote_path)
        if setup_before_sink:
            open_sink()
        if meeting_lifecycle:
            meeting_lifecycle.update(
                "local" if host_setup_failed else "live",
                transcript=str(out),
                mirror=f"{mirror_target[0]}:{mirror_target[1]}" if mirror_target else None,
            )
        if mirror_target:
            mirror = TranscriptMirror(out, *mirror_target)
            mirror.start()
        log("listening (Ctrl-C to stop)")
        while not stop:
            sink.poll_names()
            busy = False
            for src, track in tracks:
                samples = src.drain(limit=SAMPLE_RATE // 2)
                if samples.size:
                    busy = True
                    track.feed(samples)
            for u in flush_tracks(all_tracks):
                _emit(u, sink, matcher, tracks_by_name, failed_slots)
            if args.file and sources[0][0].done.is_set() and not busy and sources[0][0].queue.empty():
                break
            if not busy:
                time.sleep(0.05)
    finally:
        try:
            for src, _ in sources:
                src.stop()
            for src, track in tracks:
                track.feed(src.drain(limit=float("inf")), final=True)
            for u in flush_tracks(all_tracks, force=True):
                if sink:
                    _emit(u, sink, matcher, tracks_by_name, failed_slots)
        finally:
            if sink:
                sink.close(f"ended {datetime.now():%H:%M:%S}")
            if mirror:
                mirror.finish(timeout=30)
            elif host_setup_failed and fallback_target:
                log(f"meeting: after repairing the host setup, mirror the transcript with: "
                    f"{shlex.join(['hark', 'mirror', '--resume', str(out), fallback_target])}")
    audio = max(t.processed for _, t in tracks)
    wall = time.monotonic() - started
    log(f"done: {audio:.0f} s of audio in {wall:.0f} s (real-time factor {wall / max(audio, 1e-9):.2f})")


def _meeting(argv):
    ap = argparse.ArgumentParser(prog="hark meeting")
    ap.add_argument("--host", help="SSH alias; omit when the project is on this machine")
    ap.add_argument("--project", required=True, help="project checkout on the scribe machine")
    ap.add_argument("--under", required=True, help="parent fiber path in the felt store")
    ap.add_argument("--title", required=True)
    ap.add_argument("--agent", default="claude-opus")
    ap.add_argument("--store", default="~/loom", help="felt store on the scribe machine")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--room", action="store_true", help="mic only, diarized")
    mode.add_argument("--file", help="replay an audio file through the streaming path")
    ap.add_argument("--realtime", action="store_true", help="with --file: replay at real-time pace")
    ap.add_argument("--mic", help="input device name or index")
    ap.add_argument("--lang", help="ASR language, e.g. en-US, fr-FR")
    ap.add_argument("--latency", default="low", choices=["low", "very_low", "ultra_low"])
    ap.add_argument("--gap", type=float, default=3.0)
    args = ap.parse_args(argv)
    if args.host == "":
        ap.error("--host must be a non-empty SSH alias; omit it for local setup")
    if args.realtime and not args.file:
        ap.error("--realtime only applies to --file")
    if args.file and not Path(args.file).is_file():
        ap.error(f"audio file does not exist: {args.file}")

    now = datetime.now()
    slug = _meeting_slug(args.title)
    filename = f"{now:%Y-%m-%d_%H%M}_{slug}.txt"
    transcript_path = (f"~/.hark/meetings/{filename}" if args.host else
                       str(HOME / "meetings" / filename))
    fiber_id = f"{args.under}/meetings/{now:%Y-%m-%d-%H%M}-{slug}"
    lifecycle = MeetingLifecycle(
        title=args.title, host=args.host, project=args.project, under=args.under,
        store=args.store, fiber=fiber_id,
        started=now.astimezone().isoformat(timespec="seconds"),
    )

    try:
        when = now.astimezone().strftime("%Y-%m-%d %H:%M %Z")
        body = render_meeting_fiber(
            title=args.title, when=when, host=args.host or socket.gethostname().split(".")[0],
            transcript_path=transcript_path, mirrored=bool(args.host),
        )

        def prepare_host(out):
            setup = dict(
                project=args.project, store=args.store, fiber_id=fiber_id, under=args.under,
                title=args.title, agent=args.agent, transcript_path=transcript_path, body=body,
            )
            try:
                if args.host:
                    prepare_remote_meeting(host=args.host, **setup)
                else:
                    prepare_local_meeting(**setup)
            except Exception as error:
                failure = (f"SSH exited with status {error.returncode}"
                           if isinstance(error, subprocess.CalledProcessError) else str(error))
                if args.host:
                    recovery = (f"Repair {fiber_id} on {args.host}; hark will print the transcript "
                                "recovery command at shutdown")
                else:
                    recovery = f"Repair {fiber_id} locally"
                log(f"meeting: host setup failed ({failure}); capturing locally at {out}. {recovery}")
                return None
            log(f"meeting fiber: {fiber_id}")
            if args.host:
                log(f"watch transcript: ssh {args.host} 'tail -F {transcript_path}'")
                log(f"watch notes: ssh {args.host} 'felt -C {args.store} show {fiber_id}'")
                return f"{args.host}:{transcript_path}"
            log(f"watch transcript: tail -F {transcript_path}")
            log(f"watch notes: felt -C {args.store} show {fiber_id}")
            return ""

        capture = ["--title", args.title, "--latency", args.latency, "--gap", str(args.gap)]
        if args.room:
            capture.append("--room")
        if args.file:
            capture.extend(["--file", args.file])
        if args.realtime:
            capture.append("--realtime")
        if args.mic:
            capture.extend(["--mic", args.mic])
        if args.lang:
            capture.extend(["--lang", args.lang])
        result = main(
            capture, session_now=now,
            session_path=HOME / "meetings" / filename if not args.host else None,
            after_sources=prepare_host,
            fallback_target=f"{args.host}:{transcript_path}" if args.host else None,
            meeting_lifecycle=lifecycle,
            setup_before_sink=not bool(args.host),
        )
    except BaseException as error:
        lifecycle.update("failed", error=" ".join(str(error).split()) or type(error).__name__)
        raise
    lifecycle.update("ended")
    return result


def _resume_mirror(argv):
    ap = argparse.ArgumentParser(prog="hark mirror")
    ap.add_argument("--resume", action="store_true", required=True)
    ap.add_argument("local", type=Path)
    ap.add_argument("target", help="remote destination as HOST:PATH")
    args = ap.parse_args(argv)
    host, separator, remote_path = args.target.partition(":")
    if not separator or not host or not remote_path:
        ap.error("target must be HOST:PATH")
    mirror = TranscriptMirror(args.local, host, remote_path, resume=True)
    mirror.start()
    return 0 if mirror.finish() else 1


def _meeting_slug(title):
    slug = _slug(title)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "meeting"


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


def _name_current(speaker, name, session=None):
    if not speaker.startswith("S") or not speaker[1:].isdigit():
        raise SystemExit("speaker must be a label such as S1")
    txt = session.expanduser().resolve() if session else HOME / "current.txt"
    if not txt.exists():
        raise SystemExit(f"no session at {txt}")
    txt = txt.resolve()
    if any(line.startswith("# ended") for line in txt.read_text().splitlines()):
        raise SystemExit(f"session has ended: {txt}")
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
