"""hark — capture a conversation, append one speaker-labelled line per utterance.

    hark                 call: mic is "me", system audio (Zoom…) diarized as S1…S8
    hark --room          in person: the mic alone, diarized
    hark --phone         in person, a phone as the mic: PCM into ~/.hark/phone.sock, diarized
    hark --file x.wav    transcribe a file through the same streaming path

    hark name S2 "Ada" [--session PATH]     name a diarized speaker in the live session
    hark enroll NAME [--seconds 30 | --file x.wav] [--mic DEV]   record a voiceprint
    hark mirror --resume LOCAL HOST:PATH    finish a mirror that didn't complete

Live capture keeps each track beside the transcript as <stem>.mic.wav and
<stem>.system.wav (or <stem>.phone.wav) (16 kHz mono 16-bit PCM, t = 0 at the track's first sample,
replayable with --file); --no-save-audio opts out. At startup, saved audio under
~/.hark/meetings and ~/.hark/sessions older than 14 days is deleted; transcripts
are kept.

The live transcript is ~/.hark/current.txt (a symlink to the transcript file);
follow it with `tail -F`. A JSONL sidecar sits beside it, and <stem>.log keeps
hark's own log, timestamped, with a heartbeat per live source each minute.
A live source that goes quiet for 90 s gets a `# … lost at …` line.
"""

import argparse
import json
import os
import shlex
import signal
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

from .capture import (SAMPLE_RATE, FileSource, MicSource, PhoneSource, SystemSource, WavRecorder,
                      log, log_to)
from .health import QuietWatch
from .masking import MaskPolicy
from .mirror import TranscriptMirror
from .pause import (AUTO_RESUME_TAIL_SEC, DEFAULT_PAUSE_FOR, ManualPause, MicGate,
                    PauseMonitor, parse_patterns, read_processes)
from .transcript import Sink, Track, flush_tracks, load_models, numbered

HOME = Path(os.environ.get("HARK_DIR", Path.home() / ".hark")).expanduser().resolve()
AUDIO_RETENTION_DAYS = 14


class MeetingLifecycle:
    TRANSITIONS = {
        "loading": {"live", "stopping", "failed"},
        "live": {"stopping", "failed"},
        "stopping": {"ended", "failed"},
        "ended": set(),
        "failed": set(),
    }

    def __init__(self, *, title, started, transcript, mirror, launch=None, phone=None):
        self.path = HOME / "meeting.json"
        self.lock = threading.Lock()
        self.data = {
            "pid": os.getpid(), "phase": "loading", "title": title,
            "started": started, "transcript": transcript, "mirror": mirror,
            "launch": launch, "phone": phone, "error": None,
        }
        self._write()

    def update(self, phase=None, **values):
        with self.lock:
            changed = False
            current = self.data["phase"]
            if phase and phase in self.TRANSITIONS[current]:
                self.data["phase"] = phase
                changed = True
            for key, value in values.items():
                if self.data[key] != value:
                    self.data[key] = value
                    changed = True
            if changed:
                self._write()

    def stopping(self):
        self.update("stopping")

    def _write(self):
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


class SignalWatcher:
    """Consume live-capture signals outside the model-loading thread."""

    def __init__(self, lifecycle, stop):
        self.signals = {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}
        self.wake_signal = signal.SIGUSR1
        self.wait_signals = self.signals | {self.wake_signal}
        self.stop = stop
        self.lifecycle = lifecycle
        self.closed = threading.Event()
        self.signal_written = threading.Event()
        self.interrupts = 0
        self.previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, self.wait_signals)
        self.thread = threading.Thread(target=self._watch, name="hark-signals", daemon=True)
        try:
            self.thread.start()
        except BaseException:
            signal.pthread_sigmask(signal.SIG_SETMASK, self.previous_mask)
            raise

    def _watch(self):
        while not self.closed.is_set():
            number = signal.sigwait(self.wait_signals)
            if number == self.wake_signal:
                continue
            if number == signal.SIGINT:
                self.interrupts += 1
                if self.interrupts > 1:
                    os._exit(128 + number)
            if not self.stop.is_set():
                self.stop.set()
                try:
                    self.lifecycle.stopping()
                finally:
                    self.signal_written.set()

    def close(self):
        self.closed.set()
        signal.pthread_kill(self.thread.ident, self.wake_signal)
        self.thread.join()
        signal.pthread_sigmask(signal.SIG_SETMASK, self.previous_mask)


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


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in {"pause", "resume"}:
        return _manual_pause(argv)
    if argv and argv[0] == "processes":
        ap = argparse.ArgumentParser(prog="hark processes", description="List CoreAudio process objects")
        ap.parse_args(argv[1:])
        from dataclasses import asdict

        for process in read_processes():
            print(json.dumps(asdict(process), ensure_ascii=False))
        return 0
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
    if argv and argv[0] == "mirror":
        return _resume_mirror(argv[1:])
    ap = argparse.ArgumentParser(prog="hark", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--room", action="store_true", help="mic only, diarized")
    mode.add_argument("--phone", action="store_true",
                      help="a phone as the mic: listen for s16le 16 kHz PCM on ~/.hark/phone.sock, diarized")
    mode.add_argument("--file", help="transcribe an audio file instead of live input")
    ap.add_argument("--realtime", action="store_true", help="with --file: replay at real-time pace")
    ap.add_argument("--mic", help="input device name or index (default: system default)")
    ap.add_argument("--pause-for", default=DEFAULT_PAUSE_FOR,
                    help="mute mic for these comma-separated bundle ID substrings "
                         "(case-insensitive; default: aquavoice; none disables app detection)")
    ap.add_argument("--lang", default=None, help="ASR language, e.g. en-US, fr-FR (default: auto)")
    ap.add_argument("--latency", default="low", choices=["low", "very_low", "ultra_low"],
                    help="diarizer buffer: low=1.04 s (default), very_low=0.64 s, ultra_low=0.32 s")
    ap.add_argument("--speaker-mask", default="exclusive", choices=["shared", "exclusive"],
                    help="who hears an 80 ms frame two speakers hold: exclusive=the more "
                         "probable speaker only (default), shared=both ASR streams")
    ap.add_argument("--gap", type=float, default=3.0,
                    help="seconds of silence that end a turn when nobody else takes over")
    ap.add_argument("--title", help="appended to the session filename")
    ap.add_argument("--mirror", help="append the transcript to HOST:PATH over SSH")
    ap.add_argument("-o", "--out", type=Path, help="write the transcript here instead of ~/.hark/sessions/")
    ap.add_argument("--launch", help="launcher's id for this recording, echoed into meeting.json")
    ap.add_argument("--no-save-audio", dest="save_audio", action="store_false",
                    help="don't keep the live tracks as <stem>.<track>.wav beside the transcript")
    args = ap.parse_args(argv)
    if args.realtime and not args.file:
        ap.error("--realtime only applies to --file")
    try:
        pause_patterns = parse_patterns(args.pause_for)
    except ValueError as error:
        ap.error(str(error))

    mirror_target = None
    if args.mirror:
        host, separator, remote_path = args.mirror.partition(":")
        if not separator or not host or not remote_path:
            ap.error("--mirror must be HOST:PATH")
        mirror_target = host, remote_path

    now = datetime.now()
    out = (args.out.with_suffix(".txt").expanduser().resolve() if args.out
           else _session_path(now, args.title))
    live = not bool(args.file)
    phone_socket = HOME / "phone.sock" if args.phone else None
    if live:
        expire_audio([HOME / "meetings", HOME / "sessions"])
    lifecycle = (MeetingLifecycle(
        title=args.title, started=now.astimezone().isoformat(timespec="seconds"),
        transcript=str(out),
        mirror=f"{mirror_target[0]}:{mirror_target[1]}" if mirror_target else None,
        launch=args.launch, phone=str(phone_socket) if phone_socket else None,
    ) if live else None)
    stop = threading.Event()
    watcher = None

    def on_signal(*_):
        if not stop.is_set():
            stop.set()
        signal.signal(signal.SIGINT, signal.SIG_DFL)  # a second Ctrl-C quits hard

    def install_signal_handlers():
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, on_signal)

    what = (f"file {args.file}" if args.file else
            "room: mic diarized as S1…" if args.room else
            "phone: the phone's mic diarized as S1…" if args.phone else
            "call: me = mic, S1… = system audio")
    sources = []
    tracks = []
    all_tracks = []
    started_sources = []
    started_tracks = []
    sink = matcher = mirror = watch = None
    recorders = {}
    mic_gates = {}
    pause_monitor = None
    mirror_complete = True
    tracks_by_name = {}
    failed_slots = set()
    voices = {}

    def open_sink():
        nonlocal sink, matcher
        out.parent.mkdir(parents=True, exist_ok=True)
        log_to(out.with_suffix(".log"))
        if live:
            _point_current(out)
        sink = Sink(out, f"hark {now:%Y-%m-%d %H:%M} — {what}")
        if voices:
            from .voice import VoiceMatcher

            matcher = VoiceMatcher(voices, sink)
        log(f"transcript → {out}")

    try:
        if lifecycle:
            watcher = SignalWatcher(lifecycle, stop)
        try:
            if not stop.is_set():
                if args.file:
                    sources = [(FileSource(args.file, realtime=args.realtime), numbered)]
                elif args.phone:
                    sources = [(PhoneSource(phone_socket), numbered)]
                elif args.room:
                    sources = [(MicSource(_device(args.mic)), numbered)]
                else:
                    sources = [(MicSource(_device(args.mic)), lambda _: "me"),
                               (SystemSource(), numbered)]

            log("loading models…")
            if stop.is_set():
                asr, diar = None, None
            else:
                try:
                    asr, diar = load_models(args.latency)
                except KeyboardInterrupt:
                    if lifecycle:
                        raise
                    return 130

            if not stop.is_set():
                mask = MaskPolicy.parse(args.speaker_mask) if args.speaker_mask != "shared" else None
                tracks = [(src, Track(src.name, asr, diar, speaker_label=label,
                                      language=args.lang, gap=args.gap, mask=mask))
                          for src, label in sources]
                all_tracks = [track for _, track in tracks]
                import numpy as np

                voices = ({path.stem: np.load(path) for path in (HOME / "voices").glob("*.npy")}
                          if (HOME / "voices").exists() else {})
                if not voices:
                    for track in all_tracks:
                        track.audio = None
                tracks_by_name = {track.name: track for _, track in tracks}

            open_sink()
            if live and args.save_audio:
                recorders = {src: WavRecorder(out.with_suffix(f".{src.name}.wav")) for src, _ in tracks}
                for recorder in recorders.values():
                    log(f"audio → {recorder.path}")
            if not lifecycle:
                install_signal_handlers()

            if live and not stop.is_set():
                pause_monitor = PauseMonitor(HOME, pause_patterns)
                pause_monitor.start()
                _pause_events(pause_monitor, sink)
            capture_started = time.monotonic()
            for src, track in tracks:
                if stop.is_set():
                    break
                src.start()
                started_sources.append(src)
                started_tracks.append((src, track))
                if live and src.name == "mic":
                    mic_gates[src] = MicGate(src.anchor)
                track.t0 = (datetime.combine(now.date(), datetime.min.time()) if args.file
                            else datetime.fromtimestamp(src.anchor))
            if lifecycle and len(started_sources) == len(sources) and not stop.is_set():
                lifecycle.update("live")

            if mirror_target:
                mirror = TranscriptMirror(out, *mirror_target)
                mirror.start()
            if not stop.is_set():
                log("listening (Ctrl-C to stop)")
            watch = QuietWatch() if live else None
            spoken, said = {}, {}  # per track: wall end of the last utterance, utterances this minute
            beat = time.monotonic() + 60
            while not stop.is_set():
                sink.poll_names()
                if pause_monitor:
                    _pause_events(pause_monitor, sink, watch)
                busy = False
                for src, track in started_tracks:
                    samples = src.drain(limit=SAMPLE_RATE // 2)
                    busy = busy or bool(samples.size)
                    if src in mic_gates:
                        samples = pause_monitor.feed(mic_gates[src], samples, time.time())
                    if samples.size:
                        if src in recorders:
                            recorders[src].write(samples)
                        track.feed(samples)
                for u in flush_tracks(all_tracks):
                    _emit(u, sink, matcher, tracks_by_name, failed_slots)
                    spoken[u.track] = max(spoken.get(u.track, 0.0), u.wall_span[1])
                    said[u.track] = said.get(u.track, 0) + 1
                if watch:
                    health_now = time.time()
                    intentional = pause_monitor.health_since(health_now)
                    if intentional is not None:
                        watch.suppress("mic", intentional)
                    for event in watch.check(health_now, {src.name: (src.last_audio, src.last_sound)
                                                         for src in started_sources}, spoken):
                        sink.source(event)
                    if time.monotonic() >= beat:
                        beat += 60
                        _heartbeat(started_sources, said)
                if (args.file and sources[0][0].done.is_set() and not busy
                        and sources[0][0].queue.empty()):
                    break
                if not busy:
                    time.sleep(0.05)
        finally:
            try:
                for src in started_sources:
                    src.stop()
                if pause_monitor:
                    pause_monitor.stop()
                    _pause_events(pause_monitor, sink, watch)
                for src, track in started_tracks:
                    rest = src.drain(limit=float("inf"))
                    if src in mic_gates:
                        rest = pause_monitor.feed(mic_gates[src], rest, time.time(), final=True)
                    if src in recorders:
                        recorders[src].write(rest)
                    track.feed(rest, final=True)
                for u in flush_tracks(all_tracks, force=True):
                    if sink:
                        _emit(u, sink, matcher, tracks_by_name, failed_slots)
            finally:
                try:
                    for recorder in recorders.values():
                        recorder.close()
                    if sink:
                        for track, (since, cause) in (watch.open.items() if watch else ()):
                            log(f"{track}: still silent at the end ({cause} since "
                                f"{datetime.fromtimestamp(since):%H:%M:%S})")
                        sink.close(f"ended {datetime.now():%H:%M:%S}")
                finally:
                    if mirror:
                        mirror_complete = mirror.finish(timeout=30)

        if watcher and stop.is_set():
            watcher.signal_written.wait()
        audio = max((track.processed for _, track in tracks), default=0.0)
        wall = time.monotonic() - capture_started
        log(f"done: {audio:.0f} s of audio in {wall:.0f} s (real-time factor {wall / max(audio, 1e-9):.2f})")
        if lifecycle:
            lifecycle.update("ended", error=None if mirror_complete else (
                "mirror incomplete; resume with: "
                + shlex.join(["hark", "mirror", "--resume", str(out), lifecycle.data["mirror"]])))
    except BaseException as error:
        if watcher and stop.is_set():
            watcher.signal_written.wait()
        if lifecycle:
            lifecycle.update("failed", error=" ".join(str(error).split()) or type(error).__name__)
        raise
    finally:
        if pause_monitor:
            pause_monitor.stop()
        if watcher:
            watcher.close()
        log_to(None)


def _manual_pause(argv):
    command = argv[0]
    ap = argparse.ArgumentParser(prog=f"hark {command}")
    if command == "pause":
        ap.add_argument("--status", action="store_true", help="show the persistent manual pause state")
    args = ap.parse_args(argv[1:])
    state = ManualPause(HOME)
    if not getattr(args, "status", False):
        state.set(command == "pause")
    print("manual pause: " + ("paused" if state.read() else "resumed"))
    return 0


def _pause_events(monitor, sink, watch=None):
    for event in monitor.take_events():
        state = "paused" if event.paused else "resumed"
        detail = f" ({', '.join(event.bundles)})" if event.bundles else ""
        log(f"mic {state}: {event.reason}{detail} at "
            f"{datetime.fromtimestamp(event.at):%H:%M:%S}")
        sink.pause(event)
        if watch:
            tail = AUTO_RESUME_TAIL_SEC if event.reason == "automatic" and not event.paused else 0
            watch.suppress("mic", event.at + tail)


def _heartbeat(sources, said):
    """One log-file line per live source for the past minute."""
    for src in sources:
        stats = src.take_stats()
        log(f"heartbeat {src.name}: {stats['device'] / SAMPLE_RATE:.1f} s from the device, "
            f"{stats['padded'] / SAMPLE_RATE:.1f} s padded, peak {stats['peak']:.4f}, "
            f"{said.pop(src.name, 0)} utterances", file_only=True)


def expire_audio(roots, days=AUDIO_RETENTION_DAYS, now=None):
    """Delete saved track audio (*.wav) under `roots` last modified over `days` ago."""
    cutoff = (time.time() if now is None else now) - days * 86400
    removed = [path for root in roots if root.is_dir() for path in sorted(root.rglob("*.wav"))
               if path.is_file() and path.stat().st_mtime < cutoff]
    for path in removed:
        try:
            path.unlink()
            log(f"expired audio older than {days} days: {path}")
        except OSError as error:
            log(f"audio: could not expire {path}: {error}")
    return removed


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
                  "name": {"speaker": speaker, "as": None if name == speaker else name}}
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
