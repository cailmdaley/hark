from datetime import datetime
import threading
import time
from types import SimpleNamespace

import numpy as np

from hark.cli import _emit
from hark.transcript import Sink, Utterance


def test_voice_matching_failure_logs_once_and_preserves_lines(tmp_path, monkeypatch):
    sink = Sink(tmp_path / "meeting.txt", "session")
    messages = []
    monkeypatch.setattr("hark.cli.log", messages.append)

    class BrokenMatcher:
        calls = 0

        def finished(self, track, utterance):
            self.calls += 1
            raise ValueError("malformed embedding")

    matcher = BrokenMatcher()
    track = object()
    failed_slots = set()
    for start, text in [(0, "first line"), (2, "second line")]:
        _emit(Utterance("system", "S1", start, start + 1, datetime.now(), text),
              sink, matcher, {"system": track}, failed_slots)
    sink.close("ended")

    transcript = (tmp_path / "meeting.txt").read_text()
    assert "first line" in transcript and "second line" in transcript
    assert matcher.calls == 1
    assert messages == ["voice: disabled matching for system/S1 after error: malformed embedding"]


def test_remote_mirror_failure_does_not_abort_local_capture(tmp_path, monkeypatch, capsys):
    import hark.cli as cli
    from hark.mirror import TranscriptMirror

    class Source:
        name = "file"
        anchor = time.time()

        def __init__(self, *args, **kwargs):
            self.done = threading.Event()
            self.done.set()
            self.queue = SimpleNamespace(empty=lambda: True)

        def start(self):
            pass

        def stop(self):
            pass

        def drain(self, limit):
            return np.zeros(0, dtype=np.float32)

    class FakeTrack:
        def __init__(self, name, *args, **kwargs):
            self.name, self.processed, self.audio = name, 0.0, None

        def feed(self, *args, **kwargs):
            pass

    remote = tmp_path / "remote.txt"
    remote.write_text("stale remote bytes\n")
    real_mirror = TranscriptMirror
    monkeypatch.setattr(cli, "TranscriptMirror", lambda out, host, path: real_mirror(
        out, host, path, command=lambda offset, reset=False: ["python", "-c", "import sys; sys.stdin.buffer.read()"],
        size_command=["python", "-c", f"print({remote.stat().st_size})"], backoff=0.01))
    monkeypatch.setattr(cli, "FileSource", Source)
    monkeypatch.setattr(cli, "load_models", lambda latency: (None, None))
    monkeypatch.setattr(cli, "Track", FakeTrack)
    monkeypatch.setattr(cli, "flush_tracks", lambda tracks, force=False: [])
    monkeypatch.setattr(cli, "HOME", tmp_path)
    audio = tmp_path / "x.wav"
    audio.touch()

    cli.main(["--file", str(audio), "--mirror", f"h:{remote}", "-o", str(tmp_path / "out")])

    transcript = (tmp_path / "out.txt").read_text()
    assert transcript.startswith("# hark ")
    assert "# ended " in transcript
    output = capsys.readouterr().err
    assert "mirror: stopped:" in output
    assert "hark mirror --resume" in output


def test_mirror_resume_verb_reopens_an_existing_target(tmp_path, monkeypatch):
    import hark.cli as cli

    seen = {}

    class FakeMirror:
        def __init__(self, local, host, remote_path, *, resume):
            seen.update(local=local, host=host, remote_path=remote_path, resume=resume)

        def start(self):
            return True

        def finish(self):
            return True

    monkeypatch.setattr(cli, "TranscriptMirror", FakeMirror)
    assert cli.main(["mirror", "--resume", str(tmp_path / "local.txt"), "candide:~/notes.txt"]) == 0
    assert seen == {"local": tmp_path / "local.txt", "host": "candide",
                    "remote_path": "~/notes.txt", "resume": True}


def test_meeting_opens_capture_before_best_effort_host_setup(tmp_path, monkeypatch, capsys):
    import hark.cli as cli

    events = []
    now = datetime(2026, 9, 25, 10, 15, 30)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now

    class Source:
        name = "recording"
        anchor = 1.0

        def __init__(self, *args, **kwargs):
            self.done = threading.Event()
            self.queue = SimpleNamespace(empty=lambda: True)

        def start(self):
            events.append("source opened")
            self.done.set()

        def stop(self):
            pass

        def drain(self, limit):
            return np.zeros(0, dtype=np.float32)

    class FakeTrack:
        def __init__(self, name, *args, **kwargs):
            self.name, self.processed, self.audio = name, 0.0, None

        def feed(self, *args, **kwargs):
            pass

    def prepare(**kwargs):
        events.append("host setup")
        raise RuntimeError("ssh exited 255")

    monkeypatch.setattr(cli, "datetime", Clock)
    monkeypatch.setattr(cli, "HOME", tmp_path)
    monkeypatch.setattr(cli, "load_models", lambda latency: (events.append("models loaded") or (None, None)))
    monkeypatch.setattr(cli, "FileSource", Source)
    monkeypatch.setattr(cli, "Track", FakeTrack)
    monkeypatch.setattr(cli, "flush_tracks", lambda tracks, force=False: [])
    monkeypatch.setattr(cli, "prepare_remote_meeting", prepare)
    audio = tmp_path / "clip.wav"
    audio.touch()

    cli.main(["meeting", "--host", "candide", "--project", "/proj", "--under", "tools/hark",
              "--title", "Smoke", "--file", str(audio), "--realtime"])

    assert events == ["models loaded", "source opened", "host setup"]
    transcript = tmp_path / "sessions/2026-09-25_101530_smoke.txt"
    assert "# ended " in transcript.read_text()
    output = capsys.readouterr().err
    assert "host setup failed (ssh exited 255)" in output
    assert "hark mirror --resume" in output
    assert "2026-09-25-1015-smoke" in output
