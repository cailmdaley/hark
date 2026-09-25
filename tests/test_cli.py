from datetime import datetime
import json
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

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

    assert not (tmp_path / "meeting.json").exists()
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

    def load_models(latency):
        state = json.loads((tmp_path / "meeting.json").read_text())
        assert state["phase"] == "loading"
        assert state["transcript"] is None
        events.append("models loaded")
        return None, None

    monkeypatch.setattr(cli, "load_models", load_models)
    monkeypatch.setattr(cli, "FileSource", Source)
    monkeypatch.setattr(cli, "Track", FakeTrack)
    monkeypatch.setattr(cli, "flush_tracks", lambda tracks, force=False: [])
    monkeypatch.setattr(cli, "prepare_remote_meeting", prepare)
    audio = tmp_path / "clip.wav"
    audio.touch()

    states = []
    replace = cli.os.replace

    def record_replace(source, target):
        if target == tmp_path / "meeting.json":
            states.append(json.loads(Path(source).read_text()))
        replace(source, target)

    monkeypatch.setattr(cli.os, "replace", record_replace)
    cli.main(["meeting", "--host", "candide", "--project", "/proj", "--under", "tools/hark",
              "--title", "Smoke", "--file", str(audio), "--realtime"])

    assert events == ["models loaded", "source opened", "host setup"]
    transcript = tmp_path / "sessions/2026-09-25_101530_smoke.txt"
    assert "# ended " in transcript.read_text()
    assert [state["phase"] for state in states] == ["loading", "loading", "local", "ended"]
    assert states[1]["transcript"] == str(transcript)
    assert states[2]["transcript"] == str(transcript)
    assert states[2]["mirror"] is None
    state = json.loads((tmp_path / "meeting.json").read_text())
    assert state["phase"] == "ended"
    assert state["host"] == "candide"
    assert state["fiber"] == "tools/hark/meetings/2026-09-25-1015-smoke"
    assert state["transcript"] == str(transcript)
    assert state["mirror"] is None
    output = capsys.readouterr().err
    assert "host setup failed (ssh exited 255)" in output
    assert "hark mirror --resume" in output
    assert "2026-09-25-1015-smoke" in output


def test_local_meeting_writes_to_scribe_path_and_stops_through_lifecycle(tmp_path, monkeypatch):
    import hark.cli as cli

    now = datetime(2026, 9, 25, 10, 15, 30)
    handlers = {}
    phases = []

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now

    class Source:
        name = "mic"
        anchor = 1.0

        def __init__(self, *args, **kwargs):
            pass

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

    def local_setup(**kwargs):
        state = json.loads((tmp_path / "meeting.json").read_text())
        assert state["phase"] == "loading"
        transcript = Path(kwargs["transcript_path"])
        assert transcript == tmp_path / "meetings/2026-09-25_1015_local-meeting.txt"
        assert not transcript.exists()
        assert "held 2026-09-25 10:15" in kwargs["body"]
        assert "on cail-mac." in kwargs["body"]
        assert "mirrored from" not in kwargs["body"]
        transcript.touch()
        return kwargs["fiber_id"], kwargs["transcript_path"]

    monkeypatch.setattr(cli, "datetime", Clock)
    monkeypatch.setattr(cli, "HOME", tmp_path)
    monkeypatch.setattr(cli, "socket", SimpleNamespace(gethostname=lambda: "cail-mac.local"))
    monkeypatch.setattr(cli, "load_models", lambda latency: (None, None))
    monkeypatch.setattr(cli, "MicSource", Source)
    monkeypatch.setattr(cli, "Track", FakeTrack)
    monkeypatch.setattr(cli, "flush_tracks", lambda tracks, force=False: [])
    monkeypatch.setattr(cli, "prepare_local_meeting", local_setup)
    monkeypatch.setattr(cli.signal, "signal", lambda sig, handler: handlers.__setitem__(sig, handler))
    replace = cli.os.replace

    def record_replace(source, target):
        assert Path(source).parent == Path(target).parent
        if target == tmp_path / "meeting.json":
            phases.append(json.loads(Path(source).read_text())["phase"])
        replace(source, target)

    monkeypatch.setattr(cli.os, "replace", record_replace)

    def stop_after_live(_):
        state = json.loads((tmp_path / "meeting.json").read_text())
        assert state["phase"] == "live"
        assert state["transcript"] == str(tmp_path / "meetings/2026-09-25_1015_local-meeting.txt")
        assert state["mirror"] is None
        handlers[cli.signal.SIGINT](cli.signal.SIGINT, None)

    monkeypatch.setattr(cli.time, "sleep", stop_after_live)
    cli.main(["meeting", "--project", "/project", "--under", "tools/hark",
              "--title", "Local Meeting", "--room"])

    transcript = tmp_path / "meetings/2026-09-25_1015_local-meeting.txt"
    assert "# ended " in transcript.read_text()
    assert (tmp_path / "current.txt").resolve() == transcript
    assert phases == ["loading", "live", "stopping", "ended"]
    state = json.loads((tmp_path / "meeting.json").read_text())
    assert set(state) == {"pid", "phase", "title", "host", "project", "under", "store",
                          "fiber", "started", "transcript", "mirror", "error"}
    assert state["pid"] == os.getpid()
    assert state["phase"] == "ended"
    assert state["title"] == "Local Meeting"
    assert state["host"] is None
    assert state["project"] == "/project"
    assert state["under"] == "tools/hark"
    assert state["store"] == "~/loom"
    assert state["fiber"] == "tools/hark/meetings/2026-09-25-1015-local-meeting"
    assert state["started"] == now.astimezone().isoformat(timespec="seconds")
    assert state["mirror"] is None
    assert state["transcript"] == str(transcript)
    assert state["error"] is None


def test_remote_setup_records_live_phase_and_mirror_target(tmp_path, monkeypatch):
    import hark.cli as cli

    class Source:
        name = "file"
        anchor = 1.0

        def __init__(self, *args, **kwargs):
            self.done = threading.Event()
            self.queue = SimpleNamespace(empty=lambda: True)

        def start(self):
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

    class FakeMirror:
        def __init__(self, local, host, remote_path):
            state = json.loads((tmp_path / "meeting.json").read_text())
            assert state["phase"] == "live"
            assert state["transcript"] == str(local)
            assert state["mirror"] == f"{host}:{remote_path}"
            self.host, self.remote_path = host, remote_path

        def start(self):
            pass

        def finish(self, timeout):
            return True

    monkeypatch.setattr(cli, "HOME", tmp_path)
    monkeypatch.setattr(cli, "FileSource", Source)
    monkeypatch.setattr(cli, "Track", FakeTrack)
    monkeypatch.setattr(cli, "load_models", lambda latency: (None, None))
    monkeypatch.setattr(cli, "flush_tracks", lambda tracks, force=False: [])
    monkeypatch.setattr(cli, "TranscriptMirror", FakeMirror)
    monkeypatch.setattr(cli.signal, "signal", lambda *args: None)
    lifecycle = cli.MeetingLifecycle(
        title="Remote", host="candide", project="/project", under="tools/hark",
        store="~/loom", fiber="tools/hark/meetings/remote", started="2026-09-25T10:15:30+02:00",
    )
    audio = tmp_path / "clip.wav"
    audio.touch()

    cli.main(["--file", str(audio)], session_now=datetime(2026, 9, 25, 10, 15, 30),
             after_sources=lambda out: "candide:~/.hark/meetings/remote.txt",
             meeting_lifecycle=lifecycle)

    state = json.loads((tmp_path / "meeting.json").read_text())
    assert state["phase"] == "live"
    assert state["mirror"] == "candide:~/.hark/meetings/remote.txt"


def test_meeting_exception_records_failed_state(tmp_path, monkeypatch):
    import hark.cli as cli

    now = datetime(2026, 9, 25, 10, 15, 30)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now

    monkeypatch.setattr(cli, "datetime", Clock)
    monkeypatch.setattr(cli, "HOME", tmp_path)
    monkeypatch.setattr(cli, "MicSource", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli.signal, "signal", lambda *args: None)
    monkeypatch.setattr(cli, "load_models", lambda latency: (_ for _ in ()).throw(
        RuntimeError("model loading failed\nwith details")))

    with pytest.raises(RuntimeError, match="model loading failed"):
        cli.main(["meeting", "--project", "/project", "--under", "tools/hark",
                  "--title", "Broken", "--room"])

    state = json.loads((tmp_path / "meeting.json").read_text())
    assert state["phase"] == "failed"
    assert state["transcript"] is None
    assert state["error"] == "model loading failed with details"
