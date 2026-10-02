"""Both ears share explicit meeting ownership and independent live signal handling."""

from itertools import groupby
import json
import os
from pathlib import Path
import signal
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from hark import cli


@pytest.fixture
def capture(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "HOME", tmp_path)
    monkeypatch.setenv("HARK_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "_default_ear", lambda: "local")
    monkeypatch.setattr(cli, "api_key", lambda: "local-only")
    monkeypatch.setattr(cli, "credits_left", lambda key: 900)
    monkeypatch.setattr(cli, "load_models", lambda latency: (None, None))
    monkeypatch.setattr(cli, "flush_tracks", lambda tracks, force=False: [])
    state = SimpleNamespace(watchers=[], starts=0, signal=signal.SIGTERM)
    real_watcher = cli.SignalWatcher

    def watcher(lifecycle, stop):
        instance = real_watcher(lifecycle, stop)
        state.watchers.append(instance)
        return instance

    monkeypatch.setattr(cli, "SignalWatcher", watcher)

    class Source:
        name = "phone"
        anchor = time.time()
        last_audio = last_sound = anchor

        def __init__(self, *args, **kwargs):
            self.done = threading.Event()
            self.queue = SimpleNamespace(empty=lambda: True)
            self.signalled = False

        def start(self):
            state.starts += 1
            if self.name == "file":
                self.done.set()

        def stop(self):
            pass

        def drain(self, limit):
            if self.name != "file" and not self.signalled:
                self.signalled = True
                def send_signal():
                    signal.pthread_sigmask(signal.SIG_BLOCK, state.watchers[0].wait_signals)
                    watcher = state.watchers[0]
                    if watcher.portable:
                        os.kill(os.getpid(), state.signal)
                    else:
                        signal.pthread_kill(watcher.thread.ident, state.signal)

                sender = threading.Thread(target=send_signal)
                sender.start()
                sender.join(timeout=1)
                assert state.watchers[0].stop.wait(1), "live signal did not stop capture promptly"
            return np.zeros(0, dtype=np.float32)

    class File(Source):
        name = "file"

    class Track:
        def __init__(self, name, *args, **kwargs):
            self.name, self.processed, self.sent_seconds, self.audio = name, 0.0, 0.0, None

        def start(self, **kwargs):
            pass

        def feed(self, *args, **kwargs):
            pass

        def check(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(cli, "PhoneSource", Source)
    monkeypatch.setattr(cli, "FileSource", File)
    monkeypatch.setattr(cli, "Track", Track)
    monkeypatch.setattr(cli, "GradiumTrack", Track)
    return state


EXISTING = b'{ "phase": "live", "pid": 12345, "sentinel": "keep these bytes" }\n'


@pytest.mark.parametrize("ear", ["local", "gradium"])
@pytest.mark.parametrize("output", ["default", "elsewhere", "sessions", "sibling", "symlink-escape", "meetings-symlink-escape"])
@pytest.mark.parametrize("number", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
def test_unowned_live_capture_preserves_existing_record(ear, output, number, capture, tmp_path):
    capture.signal = number
    record = tmp_path / "meeting.json"
    record.write_bytes(EXISTING)
    options = []
    if output != "default":
        out = {"elsewhere": tmp_path / "elsewhere" / "capture.wav",
               "sessions": tmp_path / "sessions" / "capture.txt",
               "sibling": tmp_path / "meetings-other" / "capture.txt",
               "symlink-escape": tmp_path / "meetings" / "escape" / "capture.txt",
               "meetings-symlink-escape": tmp_path / "meetings" / "capture.txt"}[output]
        if output == "symlink-escape":
            (tmp_path / "meetings").mkdir()
            (tmp_path / "outside").mkdir()
            (tmp_path / "meetings" / "escape").symlink_to(tmp_path / "outside", target_is_directory=True)
        elif output == "meetings-symlink-escape":
            (tmp_path / "outside").mkdir()
            (tmp_path / "meetings").symlink_to(tmp_path / "outside", target_is_directory=True)
        options = ["-o", str(out)]
    cli.main(["--ear", ear, "--phone"] + options)
    assert record.read_bytes() == EXISTING
    assert capture.starts == 1
    assert capture.watchers[0].lifecycle is None
    assert capture.watchers[0].stop.is_set()
    transcript = (tmp_path / "current.txt").resolve()
    assert "# ended " in transcript.read_text()
    assert (tmp_path / "current.jsonl").resolve() == transcript.with_suffix(".jsonl")
    if output != "default":
        assert transcript == out.with_suffix(".txt").resolve()
    else:
        assert transcript.parent == tmp_path / "sessions"


@pytest.mark.parametrize("ear", ["local", "gradium"])
@pytest.mark.parametrize("explicit_output", [False, True])
def test_unowned_live_capture_does_not_create_record(ear, explicit_output, capture, tmp_path):
    options = ["-o", str(tmp_path / "elsewhere.txt")] if explicit_output else []
    cli.main(["--ear", ear, "--phone"] + options)
    assert not (tmp_path / "meeting.json").exists()
    assert "# ended " in (tmp_path / "current.txt").read_text()


@pytest.mark.parametrize("ear", ["local", "gradium"])
@pytest.mark.parametrize("owner", ["launch-default", "launch-elsewhere", "meetings"])
def test_owned_live_capture_records_phases(ear, owner, capture, tmp_path, monkeypatch):
    snapshots = []
    replace = cli.os.replace

    def record_replace(source, target):
        if Path(target) == tmp_path / "meeting.json":
            snapshots.append(json.loads(Path(source).read_text()))
        replace(source, target)

    monkeypatch.setattr(cli.os, "replace", record_replace)
    (tmp_path / "meeting.json").write_bytes(EXISTING)
    options = ["--launch", "test"] if owner.startswith("launch") else []
    if owner != "launch-default":
        out = tmp_path / ("meetings" if owner == "meetings" else "elsewhere") / "capture.wav"
        options += ["-o", str(out)]
    cli.main(["--ear", ear, "--phone"] + options)
    assert [phase for phase, _ in groupby(s["phase"] for s in snapshots)] == [
        "loading", "live", "stopping", "ended"]
    state = json.loads((tmp_path / "meeting.json").read_text())
    assert state["launch"] == ("test" if owner.startswith("launch") else None)
    assert state["ear"]["name"] == ear
    assert state["error"] is None
    assert Path(state["transcript"]) == (tmp_path / "current.txt").resolve()


@pytest.mark.parametrize("ear", ["local", "gradium"])
def test_file_never_owns_record_even_with_launch_and_meetings_output(ear, capture, tmp_path):
    record = tmp_path / "meeting.json"
    record.write_bytes(EXISTING)
    out = tmp_path / "meetings" / "file.wav"
    cli.main(["--ear", ear, "--file", "fake.wav", "--launch", "test", "-o", str(out)])
    assert record.read_bytes() == EXISTING
    assert not capture.watchers
    assert not (tmp_path / "current.txt").exists()
    assert "# ended " in out.with_suffix(".txt").read_text()


def test_unowned_local_signal_during_loading_stops_before_source_start(capture, monkeypatch, tmp_path):
    record = tmp_path / "meeting.json"
    record.write_bytes(EXISTING)

    def loading(latency):
        watcher = capture.watchers[0]
        if watcher.portable:
            os.kill(os.getpid(), signal.SIGTERM)
        else:
            signal.pthread_kill(watcher.thread.ident, signal.SIGTERM)
        assert watcher.stop.wait(1)
        return None, None

    monkeypatch.setattr(cli, "load_models", loading)
    cli.main(["--ear", "local", "--phone"])
    assert capture.starts == 0
    assert record.read_bytes() == EXISTING
    assert "# ended " in (tmp_path / "current.txt").read_text()
