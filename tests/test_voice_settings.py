"""Launcher-owned voice settings are reloaded without restarting capture."""

import json
import threading
import pytest

from hark import cli
from hark.live_voice import RunQuiet
from test_cli import fake_live_capture


def test_missing_invalid_or_removed_settings_use_default(tmp_path):
    path = tmp_path / "voice.json"
    setting = RunQuiet(path)
    assert setting.read() == 5
    for value in (None, True, 0, -1, "3", 10**400, float("inf"), float("nan")):
        path.write_text(json.dumps({"run_quiet_seconds": value}))
        assert setting.read() == 5
    path.write_text("{")
    assert setting.read() == 5
    path.write_text("[]")
    assert setting.read() == 5
    path.write_text(json.dumps({"run_quiet_seconds": 1.5}))
    assert setting.read() == 1.5
    path.unlink()
    assert setting.read() == 5
    assert RunQuiet(None).read() == 5


def test_settings_cache_changes_and_missing_key(tmp_path):
    path = tmp_path / "voice.json"
    path.write_text(json.dumps({"run_quiet_seconds": 2}))
    setting = RunQuiet(path)
    assert setting.read() == setting.read() == 2
    replacement = tmp_path / "replacement"
    replacement.write_text(json.dumps({"run_quiet_seconds": 7, "unrelated": True}))
    replacement.replace(path)
    assert setting.read() == 7
    path.write_text("{}")
    assert setting.read() == 5


@pytest.mark.parametrize("override", [None, "3"])
def test_running_voice_capture_reloads_settings_and_gap_overrides(tmp_path, monkeypatch, override):
    fake_live_capture(monkeypatch, tmp_path)
    path = tmp_path / "voice.json"
    path.write_text(json.dumps({"run_quiet_seconds": 1.5}))
    signals, gaps, initial = [], [], []

    class Signals:
        def __init__(self, lifecycle, stop):
            self.stop = stop
            self.signal_written = threading.Event()
            self.signal_written.set()
            signals.append(self)
        def close(self):
            pass

    class VoiceTrack:
        def __init__(self, name, *args, **kwargs):
            self.name, self.gap = name, kwargs["gap"]
            self.audio = None
            self.processed = 0
            initial.append(self.gap)
        def feed(self, *args, **kwargs):
            pass

    def flush(tracks, force=False):
        if not force:
            gaps.append(tracks[0].gap)
            path.write_text(json.dumps({"run_quiet_seconds": 8}))
            if len(gaps) == 2:
                signals[0].stop.set()
        return []

    monkeypatch.setattr(cli, "SignalWatcher", Signals)
    monkeypatch.setattr("hark.live_voice.VoiceTrack", VoiceTrack)
    monkeypatch.setattr("hark.live_voice.load_refiner", lambda **_: lambda _: "finished")
    monkeypatch.setattr(cli, "flush_tracks", flush)
    argv = ["--voice", "--settings", str(path), "--pause-for", "none", "-o", str(tmp_path / "voice")]
    cli.main(argv + (["--gap", override] if override else []))
    assert initial == ([3] if override else [1.5])
    assert gaps == ([3, 3] if override else [1.5, 8])
