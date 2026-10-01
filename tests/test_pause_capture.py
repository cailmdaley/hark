"""Live-loop integration with synthetic sources and a deterministic watcher/clock."""

import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from hark.capture import SAMPLE_RATE, Source, on_pcm16_grid
from hark.pause import AudioProcess, ManualPause, PauseMonitor
from test_cli import read_wav


@pytest.mark.parametrize("room", [False, True])
@pytest.mark.parametrize("shutdown_pause", [None, "manual", "automatic"])
def test_live_modes_feed_and_save_identical_gated_pcm_and_leave_system_untouched(
        room, shutdown_pause, tmp_path, monkeypatch):
    from hark import cli

    clock = [100.0]
    monkeypatch.setattr("time.time", lambda: clock[0])
    stop_ref = []
    app_running = []
    feeds, tracks, first_feed = {}, {}, {}
    blocks = [on_pcm16_grid(np.full(1600, (i + 1) / 32, np.float32)) for i in range(16)]

    class Signals:
        def __init__(self, lifecycle, stop):
            self.lifecycle, self.stop = lifecycle, stop
            self.signal_written = threading.Event()
            self.signal_written.set()
            stop_ref.append(self)

        def close(self):
            pass

    class Mic(Source):
        def __init__(self, *args):
            super().__init__("mic")
            self.index = 0

        def _open(self):
            pass

        def drain(self, limit):
            if self.index == len(blocks):
                stop_ref[0].stop.set()
                stop_ref[0].lifecycle.stopping()
                return np.zeros(0, np.float32)
            block = blocks[self.index]
            self.index += 1
            clock[0] = self.anchor + self.index / 10
            self.last_audio = self.last_sound = clock[0]
            return block

        def stop(self):
            # A pause starts just before watcher shutdown, without another periodic poll.
            if self.name == "mic":
                if shutdown_pause == "manual":
                    ManualPause(tmp_path).set(True)
                elif shutdown_pause == "automatic":
                    app_running.append(True)

    class System(Mic):
        def __init__(self):
            super().__init__()
            self.name = "system"

        def drain(self, limit):
            # Clock advances only on the mic; system delivery is independent of its gate.
            if self.index == len(blocks):
                return np.zeros(0, np.float32)
            block = blocks[self.index]
            self.index += 1
            self.last_audio = self.last_sound = clock[0]
            return -block

    class Track:
        def __init__(self, name, *args, **kwargs):
            self.name, self.processed, self.audio = name, 0, None
            tracks[name] = self
            feeds[name] = []

        def feed(self, samples, final=False):
            if samples.size:
                first_feed.setdefault(self.name, clock[0])
            feeds[self.name].append(samples)
            self.processed += samples.size / SAMPLE_RATE

    class ScriptedMonitor(PauseMonitor):
        def start(self):
            # No thread or CoreAudio: the clock supplies periodic observations, while stop
            # uses the real post-join poll against the current manual/app state.
            self.reader = SimpleNamespace(read=lambda: [
                AudioProcess(1, 10, "aquavoice", bool(app_running))])
            self.poll()

        def feed(self, gate, samples, now, final=False):
            if not final:
                step = round((now - gate.source.anchor) * 10)
                self.events.extend(self.state.update(
                    now, 12 <= step < 14, ("aquavoice",) if 5 <= step < 8 else ()))
            return super().feed(gate, samples, now, final=final)

    monkeypatch.setattr(cli, "HOME", tmp_path)
    monkeypatch.setattr(cli, "SignalWatcher", Signals)
    monkeypatch.setattr(cli, "MicSource", Mic)
    monkeypatch.setattr(cli, "SystemSource", System)
    monkeypatch.setattr(cli, "Track", Track)
    monkeypatch.setattr(cli, "PauseMonitor", ScriptedMonitor)
    monkeypatch.setattr(cli, "load_models", lambda *_: (None, None))
    monkeypatch.setattr(cli, "flush_tracks", lambda *_, **__: [])
    out = tmp_path / "session.txt"
    cli.main((["--room"] if room else []) + ["-o", str(out)])

    original = np.concatenate(blocks)
    expected = original.copy()
    expected[3200:17600] = 0  # automatic [0.5 - lookback, 0.8 + tail)
    expected[14400:22400] = 0  # manual [1.2 - lookback, 1.4)
    if shutdown_pause:
        expected[20800:] = 0  # final observed pause at 1.6 mutes the held [1.3, 1.6) onset
    heard = np.concatenate(feeds["mic"])
    assert np.array_equal(heard, expected)
    assert np.array_equal(read_wav(out.with_suffix(".mic.wav")), heard)
    assert tracks["mic"].t0.timestamp() == 100 and tracks["mic"].processed == pytest.approx(1.6)
    assert first_feed["mic"] == pytest.approx(100.4)
    if not room:
        assert np.array_equal(np.concatenate(feeds["system"]), -original)
        assert np.array_equal(read_wav(out.with_suffix(".system.wav")), -original)
        assert tracks["system"].t0.timestamp() == 100
        assert first_feed["system"] == pytest.approx(100.1)
    text = out.read_text()
    assert text.count("# paused\n") == 1 + (shutdown_pause == "manual")
    assert text.count("# resumed at ") == 1
    assert "aquavoice" not in text and " lost at " not in text
    records = [json.loads(line)["pause"] for line in out.with_suffix(".jsonl").read_text().splitlines()]
    expected_events = [("automatic", "paused"), ("automatic", "resumed"),
                       ("manual", "paused"), ("manual", "resumed")]
    if shutdown_pause:
        expected_events.append((shutdown_pause, "paused"))
    assert [(r["reason"], r["state"]) for r in records] == expected_events
    log = out.with_suffix(".log").read_text()
    assert "mic paused: automatic (aquavoice)" in log
    assert "mic resumed: automatic (aquavoice)" in log
