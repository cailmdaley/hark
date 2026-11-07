"""Live-loop integration with synthetic sources and a deterministic watcher/clock."""

import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from hark.capture import SAMPLE_RATE, Source, on_pcm16_grid
from hark.pause import AudioProcess, ManualPause, PauseMonitor
from test_cli import read_wav


@pytest.mark.parametrize("mode,options,system_muted", [
    ("desktop", [], False),
    ("desktop", ["--mic-only"], True),
    ("room", ["--room"], False),
    ("voice", ["--voice"], True),
    ("voice", ["--voice", "--system-audio"], False),
])
@pytest.mark.parametrize("shutdown_pause", [None, "manual", "automatic"])
def test_live_modes_feed_and_save_identical_gated_pcm_and_system_reference(
        mode, options, system_muted, shutdown_pause, tmp_path, monkeypatch):
    room = mode == "room"
    from hark import cli

    clock = [100.0]
    monkeypatch.setattr("time.time", lambda: clock[0])
    stop_ref = []
    app_running = []
    feeds, tracks, first_feed, aec_references = {}, {}, {}, []
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
            self.aqua = None  # Scripted accepted dictation; lifecycle-log parsing has its own tests.
            self.reader = SimpleNamespace(read=lambda idle=False: [
                AudioProcess(1, 10, "aquavoice", bool(app_running))])
            self.system_manual.set(self.system_paused)
            self.poll()

        def feed(self, gate, samples, now, final=False):
            if not final:
                step = round((now - gate.source.anchor) * 10)
                self.events.extend(self.state.update(
                    now, 12 <= step < 14, ("aquavoice",) if 5 <= step < 8 else ()))
            return super().feed(gate, samples, now, final=final)

    monkeypatch.setattr(cli, "_default_ear", lambda: "local")
    monkeypatch.setattr(cli, "_device_capture_available", lambda: True)
    monkeypatch.setattr(cli, "_system_capture_available", lambda: True)
    monkeypatch.setattr(cli, "HOME", tmp_path)
    monkeypatch.setattr(cli, "SignalWatcher", Signals)
    monkeypatch.setattr(cli, "MicSource", Mic)
    monkeypatch.setattr(cli, "SystemSource", System)
    monkeypatch.setattr(cli, "Track", Track)
    monkeypatch.setattr("hark.live_voice.VoiceTrack", Track)
    monkeypatch.setattr("hark.live_voice.load_refiner", lambda **kwargs: None)
    monkeypatch.setattr(cli, "PauseMonitor", ScriptedMonitor)
    from hark import aec

    class Aec:
        def __init__(self, **kwargs):
            pass

        def feed(self, mic, system, final=False):
            aec_references.append(system.copy())
            return mic

    monkeypatch.setattr(aec, "EchoCanceller", Aec)
    monkeypatch.setattr(cli, "load_models", lambda *_: (None, None))
    monkeypatch.setattr(cli, "flush_tracks", lambda *_, **__: [])
    out = tmp_path / "session.txt"
    cli.main(options + ["--ear", "local", "--launch", "test", "--keep-audio", "-o", str(out)])

    original = np.concatenate(blocks)
    expected = original.copy()
    expected[3200:17600] = 0  # automatic [0.5 - lookback, 0.8 + tail)
    expected[14400:22400] = 0  # manual [1.2 - lookback, 1.4)
    if shutdown_pause:
        expected[20800:] = 0  # final observed pause at 1.6 mutes the held [1.3, 1.6) onset
    heard = np.concatenate(feeds["mic"])
    assert heard.size == expected.size
    if room:
        assert np.array_equal(heard, expected)
    else:
        assert not heard[expected == 0].any()  # AEC tails cannot unmute a paused sample.
    assert np.array_equal(read_wav(out.with_suffix(".mic.wav")), heard)
    assert tracks["mic"].t0.timestamp() == 100 and tracks["mic"].processed == pytest.approx(1.6)
    assert first_feed["mic"] == pytest.approx(100.4)
    if not room:
        expected_system = np.zeros_like(original) if system_muted else -original
        assert np.array_equal(np.concatenate(feeds["system"]), expected_system)
        assert np.array_equal(read_wav(out.with_suffix(".system.wav")), expected_system)
        assert np.array_equal(np.concatenate(aec_references), -original)
        assert tracks["system"].t0.timestamp() == 100
        assert first_feed["system"] == pytest.approx(100.4)
    text = out.read_text()
    assert text.count("# paused\n") == 1 + (shutdown_pause == "manual")
    assert text.count("# resumed at ") == 1
    assert "aquavoice" not in text and " lost at " not in text
    records = [json.loads(line)["pause"] for line in out.with_suffix(".jsonl").read_text().splitlines()]
    expected_events = [("automatic", "paused"), ("automatic", "resumed"),
                       ("manual", "paused"), ("manual", "resumed")]
    if system_muted:
        expected_events.insert(0, ("manual", "paused"))
    if shutdown_pause:
        expected_events.append((shutdown_pause, "paused"))
    assert [(r["reason"], r["state"]) for r in records] == expected_events
    log = out.with_suffix(".log").read_text()
    assert "mic paused: automatic (aquavoice)" in log
    assert "mic resumed: automatic (aquavoice)" in log


def test_voice_system_pause_and_resume_live_leave_aec_reference_untouched(tmp_path, monkeypatch):
    from hark import aec, cli

    clock = [100.0]
    monkeypatch.setattr("time.time", lambda: clock[0])
    states, references, heard = [], [], []
    blocks = [np.full(1600, .25, np.float32) for _ in range(20)]
    signals = []

    class Signals:
        def __init__(self, lifecycle, stop):
            self.lifecycle, self.stop = lifecycle, stop
            self.signal_written = threading.Event()
            self.signal_written.set()
            signals.append(self)
        def close(self): pass

    class Mic(Source):
        def __init__(self, *args):
            super().__init__("mic")
            self.index = 0
        def _open(self): pass
        def drain(self, limit):
            if self.index == len(blocks):
                signals[0].stop.set()
                signals[0].lifecycle.stopping()
                return np.zeros(0, np.float32)
            states.append(json.loads((tmp_path / "meeting.json").read_text())["system_audio"])
            block = blocks[self.index]
            self.index += 1
            clock[0] = self.anchor + self.index / 10
            return block

    class System(Mic):
        def __init__(self):
            super().__init__()
            self.name = "system"
        def drain(self, limit):
            if self.index == len(blocks): return np.zeros(0, np.float32)
            block = blocks[self.index]
            self.index += 1
            return block

    class Track:
        def __init__(self, name, *args, **kwargs):
            self.name, self.processed, self.audio = name, 0, None
        def feed(self, samples, final=False):
            if self.name == "system": heard.append(samples.copy())

    class Monitor(PauseMonitor):
        def start(self):
            self.system_manual.set(self.system_paused)
            self.poll()
        def feed(self, gate, samples, now, final=False):
            if not final:
                step = round((now - gate.source.anchor) * 10)
                if step == 6: cli.main(["resume", "--system"])
                if step == 14: cli.main(["pause", "--system"])
                self.poll()
            return super().feed(gate, samples, now, final=final)

    class Aec:
        def __init__(self, **kwargs): pass
        def feed(self, mic, system, final=False):
            references.append(system.copy())
            return mic

    monkeypatch.setattr(cli, "HOME", tmp_path)
    monkeypatch.setattr(cli, "_default_ear", lambda: "local")
    monkeypatch.setattr(cli, "_device_capture_available", lambda: True)
    monkeypatch.setattr(cli, "_system_capture_available", lambda: True)
    monkeypatch.setattr(cli, "SignalWatcher", Signals)
    monkeypatch.setattr(cli, "MicSource", Mic)
    monkeypatch.setattr(cli, "SystemSource", System)
    monkeypatch.setattr(cli, "PauseMonitor", Monitor)
    monkeypatch.setattr(cli, "load_models", lambda *_: (None, None))
    monkeypatch.setattr(cli, "flush_tracks", lambda *_, **__: [])
    monkeypatch.setattr("hark.live_voice.VoiceTrack", Track)
    monkeypatch.setattr("hark.live_voice.load_refiner", lambda **kwargs: None)
    monkeypatch.setattr(aec, "EchoCanceller", Aec)
    cli.main(["--voice", "--pause-for", "none", "-o", str(tmp_path / "voice")])

    assert states == [False] * 6 + [True] * 8 + [False] * 6
    assert np.array_equal(np.concatenate(references), np.concatenate(blocks))
    system = np.concatenate(heard)
    assert np.array_equal(system[:9600], np.zeros(9600))
    assert np.array_equal(system[9600:17600], np.full(8000, .25))
    assert np.array_equal(system[17600:], np.zeros(14400))
    assert json.loads((tmp_path / "meeting.json").read_text())["system_audio"] is False


def test_system_pause_flag_is_independent_and_reported(tmp_path, monkeypatch, capsys):
    from hark import cli
    from hark.pause import PauseMonitor

    monkeypatch.setattr(cli, "HOME", tmp_path)
    meeting = tmp_path / "meeting.json"
    lifecycle = cli.MeetingLifecycle(title=None, started="now", transcript="x", mirror=None,
                                     system_audio=True)
    monitor = PauseMonitor(tmp_path, [])
    assert cli.main(["pause", "--system"]) == 0
    assert capsys.readouterr().out == "system audio manual pause: paused\n"
    assert (tmp_path / "system-paused").exists()
    assert not (tmp_path / "paused").exists()
    monitor.poll()
    lifecycle.update(system_audio=not monitor.system_manual.read())
    assert json.loads(meeting.read_text())["system_audio"] is False
    assert cli.main(["pause", "--system", "--status"]) == 0
    assert capsys.readouterr().out == "system audio manual pause: paused\n"
    assert cli.main(["resume", "--system"]) == 0
    assert capsys.readouterr().out == "system audio manual pause: resumed\n"
    assert not (tmp_path / "system-paused").exists()
    monitor.poll()
    lifecycle.update(system_audio=not monitor.system_manual.read())
    lifecycle.update("live")
    lifecycle.update("stopping")
    lifecycle.update("ended")
    assert json.loads(meeting.read_text())["system_audio"] is True
    assert json.loads(meeting.read_text())["phase"] == "ended"
