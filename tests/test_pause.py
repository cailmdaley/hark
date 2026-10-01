"""Pause tests use synthetic PCM, fake HAL properties, and a fake clock, never a mic or HAL."""

import ctypes
import json
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from hark.capture import SAMPLE_RATE, on_pcm16_grid
from hark.health import Quiet, QuietWatch
from hark.pause import (AUTO_RESUME_TAIL_SEC, MIC_LOOKBACK_SEC, AudioProcess,
                        CoreAudioProcesses, ManualPause, MicGate, PauseMonitor,
                        PauseState, parse_patterns, watched_bundles)
from hark.transcript import Sink


def pcm(seconds):
    return on_pcm16_grid(np.linspace(0.1, 0.8, round(seconds * SAMPLE_RATE), dtype=np.float32))


def samples_at(seconds):
    return round(seconds * SAMPLE_RATE)


def test_matching_overrides_default_ignores_self_idle_and_implicit_corespeech():
    processes = [AudioProcess(1, 10, "aquavoice.macOSBridge", True),
                 AudioProcess(2, 11, "com.vendor.SuperWhisper.helper", True),
                 AudioProcess(3, 12, "com.apple.CoreSpeech", True),
                 AudioProcess(4, 13, "com.vendor.AquaVoice", False),
                 AudioProcess(5, 99, "com.vendor.aquavoice", True)]
    assert watched_bundles(processes, parse_patterns("AQUAVOICE"), pid=99) == (
        "aquavoice.macOSBridge",)
    assert watched_bundles(processes, parse_patterns("superWHISPER"), pid=99) == (
        "com.vendor.SuperWhisper.helper",)
    assert watched_bundles(processes, parse_patterns("AQUAVOICE,superwhisper"), pid=99) == (
        "aquavoice.macOSBridge", "com.vendor.SuperWhisper.helper")
    assert watched_bundles(processes, parse_patterns("com.apple"), pid=99) == ()
    assert watched_bundles(processes, parse_patterns("corespeech"), pid=99) == (
        "com.apple.CoreSpeech",)
    assert watched_bundles(processes, parse_patterns("NoNe"), pid=99) == ()
    for invalid in ("", " , ", "none,aquavoice"):
        with pytest.raises(ValueError):
            parse_patterns(invalid)


def test_process_reader_reads_bundle_ids_even_when_input_is_off_and_skips_self():
    reader = object.__new__(CoreAudioProcesses)  # no frameworks loaded
    reader.object_ids = lambda: [1, 2, 3, 4]
    calls = []

    def get(obj, selector, value):
        calls.append((obj, selector))
        if obj == 4:
            raise OSError("process exited")
        value.value = {"ppid": {1: 10, 2: 20, 3: 99}[obj], "piri": obj != 1}[selector]
        return value

    def bundle(obj):
        calls.append((obj, "pbid"))
        return f"bundle.{obj}"

    reader._get, reader.bundle_id = get, bundle
    assert reader.read(exclude_pid=99) == [AudioProcess(1, 10, "bundle.1", False),
                                          AudioProcess(2, 20, "bundle.2", True)]
    assert (1, "pbid") in calls and (1, "piri") in calls
    assert ctypes.sizeof(ctypes.c_int32()) == 4


def test_lookback_retroactively_mutes_onset_and_tail_without_changing_timeline():
    state, gate = PauseState(), MicGate(0)
    original = pcm(1.5)
    first = gate.feed(original[:samples_at(.4)], .4, state)
    assert np.array_equal(first, original[:samples_at(.4 - MIC_LOOKBACK_SEC)])
    # Detection arrives after onset audio is captured, but before the lookback is emitted.
    state.update(.45, False, ("aquavoice",))
    middle = gate.feed(original[samples_at(.4):samples_at(.8)], .8, state)
    state.update(.8, False, ())
    last = gate.feed(original[samples_at(.8):], 1.5, state, final=True)
    heard = np.concatenate([first, middle, last])
    expected = original.copy()
    expected[samples_at(.45 - MIC_LOOKBACK_SEC):samples_at(.8 + AUTO_RESUME_TAIL_SEC)] = 0
    assert np.array_equal(heard, expected)
    assert gate.emitted == original.size and gate.pending.size == 0
    assert np.array_equal(on_pcm16_grid(heard), heard)


def test_shutdown_flushes_held_audio_and_an_active_pause_as_zeros():
    state, gate = PauseState(), MicGate(100)
    original = pcm(.2)
    assert gate.feed(original, 100.2, state).size == 0
    state.update(100.2, True)
    assert not gate.feed(original[:0], 100.2, state, final=True).any()
    assert gate.pending.size == 0 and gate.emitted == original.size


def test_disabled_detection_passes_all_original_samples_including_shutdown_lookback():
    state, gate = PauseState(), MicGate(10)
    original = pcm(1)
    first = gate.feed(original[:samples_at(.5)], 10.5, state)
    held = gate.feed(original[samples_at(.5):], 11, state)
    final = gate.feed(original[:0], 11, state, final=True)
    assert final.size == samples_at(MIC_LOOKBACK_SEC)
    assert np.array_equal(np.concatenate([first, held, final]), original)


def test_overlap_manual_resume_cannot_unmute_automatic_or_its_tail():
    state = PauseState()
    assert [(e.reason, e.paused) for e in state.update(1, True)] == [("manual", True)]
    assert [(e.reason, e.paused) for e in state.update(2, True, ("a",))] == [("automatic", True)]
    assert [(e.reason, e.paused) for e in state.update(3, False, ("a",))] == [("manual", False)]
    assert state.muted(3.5)
    event, = state.update(4, False)
    assert event.reason == "automatic" and event.since == 2 and not event.paused
    assert state.muted(4 + AUTO_RESUME_TAIL_SEC / 2)
    assert not state.muted(4 + AUTO_RESUME_TAIL_SEC)
    original = pcm(5)
    expected = original.copy()
    expected[samples_at(1 - MIC_LOOKBACK_SEC):samples_at(4 + AUTO_RESUME_TAIL_SEC)] = 0
    assert np.array_equal(state.gate(original, 0), expected)


def test_overlap_automatic_resume_cannot_unmute_manual_and_duplicate_updates_are_quiet():
    state = PauseState()
    state.update(1, False, ("a",))
    state.update(2, True, ("a",))
    event, = state.update(3, True)
    assert event.reason == "automatic"
    assert state.update(4, True) == []
    assert state.muted(10)
    event, = state.update(11, False)
    assert event.reason == "manual" and event.since == 2
    assert not state.muted(11)  # no automatic tail on a manual resume


def test_changing_watched_apps_while_capturing_is_one_automatic_episode():
    state = PauseState()
    state.update(1, False, ("a",))
    assert state.update(2, False, ("a", "b")) == []
    assert state.update(3, False, ("b",)) == []
    event, = state.update(4, False)
    assert event.since == 1 and event.bundles == ("b",)


def test_manual_commands_and_status_are_persistent_and_status_does_not_write(tmp_path, monkeypatch, capsys):
    from hark import cli

    monkeypatch.setattr(cli, "HOME", tmp_path)
    # Control commands must not load models, initialize CoreAudio, or open capture.
    monkeypatch.setattr(cli, "load_models", lambda *_: pytest.fail("models loaded"))
    monkeypatch.setattr("hark.pause.CoreAudioProcesses", lambda: pytest.fail("HAL loaded"))
    state = ManualPause(tmp_path)
    assert cli.main(["pause", "--status"]) == 0
    assert "resumed" in capsys.readouterr().out and not state.path.exists()
    assert cli.main(["pause"]) == 0
    assert "paused" in capsys.readouterr().out and state.read()
    before = state.path.read_bytes(), state.path.stat().st_mtime_ns
    assert cli.main(["pause", "--status"]) == 0
    assert "paused" in capsys.readouterr().out
    assert (state.path.read_bytes(), state.path.stat().st_mtime_ns) == before
    assert cli.main(["resume"]) == 0
    assert "resumed" in capsys.readouterr().out and not state.read()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["pause.json"]


def test_monitor_polls_manual_state_even_with_automatic_detection_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr("hark.pause.CoreAudioProcesses", lambda: pytest.fail("HAL loaded"))
    clock = [10.0]
    monkeypatch.setattr("hark.pause.time.time", lambda: clock[0])
    monitor = PauseMonitor(tmp_path, ())
    monitor.poll()
    assert monitor.take_events() == []
    ManualPause(tmp_path).set(True)
    monitor.poll()
    event, = monitor.take_events()
    assert event.reason == "manual" and event.paused
    clock[0] = 12
    ManualPause(tmp_path).set(False)
    monitor.poll()
    event, = monitor.take_events()
    assert not event.paused and event.at - event.since == 2
    # Malformed state cannot silently unmute an existing pause or kill the watcher.
    ManualPause(tmp_path).set(True)
    monitor.poll()
    monitor.manual.path.write_text('{"paused": "false"}')
    monitor.poll()
    assert monitor.muted(12)


@pytest.mark.parametrize("reason", ["automatic", "manual"])
def test_stop_polls_after_join_to_mute_shutdown_onset_and_is_idempotent(tmp_path, monkeypatch, reason):
    clock = [100.2]
    monkeypatch.setattr("hark.pause.time.time", lambda: clock[0])
    monitor = PauseMonitor(tmp_path, ("aquavoice",))
    joined, app_running, reads = [], [], []

    def read():
        reads.append(clock[0])
        return [AudioProcess(1, 10, "aquavoice", bool(app_running))]

    monitor.reader = SimpleNamespace(read=read)
    monitor.poll()
    assert monitor.take_events() == []
    gate = MicGate(100)
    original = pcm(.2)
    assert monitor.feed(gate, original, clock[0]).size == 0

    clock[0] = 100.25  # onset after the last poll, inside the final 75 ms polling window
    if reason == "manual":
        ManualPause(tmp_path).set(True)
    else:
        app_running.append(True)

    def join():
        assert monitor.stopping.is_set()
        assert reads == [100.2] and monitor.state.active == {}
        joined.append(True)

    monitor.thread = SimpleNamespace(join=join)
    monitor.stop()
    assert joined == [True] and reads == [100.2, 100.25]
    event, = monitor.take_events()
    assert event.reason == reason and event.paused and event.at == 100.25
    interval, = monitor.state.intervals
    assert interval.start == 100.25 - MIC_LOOKBACK_SEC and interval.end is None
    final = monitor.feed(gate, original[:0], clock[0], final=True)
    assert final.size == original.size and not final.any()
    # The CLI consumes final events before flushing, then calls stop defensively again.
    clock[0] = 100.3
    ManualPause(tmp_path).set(False)
    app_running.clear()
    monitor.stop()
    assert joined == [True] and reads == [100.2, 100.25]
    assert monitor.take_events() == []


def test_monitor_retains_automatic_pause_on_read_failure_and_resumes_on_success(tmp_path, monkeypatch):
    clock = [10.0]
    monkeypatch.setattr("hark.pause.time.time", lambda: clock[0])
    outcomes = iter([[AudioProcess(1, 10, "aquavoice", True)], OSError("HAL unavailable"), []])

    def read():
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monitor = PauseMonitor(tmp_path, ("aquavoice",))
    monitor.reader = SimpleNamespace(read=read)
    monitor.poll()
    event, = monitor.take_events()
    assert event.paused and event.reason == "automatic"
    clock[0] = 11
    monitor.poll()
    assert monitor.take_events() == [] and monitor.muted(11)
    clock[0] = 12
    monitor.poll()
    event, = monitor.take_events()
    assert not event.paused and event.since == 10
    assert monitor.muted(12 + AUTO_RESUME_TAIL_SEC / 2)


def test_health_suppression_resets_stale_device_times_without_lost_or_back_markers():
    watch = QuietWatch(quiet=10)
    sources = {"mic": (0, 0)}
    assert watch.check(100, sources, {"system": 100}, suppressed={"mic"}) == []
    assert watch.check(105, sources, {"system": 105}) == []
    assert watch.check(110, sources, {"system": 110}) == [Quiet("mic", "silent", 100, "no signal")]
    # An episode already open before intentional muting is cleared without a false back.
    assert watch.check(111, sources, {}, suppressed={"mic"}) == []
    assert "mic" not in watch.open
    # A pause entirely between inference-heavy loop iterations also resets the clock.
    watch.suppress("mic", 120 + AUTO_RESUME_TAIL_SEC)
    assert watch.check(125, sources, {"system": 125}) == []
    # Other sources remain watched during the mic pause.
    assert watch.check(140, {"mic": (0, 0), "system": (100, 100)}, {}, suppressed={"mic"}) == [
        Quiet("system", "silent", 100, "no signal")]


def test_health_remembers_a_pause_consumed_during_a_long_main_loop_iteration(tmp_path):
    monitor = PauseMonitor(tmp_path, ())
    monitor.state.update(90, False, ("aquavoice",))
    monitor.state.update(100, False)
    # Audio processing can consume and prune the interval before the health check runs.
    monitor.state.gate(pcm(120), 0)
    assert monitor.state.intervals == []
    watch = QuietWatch(quiet=90)
    baseline = monitor.health_since(120)
    assert baseline == 100 + AUTO_RESUME_TAIL_SEC
    watch.suppress("mic", baseline)
    assert watch.check(120, {"mic": (0, 0)}, {"system": 120}) == []
    assert watch.check(201, {"mic": (0, 0)}, {}) == [Quiet("mic", "silent", baseline, "no signal")]
    # Reusing an old suppression timestamp must not erase genuine later health episodes.
    watch.suppress("mic", monitor.health_since(202))
    assert watch.check(202, {"mic": (0, 0)}, {}) == []
    assert "mic" in watch.open


def test_manual_annotations_are_append_only_and_auto_events_only_go_to_log_jsonl(tmp_path, monkeypatch):
    from hark.cli import _pause_events

    path = tmp_path / "meeting.txt"
    sink = Sink(path, "session")
    messages = []
    monkeypatch.setattr("hark.cli.log", messages.append)
    monitor = PauseMonitor(tmp_path, ())
    start = datetime(2026, 9, 29, 12).timestamp()
    monitor.events.extend(monitor.state.update(start, False, ("aquavoice",)))
    monitor.events.extend(monitor.state.update(start + 1, True, ("aquavoice",)))
    with path.open("a") as external:
        external.write("# S2 = Cail\n")
    monitor.events.extend(monitor.state.update(start + 2, True))
    monitor.events.extend(monitor.state.update(start + 65, False))
    _pause_events(monitor, sink)
    sink.poll_names()
    assert sink.names == {"S2": "Cail"}
    sink.close("ended")
    assert path.read_text().splitlines() == ["# session", "# S2 = Cail", "# paused",
                                           "# resumed at 12:01:05 after 1m04s", "# ended"]
    events = [json.loads(line)["pause"] for line in path.with_suffix(".jsonl").read_text().splitlines()]
    assert [(e["reason"], e["state"]) for e in events] == [
        ("automatic", "paused"), ("manual", "paused"),
        ("automatic", "resumed"), ("manual", "resumed")]
    assert events[2]["duration"] == 2 and events[3]["duration"] == 64
    assert any("automatic (aquavoice)" in message for message in messages)


def test_processes_cli_exposes_structured_reader_without_loading_models(monkeypatch, capsys):
    from hark import cli

    monkeypatch.setattr(cli, "read_processes", lambda: [AudioProcess(1, 10, "idle.bundle", False)])
    assert cli.main(["processes"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "object_id": 1, "pid": 10, "bundle_id": "idle.bundle", "running_input": False}
