"""Voice capture's clock, idle cost, lifecycle and mute contracts without models/devices."""

from datetime import datetime
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from hark import cli
from hark.capture import SAMPLE_RATE
from hark.live_voice import VoiceTrack
from hark.pause import AudioProcess, ManualPause, PauseMonitor
from hark.transcript import flush_tracks, numbered
from test_cli import fake_live_capture, record_lifecycle_phases, signal_when_live


@pytest.fixture(autouse=True)
def no_batch_model(monkeypatch):
    monkeypatch.setattr("hark.live_voice.load_refiner", lambda **kwargs: None)


class Session:
    def __init__(self):
        self._mel_offset = 0
        self.calls = []
        self.tokens = []

    def feed(self, samples, final=False):
        self.calls.append((len(samples), final))
        self._mel_offset += len(samples) // 160
        tokens, self.tokens = self.tokens, []
        return [SimpleNamespace(speaker="speaker_0", tokens=tokens)] if tokens else []


def track(**kwargs):
    session = Session()
    asr = SimpleNamespace(preprocessor_config=SimpleNamespace(hop_length=160),
                          create_speaker_streaming_session=lambda *a, **kw: session)
    return VoiceTrack("mic", asr, None, speaker_label=numbered, **kwargs)


def pcm(seconds, value=0):
    return np.full(round(seconds * SAMPLE_RATE), value, np.float32)


def token(start, end, text=" hello"):
    return SimpleNamespace(start=start, end=end, text=text)


def test_idle_runs_no_models_and_retains_only_preroll_and_rolling_voice_audio():
    t = track()
    for _ in range(120):
        t.feed(pcm(30))
    assert not t.session.calls
    assert t._held.size == t.preroll
    assert len(t.audio.samples) == 45 * SAMPLE_RATE
    assert len(t._clock) == 1
    assert t.processed == pytest.approx(3599.7)


def test_resume_retains_session_and_maps_both_token_edges_to_mic_clock():
    t = track()
    t.feed(pcm(10))
    session = t.session
    # The 300 ms pre-roll starts at mic second 9.7, model second zero.
    session.tokens = [token(.3, .4)]
    t.feed(pcm(.1, .02))
    first = t.pending["speaker_0"].tokens[0]
    assert (first.start, first.end) == pytest.approx((10, 10.1))
    t.feed(pcm(20))
    assert flush_tracks([t])
    assert t._tail_left == 0
    model_start = t._sent / SAMPLE_RATE
    session.tokens = [token(model_start + .3, model_start + .4, " again")]
    t.feed(pcm(.1, .02))
    second = t.pending["speaker_0"].tokens[0]
    assert (second.start, second.end) == pytest.approx((30.1, 30.2))
    assert t.session is session
    assert t.audio_samples == round(30.2 * SAMPLE_RATE)


def test_token_end_at_skipped_gap_uses_old_segment_but_start_uses_new_segment():
    t = track()
    t.feed(pcm(.1, .02))
    t.feed(pcm(100))
    boundary = t._sent / SAMPLE_RATE
    t.session.tokens = [token(boundary - .08, boundary, " before"),
                        token(boundary, boundary + .08, " after")]
    t.feed(pcm(.1, .02))
    before, after = t.pending["speaker_0"].tokens
    assert (before.start, before.end) == pytest.approx((boundary - .08, boundary))
    assert (after.start, after.end) == pytest.approx((99.8, 99.88))


def test_clock_boundaries_remain_bounded_across_many_idle_gaps():
    t = track()
    t.audio = None
    for _ in range(1000):
        t.feed(pcm(.1, .02))
        t.feed(pcm(5))
    assert len(t._clock) <= 2
    assert t._held.size <= t.preroll


def test_preroll_quiet_tail_and_final_flush_are_delivered_once():
    t = track()
    t.feed(pcm(1))
    t.feed(pcm(.1, .02))
    assert sum(n for n, _ in t.session.calls) == int(.4 * SAMPLE_RATE)
    t.feed(pcm(5))
    # Onset resets a three-second tail, including the onset frame.
    assert sum(n for n, _ in t.session.calls) == int(3.3 * SAMPLE_RATE)
    t.feed(pcm(0), final=True)
    assert t.session.calls[-1] == (0, True)
    assert sum(n for n, _ in t.session.calls) == int(3.6 * SAMPLE_RATE)


def test_stop_set_final_drain_preserves_quiet_samples_before_backlogged_speech():
    stop = threading.Event()
    stop.set()
    t = track(stop=stop)
    t.session.tokens = [token(.3, .38)]
    samples = np.concatenate((pcm(.5), pcm(.5, .02)))
    t.feed(samples, final=True)
    assert t.audio_samples == SAMPLE_RATE
    np.testing.assert_array_equal(t.audio.samples, samples)
    first = t.pending["speaker_0"].tokens[0]
    assert (first.start, first.end) == pytest.approx((.5, .58))


def test_ungated_reference_feeds_every_sample():
    t = track(gate=False)
    t.feed(pcm(10))
    assert sum(n for n, _ in t.session.calls) == 10 * SAMPLE_RATE
    assert t.processed == 10


def test_voice_defaults_do_not_change_meeting_track_defaults():
    t = track()
    assert (t.gap, t.max_len) == (5, 75)
    assert t.t0.tzinfo is None  # CLI owns the source's wall anchor.


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("echo", [False, True])
def test_voice_desktop_starts_two_voice_tracks_aec_dedupe_and_route_cuts(tmp_path, monkeypatch, enabled, echo):
    _, Source = fake_live_capture(monkeypatch, tmp_path)
    from hark import live_voice, aec, dedupe, voice
    from hark.transcript import Utterance

    made, presets, cancellers, cuts = [], [], [], []
    ready = threading.Event()
    refiner = object()
    monkeypatch.setattr(live_voice, "load_refiner", lambda **kwargs: refiner)

    class System(Source):
        def __init__(self):
            super().__init__()
            self.name = "system"

    class FakeTrack:
        def __init__(self, name, *args, **kwargs):
            self.name, self.processed, self.audio = name, 0, None
            self.t0 = None
            self.pending = True
            made.append((self, kwargs))
        def feed(self, *args, **kwargs): pass
        def cut(self, at):
            cuts.append((self.name, at))
            if len(cuts) == 2:
                ready.set()

    class Routes:
        def __init__(self, path): self.sent = False
        def read(self):
            if self.sent: return []
            self.sent = True
            return [123.0]

    class Aec:
        def __init__(self, **kwargs): cancellers.append(kwargs)
        def feed(self, mic, system, final=False): return mic

    calls = []
    real_deduper = dedupe.EchoDeduper
    class Deduper(real_deduper):
        def feed(self, utterances, now, final=False):
            calls.append([u.track for u in utterances])
            return super().feed(utterances, now, final=final)

    def flush(tracks, force=False):
        result = []
        for t in tracks:
            if t.pending:
                t.pending = False
                text = "matching echo sentence" if echo else f"{t.name} distinct words"
                result.append(Utterance(t.name, "S1", 0, 1, t.t0, text))
        return result

    monkeypatch.setattr(live_voice, "VoiceTrack", FakeTrack)
    monkeypatch.setattr(live_voice, "RouteCuts", Routes)
    monkeypatch.setattr(cli, "load_models", lambda p: (presets.append(p), None))
    monkeypatch.setattr(cli, "flush_tracks", flush)
    monkeypatch.setattr(cli, "_system_capture_available", lambda: True)
    monkeypatch.setattr(cli, "SystemSource", System)
    monkeypatch.setattr(aec, "EchoCanceller", Aec)
    monkeypatch.setattr(dedupe, "EchoDeduper", Deduper)
    matched = []
    class Matcher:
        def __init__(self, voices, sink, name_tracks):
            assert name_tracks == {"me": {"mic"}}
        def finished(self, track, utterance):
            matched.append(utterance.track)
    monkeypatch.setattr(voice, "VoiceMatcher", Matcher)
    (tmp_path / "voices").mkdir()
    np.save(tmp_path / "voices" / "me.npy", np.ones(2))
    phases = record_lifecycle_phases(monkeypatch, cli, tmp_path)
    ManualPause(tmp_path).set(True)
    signaller, _ = signal_when_live(tmp_path, ready=ready)
    cli.main(["--voice", "--pause-for", "none", "-o", str(tmp_path / "voice")]
             + (["--system-audio"] if enabled else []))
    signaller.join(2)
    state = json.loads((tmp_path / "meeting.json").read_text())
    assert state["kind"] == "voice" and state["phase"] == "ended"
    assert all(p["system_audio"] is enabled for p in phases)
    assert ManualPause(tmp_path, "system").read() is not enabled
    assert [t.name for t, _ in made] == ["mic", "system"]
    assert all(kw["gap"] == 5 and kw["refiner"] is refiner for _, kw in made)
    assert presets == ["ultra_low"] and len(cancellers) == 1
    assert calls and ["mic", "system"] in calls
    assert cuts == [("mic", 123.0), ("system", 123.0)]
    assert ManualPause(tmp_path).read()
    records = [json.loads(line) for line in (tmp_path / "voice.jsonl").read_text().splitlines()]
    spoken = [r for r in records if "speaker" in r]
    assert {r["track"] for r in spoken} == ({"system"} if echo else {"mic", "system"})
    assert {r["speaker"] for r in spoken} == ({"S1"} if echo else {"S1", "S2"})
    assert set(matched) == {r["track"] for r in spoken}
    for r in spoken:
        t = next(t for t, _ in made if t.name == r["track"])
        assert datetime.fromisoformat(r["wall"]).tzinfo is not None
        assert r["wall"] == t.t0.isoformat(timespec="microseconds")
    assert (tmp_path / "voice.txt").read_text().splitlines()[-1].startswith("# ended ")


@pytest.mark.parametrize("enabled", [False, True])
def test_voice_without_system_capture_logs_fallback(tmp_path, monkeypatch, enabled):
    _, Source = fake_live_capture(monkeypatch, tmp_path)
    from hark import live_voice, aec

    class FakeTrack:
        def __init__(self, name, *args, **kwargs):
            self.name, self.processed, self.audio = name, 0, None
        def feed(self, *args, **kwargs): pass

    monkeypatch.setattr(live_voice, "VoiceTrack", FakeTrack)
    monkeypatch.setattr(cli, "_system_capture_available", lambda: False)
    monkeypatch.setattr(cli, "SystemSource", lambda: pytest.fail("unexpected system source"))
    monkeypatch.setattr(aec, "EchoCanceller", lambda **kw: pytest.fail("unexpected AEC"))
    signaller, _ = signal_when_live(tmp_path)
    cli.main(["--voice", "--pause-for", "none", "-o", str(tmp_path / "voice")]
             + (["--system-audio"] if enabled else []))
    signaller.join(2)
    assert json.loads((tmp_path / "meeting.json").read_text())["system_audio"] is False
    assert "system audio unavailable; using mic only" in (tmp_path / "voice.log").read_text()


@pytest.mark.parametrize("enabled", [False, True])
def test_voice_phone_uses_phone_voice_track_without_desktop_sources(tmp_path, monkeypatch, enabled):
    _, Source = fake_live_capture(monkeypatch, tmp_path)
    from hark import live_voice, aec

    made = []
    class Phone(Source):
        def __init__(self, path):
            super().__init__()
            self.name = "phone"
            made.append(path)

    class FakeTrack:
        def __init__(self, name, *args, **kwargs):
            self.name, self.processed, self.audio = name, 0, None
            made.append(name)
        def feed(self, *args, **kwargs): pass

    monkeypatch.setattr(live_voice, "VoiceTrack", FakeTrack)
    monkeypatch.setattr(cli, "_device_capture_available", lambda: False)
    monkeypatch.setattr(cli, "_system_capture_available", lambda: pytest.fail("unexpected system probe"))
    monkeypatch.setattr(cli, "MicSource", lambda *a: pytest.fail("unexpected mic source"))
    monkeypatch.setattr(cli, "PhoneSource", Phone)
    monkeypatch.setattr(aec, "EchoCanceller", lambda **kw: pytest.fail("unexpected AEC"))
    signaller, _ = signal_when_live(tmp_path)
    cli.main(["--voice", "--phone", "--pause-for", "none", "-o", str(tmp_path / "voice")]
             + (["--system-audio"] if enabled else []))
    signaller.join(2)
    state = json.loads((tmp_path / "meeting.json").read_text())
    assert state["kind"] == "voice" and state["phone"] == str(tmp_path / "phone.sock")
    assert state["system_audio"] is False and made == [tmp_path / "phone.sock", "phone"]
    assert "voice: phone" in (tmp_path / "voice.txt").read_text()


@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("token_delay", [0, .64])
@pytest.mark.parametrize("audio_on_wake", [False, True])
@pytest.mark.parametrize("queued_before_sleep", [False, True])
@pytest.mark.parametrize("resumed_blocks", [1, 6])
@pytest.mark.parametrize("mute_window", ["none", "prefix", "sleep"])
def test_sleep_gap_preserves_held_and_resumed_speech_through_real_source_and_mic_gate(
        tmp_path, monkeypatch, token_delay, audio_on_wake, queued_before_sleep, resumed_blocks,
        mute_window, external):
    from hark.capture import Source

    if external:
        def recognize(audio):
            counts = [np.count_nonzero(audio == value) for value in (1, 2)]
            words = [max(1, round(n / (SAMPLE_RATE / 10))) if n else 0 for n in counts]
            return " ".join(["before"] * words[0] + ["after"] * words[1])
        monkeypatch.setattr("hark.live_voice.load_refiner", lambda **kwargs: recognize)

    clock = [100.]
    signals = []
    monkeypatch.setattr("time.time", lambda: clock[0])
    monkeypatch.setattr(cli, "HOME", tmp_path)
    monkeypatch.setattr(cli, "_default_ear", lambda: "local")
    monkeypatch.setattr(cli, "_device_capture_available", lambda: True)

    class Signals:
        def __init__(self, lifecycle, stop):
            self.lifecycle, self.stop = lifecycle, stop
            self.signal_written = threading.Event()
            self.signal_written.set()
            signals.append(self)
        def close(self): pass

    class SleepingMic(Source):
        def __init__(self, *args):
            super().__init__("mic")
            self.index = 0
        def _open(self): pass
        def drain(self, limit):
            self.index += 1
            if self.index == 1:
                clock[0] = 100.3
                self._deliver(pcm(.3, 1))
            elif self.index == 2:
                clock[0] = 3700.4
                if audio_on_wake:
                    for i in range(resumed_blocks):
                        clock[0] = 3700.4 + i / 10
                        self._deliver(pcm(.1, 2))
            elif self.index == 3:
                clock[0] = 3700.5 + (resumed_blocks - 1) / 10 if audio_on_wake else 3700.5
                if audio_on_wake:
                    self._deliver(pcm(.1))
                else:
                    for i in range(resumed_blocks):
                        clock[0] = 3700.5 + i / 10
                        self._deliver(pcm(.1, 2))
            else:
                signals[0].stop.set()
                signals[0].lifecycle.stopping()
            samples = super().drain(limit)
            if self.index == 1 and queued_before_sleep:
                clock[0] = 100.4
                self._deliver(pcm(.1, 1))
            return samples

    class ScriptedMonitor(PauseMonitor):
        def start(self):
            super().start()
            if mute_window != "none":
                from hark.pause import MuteInterval
                interval = ((100.3, 100.4) if mute_window == "prefix" else (100.5, 3700.35))
                with self.lock:
                    self.state.intervals.append(MuteInterval(*interval))

    class Recognizer(Session):
        def feed(self, samples, final=False):
            start = self._mel_offset * .01 + token_delay
            if samples.size and samples.max() == 1:
                self.tokens.append(token(start, start + .08, " before"))
            if samples.size and samples.max() == 2:
                self.tokens.append(token(start, start + .08, " after"))
            return super().feed(samples, final=final)

    session = Recognizer()
    asr = SimpleNamespace(preprocessor_config=SimpleNamespace(hop_length=160),
                          create_speaker_streaming_session=lambda *a, **kw: session)
    monkeypatch.setattr(cli, "SignalWatcher", Signals)
    monkeypatch.setattr(cli, "MicSource", SleepingMic)
    monkeypatch.setattr(cli, "PauseMonitor", ScriptedMonitor)
    monkeypatch.setattr(cli, "load_models", lambda _: (asr, None))
    out = tmp_path / "sleep.txt"
    cli.main(["--voice", "--pause-for", "none", "--speaker-mask", "shared", "-o", str(out)])
    records = [json.loads(line) for line in out.with_suffix(".jsonl").read_text().splitlines()]
    spoken = [r for r in records if "speaker" in r]
    before = [r for r in spoken if "before" in r["text"]]
    after = [r for r in spoken if "after" in r["text"]]
    assert before and after
    assert all("after" not in r["text"] for r in before)
    assert datetime.fromisoformat(before[0]["wall"]).timestamp() == pytest.approx(100 + token_delay)
    resumed = 3700.3 if audio_on_wake else 3700.4
    assert datetime.fromisoformat(after[0]["wall"]).timestamp() == pytest.approx(resumed + token_delay)
    expected_before = 4 if queued_before_sleep and mute_window != "prefix" else 3
    assert sum(r["text"].count("before") for r in before) == expected_before
    assert sum(r["text"].count("after") for r in after) == resumed_blocks


def test_short_voice_lines_keep_distinct_slots_and_accumulate_enrolled_name_evidence(tmp_path):
    from hark.transcript import Sink, Utterance
    from hark.voice import VoiceMatcher

    t = track()
    t.feed(np.concatenate((pcm(8, .02), pcm(8, -.02))))
    t.t0 = datetime.now().astimezone()
    sink = Sink(tmp_path / "names.txt", "voice")
    matcher = VoiceMatcher({"me": np.array([1., 0.])}, sink,
                           embedder=lambda audio: np.array([1., 0.]) if audio.mean() > 0
                           else np.array([0., 1.]))
    for speaker, offset in [("speaker_0", 0), ("speaker_1", 8)]:
        for i in range(3):
            start = offset + i * 2
            u = Utterance("mic", t.speaker_label(speaker), start, start + 2,
                          t.t0, "short voice line", speech=[(start, start + 2)])
            cli._emit(u, sink, matcher, {"mic": t}, set())
    assert sink.names == {"S1": "me"}
    assert matcher.seconds[id(t), "S1"] == 6
    assert matcher.seconds[id(t), "S2"] == 6
    sink.close("ended")


def test_voice_mode_rejects_other_modes_and_requires_local_ear(monkeypatch):
    monkeypatch.setattr(cli, "_default_ear", lambda: "local")
    for options in [["--voice", "--file", "x.wav"], ["--voice", "--mic-only"],
                    ["--voice", "--call"], ["--voice", "--room"],
                    ["--phone", "--file", "x.wav"], ["--system-audio"],
                    ["--voice", "--ear", "gradium"]]:
        with pytest.raises(SystemExit) as error:
            cli.main(options)
        assert error.value.code == 2


def test_voice_pause_ignores_idle_aqua_mic_but_tracks_accepted_dictation(tmp_path, monkeypatch):
    from hark import aqua
    path = tmp_path / "main.log"
    path.write_text("")
    real_reader = aqua.AquaDictation
    monkeypatch.setattr(aqua, "AquaDictation", lambda: real_reader(path))
    monitor = PauseMonitor(tmp_path, ("aquavoice",), voice=True)
    processes = [AudioProcess(1, 10, "aquavoice.macOSBridge", True)]
    monitor.reader = SimpleNamespace(read=lambda: processes)
    monitor.poll()
    assert not monitor.automatic
    stamp = datetime.now().astimezone().isoformat()
    path.write_text(f"[{stamp}] [info] (DictationOrchestrator) dictation.start_accepted lease_id=1\n")
    monitor.poll()
    assert monitor.automatic == ("aquavoice.macOSBridge",)
    with path.open("a") as log:
        log.write(f"[{stamp}] [info] (DictationOrchestrator) dictation.completed lease_id=1\n")
    monitor.poll()
    assert not monitor.automatic


def test_unknown_detection_cannot_leave_voice_automatically_muted(tmp_path):
    monitor = PauseMonitor(tmp_path, ("aquavoice",), voice=True)
    monitor.automatic = ("aquavoice.macOSBridge",)
    def broken(): raise OSError("HAL failure")
    monitor.reader = SimpleNamespace(read=broken)
    monitor.poll()
    assert not monitor.automatic
