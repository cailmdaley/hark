import json
import re
import threading
import time
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from hark.health import QUIET_SEC, Quiet, QuietWatch
from hark.transcript import Sink

T0 = datetime(2026, 9, 29, 16, 37, 55).timestamp()


def test_no_signal_goes_quiet_unconditionally_from_the_last_delivery():
    watch = QuietWatch()
    sources = {"mic": (T0, T0, False)}
    assert watch.check(T0 + QUIET_SEC - 1, sources, {}) == []
    assert watch.check(T0 + QUIET_SEC, sources, {}) == [Quiet("mic", "silent", T0, "no signal")]


def test_digital_silence_counts_only_while_another_track_speaks():
    watch = QuietWatch()
    now = T0 + 100
    sources = {"system": (now, T0, True)}  # sound earlier, now only zeros
    assert watch.check(now, sources, {}) == []
    assert watch.check(now, sources, {"system": now - 5}) == []  # its own track doesn't count
    assert watch.check(now, sources, {"mic": now - QUIET_SEC - 1}) == []  # too long ago
    assert watch.check(now, sources, {"mic": now - 5}) == [Quiet("system", "silent", T0, "silence")]


def test_both_sources_silent_in_a_lull_is_not_a_marker():
    watch = QuietWatch()
    now = T0 + 600
    sources = {"mic": (now, T0, False), "system": (now, T0, False)}
    assert watch.check(now, sources, {"mic": T0 - 10, "system": T0 - 20}) == []


def test_one_marker_per_episode_then_back_then_a_new_episode():
    watch = QuietWatch()
    events = [e for now in np.arange(T0, T0 + 900, 0.5)
              for e in watch.check(now, {"system": (T0, T0, False)}, {"mic": now})]
    assert events == [Quiet("system", "silent", T0, "no signal")]
    back = T0 + 855
    assert watch.check(back + 1, {"system": (back, back, True)}, {}) == [
        Quiet("system", "back", T0, "no signal", back)]
    assert watch.check(back + 2, {"system": (back + 2, back, True)}, {}) == []
    assert watch.check(back + QUIET_SEC, {"system": (back + QUIET_SEC, back, True)}, {"mic": back + 80}) == [
        Quiet("system", "silent", back, "silence")]


def test_never_audible_system_stays_healthy_despite_continued_mic_utterances(monkeypatch):
    from hark.capture import Source

    clock = [T0]
    monkeypatch.setattr("hark.capture.time.time", lambda: clock[0])
    src = Source("system")
    monkeypatch.setattr(src, "_open", lambda: None)
    src.start()
    watch = QuietWatch()

    def check():
        return watch.check(clock[0], {src.name: (src.last_audio, src.last_sound, src.has_sound)},
                           {"mic": clock[0]})

    for elapsed in range(0, 601, 15):
        clock[0] = T0 + elapsed
        src._deliver(np.zeros(16, np.float32))
        assert check() == []
    assert not src.has_sound and watch.open == {}

    # Actual device sound arms the silence heuristic for the rest of the session.
    clock[0] += 1
    src._deliver(np.full(16, .01, np.float32))
    heard = clock[0]
    assert src.has_sound and check() == []
    for elapsed in range(1, 90):
        clock[0] = heard + elapsed
        src._deliver(np.zeros(16, np.float32))
        assert check() == []
    clock[0] = heard + QUIET_SEC
    src._deliver(np.zeros(16, np.float32))
    assert check() == [Quiet("system", "silent", heard, "silence")]
    clock[0] += QUIET_SEC
    src._deliver(np.zeros(16, np.float32))
    assert check() == []  # no duplicate loss, and zeros aren't recovery
    clock[0] += 1
    src._deliver(np.full(16, -.01, np.float32))
    assert check() == [Quiet("system", "back", heard, "silence", clock[0])]
    clock[0] += QUIET_SEC
    src._deliver(np.zeros(16, np.float32))
    assert check() == [Quiet("system", "silent", heard + QUIET_SEC * 2 + 1, "silence")]


@pytest.mark.parametrize("has_sound", [False, True])
def test_system_no_samples_is_lost_even_without_prior_sound_and_needs_sound_to_recover(has_sound):
    watch = QuietWatch()
    sources = {"system": (T0, T0, has_sound)}
    assert watch.check(T0 + QUIET_SEC - .01, sources, {}) == []
    assert watch.check(T0 + QUIET_SEC, sources, {}) == [
        Quiet("system", "silent", T0, "no signal")]
    # A resumed stream of zeros is not recovery, nor another loss event.
    assert watch.check(T0 + 200, {"system": (T0 + 200, T0, has_sound)}, {"mic": T0 + 200}) == []
    assert watch.check(T0 + 201, {"system": (T0 + 201, T0 + 201, True)}, {}) == [
        Quiet("system", "back", T0, "no signal", T0 + 201)]


@pytest.mark.parametrize("track", ["mic", "phone"])
def test_never_audible_other_tracks_still_use_silence_heuristic(track):
    watch = QuietWatch()
    now = T0 + QUIET_SEC
    assert watch.check(now, {track: (now, T0, False)}, {"system": now}) == [
        Quiet(track, "silent", T0, "silence")]


@pytest.mark.parametrize("peak,audible", [
    (0., False),
    (np.nextafter(1e-4, 0.), False),
    (1e-4, False),
    (np.nextafter(1e-4, np.inf), True),
    (-1e-4, False),
    (-np.nextafter(1e-4, np.inf), True),
])
def test_source_sound_threshold_is_strict_and_session_scoped(monkeypatch, peak, audible):
    from hark.capture import Source

    clock = [T0]
    monkeypatch.setattr("hark.capture.time.time", lambda: clock[0])
    src = Source("system")
    monkeypatch.setattr(src, "_open", lambda: None)
    src.start()
    clock[0] += 1
    src._deliver(np.array([peak], dtype=np.float64))
    assert src.has_sound is audible
    assert src.last_sound == (clock[0] if audible else T0)
    clock[0] += 1
    src._deliver(np.zeros(1, np.float32))
    assert src.has_sound is audible  # silence doesn't disarm a previously audible source
    clock[0] = T0 + 100
    src._deliver(np.zeros(1, np.float32))
    assert QuietWatch().check(clock[0], {
        "system": (src.last_audio, src.last_sound, src.has_sound)}, {"mic": clock[0]}) == (
            [Quiet("system", "silent", src.last_sound, "silence")] if audible else [])
    src.start()
    assert not src.has_sound  # sound in another session doesn't arm this one


def test_system_startup_silence_message_is_conditional_and_once(monkeypatch, tmp_path):
    from hark import capture

    tap = tmp_path / "audiotee"
    tap.touch()
    monkeypatch.setattr(capture, "AUDIOTEE", tap)
    clock = [T0]
    monkeypatch.setattr(capture.time, "time", lambda: clock[0])
    messages = []
    monkeypatch.setattr(capture, "log", messages.append)
    src = capture.SystemSource()
    monkeypatch.setattr(src, "_open", lambda: None)
    src.start()
    src._check_silence()
    assert messages == []
    clock[0] += 21
    src._deliver(np.zeros(16, np.float32))
    src._check_silence()
    src._check_silence()
    assert len(messages) == 1
    assert "expected if nothing is playing" in messages[0]
    assert "If playback is missing, check" in messages[0]
    assert "System Audio Recording Only" in messages[0]

    # Even sound at the start timestamp counts; timestamp equality isn't evidence of silence.
    src.start()
    src.warned = False
    src._deliver(np.full(16, .01, np.float32))
    clock[0] += 21
    src._check_silence()
    assert len(messages) == 1


def test_sink_writes_marker_lines_and_records_without_touching_names(tmp_path):
    path = tmp_path / "meeting.txt"
    sink = Sink(path, "session")
    sink.source(Quiet("system", "silent", T0, "no signal"))
    sink.source(Quiet("system", "back", T0, "no signal", T0 + 855))
    sink.source(Quiet("mic", "silent", T0, "silence"))
    sink.poll_names()
    assert sink.names == {}
    sink.close("ended")
    lines = path.read_text().splitlines()
    assert lines[1:4] == [
        "# system audio lost at 16:37:55 — no signal from the tap; nothing from the call is being transcribed",
        "# system audio back at 16:52:10 after 14m15s lost",
        "# mic lost at 16:37:55 — only silence while others speak; the mic may not be captured"]
    assert not any(re.fullmatch(r"# (S\d+) = (.+)", line) for line in lines)
    records = [json.loads(line)["source"] for line in path.with_suffix(".jsonl").read_text().splitlines()]
    assert records == [
        {"track": "system", "state": "silent", "since": "2026-09-29T16:37:55", "cause": "no signal"},
        {"track": "system", "state": "back", "since": "2026-09-29T16:37:55", "cause": "no signal",
         "back": "2026-09-29T16:52:10"},
        {"track": "mic", "state": "silent", "since": "2026-09-29T16:37:55", "cause": "silence"}]


def test_padding_is_not_device_audio(monkeypatch):
    from hark.capture import SAMPLE_RATE, Source

    src = Source("mic")
    src.anchor = src.last_audio = src.last_sound = time.time() - 3
    src._deliver(np.zeros(1600, np.float32))
    assert src.last_sound == src.anchor and not src.has_sound
    src._deliver(np.full(1600, 0.01, np.float32))
    heard = src.last_sound
    assert heard > src.anchor and src.has_sound
    src.last_audio = src.last_sound = src.anchor  # the device then falls silent for 3 s
    padded = src.drain()
    assert padded.size > SAMPLE_RATE and (src.last_audio, src.last_sound) == (src.anchor, src.anchor)
    stats = src.take_stats()
    assert stats["device"] == 3200 and stats["padded"] == padded.size - 3200 and stats["peak"] > 0.009
    assert src.take_stats() == {"device": 0, "padded": 0, "peak": 0.0}


def test_live_capture_marks_a_dead_source_and_logs_to_the_session_log(tmp_path, monkeypatch):
    from test_cli import fake_live_capture, signal_when_live

    cli, Source = fake_live_capture(monkeypatch, tmp_path)

    class Dead(Source):
        def _open(self):
            self.last_audio = self.last_sound = self.anchor - QUIET_SEC - 5

    monkeypatch.setattr(cli, "MicSource", Dead)
    signaller, _ = signal_when_live(tmp_path)
    out = tmp_path / "sessions" / "dead.txt"
    cli.main(["--ear", "local", "--launch", "test", "--room", "-o", str(out)])
    signaller.join(timeout=2)

    lost = [line for line in out.read_text().splitlines() if " lost at " in line]
    assert len(lost) == 1 and lost[0].startswith("# mic lost at ")
    assert lost[0].endswith("— no signal from the device; nothing from the mic is being transcribed")
    log = out.with_suffix(".log").read_text().splitlines()
    assert any(re.fullmatch(r"\d\d:\d\d:\d\d transcript → .*dead\.txt", line) for line in log)
    assert any("mic: still silent at the end (no signal since" in line for line in log)


def test_replay_writes_the_log_and_never_marks_sources(tmp_path, monkeypatch):
    import hark.cli as cli

    class Stale:
        name, anchor, live = "recording", 1.0, False
        last_audio = last_sound = 1.0  # would be long "quiet" if replays were watched

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

    monkeypatch.setattr(cli, "HOME", tmp_path)
    monkeypatch.setattr(cli, "_default_ear", lambda: "local")
    monkeypatch.setattr(cli, "FileSource", Stale)
    monkeypatch.setattr(cli, "Track", FakeTrack)
    monkeypatch.setattr(cli, "load_models", lambda latency: (None, None))
    monkeypatch.setattr(cli, "flush_tracks", lambda tracks, force=False: [])
    (tmp_path / "x.wav").touch()

    cli.main(["--ear", "local", "--file", str(tmp_path / "x.wav"), "-o", str(tmp_path / "replay")])

    assert " lost at " not in (tmp_path / "replay.txt").read_text()
    assert '"source"' not in (tmp_path / "replay.jsonl").read_text()
    assert "transcript → " in (tmp_path / "replay.log").read_text()
