import json
import re
import threading
from types import SimpleNamespace
from datetime import datetime

import numpy as np
import pytest

from hark.health import QUIET_SEC, Quiet, QuietWatch
from hark.transcript import Sink

T0 = datetime(2026, 9, 29, 16, 37, 55).timestamp()


@pytest.mark.parametrize("track", ["mic", "system", "phone"])
def test_no_samples_for_90_seconds_marks_loss(track):
    watch = QuietWatch()
    assert watch.check(T0 + QUIET_SEC - .01, {track: T0}) == []
    assert watch.check(T0 + QUIET_SEC, {track: T0}) == [Quiet(track, "silent", T0)]
    assert watch.check(T0 + 600, {track: T0}) == []
    back = T0 + 601
    assert watch.check(back, {track: back}) == [Quiet(track, "back", T0, at=back)]
    assert watch.check(back + 1, {track: back}) == []
    assert watch.check(back + QUIET_SEC, {track: back}) == [Quiet(track, "silent", back)]


@pytest.mark.parametrize("prior_sound", [False, True])
@pytest.mark.parametrize("track", ["system", "mic", "phone"])
def test_ten_minutes_of_device_zeros_while_mic_speaks_produces_no_marker(monkeypatch, prior_sound, track):
    from hark.capture import Source

    clock = [T0]
    monkeypatch.setattr("hark.capture.time.time", lambda: clock[0])
    src = Source(track)
    mic = Source("me")
    for source in (src, mic):
        monkeypatch.setattr(source, "_open", lambda: None)
        source.start()
    if prior_sound:
        src._deliver(np.full(16, .01, np.float32))
    watch = QuietWatch()
    for elapsed in range(1, 601):
        clock[0] = T0 + elapsed
        src._deliver(np.zeros(16, np.float32))
        mic._deliver(np.full(16, .1, np.float32))
        assert watch.check(clock[0], {src.name: src.last_audio, mic.name: mic.last_audio}) == []
    assert watch.open == {}
    clock[0] += QUIET_SEC
    assert watch.check(clock[0], {src.name: src.last_audio}) == [Quiet(track, "silent", T0 + 600)]
    clock[0] += 1
    src._deliver(np.zeros(16, np.float32))
    assert watch.check(clock[0], {src.name: src.last_audio}) == [
        Quiet(track, "back", T0 + 600, at=clock[0])]


@pytest.mark.parametrize("peak,audible", [
    (0., False),
    (np.nextafter(1e-4, 0.), False),
    (1e-4, False),
    (np.nextafter(1e-4, np.inf), True),
    (-1e-4, False),
    (-np.nextafter(1e-4, np.inf), True),
])
def test_startup_warning_sound_threshold(monkeypatch, peak, audible):
    from hark.capture import Source

    src = Source("system")
    monkeypatch.setattr(src, "_open", lambda: None)
    src.start()
    src._deliver(np.array([peak], dtype=np.float64))
    assert src.has_sound is audible
    src._deliver(np.zeros(1, np.float32))
    assert src.has_sound is audible
    src.start()
    assert not src.has_sound


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
    src.start()
    src.warned = False
    src._deliver(np.full(16, .01, np.float32))
    clock[0] += 21
    src._check_silence()
    assert len(messages) == 1


def test_sink_writes_marker_lines_and_records_without_touching_names(tmp_path):
    path = tmp_path / "meeting.txt"
    sink = Sink(path, "session")
    sink.source(Quiet("system", "silent", T0))
    sink.source(Quiet("system", "back", T0, at=T0 + 855))
    sink.poll_names()
    assert sink.names == {}
    sink.close("ended")
    lines = path.read_text().splitlines()
    assert lines[1:3] == [
        "# system audio lost at 16:37:55 — no signal from the tap; nothing from the call is being transcribed",
        "# system audio back at 16:52:10 after 14m15s lost"]
    records = [json.loads(line)["source"] for line in path.with_suffix(".jsonl").read_text().splitlines()]
    assert records == [
        {"track": "system", "state": "silent", "since": "2026-09-29T16:37:55", "cause": "no signal"},
        {"track": "system", "state": "back", "since": "2026-09-29T16:37:55", "cause": "no signal",
         "back": "2026-09-29T16:52:10"}]


def test_padding_and_empty_blocks_are_not_device_audio(monkeypatch):
    from hark.capture import SAMPLE_RATE, Source

    monkeypatch.setattr("hark.capture.time.time", lambda: T0 + 3)
    src = Source("mic")
    src.anchor = src.last_audio = T0
    src._deliver(np.zeros(0, np.float32))
    assert src.last_audio == T0
    src._deliver(np.zeros(1600, np.float32))
    assert src.last_audio == T0 + 3
    src._deliver(np.full(1600, 0.01, np.float32))
    src.last_audio = src.anchor
    padded = src.drain()
    assert padded.size > SAMPLE_RATE and src.last_audio == src.anchor
    stats = src.take_stats()
    assert stats["device"] == 3200 and stats["padded"] == padded.size - 3200 and stats["peak"] > 0.009
    assert src.take_stats() == {"device": 0, "padded": 0, "peak": 0.0}


def test_live_capture_marks_a_dead_source_and_logs_to_the_session_log(tmp_path, monkeypatch):
    from test_cli import fake_live_capture, signal_when_live

    cli, Source = fake_live_capture(monkeypatch, tmp_path)

    class Dead(Source):
        def _open(self):
            self.last_audio = self.anchor - QUIET_SEC - 5

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
        last_audio = 1.0

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
