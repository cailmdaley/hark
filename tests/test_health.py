import json
import re
import threading
import time
from datetime import datetime
from types import SimpleNamespace

import numpy as np

from hark.health import QUIET_SEC, Quiet, QuietWatch
from hark.transcript import Sink

T0 = datetime(2026, 9, 29, 16, 37, 55).timestamp()


def test_no_signal_goes_quiet_unconditionally_from_the_last_delivery():
    watch = QuietWatch()
    sources = {"mic": (T0, T0)}
    assert watch.check(T0 + QUIET_SEC - 1, sources, {}) == []
    assert watch.check(T0 + QUIET_SEC, sources, {}) == [Quiet("mic", "silent", T0, "no signal")]


def test_digital_silence_counts_only_while_another_track_speaks():
    watch = QuietWatch()
    now = T0 + 100
    sources = {"system": (now, T0)}  # samples keep arriving, all zeros
    assert watch.check(now, sources, {}) == []
    assert watch.check(now, sources, {"system": now - 5}) == []  # its own track doesn't count
    assert watch.check(now, sources, {"mic": now - QUIET_SEC - 1}) == []  # too long ago
    assert watch.check(now, sources, {"mic": now - 5}) == [Quiet("system", "silent", T0, "silence")]


def test_both_sources_silent_in_a_lull_is_not_a_marker():
    watch = QuietWatch()
    now = T0 + 600
    sources = {"mic": (now, T0), "system": (now, T0)}
    assert watch.check(now, sources, {"mic": T0 - 10, "system": T0 - 20}) == []


def test_one_marker_per_episode_then_back_then_a_new_episode():
    watch = QuietWatch()
    events = [e for now in np.arange(T0, T0 + 900, 0.5)
              for e in watch.check(now, {"system": (T0, T0)}, {"mic": now})]
    assert events == [Quiet("system", "silent", T0, "no signal")]
    back = T0 + 855
    assert watch.check(back + 1, {"system": (back, back)}, {}) == [
        Quiet("system", "back", T0, "no signal", back)]
    assert watch.check(back + 2, {"system": (back + 2, back)}, {}) == []
    assert watch.check(back + QUIET_SEC, {"system": (back + QUIET_SEC, back)}, {"mic": back + 80}) == [
        Quiet("system", "silent", back, "silence")]


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
    assert src.last_sound == src.anchor
    src._deliver(np.full(1600, 0.01, np.float32))
    heard = src.last_sound
    assert heard > src.anchor
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
    cli.main(["--room", "-o", str(out)])
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
    monkeypatch.setattr(cli, "FileSource", Stale)
    monkeypatch.setattr(cli, "Track", FakeTrack)
    monkeypatch.setattr(cli, "load_models", lambda latency: (None, None))
    monkeypatch.setattr(cli, "flush_tracks", lambda tracks, force=False: [])
    (tmp_path / "x.wav").touch()

    cli.main(["--file", str(tmp_path / "x.wav"), "-o", str(tmp_path / "replay")])

    assert " lost at " not in (tmp_path / "replay.txt").read_text()
    assert '"source"' not in (tmp_path / "replay.jsonl").read_text()
    assert "transcript → " in (tmp_path / "replay.log").read_text()
