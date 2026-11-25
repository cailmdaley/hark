"""Muted speech is held, recovered at its original times, and gone when the session ends.

Synthetic PCM, a scripted clock and a fake replay process stand in for the mic and the models."""

import json
import os
import threading
import wave
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from hark.capture import SAMPLE_RATE, Source
from hark.held import HeldSpeech
from hark.pause import ManualPause, PauseMonitor, PauseState

LOUD, QUIET = 0.25, 1e-4


def tone(seconds, level):
    return np.full(round(seconds * SAMPLE_RATE), level, np.float32)


class Replay:
    """Stands in for `hark --file`: one utterance per loud stretch of the replayed audio."""

    def __init__(self):
        self.calls = []

    def __call__(self, audio, out):
        with wave.open(str(audio), "rb") as wav:
            samples = np.frombuffer(wav.readframes(wav.getnframes()), "<i2").astype(np.float32) / 32768
        loud = np.abs(samples) > 0.1
        edges = np.flatnonzero(np.diff(np.concatenate(([0], loud.view(np.int8), [0]))))
        rows = [{"wall": "00:00", "track": audio.stem, "speaker": "S1", "start": first / SAMPLE_RATE,
                 "end": last / SAMPLE_RATE, "text": f"words {i}"}
                for i, (first, last) in enumerate(zip(edges[::2], edges[1::2]))]
        out.with_suffix(".jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.calls.append(samples)
        return SimpleNamespace(poll=lambda: 0, kill=lambda: None, wait=lambda: 0)


def test_only_speech_with_its_padding_is_held_and_indexed_by_wall_time(tmp_path):
    held = HeldSpeech(tmp_path / "s.txt", tmp_path, Replay())
    audio = np.concatenate((tone(2, QUIET), tone(.5, LOUD), tone(5, QUIET), tone(.2, LOUD), tone(1, QUIET)))
    for i in range(0, audio.size, 1600):
        held.add(1000 + i / SAMPLE_RATE, audio[i:i + 1600])
    spans = [json.loads(line) for line in (tmp_path / "s.held.jsonl").read_text().splitlines()]
    assert [row["t"] for row in spans] == pytest.approx([1001.7, 1007.2])
    # 300 ms before each loud stretch, one second after it
    assert held.seconds == pytest.approx(.3 + .5 + 1 + .3 + .2 + 1)
    assert os.path.getsize(tmp_path / "s.held.pcm") == held.samples * 2
    assert spans[1]["at"] == round(1.8 * SAMPLE_RATE)
    status = json.loads((tmp_path / "held.json").read_text())
    assert status["pid"] == os.getpid() and status["recovering"] is False


@pytest.mark.parametrize("chunk", [777, 1234, 4000, 16000])
def test_any_gate_chunking_keeps_every_loud_sample_and_little_else(chunk, tmp_path):
    held = HeldSpeech(tmp_path / "s.txt", tmp_path, Replay())
    audio = np.concatenate((tone(2, QUIET), tone(.5, LOUD), tone(5, QUIET), tone(.2, LOUD), tone(1, QUIET)))
    for i in range(0, audio.size, chunk):
        held.add(1000 + i / SAMPLE_RATE, audio[i:i + chunk])
    starts = [span[0] for span in held.spans]
    assert 1001.6 < starts[0] <= 1001.7 and 1007.1 < starts[1] <= 1007.2
    assert all(start + n / SAMPLE_RATE >= end for (start, _, n), end in zip(held.spans, (1003.5, 1008.4)))
    assert held.seconds < 3.3 + 2 * .2


def test_a_gap_in_the_mute_starts_a_new_span(tmp_path):
    held = HeldSpeech(tmp_path / "s.txt", tmp_path, Replay())
    held.add(10, tone(.4, LOUD))
    held.add(20, tone(.4, LOUD))  # unmuted in between: no tail or preroll carries over
    assert [span[0] for span in held.spans] == [10, 20]
    assert held.seconds == pytest.approx(.8)


def test_gate_holds_exactly_what_a_manual_pause_zeroes():
    state = PauseState()
    state.update(10.0, manual=True)
    state.update(10.5, manual=False, automatic=("aquavoice",))
    state.update(11.0, manual=False, automatic=())
    samples = np.arange(1, 2 * SAMPLE_RATE + 1, dtype=np.float32)
    kept = []
    out = state.gate(samples, 9.5, hold=lambda t, part: kept.append((t, part)))
    manual = (samples > (9.7 - 9.5) * SAMPLE_RATE) & (samples <= SAMPLE_RATE)  # [9.7, 10.5)
    assert [t for t, _ in kept] == pytest.approx([9.7])
    assert np.array_equal(np.concatenate([part for _, part in kept]), samples[manual])
    assert not out[manual].any()


def test_a_dead_sessions_hold_is_deleted_at_the_next_start(tmp_path):
    left = tmp_path / "old.held.pcm"
    for path in (left, tmp_path / "old.held.jsonl", tmp_path / "old.recovering.wav"):
        path.write_bytes(b"x")
    (tmp_path / "held.json").write_text(json.dumps({"pid": 2 ** 22 + 1, "audio": str(left)}))
    HeldSpeech(tmp_path / "new.txt", tmp_path, Replay())
    assert sorted(p.name for p in tmp_path.iterdir()) == ["held.json", "new.held.pcm"]
    assert json.loads((tmp_path / "held.json").read_text())["audio"] == str(tmp_path / "new.held.pcm")


def test_a_live_sessions_hold_is_left_alone(tmp_path):
    other = tmp_path / "other.held.pcm"
    other.write_bytes(b"x")
    (tmp_path / "held.json").write_text(json.dumps({"pid": os.getppid(), "audio": str(other)}))
    HeldSpeech(tmp_path / "new.txt", tmp_path, Replay())
    assert other.exists()


@pytest.mark.parametrize("mode", [["--room"], ["--voice"]])
def test_speech_said_while_paused_is_recovered_at_its_time_then_dropped_and_gone_at_stop(
        mode, tmp_path, monkeypatch):
    from hark import cli

    clock = [100.0]
    monkeypatch.setattr("time.time", lambda: clock[0])
    blocks = [np.full(1600, LOUD if step in (10, 11, 12, 28, 29, 30) else QUIET, np.float32)
              for step in range(50)]
    replay = Replay()
    seen = {}
    signals = []

    class Signals:
        def __init__(self, lifecycle, stop):
            self.lifecycle, self.stop = lifecycle, stop
            self.signal_written = threading.Event()
            self.signal_written.set()
            signals.append(self)

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
                signals[0].stop.set()
                return np.zeros(0, np.float32)
            step = self.index
            if step == 8:
                cli.main(["pause"])
            if step == 34:
                cli.main(["resume"])
            if step == 38:
                seen["held"] = json.loads((tmp_path / "held.json").read_text())
                seen["files"] = sorted(p.name for p in (tmp_path / "out").iterdir())
                cli.main(["recover"])
            if step == 45:
                seen["after"] = json.loads((tmp_path / "held.json").read_text())
            self.index += 1
            clock[0] = self.anchor + self.index / 10
            self.last_audio = clock[0]
            return blocks[step]

    class Track:
        def __init__(self, name, *args, **kwargs):
            self.name, self.processed, self.audio = name, 0, None

        def feed(self, samples, final=False):
            pass

    class Monitor(PauseMonitor):
        def start(self):
            self.poll()

        def feed(self, gate, samples, now, final=False, heard=None):
            self.poll()
            return super().feed(gate, samples, now, final=final, heard=heard)

    monkeypatch.setattr(cli, "HOME", tmp_path)
    monkeypatch.setattr(cli, "_default_ear", lambda: "local")
    monkeypatch.setattr(cli, "_device_capture_available", lambda: True)
    monkeypatch.setattr(cli, "_system_capture_available", lambda: False)
    monkeypatch.setattr(cli, "SignalWatcher", Signals)
    monkeypatch.setattr(cli, "MicSource", Mic)
    monkeypatch.setattr(cli, "PauseMonitor", Monitor)
    monkeypatch.setattr(cli, "Track", Track)
    monkeypatch.setattr(cli, "load_models", lambda *_: (None, None))
    monkeypatch.setattr(cli, "flush_tracks", lambda *_, **__: [])
    monkeypatch.setattr("hark.live_voice.VoiceTrack", Track)
    monkeypatch.setattr("hark.live_voice.load_refiner", lambda **kwargs: None)
    monkeypatch.setattr(cli, "transcribe_file", lambda *args: replay)
    ManualPause(tmp_path).set(False)
    out = tmp_path / "out" / "session.txt"
    cli.main(mode + ["--ear", "local", "--pause-for", "none", "--launch", "voice-test", "-o", str(out)])

    assert seen["held"]["launch"] == "voice-test" and seen["held"]["seconds"] >= 1
    assert seen["files"] == sorted(["session.held.jsonl", "session.held.pcm", "session.jsonl",
                                    "session.log", "session.txt"] + (["session.partial"] if mode == ["--voice"] else []))
    assert len(replay.calls) == 1
    assert seen["after"]["seconds"] == 0 and not seen["after"]["recovering"]

    lines = out.read_text().splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith("# recovered from mute at "))
    at = lambda seconds: datetime.fromtimestamp(100 + seconds)
    assert lines[header + 1:header + 3] == [
        f"{at(1.0):%H:%M:%S}-{at(1.3):%H:%M:%S} S1   [muted] words 0",
        f"{at(2.8):%H:%M:%S}-{at(3.1):%H:%M:%S} S1   [muted] words 1"]
    records = [json.loads(line) for line in out.with_suffix(".jsonl").read_text().splitlines()]
    recovered = [r for r in records if "recovered" in r]
    assert [datetime.fromisoformat(r["wall"]).timestamp() for r in recovered] == pytest.approx([101.0, 102.8])
    assert all("text" not in r and r["recovered"]["text"] for r in recovered)
    summary = next(r["recovery"] for r in records if "recovery" in r)
    # [0.7, 2.3): preroll, words, one second after; [2.5, 3.5): preroll, words, until the resume
    assert summary["utterances"] == 2 and summary["seconds"] == pytest.approx(2.6)

    assert sorted(p.name for p in out.parent.iterdir()) == sorted(
        ["session.jsonl", "session.log", "session.txt"] + (["session.partial"] if mode == ["--voice"] else []))
    assert not (tmp_path / "held.json").exists() and not (tmp_path / "recover").exists()


def test_a_failed_recovery_keeps_the_speech_for_the_next_request(tmp_path):
    from hark.transcript import Sink

    sink = Sink(tmp_path / "s.txt", "test")
    replay = Replay()
    exits = [1, 0]

    def flaky(audio, out):
        process = replay(audio, out)
        code = exits.pop(0)
        process.poll = lambda: code
        return process

    held = HeldSpeech(tmp_path / "s.txt", tmp_path, flaky)
    held.add(50, tone(3, LOUD))
    for _ in range(2):
        (tmp_path / "recover").touch()
        held.poll(sink)  # starts the replay
        held.poll(sink)  # reads its result
    assert "recovery from mute failed; 3 s of muted speech still held" in (tmp_path / "s.txt").read_text()
    assert len(replay.calls) == 2 and np.array_equal(replay.calls[0], replay.calls[1])
    assert held.seconds == 0 and (tmp_path / "s.held.pcm").stat().st_size == 0
    assert [p.name for p in tmp_path.glob("s.recovering.*")] == []
    held.close()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["s.jsonl", "s.txt"]
