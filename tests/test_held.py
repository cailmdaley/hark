"""Muted speech is held, recovered at its original times, and gone when the session ends.

Synthetic PCM, a scripted clock and a fake replay process stand in for the mic and the models."""

import json
import os
import subprocess
import threading
import wave
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from hark import held as held_module
from hark.capture import SAMPLE_RATE, Source
from hark.held import GAP_SEC, HeldSpeech, build_replay
from hark.pause import ManualPause, MicGate, PauseMonitor, PauseState
from hark.transcript import Sink

LOUD, QUIET = 0.25, 1e-4


def tone(seconds, level):
    return np.full(round(seconds * SAMPLE_RATE), level, np.float32)


def request(home):
    return home / "held" / f"{os.getpid()}.recover"


def status(home):
    return json.loads((home / "held" / f"{os.getpid()}.json").read_text())


def finished():
    return SimpleNamespace(poll=lambda: 0, kill=lambda: None, wait=lambda: 0)


class Replay:
    """Stands in for the replay process: one utterance per loud stretch of the replayed audio."""

    def __init__(self):
        self.calls = []

    def __call__(self, manifest, out):
        with wave.open(str(build_replay(manifest)), "rb") as wav:
            samples = np.frombuffer(wav.readframes(wav.getnframes()), "<i2").astype(np.float32) / 32768
        loud = np.abs(samples) > 0.1
        edges = np.flatnonzero(np.diff(np.concatenate(([0], loud.view(np.int8), [0]))))
        rows = [{"wall": "00:00", "track": "replay", "speaker": "S1", "start": first / SAMPLE_RATE,
                 "end": last / SAMPLE_RATE, "text": f"words {i}"}
                for i, (first, last) in enumerate(zip(edges[::2], edges[1::2]))]
        out.with_suffix(".jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.calls.append(samples)
        return finished()


class Muted:
    """A mic muted by hand from wall time 1000 on, through the real gate, its blocks arriving with jitter."""

    def __init__(self, held, jitter=0.0, seed=0):
        self.state = PauseState()
        self.state.update(1000.0 + 0.3, manual=True)  # muted from 1000.0
        self.gate = MicGate(SimpleNamespace(heard=(0, 1000.0)), hold=held.add)
        self.delivered = 0
        self.rng = np.random.default_rng(seed)
        self.jitter = jitter

    def say(self, samples, block=1600):
        for i in range(0, samples.size, block):
            chunk = samples[i:i + block]
            self.delivered += chunk.size
            true = 1000.0 + self.delivered / SAMPLE_RATE
            arrived = true + self.rng.uniform(-self.jitter, self.jitter)
            self.gate.feed(chunk, true + 0.5, self.state, heard=(self.delivered, arrived))


def recover(held, sink, home):
    request(home).touch()
    held.poll(sink)  # starts the replay
    held.poll(sink)  # reads its result


def recovered_walls(transcript):
    return [datetime.fromisoformat(json.loads(line)["wall"]).timestamp()
            for line in transcript.with_suffix(".jsonl").read_text().splitlines() if "recovered" in line]


def test_only_speech_with_its_padding_is_held_and_indexed_by_wall_time(tmp_path):
    held = HeldSpeech(tmp_path / "s.txt", tmp_path, Replay())
    audio = np.concatenate((tone(2, QUIET), tone(.5, LOUD), tone(5, QUIET), tone(.2, LOUD), tone(1, QUIET)))
    for i in range(0, audio.size, 1600):
        held.add(i, 1000 + i / SAMPLE_RATE, audio[i:i + 1600])
    spans = [json.loads(line) for line in (tmp_path / "s.held" / "index.jsonl").read_text().splitlines()]
    assert [row["t"] for row in spans] == pytest.approx([1001.7, 1007.2])
    # 300 ms before each loud stretch, one second after it
    assert held.seconds == pytest.approx(.3 + .5 + 1 + .3 + .2 + 1)
    assert os.path.getsize(tmp_path / "s.held" / "000000.pcm") == held.samples * 2
    assert spans[1]["at"] == round(1.8 * SAMPLE_RATE)
    assert status(tmp_path)["pid"] == os.getpid() and status(tmp_path)["recovering"] is False


@pytest.mark.parametrize("chunk", [777, 1234, 4000, 16000])
def test_any_gate_chunking_keeps_every_loud_sample_and_little_else(chunk, tmp_path):
    held = HeldSpeech(tmp_path / "s.txt", tmp_path, Replay())
    audio = np.concatenate((tone(2, QUIET), tone(.5, LOUD), tone(5, QUIET), tone(.2, LOUD), tone(1, QUIET)))
    for i in range(0, audio.size, chunk):
        held.add(i, 1000 + i / SAMPLE_RATE, audio[i:i + chunk])
    starts = [span[0] for span in held.spans]
    assert 1001.6 < starts[0] <= 1001.7 and 1007.1 < starts[1] <= 1007.2
    assert all(start + n / SAMPLE_RATE >= end for (start, _, _, n, *_), end in zip(held.spans, (1003.5, 1008.4)))
    assert held.seconds < 3.3 + 2 * .2


def test_a_gap_on_the_track_starts_a_new_span(tmp_path):
    held = HeldSpeech(tmp_path / "s.txt", tmp_path, Replay())
    held.add(0, 10, tone(.4, LOUD))
    held.add(160000, 20, tone(.4, LOUD))  # unmuted in between: no tail or preroll carries over
    assert [span[0] for span in held.spans] == [10, 20]
    assert held.seconds == pytest.approx(.8)


def test_gate_holds_exactly_what_a_manual_pause_zeroes():
    state = PauseState()
    state.update(10.0, manual=True)
    state.update(10.5, manual=False, automatic=("aquavoice",))
    state.update(11.0, manual=False, automatic=())
    samples = np.arange(1, 2 * SAMPLE_RATE + 1, dtype=np.float32)
    kept = []
    out = state.gate(samples, 9.5, hold=lambda i, t, part: kept.append((i, t, part)), index=500)
    manual = (samples > (9.7 - 9.5) * SAMPLE_RATE) & (samples <= SAMPLE_RATE)  # [9.7, 10.5)
    assert [(i, t) for i, t, _ in kept] == [(500 + round(.2 * SAMPLE_RATE), pytest.approx(9.7))]
    assert np.array_equal(np.concatenate([part for *_, part in kept]), samples[manual])
    assert not out[manual].any()


def test_a_dead_sessions_hold_is_deleted_at_the_next_start(tmp_path):
    (tmp_path / "held").mkdir()
    for pid, started in ((2 ** 22 + 1, "Mon Jan  1 00:00:00 2024"),  # no such process
                         (os.getppid(), "Mon Jan  1 00:00:00 2024")):  # a running process, but a reused pid
        audio = tmp_path / f"old-{pid}.held"
        audio.mkdir()
        (audio / "000000.pcm").write_bytes(b"x")
        (tmp_path / "held" / f"{pid}.json").write_text(json.dumps({"pid": pid, "started": started, "audio": str(audio)}))
        (tmp_path / "held" / f"{pid}.recover").touch()
    HeldSpeech(tmp_path / "new.txt", tmp_path, Replay())
    assert sorted(p.name for p in tmp_path.iterdir()) == ["held", "new.held"]
    assert sorted(p.name for p in (tmp_path / "held").iterdir()) == [f"{os.getpid()}.json"]


def test_a_second_session_leaves_the_first_ones_hold_and_status_alone(tmp_path, monkeypatch):
    sink = Sink(tmp_path / "voice.txt", "voice")
    voice = HeldSpeech(tmp_path / "voice.txt", tmp_path, Replay(), launch="voice-1")
    voice.add(0, 50, tone(1, LOUD))
    voice.poll(sink)
    other = subprocess.Popen(["sleep", "30"])
    try:
        monkeypatch.setattr(held_module.os, "getpid", lambda: other.pid)
        meeting = HeldSpeech(tmp_path / "meeting.txt", tmp_path, Replay())
        meeting.close()
    finally:
        monkeypatch.undo()
        other.kill()
        other.wait()
    assert status(tmp_path)["launch"] == "voice-1" and status(tmp_path)["seconds"] == pytest.approx(1)
    assert voice.seconds == pytest.approx(1) and (tmp_path / "voice.held" / "000000.pcm").exists()
    assert not (tmp_path / "held" / f"{other.pid}.json").exists()


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
                seen["held"] = status(tmp_path)
                seen["files"] = sorted(p.name for p in (tmp_path / "out").iterdir())
                cli.main(["recover"])
            if step == 45:
                seen["after"] = status(tmp_path)
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

    assert seen["held"]["launch"] == "voice-test" and seen["held"]["seconds"] > 0
    assert seen["files"] == sorted(["session.held", "session.jsonl", "session.log", "session.txt"]
                                   + (["session.partial"] if mode == ["--voice"] else []))
    assert len(replay.calls) == 1
    assert seen["after"]["seconds"] == 0 and not seen["after"]["recovering"]

    lines = out.read_text().splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith("# recovered from mute at "))
    at = lambda seconds: datetime.fromtimestamp(100 + seconds)
    assert lines[header + 1:header + 3] == [
        f"{at(1.0):%H:%M:%S}-{at(1.3):%H:%M:%S} mic  [muted] words 0",
        f"{at(2.8):%H:%M:%S}-{at(3.1):%H:%M:%S} mic  [muted] words 1"]
    records = [json.loads(line) for line in out.with_suffix(".jsonl").read_text().splitlines()]
    recovered = [r for r in records if "recovered" in r]
    assert [datetime.fromisoformat(r["wall"]).timestamp() for r in recovered] == pytest.approx([101.0, 102.8])
    assert all("text" not in r and r["recovered"]["text"] for r in recovered)
    summary = next(r["recovery"] for r in records if "recovery" in r)
    # [0.7, 2.3): preroll, words, one second after; [2.5, 3.5): preroll, words, until the resume
    assert summary["utterances"] == 2 and summary["seconds"] == pytest.approx(2.6)

    assert sorted(p.name for p in out.parent.iterdir()) == sorted(
        ["session.jsonl", "session.log", "session.txt"] + (["session.partial"] if mode == ["--voice"] else []))
    assert list((tmp_path / "held").iterdir()) == []


def test_a_failed_recovery_keeps_the_speech_for_the_next_request(tmp_path):
    sink = Sink(tmp_path / "s.txt", "test")
    replay = Replay()
    exits = [1, 0]

    def flaky(manifest, out):
        process = replay(manifest, out)
        code = exits.pop(0)
        process.poll = lambda: code
        return process

    held = HeldSpeech(tmp_path / "s.txt", tmp_path, flaky)
    held.add(0, 50, tone(3, LOUD))
    recover(held, sink, tmp_path)
    recover(held, sink, tmp_path)
    assert "recovery from mute stopped; 3 s of muted speech still held" in (tmp_path / "s.txt").read_text()
    assert len(replay.calls) == 2 and np.array_equal(replay.calls[0], replay.calls[1])
    assert held.seconds == 0 and held.spans == []
    assert sorted(p.name for p in (tmp_path / "s.held").iterdir()) == ["000002.pcm", "index.jsonl"]
    request(tmp_path).touch()
    held.close()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["held", "s.jsonl", "s.txt"]
    assert list((tmp_path / "held").iterdir()) == []


def test_a_replay_that_writes_nothing_is_stopped_and_the_speech_kept(tmp_path):
    sink = Sink(tmp_path / "s.txt", "test")
    killed = []
    stuck = lambda manifest, out: SimpleNamespace(poll=lambda: None, kill=lambda: killed.append(1), wait=lambda: -9)
    held = HeldSpeech(tmp_path / "s.txt", tmp_path, stuck)
    held.add(0, 50, tone(3, LOUD))
    request(tmp_path).touch()
    held.poll(sink)
    request(tmp_path).touch()  # a second request while one runs is dropped
    held.poll(sink)
    assert held.job and not request(tmp_path).exists() and status(tmp_path)["recovering"]
    old = held.job["since"] - held_module.STALL_SEC - 1
    for path in (tmp_path / "s.held").glob("replay.*"):
        os.utime(path, (old, old))
    held.job["since"], held.job["checked"] = old, held.job["checked"] - 2
    held.poll(sink)
    assert killed and held.job is None
    assert "recovery from mute stopped; 3 s of muted speech still held" in (tmp_path / "s.txt").read_text()
    assert held.seconds == pytest.approx(3) and not status(tmp_path)["recovering"]


def test_the_oldest_segments_go_past_the_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(held_module, "SEGMENT_SEC", 1)
    monkeypatch.setattr(held_module, "CAP_SEC", 2.5)
    sink = Sink(tmp_path / "s.txt", "test")
    held = HeldSpeech(tmp_path / "s.txt", tmp_path, Replay())
    held.add(0, 100, tone(5, LOUD))
    held.poll(sink)
    assert held.seconds <= 2.5 and held.spans[0][0] == pytest.approx(103)
    assert sorted(p.name for p in (tmp_path / "s.held").glob("*.pcm")) == ["000003.pcm", "000004.pcm", "000005.pcm"]


def test_a_disk_error_stops_new_holding_and_what_was_held_is_still_recovered(tmp_path):
    class Full:
        def tell(self):
            return 0

        def write(self, data):
            raise OSError(28, "No space left on device")

        def close(self):
            raise OSError(28, "No space left on device")

    sink = Sink(tmp_path / "s.txt", "test")
    held = HeldSpeech(tmp_path / "s.txt", tmp_path, Replay())
    held.add(0, 50, tone(1, LOUD))
    held.file.flush()
    held.file = Full()
    held.add(16000, 51, tone(1, LOUD))
    held.add(32000, 52, tone(1, LOUD))
    assert held.file is None and held.seconds == pytest.approx(1)
    recover(held, sink, tmp_path)
    assert recovered_walls(tmp_path / "s.txt") == pytest.approx([50])
    held.close()
    assert not (tmp_path / "s.held").exists() and list((tmp_path / "held").iterdir()) == []


def test_a_route_switch_mid_speech_splits_the_span_and_the_replay_holds_them_apart(tmp_path):
    sink = Sink(tmp_path / "s.txt", "test")
    replay = Replay()
    held = HeldSpeech(tmp_path / "s.txt", tmp_path, replay)
    held.cut(51.0)
    held.add(0, 50, tone(2, LOUD))
    assert [span[0] for span in held.spans] == [50, 51] and held.spans[1][5]
    recover(held, sink, tmp_path)
    assert replay.calls[0].size == round((.5 + 1 + GAP_SEC + 1) * SAMPLE_RATE)
    assert recovered_walls(tmp_path / "s.txt") == pytest.approx([50, 51])


def test_jittered_arrival_keeps_one_span_with_its_preroll_and_tail(tmp_path):
    held = HeldSpeech(tmp_path / "s.txt", tmp_path, Replay())
    mic = Muted(held, jitter=0.005)
    mic.say(np.concatenate((tone(.5, QUIET), tone(.3, LOUD), tone(.2, QUIET), tone(.3, LOUD), tone(.5, QUIET))))
    assert len(held.spans) == 1
    assert held.spans[0][0] == pytest.approx(1000.2, abs=0.006)
    assert held.seconds == pytest.approx(1.6)


class Pending:
    """A replay that runs until released."""

    def __init__(self):
        self.replay, self.done = Replay(), False

    def __call__(self, *args):
        process = self.replay(*args)
        process.poll = lambda: 0 if self.done else None
        return process


def test_speech_said_during_a_recovery_is_kept_for_the_next(tmp_path):
    from hark.transcript import Sink

    sink = Sink(tmp_path / "s.txt", "test")
    replay = Pending()
    held = HeldSpeech(tmp_path / "s.txt", tmp_path, replay)
    mic = Muted(held)
    mic.say(tone(1, LOUD))
    request(tmp_path).touch()
    held.poll(sink)
    mic.say(tone(5, LOUD))  # still muted, still talking
    replay.done = True
    held.poll(sink)
    assert held.seconds >= 5 and len(replay.replay.calls) == 1
    assert replay.replay.calls[0].size == round((.5 + 1) * SAMPLE_RATE)


def test_an_early_word_maps_to_its_own_span_not_the_one_before(tmp_path):
    from hark.transcript import Sink

    class Early(Replay):
        def __call__(self, manifest, out):
            process = super().__call__(manifest, out)
            rows = [json.loads(line) for line in out.with_suffix(".jsonl").read_text().splitlines()]
            rows[1]["start"] -= 0.35  # 50 ms before the span's 300 ms preroll
            out.with_suffix(".jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            return process

    sink = Sink(tmp_path / "s.txt", "test")
    held = HeldSpeech(tmp_path / "s.txt", tmp_path, Early())
    mic = Muted(held)
    mic.say(np.concatenate((tone(1, QUIET), tone(.5, LOUD), tone(60, QUIET), tone(.5, LOUD), tone(2, QUIET))))
    recover(held, sink, tmp_path)
    assert recovered_walls(tmp_path / "s.txt")[1] == pytest.approx(1000 + 61.5 - 0.3, abs=0.1)


def test_short_remarks_minutes_apart_replay_as_separate_utterances(tmp_path):
    """The real voice track, on the replayed audio, ends one remark before the next begins."""
    from hark.capture import load_audio
    from hark.transcript import Sink, flush_tracks, numbered
    from hark.live_voice import VoiceTrack

    class Recognizer:
        def __init__(self):
            self._mel_offset = 0

        def feed(self, samples, final=False):
            start = self._mel_offset * .01
            self._mel_offset += len(samples) // 160
            if samples.size and np.abs(samples).max() > .1:
                return [SimpleNamespace(speaker="speaker_0",
                                        tokens=[SimpleNamespace(start=start, end=start + .08, text=" word")])]
            return []

    def voice_replay(manifest, out):
        session = Recognizer()
        asr = SimpleNamespace(preprocessor_config=SimpleNamespace(hop_length=160),
                              create_speaker_streaming_session=lambda *a, **kw: session)
        track = VoiceTrack("replay", asr, None, speaker_label=numbered, refiner=lambda s: "a short remark")
        track.t0 = datetime(2026, 1, 1).astimezone()
        samples = load_audio(build_replay(manifest))
        said = []
        for i in range(0, samples.size, 8000):
            track.feed(samples[i:i + 8000])
            said += flush_tracks([track])
        track.feed(samples[:0], final=True)
        said += flush_tracks([track], force=True)
        out.with_suffix(".jsonl").write_text("".join(json.dumps(u.record()) + "\n" for u in said))
        return finished()

    sink = Sink(tmp_path / "s.txt", "test")
    held = HeldSpeech(tmp_path / "s.txt", tmp_path, voice_replay)
    mic = Muted(held)
    mic.say(np.concatenate((tone(1, QUIET), tone(.5, LOUD), tone(60, QUIET), tone(.5, LOUD), tone(2, QUIET))))
    recover(held, sink, tmp_path)
    recovered = [line for line in (tmp_path / "s.jsonl").read_text().splitlines() if "recovered" in line]
    assert len(recovered) == 2


def test_a_fraction_of_a_second_held_is_published(tmp_path):
    held = HeldSpeech(tmp_path / "s.txt", tmp_path, Replay())
    held.add(0, 50, tone(.3, LOUD))
    held.poll(Sink(tmp_path / "s.txt", "test"))
    assert status(tmp_path)["seconds"] == pytest.approx(.3)


def test_a_disk_error_that_also_fails_the_log_never_reaches_the_caller(tmp_path, monkeypatch):
    class Full:
        def tell(self):
            return 0

        def write(self, data):
            raise OSError(28, "No space left on device")

        def close(self):
            raise OSError(28, "No space left on device")

    def full_log(message, file_only=False):
        raise OSError(28, "No space left on device")

    held = HeldSpeech(tmp_path / "s.txt", tmp_path, Replay())
    held.add(0, 50, tone(1, LOUD))
    monkeypatch.setattr(held_module, "log", full_log)
    held.file = Full()
    held.add(16000, 51, tone(1, LOUD))
    held.add(32000, 52, tone(1, LOUD))


def test_speech_held_before_a_disk_error_is_recovered_once_it_clears(tmp_path):
    sink = Sink(tmp_path / "s.txt", "test")
    held = HeldSpeech(tmp_path / "s.txt", tmp_path, Replay())
    held.add(0, 50, tone(1, LOUD))
    (tmp_path / "s.held" / "replay.json").mkdir()  # the manifest cannot be written
    recover(held, sink, tmp_path)
    assert held.seconds == pytest.approx(1)
    (tmp_path / "s.held" / "replay.json").rmdir()
    recover(held, sink, tmp_path)
    for _ in range(3):
        held.poll(sink)
    assert recovered_walls(tmp_path / "s.txt") == pytest.approx([50])
    assert held.seconds == 0 and not request(tmp_path).exists()


def test_a_large_hold_recovers_in_bounded_replays_as_one_recovery(tmp_path, monkeypatch):
    monkeypatch.setattr(held_module, "BATCH_SEC", 10)
    sink = Sink(tmp_path / "s.txt", "test")
    replay = Replay()
    held = HeldSpeech(tmp_path / "s.txt", tmp_path, replay)
    for k in range(3):
        held.add(k * 16000 * 100, 1000 + 100 * k, tone(1, LOUD))
    request(tmp_path).touch()
    for _ in range(8):
        held.poll(sink)
    assert len(replay.calls) == 3 and all(call.size <= 10 * SAMPLE_RATE for call in replay.calls)
    assert recovered_walls(tmp_path / "s.txt") == pytest.approx([1000, 1100, 1200])
    rows = [json.loads(line) for line in (tmp_path / "s.jsonl").read_text().splitlines()]
    assert [r["recovery"]["utterances"] for r in rows if "recovery" in r] == [3]
    assert held.seconds == 0 and not status(tmp_path)["recovering"]


@pytest.mark.parametrize("after", [0.2, 0.0])
def test_a_route_switch_splits_held_speech_at_its_sample(tmp_path, after):
    held = HeldSpeech(tmp_path / "s.txt", tmp_path, Replay())
    held.cut(50.05)
    held.add(0, 50.0, tone(0.1 + after, LOUD))  # the switch falls inside the first frame, the last one held when after = 0
    assert [span[0] for span in held.spans][:2] == pytest.approx([50.0, 50.05])
    assert held.spans[0][3] == round(.05 * SAMPLE_RATE)


def test_speech_held_after_a_short_first_reading_is_published(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(held_module, "time", SimpleNamespace(monotonic=lambda: clock[0], time=lambda: 1e9 + clock[0]))
    sink = Sink(tmp_path / "s.txt", "test")
    held = HeldSpeech(tmp_path / "s.txt", tmp_path, Replay())
    held.add(0, 50, tone(.01, LOUD))
    held.poll(sink)
    held.add(160, 50.01, tone(.3, LOUD))
    clock[0] = 2
    held.poll(sink)
    assert status(tmp_path)["seconds"] == pytest.approx(.31)


def test_the_cap_holds_during_a_recovery_without_touching_the_batch_being_replayed(tmp_path, monkeypatch):
    monkeypatch.setattr(held_module, "SEGMENT_SEC", 1)
    monkeypatch.setattr(held_module, "CAP_SEC", 2.5)
    sink = Sink(tmp_path / "s.txt", "test")
    replay = Pending()
    held = HeldSpeech(tmp_path / "s.txt", tmp_path, replay)
    held.add(0, 100, tone(2, LOUD))
    request(tmp_path).touch()
    held.poll(sink)
    replaying = [span[:] for span in held.job["spans"]]
    held.add(32000, 102, tone(5, LOUD))  # still muted, still talking, while the replay runs
    held.poll(sink)
    assert held.seconds <= 2.5
    assert held.spans[:len(replaying)] == replaying
    assert all(held._segment(span[1]).exists() for span in replaying)
    assert "muted speech said before" in (tmp_path / "s.txt").read_text()
    replay.done = True
    held.poll(sink)
    assert recovered_walls(tmp_path / "s.txt")


def test_recover_refuses_a_session_whose_process_is_gone(tmp_path, monkeypatch, capsys):
    from hark import cli

    monkeypatch.setattr(cli, "HOME", tmp_path)
    (tmp_path / "held").mkdir()
    transcript = tmp_path / "gone.txt"
    transcript.touch()
    (tmp_path / "held" / "9999999.json").write_text(json.dumps(
        {"pid": 9999999, "started": "Mon Jan  1 00:00:00 2024", "transcript": str(transcript), "seconds": 4.0}))
    assert cli.main(["recover", "--session", str(transcript)]) == 1
    assert not (tmp_path / "held" / "9999999.recover").exists()
    assert "no live session holds muted speech" in capsys.readouterr().out


def test_a_status_that_could_not_be_put_in_place_is_gone_after_close(tmp_path, monkeypatch):
    sink = Sink(tmp_path / "s.txt", "test")
    held = HeldSpeech(tmp_path / "s.txt", tmp_path, Replay())
    held.add(0, 50, tone(1, LOUD))

    def refused(*args):
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(held_module.os, "replace", refused)
    held.poll(sink)
    monkeypatch.undo()
    assert (tmp_path / "held" / f"{os.getpid()}.json.tmp").exists()
    held.close()
    assert list((tmp_path / "held").iterdir()) == []
