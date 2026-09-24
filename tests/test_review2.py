"""Regression tests for transcript timing, buffering, and speaker matching."""

import time
from datetime import datetime
from types import SimpleNamespace

import numpy as np

from hark.capture import SAMPLE_RATE, Source
from hark.transcript import Sink, Utterance, flush_tracks
from hark.voice import AudioBuffer, VoiceMatcher
from tests.test_turns import track

FRAME = 0.08  # every AlignedToken lasts exactly one encoder frame


def tok(start, text):
    return SimpleNamespace(text=text, start=start, end=start + FRAME)


def spoken(start, end, text):
    ws = text.split()
    step = (end - start) / len(ws)
    return [tok(start + i * step, " " + w) for i, w in enumerate(ws)]


# ── Transcript name polling ──

def test_poll_names_survives_non_ascii_transcript(tmp_path):
    sink = Sink(tmp_path / "réunion.txt", "session")
    for i in range(6):
        sink.write(Utterance("system", "S1", i, i + 1, datetime.now(), "é" * (i + 1)))
    with (tmp_path / "réunion.txt").open("a") as external:
        external.write("# S1 = Hélène\n")
    sink.poll_names()  # Complete UTF-8 lines remain readable from a byte offset.
    assert sink.names == {"S1": "Hélène"}


# ── Source buffering ──

def test_drain_does_not_pad_while_backlog_is_queued():
    src = Source("mic")
    src.anchor = time.time() - 2.0  # the loop stalled 2 s (first MLX inference, an ONNX embed…)
    for _ in range(20):
        src.queue.put(np.ones(1600, np.float32))  # …while the device delivered all 2 s
    out = src.drain(limit=SAMPLE_RATE // 2)
    assert np.count_nonzero(out == 0) == 0, f"{np.count_nonzero(out == 0) / SAMPLE_RATE:.2f} s of fake silence"


# ── Audio buffer bounds ──

def test_slice_before_buffer_is_empty():
    buffer = AudioBuffer(seconds=45)
    buffer.append(0, np.arange(60 * 16000, dtype=np.float32))  # keeps [15, 60)
    assert buffer.slice(10, 12).size == 0


# ── Token timing ──

def tuples(tokens):
    return [(t.start, t.end, t.text) for t in tokens]


def realistic_utterance(start, seconds, rate=3.0):
    tokens = [(start + i / rate, start + i / rate + FRAME) for i in range(int(seconds * rate))]
    return Utterance("system", "S1", tokens[0][0], tokens[-1][1], datetime.now(), "speech", speech=tokens)


def test_voice_is_named_after_a_minute_of_speech(tmp_path):
    sink = Sink(tmp_path / "s.txt", "session")
    buffer = AudioBuffer(seconds=10_000)
    buffer.append(0, np.ones(200 * 16000, dtype=np.float32))
    matcher = VoiceMatcher({"ada": np.array([1.0, 0.0])}, sink, embedder=lambda _: np.array([1.0, 0.0]))
    for i in range(6):  # six 10-s turns = 60 s of continuous talk
        matcher.finished(SimpleNamespace(audio=buffer), realistic_utterance(20 * i, 10))
    assert sink.names == {"S1": "ada"}


# ── Bounded voice-match history ──

def test_segments_stay_bounded(tmp_path):
    sink = Sink(tmp_path / "s.txt", "session")
    buffer = AudioBuffer()
    buffer.append(0, np.ones(45 * 16000, dtype=np.float32))
    matcher = VoiceMatcher({"ada": np.array([1.0, 0.0])}, sink, embedder=lambda _: np.array([0.0, 1.0]))
    t = SimpleNamespace(audio=buffer)
    for i in range(2000):  # a long meeting with someone not enrolled
        matcher.finished(t, Utterance("system", "S1", i, i + 1, datetime.now(), "x", speech=[(i, i + 1)]))
    assert max(len(d) for d in matcher.segments.values()) < 100


# ── Overlapping speech ──

def test_crosstalk_does_not_fragment_into_single_tokens():
    a = spoken(0, 6, "so the thing about the covariance is that it depends")
    b = [tok(t.start + 0.25, " yeah") for t in a[::2]]  # B keeps talking over A
    room = track("room", {"A": [], "B": []}, processed=0)
    out, pending = [], {"A": list(a), "B": list(b)}
    for step in range(1, 12):
        p = step * 0.56
        room.session._mel_offset = p * SAMPLE_RATE
        for s in pending:
            room.pending[s].tokens += [x for x in pending[s] if x.end <= p]
            pending[s] = [x for x in pending[s] if x.end > p]
        out += flush_tracks([room])
    out += flush_tracks([room], force=True)
    assert len([u for u in out if u.speaker == "A"]) <= 2, [u.text for u in out]


# ── Repeated cuts at final flush ──

def test_final_flush_splits_a_b_a_b_a_turns():
    a = (spoken(0, 2, "first part here") + spoken(3.6, 5.4, "and more words here")
         + spoken(7, 8.4, "final part here"))
    b = spoken(2.2, 3.4, "wait hold on") + spoken(5.6, 6.8, "another wait here")
    room = track("room", {"A": tuples(a), "B": tuples(b)}, processed=8.5)
    out = flush_tracks([room], force=True)
    assert [u.text for u in out] == ["first part here", "wait hold on", "and more words here",
                                    "another wait here", "final part here"]
