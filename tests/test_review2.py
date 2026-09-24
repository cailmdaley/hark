"""Review demos for 83911ce + b9a8e15. Every test here asserts the *intended* behaviour and fails at b9a8e15."""

import queue
import time
from datetime import datetime
from types import SimpleNamespace

import numpy as np

from hark.capture import SAMPLE_RATE, Source
from hark.transcript import EchoGate, Sink, Utterance, flush_tracks
from hark.voice import AudioBuffer, VoiceMatcher
from tests.test_turns import track

CHUNK = 1.12  # SpeakerStreamingSession chunk: (att_context[1] + 1) * 8 mel frames = 112 * 10 ms
FRAME = 0.08  # every AlignedToken lasts exactly one encoder frame


def tok(start, text):
    return SimpleNamespace(text=text, start=start, end=start + FRAME)


def spoken(start, end, text):
    ws = text.split()
    step = (end - start) / len(ws)
    return [tok(start + i * step, " " + w) for i, w in enumerate(ws)]


def live_call(mic_tokens, system_tokens, until, *, mic_t0=100, system_t0=100):
    """Drive capture → release → flush the way cli.main does, one session chunk per iteration."""
    mic = track("mic", {}, processed=0, start=mic_t0)
    system = track("system", {}, processed=0, start=system_t0)
    future = {id(mic): ("me", list(mic_tokens)), id(system): ("S1", list(system_tokens))}
    gate, out, p = EchoGate(), [], 0.0

    def emit_until(t, p):
        t.session._mel_offset = p * SAMPLE_RATE
        speaker, tokens = future[id(t)]
        ready = [x for x in tokens if x.end <= p]
        future[id(t)] = (speaker, [x for x in tokens if x.end > p])
        t.pending.setdefault(speaker, SimpleNamespace(tokens=[])).tokens.extend(ready)

    while p < until:
        p = round(p + CHUNK, 2)
        emit_until(mic, p), emit_until(system, p)
        gate.capture(mic, system)
        gate.release(mic, system._wall_time(system.processed))
        out += flush_tracks([mic, system])
    gate.capture(mic, system)
    gate.release(mic, float("inf"), final=True)
    return out + flush_tracks([mic, system], force=True)


# ── 1. EchoGate recirculates kept mic tokens behind still-waiting ones → word order scrambled ──

def test_live_mic_speech_keeps_word_order():
    words = "one two three four five six seven eight nine ten"
    out = live_call(spoken(0, 5, words), [], until=12)
    assert [(u.speaker, u.text) for u in out] == [("me", words)]


# ── 2. Echo removal in the real loop (chunked watermark, recirculation) ──

def test_live_echo_is_fully_removed():
    remote = "we should really start the review of the shear pipeline now"
    system_tokens = spoken(0, 5, remote)
    echo = [tok(t.start + 0.12, t.text) for t in system_tokens]
    out = live_call(echo, system_tokens, until=12)
    assert [(u.speaker, u.text) for u in out] == [("S1", remote)]


# ── 3. Sink.poll_names adds a *character* count to a byte-position cookie ──

def test_poll_names_survives_non_ascii_transcript(tmp_path):
    sink = Sink(tmp_path / "réunion.txt", "session")
    for i in range(6):
        sink.write(Utterance("system", "S1", i, i + 1, datetime.now(), "é" * (i + 1)))
    with (tmp_path / "réunion.txt").open("a") as external:
        external.write("# S1 = Hélène\n")
    sink.poll_names()  # UnicodeDecodeError: seek lands inside a multibyte character
    assert sink.names == {"S1": "Hélène"}


# ── 4. Source.drain pads with silence while real audio is still queued ──

def test_drain_does_not_pad_while_backlog_is_queued():
    src = Source("mic")
    src.anchor = time.time() - 2.0  # the loop stalled 2 s (first MLX inference, an ONNX embed…)
    for _ in range(20):
        src.queue.put(np.ones(1600, np.float32))  # …while the device delivered all 2 s
    out = src.drain(limit=SAMPLE_RATE // 2)
    assert np.count_nonzero(out == 0) == 0, f"{np.count_nonzero(out == 0) / SAMPLE_RATE:.2f} s of fake silence"


# ── 5. AudioBuffer.slice with a span before the buffer returns unrelated audio ──

def test_slice_before_buffer_is_empty():
    buffer = AudioBuffer(seconds=45)
    buffer.append(0, np.arange(60 * 16000, dtype=np.float32))  # keeps [15, 60)
    assert buffer.slice(10, 12).size == 0


# ── 6. Real tokens are 80 ms frames: "speech" and the embedded audio are ~4x too short ──

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


# ── 7. Unnamed slots keep every segment forever once past the 20-s checkpoint ──

def test_segments_stay_bounded(tmp_path):
    sink = Sink(tmp_path / "s.txt", "session")
    buffer = AudioBuffer()
    buffer.append(0, np.ones(45 * 16000, dtype=np.float32))
    matcher = VoiceMatcher({"ada": np.array([1.0, 0.0])}, sink, embedder=lambda _: np.array([0.0, 1.0]))
    t = SimpleNamespace(audio=buffer)
    for i in range(2000):  # a long meeting with someone not enrolled
        matcher.finished(t, Utterance("system", "S1", i, i + 1, datetime.now(), "x", speech=[(i, i + 1)]))
    assert max(len(d) for d in matcher.segments.values()) < 100


# ── 8. EchoGate maps system tokens through the *mic* clock ──

def test_echo_uses_each_tracks_own_clock():
    remote = "we should really start the review of the shear pipeline now"
    system_tokens = spoken(0, 5, remote)  # system t0 = 102: remote speech at wall 102–107
    echo = [tok(t.start + 2.1, t.text) for t in system_tokens]  # mic t0 = 100: same wall instants
    mic = track("mic", {"me": tuples(echo)}, processed=12, start=100)
    system = track("system", {"S1": tuples(system_tokens)}, processed=10, start=102)
    gate = EchoGate()
    gate.capture(mic, system)
    assert gate.release(mic, float("inf"), final=True) == []


# ── 9. Overlapping speech: the new cut-at-interruption fragments both speakers ──

def test_crosstalk_does_not_fragment_into_single_words():
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


# ── 10. The final force-flush ignores turn cuts: A–B–A at stop merges A's two turns ──

def test_final_flush_still_splits_turns():
    a = spoken(0, 2, "first part here") + spoken(3.6, 4.0, "and more")
    b = spoken(2.2, 3.4, "wait hold on")
    room = track("room", {"A": tuples(a), "B": tuples(b)}, processed=4.1)
    out = flush_tracks([room], force=True)
    assert [u.text for u in out] == ["first part here", "wait hold on", "and more"]
