from types import SimpleNamespace

from hark.capture import SAMPLE_RATE
from hark.transcript import EchoGate, Sink, _monologue_cut, flush_tracks
from hark.cli import _name_current
from tests.test_turns import track


def words(start, end, text):
    ws = text.split()
    step = (end - start) / len(ws)
    return [(start + i * step, start + (i + 1) * step - 0.05, " " + w) for i, w in enumerate(ws)]


def gated(mic, system, *, watermark=float("inf")):
    gate = EchoGate()
    gate.capture(mic, system)
    gate.release(mic, watermark, final=watermark == float("inf"))
    return flush_tracks([mic, system], force=True)


def test_echo_is_removed_before_system_line_flush():
    text = "we should really start the review of the shear pipeline now please everyone"
    remote = words(0, 10, text)
    mic = track("mic", {"me": [t for t in remote if t[1] < 8.5]}, processed=12)
    system = track("system", {"S1": remote}, processed=10)
    result = gated(mic, system, watermark=112)
    assert [u for u in result if u.track == "mic"] == []


def test_echo_waits_until_system_asr_watermark_passes_token():
    mic = track("mic", {"me": words(0, 4, "hello world")}, processed=5)
    system = track("system", {"S1": words(0, 4, "hello world")}, processed=3)
    gate = EchoGate()
    gate.capture(mic, system)
    assert gate.release(mic, 102) == []
    assert len(gate.pending) == 2
    assert gate.release(mic, 110) == []
    assert gate.pending == []
    assert not mic.pending["me"].tokens


def test_echo_does_not_cut_system_turn():
    remote = words(0, 5, "one two three four five")
    mic = track("mic", {"me": remote + [(5.1, 5.5, " six"), (5.6, 6.2, " seven")]}, processed=7)
    system = track("system", {"S1": remote}, processed=5.3)
    gate = EchoGate()
    gate.capture(mic, system)
    gate.release(mic, 105.3)
    assert [u.track for u in flush_tracks([mic, system])] == []


def test_echo_match_is_local_in_time_and_needs_two_words():
    mic = track("mic", {"me": words(20, 21, "yeah I agree")}, processed=4, start=100)
    remote = "yeah I agree " + "the covariance is fine and the pipeline looks reasonable " * 3
    system = track("system", {"S1": words(0, 30, remote)}, processed=31, start=100)
    result = gated(mic, system)
    assert [(u.track, u.text) for u in result if u.track == "mic"] == [("mic", "yeah I agree")]


def test_single_common_word_is_not_removed():
    mic = track("mic", {"me": [(1, 1.2, " the")]}, processed=4)
    system = track("system", {"S1": [(1, 1.2, " the")]}, processed=4)
    assert [u.track for u in gated(mic, system) if u.track == "mic"] == ["mic"]


def test_partial_name_line_waits_for_newline(tmp_path):
    path = tmp_path / "meeting.txt"
    sink = Sink(path, "session")
    with path.open("a") as external:
        external.write("# S2 = Mi")
        external.flush()
        sink.poll_names()
        assert sink.names == {}
        external.write("ke Hudson\n")
    sink.poll_names()
    assert sink.names == {"S2": "Mike Hudson"}
    sink.close("ended")


def test_monologue_cut_never_returns_zero():
    tokens = [SimpleNamespace(text=" super", start=0, end=10),
              SimpleNamespace(text="cali", start=10, end=20),
              SimpleNamespace(text="fragilistic", start=20, end=31)]
    assert _monologue_cut(tokens) == len(tokens)


def test_name_refuses_ended_current_but_accepts_explicit_session(tmp_path, monkeypatch):
    monkeypatch.setattr("hark.cli.HOME", tmp_path)
    ended = tmp_path / "ended.txt"
    ended.write_text("# session\n# ended 12:00:00\n")
    (tmp_path / "current.txt").symlink_to(ended)
    try:
        _name_current("S1", "Ada")
    except SystemExit as error:
        assert "ended" in str(error)
    else:
        raise AssertionError("ended session accepted")
    active = tmp_path / "active.txt"
    active.write_text("# session\n")
    _name_current("S2", "Grace", active)
    assert active.read_text().endswith("# S2 = Grace\n")


def test_interleaved_turns_cut_a_before_b_and_keep_a_resumption():
    a = words(0, 4, "first part of a") + words(6, 8, "and back again")
    b = words(4.3, 5.8, "wait no stop")
    room = track("room", {"A": a, "B": b}, processed=8.2)
    out = flush_tracks([room])
    assert [(u.speaker, u.text) for u in out] == [("A", "first part of a"), ("B", "wait no stop")]
    room.session._mel_offset = 11.5 * SAMPLE_RATE
    out += flush_tracks([room])
    assert [(u.speaker, u.text) for u in out] == [("A", "first part of a"), ("B", "wait no stop"),
                                                    ("A", "and back again")]
