from types import SimpleNamespace

from hark.capture import SAMPLE_RATE
from hark.transcript import Sink, _monologue_cut, flush_tracks
from hark.cli import _name_current
from tests.test_turns import track


def words(start, end, text):
    ws = text.split()
    step = (end - start) / len(ws)
    return [(start + i * step, start + (i + 1) * step - 0.05, " " + w) for i, w in enumerate(ws)]


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
