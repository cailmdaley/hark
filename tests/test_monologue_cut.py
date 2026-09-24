from types import SimpleNamespace

from hark.transcript import _monologue_cut


def test_long_monologue_cuts_at_largest_late_word_boundary_pause():
    tokens = [
        SimpleNamespace(text="hello", start=0, end=2),
        SimpleNamespace(text=" there", start=2.1, end=5),
        SimpleNamespace(text=" everyone", start=5.1, end=10),
        SimpleNamespace(text=" we", start=10.1, end=14),
        SimpleNamespace(text=" are", start=16, end=19),
        SimpleNamespace(text=" nearly", start=19.1, end=23),
        SimpleNamespace(text=" done", start=26, end=30),
    ]
    assert _monologue_cut(tokens) == 6


def test_long_monologue_falls_back_to_last_word_boundary():
    tokens = [
        SimpleNamespace(text="hello", start=0, end=5),
        SimpleNamespace(text=" there", start=5.1, end=10),
        SimpleNamespace(text=" friend", start=10.1, end=30),
    ]
    assert _monologue_cut(tokens) == 2
