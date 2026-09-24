from datetime import datetime
from types import SimpleNamespace

from hark.capture import SAMPLE_RATE
from hark.transcript import Track, flush_tracks


def track(name, speaker_tokens, *, processed, start=100):
    instance = Track.__new__(Track)
    instance.name = name
    instance.t0 = datetime.fromtimestamp(start)
    instance.speaker_label = lambda speaker: speaker
    instance.gap = 3.0
    instance.max_len = 30.0
    instance.hop = 1
    instance.session = SimpleNamespace(_mel_offset=processed * SAMPLE_RATE)
    instance.pending = {
        speaker: SimpleNamespace(tokens=[SimpleNamespace(text=text, start=begin, end=end)
                                         for begin, end, text in tokens])
        for speaker, tokens in speaker_tokens.items()
    }
    return instance


def test_sustained_other_speaker_takes_turn_but_backchannel_does_not():
    own = track("room", {"S1": [(0, 0.5, "hello"), (1, 2, " there")]}, processed=2.5)
    backchannel = track("room", {"S2": [(2.2, 2.4, " yeah")]}, processed=2.5)
    assert flush_tracks([own, backchannel]) == []

    sustained = track("room", {"S2": [(2.2, 2.4, " I"), (3.4, 3.6, " agree")]}, processed=3.7)
    result = flush_tracks([own, sustained])
    assert [(line.speaker, line.text) for line in result] == [("S1", "hello there")]


def test_gap_flushes_when_nobody_takes_over():
    own = track("room", {"S1": [(0, 0.5, "hello"), (1, 2, " there")]}, processed=5.1)
    assert [(line.speaker, line.text) for line in flush_tracks([own])] == [("S1", "hello there")]


def test_turn_change_is_detected_across_tracks():
    mic = track("mic", {"me": [(0, 0.5, "let's"), (1, 2, " begin")]}, processed=2.5)
    remote = track("system", {"speaker_0": [(2.2, 2.4, " I"), (3.4, 3.6, " agree")]}, processed=3.7)
    result = flush_tracks([mic, remote])
    assert [(line.track, line.speaker, line.text) for line in result] == [("mic", "me", "let's begin")]
