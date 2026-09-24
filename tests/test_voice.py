from datetime import datetime
from types import SimpleNamespace

import numpy as np

from hark.voice import AudioBuffer, VoiceMatcher
from hark.transcript import Sink, Utterance


def setup(tmp_path, voices, vector=(1, 0)):
    sink = Sink(tmp_path / "session.txt", "session")
    buffer = AudioBuffer()
    buffer.append(0, np.ones(10 * 16000, dtype=np.float32))
    track = SimpleNamespace(audio=buffer)
    matcher = VoiceMatcher(voices, sink, embedder=lambda _: np.asarray(vector, dtype=np.float32))
    return matcher, sink, track


def utterance(speaker, start=0):
    return Utterance("system", speaker, start, start + 5, datetime.now(), "speech",
                     speech=[(start, start + 5)])


def test_names_slot_at_five_seconds_and_threshold(tmp_path):
    matcher, sink, track = setup(tmp_path, {"me": np.array([0.415, np.sqrt(1 - 0.415**2)])})
    matcher.finished(track, utterance("S1"))
    assert sink.names == {"S1": "me"}
    sink.close("ended")
    assert "# S1 = me" in (tmp_path / "session.txt").read_text()


def test_runner_up_margin_abstains(tmp_path):
    matcher, sink, track = setup(tmp_path, {"me": np.array([0.8, 0.6]),
                                           "other": np.array([0.75, np.sqrt(1 - 0.75**2)])})
    matcher.finished(track, utterance("S1"))
    assert sink.names == {}
    sink.close("ended")


def test_manual_name_is_never_overridden(tmp_path):
    matcher, sink, track = setup(tmp_path, {"me": np.array([1, 0])})
    sink.name("S1", "Mike")
    matcher.finished(track, utterance("S1"))
    assert sink.names == {"S1": "Mike"}
    sink.close("ended")


def test_non_diarized_call_mic_is_not_voice_named(tmp_path):
    matcher, sink, track = setup(tmp_path, {"me": np.array([1, 0])})
    matcher.finished(track, utterance("me"))
    assert sink.names == {}
    sink.close("ended")


def test_two_slots_can_receive_the_same_name(tmp_path):
    matcher, sink, track = setup(tmp_path, {"me": np.array([1, 0])})
    matcher.finished(track, utterance("S1"))
    matcher.finished(track, utterance("S2"))
    assert sink.names == {"S1": "me", "S2": "me"}
    sink.close("ended")


def test_empty_voice_bank_does_not_import_onnxruntime(monkeypatch, tmp_path):
    import builtins

    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "onnxruntime":
            raise AssertionError("onnxruntime imported without an enrolled voice")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    voices = {}
    matcher = VoiceMatcher(voices, Sink(tmp_path / "session.txt", "session")) if voices else None
    assert matcher is None
