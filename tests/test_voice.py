import json
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

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
    matcher, sink, track = setup(tmp_path, {"me": np.array([0.55, np.sqrt(1 - 0.55**2)])})
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


def at(cosine):
    return np.array([cosine, np.sqrt(1 - cosine**2)], dtype=np.float32)


def slots(tmp_path, voices, heard):
    """A matcher whose embedder hears each slot at the cosine given in `heard`."""
    sink = Sink(tmp_path / "session.txt", "session")
    buffer = AudioBuffer()
    buffer.append(0, np.ones(10 * 16000, dtype=np.float32))
    track = SimpleNamespace(audio=buffer)
    current = {}
    matcher = VoiceMatcher(voices, sink, embedder=lambda _: heard[current["slot"]])

    def speak(slot):
        current["slot"] = slot
        matcher.finished(track, utterance(slot))

    return matcher, sink, speak


def test_below_floor_abstains_with_one_enrolled_voice(tmp_path):
    matcher, sink, speak = slots(tmp_path, {"me": np.array([1, 0])}, {"S1": at(0.45)})
    speak("S1")
    assert sink.names == {}
    sink.close("ended")


@pytest.mark.parametrize("order", [("S1", "S2"), ("S2", "S1")])
def test_one_voice_two_slots_names_only_the_true_slot(tmp_path, order):
    # 2026-09-28 room meeting: S1 (Martin, with bleed) 0.45, S2 (Cail) 0.65, "me" the only print
    matcher, sink, speak = slots(tmp_path, {"me": np.array([1, 0])}, {"S1": at(0.45), "S2": at(0.65)})
    for slot in order:
        speak(slot)
    assert sink.names == {"S2": "me"}
    sink.close("ended")
    assert "# S1 = me" not in (tmp_path / "session.txt").read_text()


def test_a_name_goes_to_one_slot_only(tmp_path):
    matcher, sink, speak = slots(tmp_path, {"me": np.array([1, 0])}, {"S1": at(0.9), "S2": at(0.85)})
    speak("S1")
    speak("S2")
    assert sink.names == {"S1": "me"}
    sink.close("ended")


def test_ambiguous_second_slot_does_not_move_the_name(tmp_path):
    matcher, sink, speak = slots(tmp_path, {"me": np.array([1, 0])}, {"S1": at(0.6), "S2": at(0.75)})
    speak("S1")
    speak("S2")
    assert sink.names == {"S1": "me"}
    sink.close("ended")


def test_clearly_stronger_slot_takes_the_name_and_retracts_the_holder(tmp_path):
    matcher, sink, speak = slots(tmp_path, {"me": np.array([1, 0])}, {"S1": at(0.6), "S2": at(0.95)})
    speak("S1")
    speak("S2")
    assert sink.names == {"S2": "me"}
    sink.close("ended")
    text = (tmp_path / "session.txt").read_text()
    assert text.index("# S1 = me") < text.index("# S1 = S1") < text.index("# S2 = me")
    records = [json.loads(line) for line in (tmp_path / "session.jsonl").read_text().splitlines()]
    assert [r["name"] for r in records if "name" in r] == [
        {"speaker": "S1", "as": "me"}, {"speaker": "S1", "as": None}, {"speaker": "S2", "as": "me"}]


def test_manually_given_name_is_not_handed_to_another_slot(tmp_path):
    matcher, sink, speak = slots(tmp_path, {"me": np.array([1, 0])}, {"S2": at(0.95)})
    sink.name("S1", "me")
    speak("S2")
    assert sink.names == {"S1": "me"}
    sink.close("ended")


def test_retraction_line_returns_a_slot_to_its_label(tmp_path):
    sink = Sink(tmp_path / "session.txt", "session")
    with (tmp_path / "session.txt").open("a") as f:
        f.write("# S1 = Martin\n")
    sink.poll_names()
    assert sink.names == {"S1": "Martin"}
    with (tmp_path / "session.txt").open("a") as f:
        f.write("# S1 = S1\n")
    u = Utterance("mic", "S1", 0, 1, datetime.now(), "hello")
    sink.write(u)
    assert sink.names == {} and u.name is None and "name" not in u.record()
    sink.close("ended")


def test_human_retraction_of_a_voice_name_sticks(tmp_path):
    matcher, sink, speak = slots(tmp_path, {"me": np.array([1, 0])}, {"S1": at(0.9)})
    speak("S1")
    sink.name("S1", "S1")
    for _ in range(3):
        matcher.checked.clear()
        speak("S1")
    assert sink.names == {}
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
