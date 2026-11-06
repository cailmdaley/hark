"""Physical-run finalization, external-ear failures and private display-only previews."""

from datetime import datetime
import numpy as np
import pytest

from hark.capture import SAMPLE_RATE
from hark.live_voice import Partial, _align, _word_anchors, quiet_boundary
from hark.transcript import _Pending, flush_tracks
from test_live_voice import pcm, token, track


def finished_track(recognize, **kwargs):
    kwargs.setdefault("gap", 2)
    t = track(refiner=recognize, **kwargs)
    t.t0 = datetime.now().astimezone()
    return t


def test_physical_run_calls_ear_once_despite_overlapping_speaker_slots():
    calls = []
    t = finished_track(lambda audio: calls.append(audio.copy()) or "Hello corrected world")
    # Two overlapping diarizer slots must not send duplicate audio to the ear.
    t.session.tokens = [token(.1, .3, " Hello"), token(.4, .6, " wrong")]
    t.feed(pcm(1, .02))
    t.pending["speaker_1"] = _Pending([token(.5, .8, " world")])
    assert "wrong" in t.partial and not calls
    t.feed(pcm(3))
    out = flush_tracks([t])
    assert len(calls) == 1
    assert " ".join(u.text for u in out) == "Hello corrected world"
    assert [u.speaker for u in out] == ["S1", "S2"]
    assert out[0].start == .1 and out[-1].end == .8
    assert out[0].wall == t.t0 + __import__("datetime").timedelta(seconds=.1)
    assert not t.partial
    assert not flush_tracks([t], force=True)
    assert len(calls) == 1


@pytest.mark.parametrize("failure", [ValueError("secret-in-exception"), None])
def test_failed_or_empty_call_writes_raw_once_without_logging_exception_text(failure):
    calls, messages = [], []

    def failed(audio):
        calls.append(len(audio))
        if failure:
            raise failure
        return ""

    t = finished_track(failed, report=messages.append)
    t.session.tokens = [token(.1, .5, " preserve my words")]
    t.feed(pcm(1, .02))
    t.feed(pcm(3))
    out = flush_tracks([t], force=True)
    assert [u.text for u in out] == ["preserve my words"]
    assert len(calls) == len(messages) == 1
    assert "ValueError" in messages[0]
    assert "secret-in-exception" not in messages[0]
    assert not flush_tracks([t], force=True)


def test_idle_never_calls_external_ear_and_memory_stays_bounded():
    t = finished_track(lambda _: pytest.fail("idle audio reached external ear"))
    for _ in range(100):
        t.feed(pcm(30))
        assert flush_tracks([t]) == []
    assert len(t.audio.samples) == 90 * SAMPLE_RATE
    assert not t._ready and not t.partial and t._run_start is None


def test_audio_is_partitioned_once_across_run_caps_and_final_drain():
    calls = []
    t = finished_track(lambda audio: calls.append(len(audio)) or "word", max_len=2)
    # Introduce tokens at each source half-second, so the cap sees committed speech.
    for i in range(10):
        t.session.tokens = [token(i / 2 + .1, i / 2 + .2, " word")]
        t.feed(pcm(.5, .02))
        flush_tracks([t])
    t.feed(pcm(0), final=True)
    flush_tracks([t], force=True)
    assert len(calls) == 3
    assert sum(calls) == 5 * SAMPLE_RATE
    assert not any(p.tokens for p in t.pending.values())


def test_false_slot_switches_do_not_duplicate_final_words():
    pending = {
        "speaker_0": _Pending([token(0, .1, " a"), token(.2, .3, " typo")]),
        "speaker_1": _Pending([token(.25, .35, " typo"), token(.4, .5, " c")]),
    }
    aligned = _align("a corrected c", _word_anchors(pending))
    assert [word for _, word in aligned] == ["a", "corrected", "c"]
    assert len(aligned) == 3
    assert {anchor[0] for anchor, _ in aligned} <= set(pending)


def test_insertions_and_deletions_inherit_existing_word_anchors():
    pending = {"speaker_0": _Pending([token(1, 1.1, " hello"), token(2, 2.1, " bad"),
                                       token(3, 3.1, " world")])}
    raw = _word_anchors(pending)
    aligned = _align("well hello wonderful world today", raw)
    assert [word for _, word in aligned] == ["well", "hello", "wonderful", "world", "today"]
    assert all(anchor in raw for anchor, _ in aligned)
    assert len(_align("hello world", raw)) == 2


def test_boundary_prefers_quiet_audio_near_cut():
    t = track()
    audio = pcm(2, .02)
    audio[round(.8 * SAMPLE_RATE):round(1.0 * SAMPLE_RATE)] = 0
    t.audio.append(0, audio)
    assert .8 <= quiet_boundary(t.audio, .9) <= 1


def test_partial_is_private_atomic_and_independent_of_transcript(tmp_path):
    transcript = tmp_path / "voice.txt"
    transcript.write_text("final words\n")
    partial = Partial(transcript)
    assert partial.path.read_text() == ""
    first = partial.path.stat().st_ino
    partial.write("raw provisional words")
    assert partial.path.read_text() == "raw provisional words"
    assert partial.path.stat().st_ino != first
    assert partial.path.stat().st_mode & 0o777 == 0o600
    stamp = partial.path.stat().st_mtime_ns
    partial.write("raw provisional words")
    assert partial.path.stat().st_mtime_ns == stamp
    partial.write("")
    assert partial.path.read_text() == ""
    assert transcript.read_text() == "final words\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["voice.partial", "voice.txt"]


def test_operator_module_loads_once_and_has_no_service_contract(tmp_path, monkeypatch):
    # Use the actual loader, independent of the no-model CLI fixture.
    module = tmp_path / "ear.py"
    module.write_text("def create(language=None):\n    return lambda pcm: language or 'auto'\n")
    monkeypatch.setenv("HARK_VOICE_EAR", str(module))
    from hark.live_voice import load_refiner
    recognize = load_refiner(language="en-US")
    module.unlink()
    assert recognize(pcm(1)) == "en-US"


def test_adapter_setup_failure_is_sanitized(tmp_path, monkeypatch):
    module = tmp_path / "ear.py"
    module.write_text("raise RuntimeError('private-secret')\n")
    monkeypatch.setenv("HARK_VOICE_EAR", str(module))
    from hark.live_voice import load_refiner
    with pytest.raises(RuntimeError) as error:
        load_refiner()
    assert "private-secret" not in str(error.value)
    assert error.value.__suppress_context__


def test_raw_empty_speech_still_reaches_external_ear_without_claiming_an_identity():
    calls = []
    t = finished_track(lambda audio: calls.append(len(audio)) or "recovered phrase")
    t.feed(pcm(1, .02))
    t.feed(pcm(3))
    out = flush_tracks([t], force=True)
    assert len(calls) == 1
    assert [u.text for u in out] == ["recovered phrase"]
    assert out[0].speaker == "S?"
    assert not any(p.tokens for p in t.pending.values())


def test_sleep_preroll_without_word_anchors_advances_and_preserves_resumed_wall_time():
    calls = []
    t = finished_track(lambda audio: calls.append(len(audio)) or "finished")
    t.session.tokens = [token(.1, .2, " before")]
    t.feed(pcm(.3, .02))
    t.add_wall_gap(5.3, t.t0.timestamp() + 3600)
    t.feed(pcm(5))
    assert flush_tracks([t])
    model_start = t._sent / SAMPLE_RATE
    t.session.tokens = [token(model_start + .3, model_start + .38, " after")]
    t.feed(pcm(.1, .02))
    t.feed(pcm(3))
    t.feed(pcm(0), final=True)
    out = flush_tracks([t], force=True)
    assert [u.text for u in out] == ["finished"]
    assert out[0].wall.timestamp() == pytest.approx(t.t0.timestamp() + 3600 + out[0].start)
    assert len(calls) == 2
    assert not any(p.tokens for p in t.pending.values())
    assert t._run_start is None


def test_punctuation_only_capped_prefix_cannot_block_a_later_word():
    calls = []
    t = finished_track(lambda audio: calls.append(len(audio)) or "finished", max_len=25)
    t.session.tokens = [token(.1, .2, ".")]
    t.feed(pcm(26, .02))
    t.session.tokens = [token(26.1, 26.2, " word")]
    t.feed(pcm(.5, .02))
    t.feed(pcm(3))
    t.feed(pcm(0), final=True)
    out = flush_tracks([t], force=True)
    assert len(calls) == 2
    assert len(out) == 2
    assert not any(p.tokens for p in t.pending.values())
    assert t._run_start is None


def test_delayed_tokens_after_finished_run_do_not_call_ear_on_a_silent_tail():
    calls = []
    t = finished_track(lambda audio: calls.append(len(audio)) or "finished")
    t.session.tokens = [token(.1, .2, " first")]
    t.feed(pcm(1, .02))
    t.feed(pcm(3))
    assert flush_tracks([t])
    t.session.tokens = [token(3.1, 3.2, " delayed")]
    t.feed(pcm(.5))
    t.feed(pcm(0), final=True)
    assert not flush_tracks([t], force=True)
    assert len(calls) == 1


def test_final_flush_drains_every_cap_even_without_raw_words():
    calls = []
    t = finished_track(lambda audio: calls.append(len(audio)) or "finished", max_len=2)
    # Delay the processed clock so all eight seconds are still pending at shutdown.
    t.session.feed = lambda *args, **kwargs: []
    t.feed(pcm(8, .02))
    assert not calls
    t.feed(pcm(0), final=True)
    out = flush_tracks([t], force=True)
    assert len(calls) == len(out) == 4
    assert sum(calls) == 8 * SAMPLE_RATE
    assert t._run_start is None


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("force", [False, True])
def test_delayed_anchors_belong_to_finished_physical_run(failure, force):
    calls = []

    def recognize(audio):
        calls.append(len(audio))
        if failure:
            raise ValueError("failed")
        return "finished words"

    t = finished_track(recognize, gap=.5, report=lambda _: None)
    t.session.tokens = [token(.64, .72, " raw words")]
    t.feed(pcm(.1, .02))
    if force:
        t.feed(pcm(0), final=True)
    else:
        # Quiet finalization can precede a delayed raw timestamp.
        t.feed(pcm(3.5))
    out = flush_tracks([t], force=force)
    assert [u.text for u in out] == ["raw words" if failure else "finished words"]
    assert out[0].speaker == "S1"
    assert (out[0].start, out[0].end) == (.64, .72)
    assert out[0].wall.timestamp() == pytest.approx(t.t0.timestamp() + .64)
    assert t._run_start is None
    assert not any(p.tokens for p in t.pending.values())
    t.feed(pcm(6), final=True)
    assert not flush_tracks([t], force=True)
    assert len(calls) == 1


@pytest.mark.parametrize("force", [False, True])
def test_crossing_speaker_end_times_preserve_external_word_order(force):
    t = finished_track(lambda _: "one two three four")
    t.session.tokens = [token(.1, .3, " one"), token(.5, .7, " three")]
    t.feed(pcm(1, .02))
    t.pending["speaker_1"] = _Pending([token(.2, .8, " two"), token(.6, .9, " four")])
    t.feed(pcm(0) if force else pcm(3), final=force)
    out = flush_tracks([t], force=force)
    assert [u.text for u in out] == ["one", "two", "three", "four"]
    assert [u.speaker for u in out] == ["S1", "S2", "S1", "S2"]
    assert [(u.start, u.end) for u in out] == [(.1, .3), (.2, .8), (.5, .7), (.6, .9)]


def test_acoustic_cap_never_exceeds_selected_hard_boundary():
    calls = []
    t = finished_track(lambda a: calls.append(a.copy()) or "finished", max_len=2)
    t.session.feed = lambda *args, **kwargs: []
    samples = pcm(3, .02)
    samples[round(2.08 * SAMPLE_RATE):round(2.16 * SAMPLE_RATE)] = 0
    t.feed(samples, final=True)
    flush_tracks([t], force=True)
    assert all(len(a) <= 2 * SAMPLE_RATE for a in calls)
    np.testing.assert_array_equal(np.concatenate(calls), samples)


def test_external_voice_requires_acoustic_gate():
    with pytest.raises(ValueError, match="gate"):
        finished_track(lambda _: "finished", gate=False)


@pytest.mark.parametrize("raw", [" hello claw", " word" * 200])
def test_raw_name_or_word_budget_closes_a_run_before_quiet(raw):
    calls = []
    t = finished_track(lambda a: calls.append(len(a)) or "finished")
    t.session.tokens = [token(.1, .2, raw)]
    t.feed(pcm(.5, .02))
    assert [u.text for u in flush_tracks([t])] == ["finished"]
    assert len(calls) == 1


def test_voice_run_defaults_and_rolling_pcm_bound():
    t = track(refiner=lambda _: "finished")
    assert (t.gap, t.words, t.max_len) == (5, 200, 75)
    t.feed(pcm(100))
    assert len(t.audio.samples) == 90 * SAMPLE_RATE


def test_echo_cancelled_mic_withholds_runs_the_local_recognizer_did_not_hear():
    calls = []
    t = finished_track(lambda audio: calls.append(audio) or "phantom call audio", local_words_required=True)
    t.feed(pcm(1, .02))
    t.feed(pcm(3))
    assert flush_tracks([t]) == [] and not calls
    t.session.tokens = [token(4.1, 4.3, " hello")]
    t.feed(pcm(1, .02))
    t.feed(pcm(3))
    assert [u.text for u in flush_tracks([t])] == ["phantom call audio"] and len(calls) == 1


def test_a_stalled_ear_rests_but_an_empty_reply_does_not(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("hark.live_voice.time.monotonic", lambda: now[0])
    calls, stall = [], [False]

    def recognize(audio):
        calls.append(audio)
        now[0] += 30 if stall[0] else 1
        if stall[0]:
            raise TimeoutError
        return ""

    t = finished_track(recognize)
    def speak():
        t.session.tokens.append(token(t.audio_samples / SAMPLE_RATE + .1, t.audio_samples / SAMPLE_RATE + .3))
        t.feed(pcm(1, .02))
        t.feed(pcm(3))
        return flush_tracks([t])

    assert speak() and speak() and len(calls) == 2  # Empty replies leave the next run its call.
    stall[0] = True
    assert speak() and len(calls) == 3
    assert speak() and len(calls) == 3  # Rested: raw words, no call.
    now[0] += 61
    assert speak() and len(calls) == 4


def test_residual_echo_above_the_gate_cannot_hold_a_run_open():
    calls = []
    t = finished_track(lambda audio: calls.append(audio) or "said in the room", local_words_required=True, gap=5)
    t.session.tokens = [token(.1, .3, " hello")]
    t.feed(pcm(30, .02))  # Loud throughout, with one heard word at the start.
    out = flush_tracks([t])
    assert len(calls) == 1 and len(calls[0]) < 8 * SAMPLE_RATE
    assert [u.text for u in out] == ["said in the room"]


def test_short_quiet_settings_keep_a_late_word():
    # Three seconds of the model's drain close a run, however short the quiet setting.
    calls = []
    t = finished_track(lambda audio: calls.append(audio) or "word", gap=1)
    t.feed(pcm(.5, .02))
    t.feed(pcm(2))
    assert not calls
    t.pending["speaker_0"] = _Pending([token(.1, .3)])
    t.feed(pcm(2))
    assert [u.text for u in flush_tracks([t])] == ["word"]
