"""Unified capture selection and session-wide attribution without audio devices."""

from datetime import datetime
import json

import numpy as np
import pytest

from hark import cli, gradium
from hark.transcript import Sink, Utterance
from hark.voice import AudioBuffer, VoiceMatcher
from test_cli import fake_live_capture, feed_chunks, signal_when_live


@pytest.mark.parametrize("ear", ["local", "gradium"])
@pytest.mark.parametrize("mode,available,expected", [
    ([], True, ["mic", "system"]),
    (["--call"], True, ["mic", "system"]),
    ([], False, ["mic"]),
    (["--call"], False, ["mic"]),
    (["--room"], True, ["mic"]),
    (["--phone"], True, ["phone"]),
])
def test_live_selection_diarizes_every_source_and_allocates_unique_slots(
        tmp_path, monkeypatch, ear, mode, available, expected):
    _, Source = fake_live_capture(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "_system_capture_available", lambda: available)
    for attribute, name in [("MicSource", "mic"), ("SystemSource", "system"), ("PhoneSource", "phone")]:
        source, _ = feed_chunks(Source, name, [])
        monkeypatch.setattr(cli, attribute, source)
    voices = tmp_path / "voices"
    voices.mkdir()
    np.save(voices / "me.npy", [1., 0.])
    # No embeddings or model downloads; the voice-bank handoff is checked below.
    monkeypatch.setattr("hark.voice.Embedder", lambda: lambda _: np.array([1., 0.]))
    made = []

    class Track:
        def __init__(self, name, *args, **kwargs):
            self.name, self.processed, self.audio = name, 0., None
            self.sent_seconds = 0.
            self.label = kwargs.get("speaker_label", lambda _: "S1")
            self.pending = True
            made.append((name, kwargs))

        def start(self, **kwargs): pass
        def feed(self, *args, **kwargs): pass
        def check(self): pass
        def close(self): pass

    def flush(tracks, force=False):
        result = []
        for track in tracks:
            # Exercise shutdown attribution as well as the normal live path.
            if track.pending and (force or track.name != "system"):
                track.pending = False
                result.append(Utterance(track.name, track.label("speaker_0"),
                                        0, 1, datetime.now(), "hello"))
        return result

    monkeypatch.setattr(cli, "Track", Track)
    monkeypatch.setattr(gradium, "GradiumTrack", Track)
    monkeypatch.setattr(gradium, "credits_left", lambda _: None)
    monkeypatch.setattr(cli, "flush_tracks", flush)
    out = tmp_path / "session.txt"
    signaller, _ = signal_when_live(tmp_path)
    cli.main(mode + ["--ear", ear, "--launch", "test", "-o", str(out)])
    signaller.join(2)
    assert not signaller.is_alive()
    assert [name for name, _ in made] == expected
    for name, options in made:
        assert options.get("fixed_speaker") is None
        if ear == "local":
            assert options["speaker_label"]("speaker_1") == "S2"
        else:
            # Anchors receive the full bank on both tracks, independently of naming.
            np.testing.assert_array_equal(options["voices"]["me"], [1., 0.])
    records = [json.loads(line) for line in out.with_suffix(".jsonl").read_text().splitlines()]
    utterances = [r for r in records if "speaker" in r]
    assert [(r["track"], r["speaker"]) for r in utterances] == list(
        zip(expected, [f"S{i + 1}" for i in range(len(expected))]))
    assert "me = mic" not in out.read_text()
    assert out.read_text().splitlines()[-1].startswith("# ended ")
    assert sorted(p.suffixes[-2] for p in tmp_path.glob("*.wav")) == sorted(f".{s}" for s in expected)


def test_global_slots_remain_stable_across_tracks_and_more_than_eight_speakers(tmp_path):
    sink = Sink(tmp_path / "session.txt", "test")
    slots = cli.SpeakerSlots()
    for track in ["mic", "system", "mic", "system"]:
        for number in range(8):
            cli._emit(Utterance(track, f"S{number + 1}", 0, 1, datetime.now(), "speech"),
                      sink, None, {}, set(), slots)
    sink.close("ended")
    records = [json.loads(line) for line in sink.path.with_suffix(".jsonl").read_text().splitlines()]
    labels = [r["speaker"] for r in records if "speaker" in r]
    assert labels == [f"S{i + 1}" for i in range(16)] * 2


def test_me_is_matched_on_mic_not_system_and_cannot_be_stolen(tmp_path):
    sink = Sink(tmp_path / "session.txt", "test")
    matcher = VoiceMatcher({"me": np.array([1., 0.]), "Ada": np.array([0., 1.])},
                           sink, embedder=lambda _: np.array([1., 0.]))
    tracks = {}
    for name in ["mic", "system"]:
        audio = AudioBuffer()
        audio.append(0, np.ones(6 * 16000, np.float32))
        tracks[name] = type("Track", (), {"audio": audio})()
    slots = cli.SpeakerSlots()
    for name in ["system", "mic", "system"]:
        cli._emit(Utterance(name, "S1", 0, 5, datetime.now(), "speech", speech=[(0, 5)]),
                  sink, matcher, tracks, set(), slots)
    assert sink.names == {"S2": "me"}
    assert matcher.given == {"S2": "me"}
    sink.name("S1", "Ada")
    assert sink.names == {"S1": "Ada", "S2": "me"}
    sink.close("ended")


@pytest.mark.parametrize("me_score", [1., .7])
def test_system_me_match_still_prevents_a_false_other_name(tmp_path, me_score):
    sink = Sink(tmp_path / "session.txt", "test")
    # Ada clears the floor, but doesn't beat me by the ordinary bank margin.
    matcher = VoiceMatcher({"me": np.array([me_score, np.sqrt(1 - me_score ** 2)]),
                            "Ada": np.array([.8, .6])},
                           sink, embedder=lambda _: np.array([1., 0.]))
    audio = AudioBuffer()
    audio.append(0, np.ones(6 * 16000, np.float32))
    track = type("Track", (), {"audio": audio})()
    matcher.finished(track, Utterance("system", "S1", 0, 5, datetime.now(), "speech",
                                      speech=[(0, 5)]))
    assert sink.names == {}
    assert next(iter(matcher.scores.values()))["me"] == me_score
    sink.close("ended")


def test_system_availability_checks_executable_without_opening_devices(tmp_path, monkeypatch):
    tap = tmp_path / "audiotee"
    monkeypatch.setattr(cli, "AUDIOTEE", tap)
    monkeypatch.setattr(cli, "_device_capture_available", lambda: True)
    assert not cli._system_capture_available()
    tap.write_text("placeholder")
    tap.chmod(0o600)
    assert not cli._system_capture_available()
    tap.chmod(0o700)
    assert cli._system_capture_available()
    monkeypatch.setattr(cli, "_device_capture_available", lambda: False)
    assert not cli._system_capture_available()


def test_call_is_accepted_but_hidden_from_help(capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["--help"])
    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert "--call" not in help_text
    assert "--room" in help_text and "mic + system audio" in help_text
