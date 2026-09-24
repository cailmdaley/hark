from datetime import datetime

from hark.cli import _emit
from hark.transcript import Sink, Utterance


def test_voice_matching_failure_logs_once_and_preserves_lines(tmp_path, monkeypatch):
    sink = Sink(tmp_path / "meeting.txt", "session")
    messages = []
    monkeypatch.setattr("hark.cli.log", messages.append)

    class BrokenMatcher:
        calls = 0

        def finished(self, track, utterance):
            self.calls += 1
            raise ValueError("malformed embedding")

    matcher = BrokenMatcher()
    track = object()
    failed_slots = set()
    for start, text in [(0, "first line"), (2, "second line")]:
        _emit(Utterance("system", "S1", start, start + 1, datetime.now(), text),
              sink, matcher, {"system": track}, failed_slots)
    sink.close("ended")

    transcript = (tmp_path / "meeting.txt").read_text()
    assert "first line" in transcript and "second line" in transcript
    assert matcher.calls == 1
    assert messages == ["voice: disabled matching for system/S1 after error: malformed embedding"]
