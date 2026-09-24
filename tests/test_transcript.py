import json
from datetime import datetime
from hark.transcript import EchoGate, Sink, Utterance


def utterance(track, start, end, text, speaker):
    return Utterance(track, speaker, start, end, datetime.fromtimestamp(start), text)


def test_speaker_slot_column_is_four_characters_and_names_are_unpadded():
    slot = utterance("system", 10, 11, "Hello", "S2")
    assert slot.line().endswith(" S2   Hello")
    slot.name = "Alexander Hamilton"
    assert slot.line().endswith(" Alexander Hamilton Hello")


def test_echo_gate_drops_overlapping_duplicate():
    gate = EchoGate()
    gate.push(utterance("mic", 10, 12, "We should start now", "me"))
    gate.push(utterance("system", 10.2, 12, "We should start now", "S1"))
    assert gate.release(13) == []


def test_echo_gate_keeps_genuine_me_speech():
    gate = EchoGate()
    me = utterance("mic", 10, 12, "I will check the pipeline", "me")
    gate.push(me)
    gate.push(utterance("system", 10, 12, "Can everyone see the slides", "S1"))
    assert gate.release(13) == [me]


def test_echo_gate_keeps_me_talking_over_remote_partial_overlap():
    gate = EchoGate()
    me = utterance("mic", 10, 13, "yes that seems right", "me")
    gate.push(me)
    gate.push(utterance("system", 11, 13, "yes", "S1"))
    assert gate.release(14) == [me]


def test_echo_gate_final_flush_releases_held_lines():
    gate = EchoGate()
    me = utterance("mic", 10, 12, "Something unique", "me")
    gate.push(me)
    assert gate.release(10) == []
    assert gate.release(10, final=True) == [me]


def test_external_name_line_applies_to_next_line_and_jsonl(tmp_path):
    path = tmp_path / "meeting.txt"
    sink = Sink(path, "session")
    with path.open("a") as external:
        external.write("# S2 = Mike Hudson\n")
    named = utterance("system", 10, 11, "Hello there", "S2")
    sink.write(named)
    sink.close("ended")
    assert "Mike Hudson" in path.read_text().splitlines()[-2]
    assert path.read_text().splitlines()[-2].endswith("Hello there")
    records = [json.loads(line) for line in path.with_suffix(".jsonl").read_text().splitlines()]
    assert records == [{"wall": named.wall.isoformat(timespec="seconds"), "track": "system",
                        "speaker": "S2", "start": 10, "end": 11, "text": "Hello there",
                        "name": "Mike Hudson"}]


def test_name_appended_between_poll_and_write_is_seen_next_time(tmp_path):
    path = tmp_path / "meeting.txt"
    sink = Sink(path, "session")
    poll = sink.poll_names
    appended = False

    def racing_poll():
        nonlocal appended
        poll()
        if not appended:
            with path.open("a") as external:
                external.write("# S3 = Grace Hopper\n")
            appended = True

    sink.poll_names = racing_poll
    first = utterance("system", 10, 11, "Initial", "S3")
    second = utterance("system", 12, 13, "Named", "S3")
    sink.write(first)
    sink.write(second)
    sink.close("ended")
    assert first.name is None
    assert second.name == "Grace Hopper"


def test_sink_name_writes_mapping_record(tmp_path):
    path = tmp_path / "meeting.txt"
    sink = Sink(path, "session")
    sink.name("S1", "Ada")
    item = utterance("system", 10, 11, "Hello", "S1")
    sink.write(item)
    sink.close("ended")
    record = json.loads(path.with_suffix(".jsonl").read_text().splitlines()[-1])
    assert item.name == "Ada"
    assert record["speaker"] == "S1" and record["name"] == "Ada"
    assert '"speaker": "S1", "as": "Ada"' in path.with_suffix(".jsonl").read_text()
