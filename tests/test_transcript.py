import json
from datetime import datetime
from hark.transcript import Sink, Utterance


def utterance(track, start, end, text, speaker):
    return Utterance(track, speaker, start, end, datetime.fromtimestamp(start), text)


def test_speaker_slot_column_is_four_characters_and_names_are_unpadded():
    slot = utterance("system", 10, 11, "Hello", "S2")
    assert slot.line().endswith(" S2   Hello")
    slot.name = "Alexander Hamilton"
    assert slot.line().endswith(" Alexander Hamilton Hello")


def test_line_range_matches_jsonl_wall_and_audio_times(tmp_path):
    path = tmp_path / "meeting.txt"
    sink = Sink(path, "session")
    item = Utterance("system", "S2", 10.125, 19.875,
                     datetime(2026, 9, 24, 14, 3, 12, 250_000), "Hello")
    sink.write(item)
    sink.close("ended")

    line = path.read_text().splitlines()[-2]
    record = json.loads(path.with_suffix(".jsonl").read_text().splitlines()[-1])
    start, end = line.split(" ", 1)[0].split("-")
    wall = datetime.fromisoformat(record["wall"])
    wall_end = datetime.fromtimestamp(wall.timestamp() + record["end"] - record["start"])
    assert line == "14:03:12-14:03:22 S2   Hello"
    assert start == wall.strftime("%H:%M:%S")
    assert end == wall_end.strftime("%H:%M:%S")
    assert (record["start"], record["end"]) == (10.12, 19.88)


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
    assert records == [{"wall": named.wall.isoformat(timespec="milliseconds"), "track": "system",
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
