from hark.follow import Batcher, follow


def test_addressed_line_flushes_everything_pending():
    batcher = Batcher(words=150, seconds=15)
    assert batcher.add("00:00:01 me   just talking about the covariance", 0) is None
    batch = batcher.add("00:00:02 me   hey Claude, can you check the redshift bins?", 1)
    assert batch == [
        "00:00:01 me   just talking about the covariance",
        "00:00:02 me   hey Claude, can you check the redshift bins?",
    ]
    assert batcher.pending == []


def test_addressed_match_is_case_insensitive_and_catches_mishearings():
    for word in ["Claude", "cloud", "Clawed", "KLAUD"]:
        batcher = Batcher(words=150, seconds=15)
        batch = batcher.add(f"00:00:01 me   {word}, what do you think?", 0)
        assert batch is not None, word


def test_word_threshold_flushes_once_reached():
    batcher = Batcher(words=10, seconds=1000)
    six_words = " ".join(["word"] * 6)
    assert batcher.add(f"00:00:01 me   {six_words}", 0) is None
    batch = batcher.add(f"00:00:02 me   {six_words}", 1)
    assert batch is not None
    assert len(batch) == 2


def test_time_threshold_flushes_after_seconds_since_first_pending_line():
    batcher = Batcher(words=1000, seconds=15)
    assert batcher.add("00:00:01 me   hi", 100.0) is None
    assert batcher.tick(114.0) is None
    batch = batcher.tick(115.0)
    assert batch == ["00:00:01 me   hi"]


def test_ended_line_flushes_and_marks_ended():
    batcher = Batcher(words=1000, seconds=1000)
    batcher.add("00:00:01 me   hi", 0)
    batch = batcher.add("# ended 00:00:02", 1)
    assert batch == ["00:00:01 me   hi", "# ended 00:00:02"]
    assert batcher.ended is True


def test_naming_line_rides_along_without_counting_or_triggering():
    batcher = Batcher(words=1000, seconds=1000)
    batcher.add("00:00:01 me   hello there", 0)
    batcher.add("# S2 = Martin", 1)
    batch = batcher.add("00:00:02 me   claude", 2)
    assert batch == [
        "00:00:01 me   hello there",
        "# S2 = Martin",
        "00:00:02 me   claude",
    ]


def test_follow_prints_existing_contents_as_first_batch(tmp_path):
    path = tmp_path / "live.txt"
    path.write_text("# hark session\n00:00:01 me   hello\n")
    lines = []
    calls = {"n": 0}

    def sleep(_):
        calls["n"] += 1
        if calls["n"] == 1:
            path.write_text(path.read_text() + "# ended 00:00:02\n")

    follow(path, Batcher(), poll_interval=0, sleep=sleep, clock=lambda: 0.0, emit=lines.append)
    assert lines == ["# hark session", "00:00:01 me   hello", "", "# ended 00:00:02", ""]


def test_follow_waits_for_the_file_to_appear(tmp_path):
    path = tmp_path / "live.txt"
    lines = []
    calls = {"n": 0}

    def sleep(_):
        calls["n"] += 1
        if calls["n"] == 3:
            path.write_text("# hark session\n00:00:01 me   hello\n# ended 00:00:02\n")

    follow(path, Batcher(), poll_interval=0, sleep=sleep, clock=lambda: 0.0, emit=lines.append)
    assert lines == ["# hark session", "00:00:01 me   hello", "# ended 00:00:02", ""]
    assert calls["n"] >= 3


def test_follow_handles_truncation_by_rereading_from_zero(tmp_path):
    path = tmp_path / "live.txt"
    path.write_text("# hark session\n00:00:01 me   this line will be lost\n")
    lines = []
    calls = {"n": 0}

    def sleep(_):
        calls["n"] += 1
        if calls["n"] == 1:
            path.write_text("# hark session (restarted)\n# ended 00:00:02\n")

    follow(path, Batcher(), poll_interval=0, sleep=sleep, clock=lambda: 0.0, emit=lines.append)
    assert lines == ["# hark session", "00:00:01 me   this line will be lost", ""] + [
        "# hark session (restarted)", "# ended 00:00:02", "",
    ]
