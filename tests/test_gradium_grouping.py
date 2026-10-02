"""Cost-aware requests preserve the padded source clock, not the compact cloud clock."""

from datetime import datetime
from pathlib import Path
import socket
import tempfile
import time
import wave

import numpy as np
import pytest

from gradium_mock import MockGradium
from hark.capture import PhoneSource, WavRecorder, to_pcm16
from hark.gradium import GradiumError, GradiumTrack
from hark.transcript import Sink, flush_tracks
from hark.voice import OnlineCluster, VoiceMatcher
from test_gradium import finish, speech, wait


def default_track(mock, **options):
    # Timing limits are accelerated; request, phrase, gate and speaker defaults are real.
    t = GradiumTrack("phone", key="mock-key", url=mock.url,
                     timeout=.3, backoff=.01, shutdown_timeout=3, **options)
    t.cluster.embedder = lambda _: np.array([1., 0.])
    return t


def collect(t, count=1, timeout=1):
    out = []
    deadline = time.monotonic() + timeout
    while len(out) < count and time.monotonic() < deadline:
        out.extend(flush_tracks([t]))
        time.sleep(.01)
    return out


def test_default_phone_gaps_share_request_and_preserve_wav_jsonl_and_disjoint_samples(
        tmp_path, monkeypatch):
    clock = [1000.]
    monkeypatch.setattr("hark.capture.time.time", lambda: clock[0])
    plans = [[("first", .32, 2.32), ("joined", .32, 5.44)], [("third", .32, 1.32)]]
    embedded = []
    with tempfile.TemporaryDirectory(prefix="hk-phone-", dir="/tmp") as home, MockGradium(plans=plans) as mock:
        t = default_track(mock)
        t.cluster.embedder = lambda x: embedded.append(x.copy()) or np.array([1., 0.])
        source = PhoneSource(Path(home) / "phone.sock")
        recorder = WavRecorder(tmp_path / "phone.wav")
        sink = Sink(tmp_path / "phone.txt", "phone")
        t.start()
        source.start()
        t.t0 = datetime.fromtimestamp(source.anchor)

        def drain():
            samples = source.drain(limit=float("inf"))
            recorder.write(samples)
            t.feed(samples)
            return samples

        def padding(seconds):
            clock[0] = source.anchor + source.delivered / 16000 + seconds + .5
            samples = drain()
            assert samples.size == round(seconds * 16000)
            assert not samples.any()

        def device(seconds):
            before = source.stats["device"]
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.connect(str(source.path))
                client.sendall(to_pcm16(speech(seconds)).tobytes())
                client.shutdown(socket.SHUT_WR)
                wait(lambda: source.stats["device"] == before + round(seconds * 16000))
                wait(lambda: source.conn is None)
            assert drain().size == round(seconds * 16000)

        try:
            padding(2)
            assert len(mock.connections) == 1 and t.sent_seconds == 0
            device(2)
            padding(4)
            first = collect(t)
            assert [u.text for u in first] == ["first"]  # Short phrase, before request EOS.
            assert first[0].start == 2 and first[0].end == 4
            assert len(mock.connections) == 2
            assert t.request is not None and not t.active
            assert not any(m["type"] == "end_of_stream" for m in mock.connections[1]["messages"])
            assert embedded == []  # The default CPU speaker minimum is four real seconds.

            device(2)
            padding(12)
            joined = collect(t)
            assert [u.text for u in joined] == ["joined"]
            u = joined[0]
            assert (u.start, u.end) == (2, 10)
            assert u.speech == [(2, 4.8), (7.68, 10)]
            assert sum(b - a for a, b in u.speech) == pytest.approx(5.12)
            expected_clip = np.concatenate([t.audio.slice(a, b) for a, b in u.speech])
            assert len(embedded) == 1
            np.testing.assert_array_equal(embedded[0], expected_clip)
            assert embedded[0].size == 81920
            assert not any(m["type"] == "end_of_stream" for m in mock.connections[1]["messages"])
            assert mock.connections[1]["samples"] == 99840  # 6.24 s, not 10.8 source seconds.
            assert [m["flush_id"] for m in mock.connections[1]["messages"] if m["type"] == "flush"] == [1, 2]

            # Voice matching consumes the same disjoint intervals, with its normal .1 s context.
            matched = []
            matcher = VoiceMatcher({"me": np.array([1., 0.])}, sink,
                                   embedder=lambda x: matched.append(x.copy()) or np.array([1., 0.]))
            matcher.finished(t, u)
            assert matcher.seconds[(id(t), "S1")] == pytest.approx(5.12)
            np.testing.assert_array_equal(matched[0], np.concatenate([
                t.audio.slice(a - .1, b + .1) for a, b in u.speech]))
            assert matched[0].size == 88320
            assert sink.names == {"S1": "me"}

            t._end_request()
            device(1.00625)
            third = finish(t)
            assert [u.text for u in third] == ["third"]
            assert (third[0].start, third[0].end) == (22, 23)
            assert len(mock.connections) == 3  # Startup, grouped dialogue, fresh third request.
            assert t.sent_seconds == pytest.approx(6.24 + 1.36)
            for u in first + joined + third:
                sink.write(u)
            sink.close("ended")
            import json
            records = [record for line in sink.path.with_suffix(".jsonl").read_text().splitlines()
                       if "start" in (record := json.loads(line))]
            assert [(r["start"], r["end"]) for r in records] == [(2, 4), (2, 10), (22, 23)]
            assert records[-1]["name"] == "me" and records[-1]["track"] == "phone"
            recorder.close()
            with wave.open(str(recorder.path)) as wav:
                assert wav.getnframes() == t.audio_samples == 368100
                raw = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
            assert not raw[4 * 16000:8 * 16000].any()
            assert not raw[10 * 16000:22 * 16000].any()
            assert np.all(raw[22 * 16000:23 * 16000] == 98)
        finally:
            source.stop()
            recorder.close()
            t.close()


def test_cloud_join_boundaries_map_end_left_and_start_right():
    with MockGradium(plans=[[("left", 2.96, 3.12), ("right", 3.12, 3.28)]]) as mock:
        t = default_track(mock)
        t.start()
        try:
            t.feed(np.zeros(2 * 16000, np.float32))
            t.feed(speech(2))
            t.feed(np.zeros(4 * 16000, np.float32))
            t.feed(speech(2))
            output = finish(t)
            assert [u.text for u in output] == ["left", "right"]
            assert output[0].speech == [(4.64, 4.8)]
            assert output[1].speech == [(7.68, 7.84)]
            assert (output[0].end, output[1].start) == (4.8, 7.68)
            assert len(mock.connections) == 2
        finally:
            t.close()


def test_short_discarded_quiet_preroll_does_not_duplicate_sent_hangover():
    with MockGradium(plans=[[("turn", 0, 1)]]) as mock:
        t = default_track(mock)
        t.start()
        try:
            t.feed(speech(1.04))
            t.feed(np.zeros(round(.96 * 16000), np.float32))
            request = t.request
            t.feed(speech(.8))
            assert t.request is request and len(request.runs) == 1
            finish(t)
            assert len(mock.connections) == 2
            assert t.sent_seconds == pytest.approx(2.8)
            assert request.samples == 44800
        finally:
            t.close()


def test_pending_words_publish_at_speech_flush_on_each_side_of_gap():
    plan = [("before", 1.6, 2), ("tail", 3.12, 5.12)]
    with MockGradium(plans=[plan], dangling_words={0}, dangling_last=True) as mock:
        t = default_track(mock)
        t.start()
        try:
            t.feed(speech(2))
            t.feed(np.zeros(4 * 16000, np.float32))
            before = collect(t)
            assert [u.text for u in before] == ["before"]
            assert before[0].speech == [(1.6, 2.8)]
            assert len(mock.connections) == 2
            t.feed(speech(2))
            t.feed(np.zeros(4 * 16000, np.float32))
            tail = collect(t)
            assert [u.text for u in tail] == ["tail"]
            assert finish(t) == []
            assert tail[0].speech == [(6, 8.8)]
            assert (tail[0].start, tail[0].end) == (6, 8.8)
            assert t.cluster.counts == []
        finally:
            t.close()


def test_quiet_source_clock_retires_request_and_publishes_pending_tail():
    with MockGradium(plans=[[("tail", 0, 1)]], dangling_last=True) as mock:
        t = default_track(mock)
        t.start()
        try:
            t.feed(speech(1.04))
            t.feed(np.zeros(9 * 16000, np.float32))
            assert t.request is not None
            output = collect(t)
            request = t.request
            t.feed(np.zeros(600 * 16000, np.float32))
            assert t.request is None and request.done
            assert [u.text for u in output] == ["tail"]
            assert output[0].end == pytest.approx(1.84)
            assert t.sent_seconds == pytest.approx(1.84)
            assert finish(t) == []
        finally:
            t.close()


def test_midrequest_drop_replays_only_post_flush_mapping_across_gap():
    plans = [[("one", .32, 2.32)], [("two", .32, 2.32)]]
    with MockGradium(plans=plans, drop_first_at=4) as mock:
        t = default_track(mock)
        t.start()
        try:
            t.feed(np.zeros(2 * 16000, np.float32))
            t.feed(speech(2))
            t.feed(np.zeros(4 * 16000, np.float32))
            one = collect(t)
            assert [u.text for u in one] == ["one"]
            t.feed(speech(2))
            t.feed(np.zeros(16000, np.float32))
            two = finish(t)
            assert [u.text for u in two] == ["two"]
            assert (one[0].start, one[0].end) == (2, 4)
            assert two[0].speech == [(8, 10)]
            assert len(mock.connections) == 3
            assert mock.connections[2]["samples"] == 49920  # Only the second speech and its context.
            assert mock.connections[1]["samples"] == 4 * 16000
            # Writes accepted by the socket can race ahead of the server's observed drop.
            observed = sum(c["samples"] for c in mock.connections) / 16000
            assert observed <= t.sent_seconds <= 2 * 6.24 + 1e-9
            assert t.backlog_samples == 0
        finally:
            t.close()


def test_idle_after_speech_flush_is_healthy_and_eos_gets_fresh_progress_budget():
    with MockGradium(plans=[[("short", 0, 1)]], finish_delay=.06) as mock:
        t = default_track(mock, retries=0)
        t.timeout = .1
        t.start()
        try:
            t.feed(speech(1.04))
            t.feed(np.zeros(16000, np.float32))
            assert [u.text for u in collect(t)] == ["short"]
            time.sleep(.25)
            assert t.error is None and not t.finished.is_set()
            assert len(mock.connections) == 2
            assert finish(t) == []
            assert len(mock.connections) == 2
        finally:
            t.close()


def test_unchanged_eos_heartbeats_after_idle_still_timeout():
    with MockGradium(plans=[[("short", 0, 1)]], stalled_eos=True) as mock:
        t = default_track(mock, retries=0)
        t.timeout = .1
        t.start()
        try:
            t.feed(speech(1.04))
            t.feed(np.zeros(16000, np.float32))
            assert [u.text for u in collect(t)] == ["short"]
            time.sleep(.25)
            assert t.error is None
            began = time.monotonic()
            finish(t)
            assert t.error is None
            assert .08 <= time.monotonic() - began < 1
            assert not t.thread.is_alive()
        finally:
            t.close()


def test_unchanged_flush_heartbeats_do_not_buy_a_progress_budget():
    with MockGradium(plans=[[("short", 0, 1)]], stalled_flush=True) as mock:
        t = default_track(mock, retries=0)
        t.timeout = .1
        t.start()
        try:
            t.feed(speech(1.04))
            t.feed(np.zeros(16000, np.float32))
            wait(lambda: len(mock.connections) == 2 and any(
                m["type"] == "flush" for m in mock.connections[1]["messages"]))
            time.sleep(.35)
            assert t.degraded, "unchanged steps kept a pending flush alive"
            assert t.error is None
            assert not t.finished.is_set()
            finish(t)
            assert not t.thread.is_alive()
        finally:
            t.close()


def test_new_audio_after_idle_gets_fresh_progress_budget_without_reconnect():
    with MockGradium(plans=[[("first", 0, 1), ("second", 2.04, 3.04)]]) as mock:
        t = default_track(mock, retries=0)
        t.timeout = .1
        t.start()
        try:
            t.feed(speech(1.04))
            t.feed(np.zeros(16000, np.float32))
            assert [u.text for u in collect(t)] == ["first"]
            time.sleep(.25)
            mock.delay = .03  # Fresh input has a real response delay, shorter than its budget.
            t.feed(speech(1.04))
            t.feed(np.zeros(16000, np.float32))
            output = collect(t, timeout=2)
            assert [u.text for u in output] == ["second"]
            assert output[0].speech == [(2.04, 3.04)]
            assert len(mock.connections) == 2
            assert finish(t) == []
        finally:
            t.close()


def test_padded_final_tail_after_gap_infers_only_valid_mapped_samples():
    plans = [[("first", 0, 1), ("tail", 3.12, 4.12625)]]
    with MockGradium(plans=plans, dangling_last=True) as mock:
        t = default_track(mock)
        t.start()
        try:
            t.feed(speech(2))
            t.feed(np.zeros(4 * 16000, np.float32))
            t.feed(speech(1.00625))
            output = finish(t)
            assert [u.text for u in output] == ["first", "tail"]
            assert output[1].speech == [(6, 7.00625)]
            assert output[1].end == t.processed == 7.00625
            assert mock.connections[1]["samples"] == 66560
            assert t.sent_seconds == pytest.approx(4.16)
        finally:
            t.close()


def test_default_58_submitted_second_rotation_bounds_continuous_replay():
    with MockGradium(plans=[[("part", 0, 1)]]) as mock:
        t = default_track(mock, realtime=False)
        t.start()
        try:
            assert t.max_frames * 1280 == 58 * 16000
            t.feed(speech(117))
            output = finish(t)
            assert [u.start for u in output] == [0, 58, 116]
            assert [c["samples"] for c in mock.connections[1:]] == [928000, 928000, 16640]
            assert t.sent_seconds == pytest.approx(117.04)
            assert t.backlog_samples == 0
        finally:
            t.close()


def test_sparse_grouped_request_is_bounded_by_submitted_samples_not_source_age():
    with MockGradium(plans=[[("part", 0, .08)]]) as mock:
        t = default_track(mock, max_duration=30, realtime=False)
        t.start()
        try:
            t.feed(speech(.08))
            first = t.request
            t.feed(np.zeros(round(7.92 * 16000), np.float32))
            for _ in range(14):
                t.feed(speech(.08))
                t.feed(np.zeros(round(7.92 * 16000), np.float32))
            assert t.position == 120 * 16000
            assert not first.done and t.request is first
            assert first.samples == round(17.68 * 16000)
            assert len(first.runs) == 15
            t.feed(speech(.08))
            t.feed(np.zeros(round(7.92 * 16000), np.float32))
            output = finish(t)
            assert [u.start for u in output] == [0]
            assert len(mock.connections) == 2
            assert len(first.runs) == 16
            assert t.sent_seconds == pytest.approx(18.88)
        finally:
            t.close()


def test_overlapping_word_spans_count_unique_source_samples_for_phrase_and_embedding():
    plan = [("one", 0, 2), ("two", 1, 3), ("three", 2, 3)]
    with MockGradium(plans=[plan]) as mock:
        t = default_track(mock)
        t.cluster.embedder = lambda _: pytest.fail("overlap manufactured four seconds of speech")
        t.start()
        try:
            t.feed(speech(3.04))
            wait(lambda: t.results.qsize() >= 3)
            assert flush_tracks([t]) == []  # Raw duration sums to 5 s; union is only 3 s.
            output = finish(t)
            assert [u.text for u in output] == ["one two three"]
            assert output[0].speech == [(0, 3)]
            assert t.cluster.counts == []
        finally:
            t.close()


def test_empty_and_zero_length_word_spans_are_safe():
    with MockGradium(plans=[[("", 0, 0), ("zero", 0, 0), ("ok", 0, 1)]]) as mock:
        t = default_track(mock)
        t.start()
        try:
            t.feed(speech(1.04))
            output = finish(t)
            assert [u.text for u in output] == ["zero ok"]
            assert output[0].speech == [(0, 1)]
            assert t.cluster.counts == []
        finally:
            t.close()
