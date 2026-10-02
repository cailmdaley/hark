from datetime import datetime
import json
from pathlib import Path
import socket
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import pytest

from gradium_mock import MockGradium
from hark.capture import PhoneSource, to_pcm16
from hark.gradium import GradiumError, GradiumTrack, api_key
from hark.transcript import Sink, flush_tracks
from hark.voice import OnlineCluster, VoiceMatcher


def speech(seconds, level=0.003):
    return np.full(round(seconds * 16000), level, np.float32)


def wait(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    pytest.fail("condition timed out")


def track(mock, **kwargs):
    options = dict(timeout=0.5, backoff=0.01, shutdown_timeout=5, phrase_seconds=2)
    options.update(kwargs)
    return GradiumTrack("phone", key="mock-key", url=mock.url,
                        cluster=OnlineCluster(lambda _: np.array([1., 0.])), **options)


def finish(t):
    t.feed(np.zeros(0, np.float32), final=True)
    return flush_tracks([t], force=True)


def test_quiet_speech_is_gated_with_preroll_hangover_and_source_clock(tmp_path):
    plan = [[(["Bon", "jour"], .32, 2.32), ("monde", 2.32, 4.32, 1), ("!", 4.32, 4.4)]]
    with MockGradium(plans=plan) as mock:
        t = track(mock, language="fr-FR")
        t.t0 = datetime(2026, 10, 2, 12)
        sink = Sink(tmp_path / "meeting.txt", "test")
        t.start()
        try:
            t.feed(np.zeros(10 * 16000, np.float32))
            assert len(mock.connections) == 1
            assert mock.connections[0]["samples"] == 0
            t.feed(speech(4.08))
            t.feed(np.zeros(15 * 16000, np.float32))
            lines = finish(t)
            assert [u.text for u in lines] == ["Bonjour", "monde", "!"]
            u = lines[0]
            assert u.start == pytest.approx(10)
            assert lines[-1].end == pytest.approx(14.08)
            assert u.wall == datetime(2026, 10, 2, 12, 0, 10)
            assert lines[-1].wall_end == datetime(2026, 10, 2, 12, 0, 14, 80000)
            assert t.sent_seconds == pytest.approx(5.2)
            for u in lines:
                sink.write(u)
            sink.close("ended")
            assert "12:00:10-12:00:12 S1" in sink.path.read_text()
            messages = mock.connections[1]["messages"]
            assert [m["type"] for m in messages][-2:] == ["flush", "end_of_stream"]
            assert mock.connections[0]["key"] == "mock-key"
            assert messages[0]["json_config"]["language"] == "fr"
            assert np.all(np.frombuffer(mock.connections[1]["nonzero_head"], dtype="<i2") == 98)
        finally:
            t.close()


def test_final_segment_is_visible_before_burst_end_or_eos():
    with MockGradium(plans=[[("live", 0, 2), ("later", 2, 4)]]) as mock:
        t = track(mock)
        t.start()
        try:
            t.feed(speech(2))
            wait(lambda: not t.results.empty())
            output = flush_tracks([t])
            assert [u.text for u in output] == ["live"]
            assert not t.request.done
            assert not any(m["type"] == "end_of_stream" for m in mock.connections[1]["messages"])
            t.feed(speech(2))
            assert [u.text for u in finish(t)] == ["later"]
        finally:
            t.close()


def test_startup_auth_checked_even_all_silent_and_no_audio_sent():
    with MockGradium() as mock:
        t = track(mock)
        t.start()
        try:
            t.feed(np.zeros(60 * 16000, np.float32))
            assert finish(t) == []
            assert len(mock.connections) == 1
            assert mock.connections[0]["samples"] == 0
            assert t.sent_seconds == 0
        finally:
            t.close()


def test_drop_replays_only_uncommitted_burst_and_counts_retries():
    plans = [[("one", .32, 2.32), ("two", 2.32, 4.32)]]
    with MockGradium(plans=plans, drop_first_at=2.4) as mock:
        t = track(mock)
        t.start()
        try:
            t.feed(np.zeros(3 * 16000, np.float32))
            t.feed(speech(4))
            t.feed(np.zeros(5 * 16000, np.float32))
            output = finish(t)
            assert [u.text for u in output] == ["one", "two"]
            assert output[0].start == pytest.approx(2.96)
            assert len(mock.connections) == 3
            sent = sum(c["samples"] / 16000 for c in mock.connections)
            assert sent <= t.sent_seconds <= 2 * 5.12 + 1e-9
            assert t.sent_seconds > 5.12
        finally:
            t.close()


def test_reconnect_between_bursts_preserves_discarded_gap():
    with MockGradium(plans=[[("first", .32, 2.32)], [("second", .32, 2.32)]]) as mock:
        t = track(mock)
        t.start()
        try:
            t.feed(np.zeros(16000, np.float32))
            t.feed(speech(2))
            t.feed(np.zeros(8 * 16000, np.float32))
            wait(lambda: not t.results.empty())
            first = flush_tracks([t])
            t._end_request()
            t.feed(speech(2))
            second = finish(t)
            assert [u.text for u in first + second] == ["first", "second"]
            assert [u.start for u in first + second] == pytest.approx([0.96, 10.96])
            assert [u.end for u in first + second] == pytest.approx([2.96, 12.96])
            assert len(mock.connections) == 3
        finally:
            t.close()


def test_phone_socket_padding_through_track_and_sink(tmp_path):
    path = Path(tempfile.mkdtemp(prefix="hk-", dir="/tmp")) / "phone.sock"
    with MockGradium(plans=[[("phone", .32, 2.32)]]) as mock:
        t = track(mock)
        src = PhoneSource(path)
        t.start()
        src.start()
        t.t0 = datetime.fromtimestamp(src.anchor)
        sink = Sink(tmp_path / "phone.txt", "phone")
        try:
            # The actual Source pads a four-second device gap on its own clock.
            src.anchor -= 4
            t.t0 = datetime.fromtimestamp(src.anchor)
            gap = src.drain()
            assert gap.size > 3 * 16000
            t.feed(gap)
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.connect(str(path))
            client.sendall(to_pcm16(speech(2.4)).tobytes())
            client.close()
            wait(lambda: src.stats["device"] == 2.4 * 16000)
            device = src.drain(limit=float("inf"))
            t.feed(device)
            src.anchor -= 5
            t.feed(src.drain())
            output = finish(t)
            assert len(output) == 1
            u = output[0]
            # Burst begins four frames before the first voiced frame; start_s .32 cancels it.
            expected = (gap.size // 1280) * .08
            assert u.start == pytest.approx(expected)
            assert u.wall.timestamp() == pytest.approx(t.t0.timestamp() + expected)
            sink.write(u)
            sink.close("ended")
            record = json.loads(sink.path.with_suffix(".jsonl").read_text())
            assert record["start"] == round(expected, 2)
            assert record["track"] == "phone" and record["text"] == "phone"
        finally:
            src.stop()
            t.close()
    assert not path.exists()


def test_segment_clustering_and_existing_voice_matcher_name_me(tmp_path):
    with MockGradium(plans=[[("A", 0, 2), ("B", 2, 4), ("A again", 4, 6), ("yes", 6, 6.4)]]) as mock:
        vectors = iter([[1., 0.], [0., 1.], [1., 0.]])
        t = track(mock)
        t.cluster = OnlineCluster(lambda _: np.array(next(vectors)))
        t.start()
        sink = Sink(tmp_path / "room.txt", "room")
        matcher = VoiceMatcher({"me": np.array([1., 0.])}, sink,
                               embedder=lambda _: np.array([1., 0.]))
        try:
            t.feed(speech(6.4, level=.02))
            output = finish(t)
            assert [u.speaker for u in output] == ["S1", "S2", "S1", "S1"]
            assert output[-1].text == "yes"
            # Repeated segments accumulate the five seconds required by VoiceMatcher.
            for u in output:
                matcher.finished(t, u)
            matcher.finished(t, output[0])
            assert sink.names == {"S1": "me"}
            sink.write(output[-1])
            assert output[-1].name == "me"
            sink.close("ended")
        finally:
            t.close()


def test_fixed_call_mic_label_never_clusters():
    with MockGradium(plans=[[("me", 0, 2)]]) as mock:
        t = track(mock, fixed_speaker="me")
        t.cluster = SimpleNamespace(assign=lambda _: pytest.fail("mic clustered"))
        t.start()
        try:
            t.feed(speech(2))
            assert finish(t)[0].speaker == "me"
        finally:
            t.close()


@pytest.mark.parametrize("error,ready,finish_,match", [
    (("invalid authentication", 1008), True, True, "invalid authentication"),
])
def test_failures_are_bounded_and_threads_stop(error, ready, finish_, match):
    with MockGradium(error=error, ready=ready, finish=finish_) as mock:
        t = track(mock)
        began = time.monotonic()
        try:
            with pytest.raises(GradiumError, match=match):
                t.start()
                t.feed(speech(2))
                finish(t)
            assert time.monotonic() - began < 4
            assert len(mock.connections) == (1 if error and error[1] == 1008 else 4 if not finish_ else 3)
        finally:
            t.close()
        assert not t.thread.is_alive()


def test_committed_segments_survive_later_failure_without_flush_first():
    with MockGradium(plans=[[("kept", 0, 2)]], error=("down", 1011), error_from=2) as mock:
        t = track(mock)
        t.start()
        try:
            t.feed(speech(2))
            t.feed(np.zeros(16000, np.float32))
            wait(lambda: not t.results.empty())
            t._end_request()
            t.feed(speech(2))
            wait(lambda: t.degraded)
            assert [u.text for u in flush_tracks([t])] == ["kept"]
            t.check()
            assert flush_tracks([t]) == []
            finish(t)
        finally:
            t.close()


@pytest.mark.parametrize("status,attempts", [(401, 1), (403, 1)])
def test_handshake_auth_is_terminal_but_refusal_retries(status, attempts):
    with MockGradium(http_status=status) as mock:
        t = track(mock)
        try:
            with pytest.raises(GradiumError, match="authentication refused" if status != 503 else "persistent failure"):
                t.start()
            assert mock.handshakes == attempts
        finally:
            t.close()


def test_queue_overload_drops_recognition_not_capture():
    with MockGradium(finish=False) as mock:
        t = track(mock, queue_size=1, max_duration=1)
        t.start()
        try:
            t.feed(speech(10))
            assert t.audio_samples == 10 * 16000
            assert t.missed is not None
            assert t.jobs.qsize() <= 1
            assert t.backlog_samples <= t.backlog_limit
        finally:
            t.close()


def test_duration_rotation_keeps_long_speech_bounded_and_contiguous():
    with MockGradium(plans=[[("part", 0, 2)]]) as mock:
        t = track(mock, max_duration=2, realtime=False)
        t.start()
        try:
            t.feed(speech(10))
            output = finish(t)
            assert [u.start for u in output] == [0, 2, 4, 6, 8]
            assert len(mock.connections) == 6
            assert all(c["samples"] == 2 * 16000 for c in mock.connections[1:])
            assert t.sent_seconds == pytest.approx(10)
        finally:
            t.close()


def test_character_rotation_at_completed_segment_boundary():
    with MockGradium(plans=[[("a" * 1200, 0, 2)], [("tail", 0, 2)]]) as mock:
        t = track(mock)
        t.start()
        try:
            t.feed(speech(2))
            wait(lambda: t.request.rotate)
            t.feed(speech(2))
            output = finish(t)
            assert len(mock.connections) == 3
            assert [u.start for u in output] == [0, 2]
            assert output[1].text == "tail"
        finally:
            t.close()


def test_overlapping_happy_path_segments_are_not_suppressed():
    with MockGradium(plans=[[("Hello", 0, 2), ("world", 1.92, 4)]]) as mock:
        t = track(mock)
        t.start()
        try:
            t.feed(speech(4))
            assert [u.text for u in finish(t)] == ["Hello", "world"]
        finally:
            t.close()


def test_replay_boundary_change_is_suppressed_and_logged(capsys):
    with MockGradium(plans=[[("Hello", 0, 2)], [("Hello world", 0, 4)]], drop_first_at=2.4) as mock:
        t = track(mock)
        t.start()
        try:
            t.feed(speech(4))
            assert [u.text for u in finish(t)] == ["Hello"]
            assert "replay skips segment 0.00–4.00 before committed horizon 2.00" in capsys.readouterr().err
        finally:
            t.close()


def test_many_short_bursts_survive_retryable_setup_delay():
    with MockGradium(plans=[[("turn", 0, .32)]], stall_setup={1: .3}) as mock:
        t = track(mock, timeout=.1)
        t.start()
        try:
            for _ in range(20):
                t.feed(speech(.4))
                t.feed(np.zeros(round(.96 * 16000), np.float32))
                t._end_request()
            assert t.jobs.qsize() > 4
            output = finish(t)
            assert len(output) == 20
            assert t.backlog_samples == 0
        finally:
            t.close()


def test_audio_seconds_backlog_bound_omits_only_recognition():
    with MockGradium() as mock:
        t = track(mock, backlog_seconds=1)
        t.start()
        try:
            t.feed(speech(20))
            assert t.audio_samples == 20 * 16000
            assert t.missed is not None
            assert t.backlog_samples <= 16000
        finally:
            t.close()


def test_retries_back_off_without_terminal_failure():
    with MockGradium(http_status=503) as mock:
        t = track(mock, backoff=.04)
        began = time.monotonic()
        try:
            t.start()
            assert time.monotonic() - began < .2
            wait(lambda: mock.handshakes >= 3)
            assert time.monotonic() - began >= .11
            assert t.degraded and t.error is None
        finally:
            t.close()


def test_progress_keeps_flush_and_final_alive_beyond_idle_timeout():
    with MockGradium(plans=[[("part", 0, 2)]], delay=.005) as mock:
        t = track(mock, max_duration=2, realtime=False, timeout=.04, shutdown_timeout=.06)
        t.start()
        began = time.monotonic()
        try:
            t.feed(speech(10))
            output = finish(t)
            assert len(output) == 5
            assert time.monotonic() - began > 5 * t.shutdown_timeout
            assert len(mock.connections) == 6  # startup plus five successful bursts, no retry
        finally:
            t.close()


def test_track_warms_default_embedder_before_capture(monkeypatch):
    calls = []
    monkeypatch.setattr("hark.voice.Embedder", lambda: lambda samples: calls.append(len(samples)) or np.array([1., 0.]))
    with MockGradium() as mock:
        t = track(mock)
        t.cluster = OnlineCluster()
        t.start()
        try:
            assert calls == [16000]
            assert t.audio_samples == 0
            assert finish(t) == []
        finally:
            t.close()


def test_startup_wait_observes_stop():
    import threading
    with MockGradium(ready=False) as mock:
        t = track(mock)
        stop = threading.Event()
        timer = threading.Timer(.05, stop.set)
        timer.start()
        began = time.monotonic()
        try:
            t.start(stop=stop)
            assert time.monotonic() - began < 1
            assert not t.thread.is_alive()
        finally:
            timer.join()
            t.close()


def test_cluster_failure_preserves_all_text_and_logs_once(capsys):
    with MockGradium(plans=[[("first", 0, 2), ("second", 2, 4)]]) as mock:
        t = track(mock)
        def broken(_):
            raise ValueError("embedding exploded")
        t.cluster = OnlineCluster(broken)
        t.start()
        try:
            t.feed(speech(4))
            output = finish(t)
            assert [u.text for u in output] == ["first", "second"]
            assert [u.speaker for u in output] == ["S1", "S1"]
            assert capsys.readouterr().err.count("speaker clustering disabled") == 1
        finally:
            t.close()


def test_word_phrases_are_live_and_dangling_last_word_flushes_at_eos():
    words = [(w, i * .4, (i + 1) * .4) for i, w in enumerate(["One", "two", "three", "four", "five", "six."])]
    with MockGradium(plans=[words], dangling_last=True) as mock:
        t = track(mock)
        t.start()
        try:
            t.feed(speech(2))
            wait(lambda: t.results.qsize() >= 5)
            live = flush_tracks([t])
            assert [u.text for u in live] == ["One two three four five"]
            assert live[0].speaker == "S1" and t.cluster.counts == [1]
            assert not t.request.done
            t.feed(speech(.4))
            rest = finish(t)
            assert [u.text for u in rest] == ["six."]
            assert rest[0].end == pytest.approx(2.4)
            assert t.cluster.counts == [1]
        finally:
            t.close()


def test_calibrated_default_phrase_and_cluster_minimum():
    t = GradiumTrack("phone", key="mock-key")
    try:
        assert t.phrase_seconds == 4
        assert t.cluster.minimum_duration == 4
        assert t.cluster.threshold == .55
    finally:
        t.close()


def test_sparse_phrase_has_eight_second_wall_ceiling():
    words = [("word", i * .64, i * .64 + .08) for i in range(14)]
    with MockGradium(plans=[words]) as mock:
        t = track(mock, phrase_seconds=4)
        t.start()
        try:
            t.feed(speech(9))
            wait(lambda: t.results.qsize() >= 14)
            output = flush_tracks([t])
            assert len(output) == 1 and len(output[0].speech) == 14
            assert output[0].end == pytest.approx(8.4)
            assert not t.request.done
            assert t.cluster.counts == []
            finish(t)
        finally:
            t.close()


def test_unchanged_step_heartbeats_do_not_prevent_failure_or_leave_thread_alive():
    with MockGradium(stalled_flush=True) as mock:
        t = track(mock, timeout=.1, shutdown_timeout=.5)
        t.start()
        began = time.monotonic()
        try:
            t.feed(speech(2))
            finish(t)
            assert t.error is None
            assert time.monotonic() - began < 2
            assert t.finished.is_set() and not t.thread.is_alive()
            assert t.thread.daemon
        finally:
            t.close()


def test_cancel_interrupts_a_sender_waiting_for_flush():
    with MockGradium(stalled_flush=True) as mock:
        t = track(mock, timeout=3)
        t.start()
        try:
            t.feed(speech(2))
            t.feed(np.zeros(16000, np.float32))
            wait(lambda: len(mock.connections) > 1 and
                 any(m["type"] == "flush" for m in mock.connections[1]["messages"]))
            began = time.monotonic()
            t.abort()
            assert time.monotonic() - began < .5
            assert not t.thread.is_alive()
        finally:
            t.close()


def test_inferred_tail_end_never_exceeds_unpadded_track():
    with MockGradium(plans=[[("in.", .4, 1.00625)]], dangling_last=True) as mock:
        t = track(mock)
        t.start()
        try:
            t.feed(speech(1.00625))
            output = finish(t)
            assert len(output) == 1
            assert output[0].end == t.processed == pytest.approx(1.00625)
            assert output[0].speech == [(.4, 1.00625)]
            assert mock.connections[1]["samples"] / 16000 == 1.04
        finally:
            t.close()


def test_key_environment_precedence_permissions_and_lines(tmp_path, monkeypatch):
    path = tmp_path / "gradium.key"
    path.write_text("file-secret\n")
    path.chmod(0o644)
    monkeypatch.setenv("GRADIUM_API_KEY", "env-secret")
    assert api_key(path) == "env-secret"
    monkeypatch.delenv("GRADIUM_API_KEY")
    with pytest.raises(GradiumError, match="mode 600"):
        api_key(path)
    path.chmod(0o600)
    assert api_key(path) == "file-secret"
    path.write_text("one\ntwo\n")
    with pytest.raises(GradiumError, match="one non-empty line"):
        api_key(path)
    path.unlink()
    with pytest.raises(GradiumError, match="missing API key"):
        api_key(path)
