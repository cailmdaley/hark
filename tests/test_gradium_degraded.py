import json
import os
from pathlib import Path
import signal
import socket
import tempfile
import threading
import time
import wave

import numpy as np
import pytest

from gradium_mock import MockGradium
from hark import cli
from hark.capture import to_pcm16
from test_gradium import track, speech, wait, finish
from test_gradium_cli import configure


def test_startup_outage_keeps_actual_phone_wav_growing_then_recovers(monkeypatch):
    with tempfile.TemporaryDirectory(prefix='hk-', dir='/tmp') as directory, MockGradium(error=('no workers', 1011), plans=[[('recovered', 0, 2)]]) as mock:
        home = Path(directory)
        configure(monkeypatch, home, mock)
        monkeypatch.setattr(cli.sys, 'platform', 'linux')
        failures = []
        def sender():
            try:
                wait(lambda: (home / 'phone.sock').exists())
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
                    conn.connect(str(home / 'phone.sock'))
                    deadline = time.monotonic() + 1.7  # Longer than the finite retry window.
                    while time.monotonic() < deadline:
                        conn.sendall(to_pcm16(speech(.1)).tobytes())
                        time.sleep(.03)
                    with wave.open(str(home / 'meeting.phone.wav')) as wav:
                        assert wav.getnframes() >= 4 * 16000
                    assert json.loads((home / 'meeting.json').read_text())['phase'] == 'live'
                    assert (home / 'meeting.txt').read_text().count('# gradium lost at') == 1
                    mock.error = None
                    wait(lambda: '# gradium back at' in (home / 'meeting.txt').read_text())
                    conn.sendall(to_pcm16(speech(2.4)).tobytes())
                    wait(lambda: 'recovered' in (home / 'meeting.txt').read_text())
            except BaseException as error:
                failures.append(error)
            finally:
                os.kill(os.getpid(), signal.SIGTERM)
        worker = threading.Thread(target=sender)
        worker.start()
        assert cli.main(['--ear', 'gradium', '--phone', '--launch', 'test', '-o', str(home / 'meeting.txt')]) is None
        worker.join(3)
        assert not failures and not worker.is_alive()
        text = (home / 'meeting.txt').read_text()
        assert text.count('# gradium lost at') == text.count('# gradium back at') == 1
        assert text.splitlines()[-1].startswith('# ended ')
        assert json.loads((home / 'meeting.json').read_text())['phase'] == 'ended'


def test_bounded_backlog_drops_only_stt_and_offline_final_cancels_30_second_backoff():
    with MockGradium(error=('rate limit', 1008)) as mock:
        t = track(mock, backlog_seconds=1, backoff=30)
        t.start()
        try:
            t.feed(speech(130))
            assert t.audio_samples == 130 * 16000
            assert t.audio.slice(129, 130).size == 16000
            assert t.backlog_samples <= 16000
            assert t.jobs.qsize() <= t.jobs.maxsize
            began = time.monotonic()
            finish(t)
            assert time.monotonic() - began < .5
            assert t.error is None and not t.thread.is_alive()
            notices = t.take_notices()
            assert sum('lost at' in n for n in notices) == 1
            assert any('missed recognition' in n for n in notices)
        finally:
            t.close()


def test_pending_words_survive_abort_after_acknowledged_drop():
    with MockGradium(plans=[[('pending words', 0, 2)]], dangling_last=True, drop_first_at=2.4) as mock:
        t = track(mock, backoff=30)
        t.start()
        try:
            t.feed(speech(2.4))
            wait(lambda: t.unavailable)
            assert t.flush(force=True) == []  # A step ACK isn't a finalized text boundary.
            began = time.monotonic()
            t.abort()
            assert time.monotonic() - began < .5
            assert [u.text for u in t.flush(force=True)] == ['pending words']
            assert t.take_notices() == []
            assert len(mock.connections) == 2
        finally:
            t.close()


def test_sustained_outage_full_backlog_admits_no_empty_jobs_and_recovers():
    with MockGradium(error=('no workers', 1011), plans=[[('word', 0, .08)]]) as mock:
        t = track(mock, backlog_seconds=.16, max_duration=.08)
        t.start()
        try:
            t.feed(speech(20))
            assert t.backlog_samples == 2560
            assert t.jobs.qsize() == 2
            assert all(job.wire_samples == 1280 and job.frames for job in t.jobs.queue)
            mock.error = None
            wait(lambda: t.backlog_samples == 0)
            assert t.jobs.empty()
            assert len([c for c in mock.connections if c['samples']]) == 2
            t.feed(speech(.08))
            output = finish(t)
            assert [u.text for u in output] == ['word', 'word', 'word']
            assert [u.start for u in output] == [0, .08, 20]
            assert len([c for c in mock.connections if c['samples']]) == 3
        finally:
            t.close()


@pytest.mark.parametrize('drop_at', [None, 1])
def test_nonrealtime_file_preserves_all_twenty_seconds_through_transient_outage(drop_at):
    with MockGradium(plans=[[('part', 0, 2)]], drop_first_at=drop_at) as mock:
        t = track(mock, max_duration=2, realtime=False, backlog_seconds=4, backoff=.2)
        t.start()
        try:
            t.feed(speech(20))
            output = finish(t)
            assert [u.text for u in output] == ['part'] * 10
            assert [u.start for u in output] == list(range(0, 20, 2))
            assert t.backlog_samples == 0 and t.missed is None
            assert not any('missed recognition' in notice for notice in t.take_notices())
        finally:
            t.close()


def test_nonrealtime_file_full_job_queue_waits_through_transient_outage():
    with MockGradium(plans=[[('part', 0, 2)]], drop_first_at=1) as mock:
        t = track(mock, max_duration=2, realtime=False, queue_size=1, backlog_seconds=6, backoff=.2)
        t.start()
        try:
            t.feed(speech(20))
            output = finish(t)
            assert [u.start for u in output] == list(range(0, 20, 2))
            assert t.backlog_samples == 0 and t.missed is None
        finally:
            t.close()


def test_nonrealtime_final_retries_beyond_live_stop_attempt_limit():
    with MockGradium(plans=[[('part', 0, 2)]], stall_setup={1: .1, 2: .1, 3: .1}) as mock:
        t = track(mock, realtime=False, retries=0, timeout=.05, shutdown_timeout=1)
        t.start()
        try:
            t.feed(speech(2))
            assert [u.text for u in finish(t)] == ['part']
            assert len(mock.connections) == 5
        finally:
            t.close()


def test_nonrealtime_final_permanent_outage_is_bounded():
    with MockGradium(http_status=503) as mock:
        t = track(mock, realtime=False, shutdown_timeout=.15)
        t.start()
        try:
            t.feed(speech(.08))
            began = time.monotonic()
            assert finish(t) == []
            assert .14 <= time.monotonic() - began < .6
            assert not t.thread.is_alive()
        finally:
            t.close()


def test_final_does_not_join_again_after_abort_times_out():
    t = track(type('Mock', (), {'url': 'ws://127.0.0.1:1'})())
    joins = []
    class StuckThread:
        def join(self, timeout): joins.append(timeout)
    t.thread = StuckThread()
    t.degraded = True
    try:
        finish(t)
        assert joins == [t.timeout + 2]
    finally:
        t.close()


def test_speaker_warm_failure_falls_back_without_stopping_capture():
    with MockGradium(plans=[[('words', 0, 2)]]) as mock:
        t = track(mock)
        def failed(): raise OSError('offline embedding weights')
        t.cluster.warm = failed
        t.start()
        try:
            t.feed(speech(2))
            assert finish(t)[0].text == 'words'
            assert t.cluster_failed
        finally:
            t.close()
