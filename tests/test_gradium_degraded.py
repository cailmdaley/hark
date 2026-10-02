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


def test_pending_words_survive_abort_during_backoff():
    with MockGradium(plans=[[('pending words', 0, 2)]], dangling_last=True, drop_first_at=2.4) as mock:
        t = track(mock, backoff=30)
        t.start()
        try:
            t.feed(speech(2.4))
            wait(lambda: t.degraded)
            began = time.monotonic()
            t.abort()
            assert time.monotonic() - began < .5
            assert [u.text for u in t.flush(force=True)] == ['pending words']
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
