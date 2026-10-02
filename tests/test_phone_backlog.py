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

from hark import cli
from hark.capture import PhoneSource, SAMPLE_RATE, to_pcm16
from test_phone import wait_for


def test_actual_socket_accepts_more_than_32_seconds_of_worklet_frames():
    with tempfile.TemporaryDirectory(prefix='hk-', dir='/tmp') as home:
        src = PhoneSource(Path(home) / 'phone.sock')
        src.live = False
        src.start()
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
                conn.connect(str(src.path))
                for index in range(400):
                    conn.sendall(b'\1\0' * 1600)
                    deadline = time.monotonic() + 2
                    while src.queued_samples < (index + 1) * 1600:
                        assert time.monotonic() < deadline
                        time.sleep(.0001)
                assert src.queue.qsize() == 400  # Each worklet frame really reached the queue separately.
                assert src.queued_samples == 40 * SAMPLE_RATE
            assert src.error is None
            assert src.drain(limit=float('inf')).size == 40 * SAMPLE_RATE
            assert src.queued_samples == 0
        finally:
            src.stop()


def test_overflow_tail_is_fully_recorded_before_error(monkeypatch):
    with tempfile.TemporaryDirectory(prefix='hk-', dir='/tmp') as home:
        home = Path(home)
        monkeypatch.setattr(cli, 'HOME', home)
        source = PhoneSource(home / 'phone.sock')
        source.sample_limit = 3200
        source.live = False
        def start():
            source.anchor = source.last_sound = 0
            for _ in range(3): source._deliver(np.ones(1600, np.float32) / 32768)
        monkeypatch.setattr(source, 'start', start)
        monkeypatch.setattr(cli, 'PhoneSource', lambda _: source)
        monkeypatch.setattr(cli, '_default_ear', lambda: 'local')
        monkeypatch.setattr(cli, 'load_models', lambda _: (None, None))
        class Track:
            processed = 0
            def __init__(self, name, *args, **kwargs): self.name = name
            def feed(self, *args, **kwargs): pass
            def flush(self, force=False): return []
        monkeypatch.setattr(cli, 'Track', Track)
        import pytest
        with pytest.raises(RuntimeError, match='capture backlog full'):
            cli.main(['--ear', 'local', '--phone', '--launch', 'test', '-o', str(home / 'out.txt')])
        with wave.open(str(home / 'out.phone.wav')) as wav:
            assert wav.getnframes() == 3200
            assert wav.readframes(3200) == b'\1\0' * 3200
        assert json.loads((home / 'meeting.json').read_text())['phase'] == 'failed'
