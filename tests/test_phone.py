import socket
import tempfile
import time
from pathlib import Path

import numpy as np

from hark.capture import PhoneSource, to_pcm16


def wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def collect(src, n):
    got = []
    wait_for(lambda: (got.append(src.drain(limit=n)) or sum(g.size for g in got) >= n))
    return np.concatenate(got)


def send(path, samples):
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.connect(str(path))
    conn.sendall(to_pcm16(samples).tobytes())
    return conn


def test_phone_source_delivers_pcm_and_survives_a_reconnect():
    # A short directory: macOS caps a Unix socket path at 104 bytes.
    path = Path(tempfile.mkdtemp(prefix="hk")) / "phone.sock"
    src = PhoneSource(path)
    src.live = False  # no wall-clock padding, so the test sees exactly what was sent
    src.start()
    try:
        first = (np.arange(1600, dtype=np.float32) / 32768.0)
        a = send(path, first)
        assert np.array_equal(collect(src, first.size), first)
        # an odd byte split across sends still lands on whole samples
        b = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        b.connect(str(path))  # replaces the first sender
        raw = to_pcm16(-first).tobytes()
        b.sendall(raw[:1001])
        time.sleep(0.05)
        b.sendall(raw[1001:])
        assert np.array_equal(collect(src, first.size), -first)
        assert src.last_sound is not None
        a.close()
        b.close()
    finally:
        src.stop()
    assert not path.exists()
