"""Process capture shutdown fences delivery before the final recording drain."""

import io
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from hark import capture


@pytest.mark.parametrize("blocked_in", ["select", "read"])
def test_stop_joins_blocked_pump_and_prevents_late_delivery(monkeypatch, blocked_in):
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    pcm = np.full(1600, 8192, dtype="<i2").tobytes()

    class Pipe(io.BytesIO):
        def fileno(self): return 42

    class Child:
        def __init__(self, *args, **kwargs):
            self.stdout, self.stderr = Pipe(), io.BytesIO()
            self.code = None

        def poll(self): return self.code
        def terminate(self): self.code = -15
        def kill(self): self.code = -9
        def wait(self, timeout=None): return self.code

    def block():
        entered.set()
        assert release.wait(3), "test didn't release blocked pump"

    def select(*args):
        if blocked_in == "select":
            block()
        return [42], [], []

    def read(*args):
        if blocked_in == "read":
            block()
        return pcm

    monkeypatch.setattr(capture.subprocess, "Popen", Child)
    monkeypatch.setattr(capture.select, "select", select)
    monkeypatch.setattr(capture.os, "read", read)
    mic = capture.MicSource()
    mic.anchor = mic.last_audio = mic.last_sound = time.time()
    helper = capture.PCMProcessSource("mic AEC", ["unused"], owner=mic)
    helper._open()
    assert entered.wait(2)

    def stop():
        helper.stop()
        returned.set()

    stopper = threading.Thread(target=stop)
    stopper.start()
    try:
        assert helper.stopping.wait(1)
        assert not returned.wait(.05), "stop returned while the pump was blocked"
    finally:
        release.set()
        stopper.join(2)
    assert returned.is_set()
    assert not helper.supervisor.is_alive()
    assert helper.proc.poll() is not None
    assert mic.drain().size == 0
    assert mic.stats["device"] == 0
    assert mic.queue.empty()


def test_stop_waits_for_already_admitted_delivery(monkeypatch):
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    mic = capture.MicSource()
    mic.anchor = mic.last_audio = mic.last_sound = time.time()
    helper = capture.PCMProcessSource("mic AEC", ["unused"], owner=mic)
    deliver = mic._deliver

    def admitted(samples):
        entered.set()
        assert release.wait(3)
        deliver(samples)

    monkeypatch.setattr(mic, "_deliver", admitted)
    monkeypatch.setattr(capture.select, "select", lambda *args: ([42], [], []))
    chunks = iter([np.full(1600, 8192, dtype="<i2").tobytes(), b""])
    monkeypatch.setattr(capture.os, "read", lambda *args: next(chunks))

    class Pipe:
        def fileno(self): return 42

    class Child:
        stdout = Pipe()

    helper.supervisor = threading.Thread(target=helper._pump, args=(Child(),))
    helper.supervisor.start()
    assert entered.wait(2)

    def stop():
        helper.stop()
        returned.set()

    stopper = threading.Thread(target=stop)
    stopper.start()
    try:
        assert not returned.wait(.05)
    finally:
        release.set()
        stopper.join(2)
    assert returned.is_set() and not helper.supervisor.is_alive()
    np.testing.assert_array_equal(mic.drain(), np.full(1600, .25, np.float32))
    assert mic.queue.empty()


def test_stop_reaps_real_child_that_ignores_terminate():
    # Only stderr carries a readiness marker; the real PCM pump receives no audio.
    ready = threading.Event()
    helper = capture.PCMProcessSource("test child", [sys.executable, "-u", "-c",
        "import signal,sys,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print('ready', file=sys.stderr, flush=True); time.sleep(30)"])

    def stderr(proc):
        assert proc.stderr.readline() == b"ready\n"
        ready.set()
        proc.stderr.read()
        proc.stderr.close()

    helper._stderr = stderr
    helper._open()
    try:
        assert ready.wait(3)
        started = time.monotonic()
        helper.stop()
        assert time.monotonic() - started < 2.5
        assert helper.proc.poll() is not None
        assert not helper.supervisor.is_alive()
        assert helper.proc.stdout.closed
        assert helper.stats["device"] == 0 and helper.queue.empty()
    finally:
        helper.stop()


def test_stop_kills_child_after_terminate_timeout(monkeypatch):
    calls = []

    class Child:
        def poll(self): return None
        def terminate(self): calls.append("terminate")
        def kill(self): calls.append("kill")
        def wait(self, timeout):
            calls.append(("wait", timeout))
            if "kill" not in calls:
                raise subprocess.TimeoutExpired("test", timeout)
            return -9

    helper = capture.PCMProcessSource("test child", ["unused"])
    helper.proc = Child()
    helper.stop()
    assert calls == ["terminate", ("wait", .5), "kill", ("wait", .5)]


def test_stop_does_not_join_current_supervisor():
    helper = capture.PCMProcessSource("test child", ["unused"])
    helper.supervisor = threading.current_thread()
    helper.stop()
    assert helper.stopping.is_set()
