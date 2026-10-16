"""AEC routing and PCM supervision, without audio devices or helper builds."""

import io
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from hark import capture


@pytest.fixture(autouse=True)
def no_device(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(
        InputStream=lambda **kw: pytest.fail("audio device opened")))


def test_helper_availability_and_env_override(tmp_path, monkeypatch):
    default = tmp_path / "default"
    override = tmp_path / "override"
    monkeypatch.setattr(capture, "VOICE_MIC", default)
    monkeypatch.delenv("HARK_VOICE_MIC", raising=False)
    assert capture.voice_mic_path() == default
    assert not capture.voice_mic_available()
    default.write_text("placeholder")
    default.chmod(0o700)
    assert capture.voice_mic_available()
    monkeypatch.setenv("HARK_VOICE_MIC", str(override))
    assert capture.voice_mic_path() == override
    assert not capture.voice_mic_available()
    override.mkdir()
    assert not capture.voice_mic_available()
    override.rmdir()
    override.write_text("placeholder")
    override.chmod(0o600)
    assert not capture.voice_mic_available()
    override.chmod(0o700)
    assert capture.voice_mic_available()


def test_exported_path_honors_env_at_import(tmp_path):
    helper = tmp_path / "voice-mic"
    env = dict(os.environ, HARK_VOICE_MIC=str(helper))
    result = subprocess.run([sys.executable, "-c",
                             "from hark.capture import VOICE_MIC; print(VOICE_MIC)"],
                            env=env, capture_output=True, text=True, check=True)
    assert Path(result.stdout.strip()) == helper


@pytest.mark.parametrize("aec,device,available,uses_helper,warning", [
    (True, None, True, True, None),
    (True, None, False, False, "not executable"),
    (True, "USB", True, False, "explicit --mic"),
    (True, 0, True, False, "explicit --mic"),
    (False, None, True, False, None),  # room and enrollment
])
def test_mic_routes_without_changing_constructor(
        monkeypatch, capsys, aec, device, available, uses_helper, warning):
    monkeypatch.setattr(capture, "voice_mic_available", lambda: available)
    opened, raw = [], []
    monkeypatch.setattr(capture.PCMProcessSource, "_open", lambda self: opened.append(self))
    monkeypatch.setattr(capture.MicSource, "_stream", lambda self: raw.append(self))
    monkeypatch.setattr(capture.MicSource, "_watch", lambda self: None)
    mic = capture.MicSource(device)
    mic.echo_cancel = aec
    mic.start()
    assert bool(opened) == uses_helper
    assert bool(raw) != uses_helper
    if uses_helper:
        assert mic.helper.owner is mic
        assert mic.helper.command == [str(capture.voice_mic_path())]
        mic.stop()
        assert mic.helper.stopping.is_set()
    text = capsys.readouterr().err
    if warning:
        assert "AEC unavailable" in text and warning in text and "raw input" in text
    elif not aec:
        assert not text


def fake_pcm(monkeypatch, chunks):
    chunks = iter(chunks)
    monkeypatch.setattr(capture.select, "select", lambda *args: ([42], [], []))
    monkeypatch.setattr(capture.os, "read", lambda *args: next(chunks))
    return SimpleNamespace(stdout=SimpleNamespace(fileno=lambda: 42))


def test_helper_pcm_uses_mic_stats_timestamps_queue_and_errors(monkeypatch):
    monkeypatch.setattr(capture.time, "time", lambda: 100.)
    mic = capture.MicSource()
    mic.anchor = mic.last_audio = mic.last_sound = 99.
    helper = capture.PCMProcessSource("mic AEC", ["unused"], owner=mic)
    pcm = np.array([-32768, 0, 12345, 32767], dtype="<i2")
    raw = pcm.tobytes()
    proc = fake_pcm(monkeypatch, [raw[:1], raw[1:5], raw[5:], b""])
    assert helper._pump(proc) == (True, "hit EOF")
    np.testing.assert_array_equal(mic.drain(), pcm.astype(np.float32) / 32768)
    assert mic.heard == (4, 100.)
    assert mic.last_audio == mic.last_sound == 100.
    assert mic.take_stats() == {"device": 4, "padded": 0, "peak": 1.}
    assert mic.queued_samples == 0
    assert helper.stats["device"] == 0
    # The parent track's backlog gate is also authoritative for helper samples.
    mic.sample_limit = 0
    helper._pump(fake_pcm(monkeypatch, [raw, b""]))
    with pytest.raises(RuntimeError, match="mic: capture backlog full"):
        mic.drain()


def test_helper_delivery_obeys_mic_pause_gate(monkeypatch):
    from hark.pause import MicGate, PauseState

    mic = capture.MicSource()
    mic.anchor = mic.last_audio = mic.last_sound = 100.
    mic.heard = (0, 100.)
    helper = capture.PCMProcessSource("mic AEC", ["unused"], owner=mic)
    state, gate = PauseState(), MicGate(mic)
    monkeypatch.setattr(capture.time, "time", lambda: 100.2)
    pcm = np.full(3200, 8192, dtype="<i2")
    helper._pump(fake_pcm(monkeypatch, [pcm.tobytes(), b""]))
    assert gate.feed(mic.drain(), 100.2, state).size == 0
    state.update(100.2, True)
    muted = gate.feed(mic.drain(), 100.2, state, final=True)
    assert muted.size == pcm.size and not muted.any()
    assert mic.last_sound == 100.2
    assert mic.take_stats()["device"] == pcm.size


def test_stalled_helper_does_not_invent_device_samples(monkeypatch):
    mic = capture.MicSource()
    mic.anchor = mic.last_audio = mic.last_sound = 100.
    clock = iter([0., 6.])
    monkeypatch.setattr(capture.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(capture.select, "select", lambda *args: ([], [], []))
    helper = capture.PCMProcessSource("mic AEC", ["unused"], owner=mic)
    proc = SimpleNamespace(stdout=SimpleNamespace(fileno=lambda: 42))
    assert helper._pump(proc) == (False, "stalled (5 s without data)")
    monkeypatch.setattr(capture.time, "time", lambda: 103.)
    assert mic.drain().size == 40000
    assert mic.last_audio == mic.last_sound == 100.
    assert mic.take_stats() == {"device": 0, "padded": 40000, "peak": 0.}


class RetryEvent:
    def __init__(self, attempts):
        self.delays = []
        self.attempts = attempts

    def is_set(self):
        return len(self.delays) >= self.attempts

    def wait(self, delay):
        self.delays.append(delay)
        return self.is_set()


def test_spawn_failure_logs_aec_outage_and_retries_without_raw(monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise PermissionError("cannot execute")

    monkeypatch.setattr(capture.subprocess, "Popen", fail)
    helper = capture.PCMProcessSource("mic AEC (VoiceProcessingIO)", ["unused"])
    helper.stopping = RetryEvent(7)
    helper._supervise()
    assert helper.stopping.delays == [2., 4., 8., 16., 30., 30., 30.]
    text = capsys.readouterr().err
    assert "mic AEC (VoiceProcessingIO)" in text
    assert "cannot execute" in text and "capture unavailable" in text
    assert "restarting" in text


@pytest.mark.parametrize("reason", ["hit EOF", "stalled (5 s without data)"])
def test_exit_or_stall_is_reaped_and_restarted(monkeypatch, capsys, reason):
    made = []

    class Process:
        def __init__(self, *args, **kwargs):
            self.stdout, self.stderr = io.BytesIO(), io.BytesIO(b"AEC initialization failed\n")
            self.killed = self.reaped = False
            made.append(self)

        def poll(self): return None
        def kill(self): self.killed = True
        def wait(self): self.reaped = True; return 1

    monkeypatch.setattr(capture.subprocess, "Popen", Process)
    # Keep stderr consumption deterministic rather than starting real threads.
    monkeypatch.setattr(capture.threading, "Thread", lambda target, args, **kw:
                        SimpleNamespace(start=lambda: target(*args)))
    helper = capture.PCMProcessSource("mic AEC", ["unused"])
    helper.stopping = RetryEvent(2)
    monkeypatch.setattr(helper, "_pump", lambda proc: (False, reason))
    helper._supervise()
    assert len(made) == 2
    assert all(p.killed and p.reaped and p.stdout.closed and p.stderr.closed for p in made)
    text = capsys.readouterr().err
    assert reason in text and "AEC initialization failed" in text


def test_stop_interrupts_helper_pump_without_delivery(monkeypatch):
    helper = capture.PCMProcessSource("mic AEC", ["unused"])
    stopped = []
    helper.proc = SimpleNamespace(poll=lambda: None, terminate=lambda: stopped.append(True))
    helper.stop()
    assert stopped == [True]
    assert helper._pump(SimpleNamespace(stdout=SimpleNamespace(fileno=lambda: 42))) == (False, "stopped")
