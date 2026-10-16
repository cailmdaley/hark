"""Playback probes refuse active meetings and quantify known synthetic leakage."""
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def probe(monkeypatch):
    spec = importlib.util.spec_from_file_location("aec_probe", Path(__file__).parents[1] / "scripts/aec-probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **kw: "")
    return module


@pytest.mark.parametrize("meeting", [None, {"state": "ended"}, {"state": "failed"}])
def test_playback_requires_explicit_inactive_state(probe, monkeypatch, meeting):
    monkeypatch.setattr(probe.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(json.dumps({"meeting": meeting}).encode()))
    assert probe.playback_allowed("http://local")["meeting"] == meeting


@pytest.mark.parametrize("payload", [{}, {"meeting": {}}, {"meeting": {"state": "live"}},
                                     {"meeting": {"state": "loading"}}, {"meeting": {"state": "stopping"}}])
def test_live_or_unknown_meeting_refuses_playback(probe, monkeypatch, payload):
    monkeypatch.setattr(probe.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(json.dumps(payload).encode()))
    with pytest.raises(RuntimeError, match="refusing playback"):
        probe.playback_allowed("http://local")


def test_metrics_measure_known_attenuation_and_delay(probe):
    reference = np.random.default_rng(1).normal(size=16000).astype(np.float32)
    audio = np.concatenate([np.zeros(3200), 0.01 * reference, np.zeros(1000)])
    result = probe.metrics(audio, reference)
    assert result["absolute_peak_correlation"] == pytest.approx(1, abs=1e-6)
    assert result["reference_projection_gain"] == pytest.approx(0.01, rel=1e-5)
    assert result["best_offset_seconds"] == pytest.approx(0.2)


@pytest.mark.parametrize("command,live", [
    ("/opt/bin/hark --call", True),
    ("/usr/bin/python3 /opt/bin/hark --room", True),
    ("/usr/bin/python3 -m hark --phone", True),
    ("/usr/bin/python3 -m hark --file recording.wav", False),
    ("/opt/bin/hark --file=recording.wav", False),
    ("/opt/bin/hark enroll me", False),
    ("/usr/bin/python3 scripts/aec-probe.py --raw-worker", False),
    ("/bin/bash -c 'hark --call'", False),
])
def test_local_recording_command_detection(probe, command, live):
    assert probe.live_hark_command(command) == live


def test_standalone_capture_blocks_playback_even_with_no_managed_meeting(probe, monkeypatch):
    monkeypatch.setattr(probe.subprocess, "check_output", lambda *a, **kw: "/usr/bin/python3 /opt/bin/hark --room\n")
    monkeypatch.setattr(probe.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(b'{"meeting":null}'))
    with pytest.raises(RuntimeError, match="local hark may be recording"):
        probe.playback_allowed("http://local")


def test_all_zero_capture_is_inconclusive(probe):
    with pytest.raises(ValueError, match="all-zero PCM"):
        probe.metrics(np.zeros(200), np.ones(100))


def test_silent_or_short_reference_is_not_a_measurement(probe):
    with pytest.raises(ValueError):
        probe.metrics(np.zeros(100), np.zeros(100))
    with pytest.raises(ValueError):
        probe.metrics(np.zeros(10), np.ones(100))
