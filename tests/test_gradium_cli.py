import builtins
import json
import os
from pathlib import Path
import signal
import socket
import threading
import time
from urllib.request import Request, urlopen

import numpy as np
import pytest

from gradium_mock import MockGradium
from hark import cli
from hark.capture import WavRecorder, to_pcm16
from hark.gradium import GradiumError, GradiumTrack, credits_left
from hark.voice import OnlineCluster
from test_gradium import speech, wait


def prohibit_mlx(monkeypatch):
    real_import = builtins.__import__
    def no_mlx(name, *args, **kwargs):
        if name == "mlx" or name.startswith("mlx.") or name.startswith("mlx_audio"):
            raise AssertionError("Gradium imported MLX")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_mlx)


def configure(monkeypatch, tmp_path, mock=None):
    monkeypatch.setattr(cli, "HOME", tmp_path)
    monkeypatch.setenv("GRADIUM_API_KEY", "mock-key")
    metered = []
    monkeypatch.setattr(cli, "credits_left", lambda key: metered.append(key) or (900 - len(metered)))
    if mock:
        def make_track(name, **kwargs):
            return GradiumTrack(name, **kwargs, url=mock.url, timeout=.3, shutdown_timeout=3, phrase_seconds=2,
                                backoff=.01, cluster=OnlineCluster(lambda _: np.array([1., 0.])))
        monkeypatch.setattr(cli, "GradiumTrack", make_track)
    return metered


@pytest.mark.parametrize("failure", ["missing-key", "auth", "refusal", "http-auth", "no-ready"])
def test_cli_startup_failure_writes_failed_lifecycle_comment_ended_and_mirrors(
        failure, tmp_path, monkeypatch):
    options = {"auth": {"error": ("authentication rejected", 1008)},
               "refusal": {"error": ("no workers", 1011)},
               "http-auth": {"http_status": 401}, "no-ready": {"ready": False}}
    mirrors = []

    class Mirror:
        def __init__(self, path, *args):
            self.path = path
        def start(self):
            mirrors.append("started")
        def finish(self, timeout):
            mirrors.append(self.path.read_text())
            return True

    with MockGradium(**options.get(failure, {})) as mock:
        metered = configure(monkeypatch, tmp_path, mock)
        monkeypatch.setattr(cli, "TranscriptMirror", Mirror)
        if failure == "missing-key":
            def missing():
                raise GradiumError("missing API key")
            monkeypatch.setattr(cli, "api_key", missing)
        out = tmp_path / "failure.txt"
        assert cli.main(["--ear", "gradium", "--launch", "test", "--phone", "-o", str(out), "--mirror", "h:notes"]) == 1
        lines = out.read_text().splitlines()
        assert lines[-2].startswith("# gradium ") and lines[-1].startswith("# ended ")
        state = json.loads((tmp_path / "meeting.json").read_text())
        assert state["phase"] == "failed" and state["error"]
        assert state["ear"] == {"name": "gradium", "seconds": 0.0,
                                "credits_left": 898 if failure != "missing-key" else None}
        assert metered == (["mock-key", "mock-key"] if failure != "missing-key" else [])
        assert mirrors == ["started", out.read_text()]
        assert not (tmp_path / "phone.sock").exists()


def test_real_phone_cli_emits_live_and_handles_signal_without_mlx(tmp_path, monkeypatch):
    # macOS has a short AF_UNIX path limit, so this test uses a short source home.
    import tempfile
    tmp_path = Path(tempfile.mkdtemp(prefix="hkc"))
    failures = []
    with MockGradium(plans=[[("live phone", 0, 2)]]) as mock:
        metered = configure(monkeypatch, tmp_path, mock)
        prohibit_mlx(monkeypatch)
        monkeypatch.setattr(cli.sys, "platform", "linux")
        out = tmp_path / "phone.txt"
        def phone():
            signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGUSR1})
            try:
                wait(lambda: (tmp_path / "meeting.json").exists() and
                     json.loads((tmp_path / "meeting.json").read_text())["phase"] == "live")
                client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                client.connect(str(tmp_path / "phone.sock"))
                client.sendall(to_pcm16(speech(2.4)).tobytes())
                wait(lambda: out.with_suffix(".jsonl").exists() and "live phone" in out.with_suffix(".jsonl").read_text())
                assert not any(m["type"] == "end_of_stream" for m in mock.connections[1]["messages"])
                client.close()
            except BaseException as error:
                failures.append(error)
            finally:
                os.kill(os.getpid(), signal.SIGTERM)
        sender = threading.Thread(target=phone)
        sender.start()
        assert cli.main(["--ear", "gradium", "--launch", "test", "--phone", "-o", str(out)]) is None
        sender.join(timeout=3)
        assert not failures and not sender.is_alive()
        state = json.loads((tmp_path / "meeting.json").read_text())
        assert state["phase"] == "ended" and state["error"] is None
        assert state["ear"]["seconds"] >= 2.4
        assert state["ear"]["credits_left"] == 898
        assert metered == ["mock-key", "mock-key"]
        assert "live phone" in out.read_text() and out.read_text().splitlines()[-1].startswith("# ended ")
        assert out.with_suffix(".phone.wav").exists()
        assert not (tmp_path / "phone.sock").exists()


def test_cli_preserves_committed_text_on_later_persistent_failure(tmp_path, monkeypatch):
    import tempfile
    home = Path(tempfile.mkdtemp(prefix="hkf"))
    failures = []
    with MockGradium(plans=[[("kept", 0, 2)]], error=("down", 1011), error_from=2) as mock:
        configure(monkeypatch, home, mock)
        monkeypatch.setattr(cli.sys, "platform", "linux")
        def phone():
            try:
                wait(lambda: (home / "meeting.json").exists() and
                     json.loads((home / "meeting.json").read_text())["phase"] == "live")
                client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                client.connect(str(home / "phone.sock"))
                samples = np.concatenate([speech(2), np.zeros(16000, np.float32), speech(2)])
                client.sendall(to_pcm16(samples).tobytes())
                client.close()
            except BaseException as error:
                failures.append(error)
        sender = threading.Thread(target=phone)
        sender.start()
        assert cli.main(["--ear", "gradium", "--launch", "test", "--phone", "-o", str(home / "meeting.txt")]) == 1
        sender.join(3)
        assert not failures and not sender.is_alive()
        text = (home / "meeting.txt").read_text()
        assert text.count("kept") == 1
        lines = text.splitlines()
        assert lines[-2].startswith("# gradium ") and lines[-1].startswith("# ended ")
        state = json.loads((home / "meeting.json").read_text())
        assert state["phase"] == "failed" and "persistent failure" in state["error"]


def test_linux_file_cli_and_enrollment_use_portable_audio_without_mlx(tmp_path, monkeypatch):
    with MockGradium(plans=[[("file", 0, 2)]]) as mock:
        configure(monkeypatch, tmp_path, mock)
        prohibit_mlx(monkeypatch)
        monkeypatch.setattr(cli.sys, "platform", "linux")
        audio = tmp_path / "audio.wav"
        recorder = WavRecorder(audio)
        recorder.write(speech(6))
        recorder.close()
        out = tmp_path / "file.txt"
        assert cli.main(["--ear", "gradium", "--file", str(audio), "-o", str(out)]) is None
        assert "00:00:00-00:00:02 S1" in out.read_text()
        assert not (tmp_path / "meeting.json").exists()
        monkeypatch.setattr("hark.voice.Embedder", lambda: lambda _: np.array([1., 0.]))
        cli.main(["enroll", "me", "--file", str(audio)])
        assert np.array_equal(np.load(tmp_path / "voices/me.npy"), [1, 0])
        assert (tmp_path / "voices/me.wav").exists()


def test_default_ear_follows_importability_and_explicit_choice_is_lazy(monkeypatch):
    original = builtins.__import__
    def unavailable(name, *args, **kwargs):
        if name.startswith("mlx"):
            raise ImportError("MLX unavailable")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", unavailable)
    assert cli._default_ear() == "gradium"
    monkeypatch.setattr(builtins, "__import__", lambda name, *a, **kw: object() if name == "mlx.core" else original(name, *a, **kw))
    assert cli._default_ear() == "local"


@pytest.mark.parametrize("arguments", [[], ["--room"], ["--ear", "local", "--room"]])
def test_linux_device_modes_fail_fast(arguments, monkeypatch):
    monkeypatch.setattr(cli.sys, "platform", "linux")
    monkeypatch.setattr(cli, "MicSource", lambda *a: pytest.fail("device opened"))
    monkeypatch.setattr(cli, "SystemSource", lambda *a: pytest.fail("device opened"))
    with pytest.raises(SystemExit) as error:
        cli.main(["--ear", "gradium"] + arguments)
    assert error.value.code == 2


def test_metering_uses_exact_documented_endpoint_schema_and_header(monkeypatch):
    calls = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def read(self):
            return b'{"remaining_credits":123,"allocated_credits":45000,"billing_period":"monthly","next_rollover_date":null,"plan_name":"free"}'
    def get(request, timeout):
        calls.append((request.full_url, request.get_header("X-api-key"), timeout))
        return Response()
    monkeypatch.setattr("hark.gradium.urlopen", get)
    assert credits_left("secret") == 123
    assert calls == [("https://api.gradium.ai/api/usages/credits", "secret", 3)]


def test_unavailable_explicit_local_fails_before_lifecycle(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_default_ear", lambda: "gradium")
    monkeypatch.setattr(cli, "HOME", tmp_path)
    monkeypatch.setattr(cli, "load_models", lambda *a: pytest.fail("models loaded"))
    with pytest.raises(SystemExit) as error:
        cli.main(["--ear", "local", "--phone"])
    assert error.value.code == 2
    assert "MLX on Apple Silicon; use --ear gradium" in capsys.readouterr().err
    assert not (tmp_path / "meeting.json").exists()


def test_real_cli_subprocess_sigterm_after_blas_threads_writes_ended(tmp_path):
    import subprocess
    import sys
    import tempfile

    home = Path(tempfile.mkdtemp(prefix="hks"))
    command = [sys.executable, "scripts/gradium-smoke.py", "--synthetic-cluster", "--home", str(home)]
    env = dict(os.environ, HARK_DIR=str(home), OPENBLAS_NUM_THREADS="4")
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        wait(lambda: (home / "meeting.json").exists() and
             json.loads((home / "meeting.json").read_text())["phase"] == "live", timeout=10)
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(str(home / "phone.sock"))
        client.sendall(to_pcm16(speech(2.4)).tobytes())
        wait(lambda: "Hello" in (home / "meeting.txt").read_text())
        process.send_signal(signal.SIGTERM)
        output, error = process.communicate(timeout=10)
        client.close()
        assert process.returncode == 0, error
        state = json.loads((home / "meeting.json").read_text())
        assert state["phase"] == "ended" and state["ear"]["seconds"] >= 2.4
        assert (home / "meeting.txt").read_text().splitlines()[-1].startswith("# ended ")
        assert not (home / "phone.sock").exists()
        if sys.platform == "linux":
            assert json.loads(output.splitlines()[0])["threads"] >= 4
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_metering_outage_is_optional(monkeypatch):
    def down(*a, **kw): raise OSError("down")
    monkeypatch.setattr("hark.gradium.urlopen", down)
    assert credits_left("mock") is None
