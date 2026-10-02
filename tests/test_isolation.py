"""Isolation guards fail safely, including when their negative controls are mutated."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import wave

import pytest

from hark import cli


ROOT = Path(__file__).resolve().parents[1]


def test_suite_home_and_environment_are_isolated():
    default = (Path.home() / ".hark").resolve()
    assert cli.HOME.resolve() != default
    assert Path(os.environ["HARK_DIR"]).resolve() != default


def test_cli_guard_refuses_overridden_default_home(monkeypatch):
    # --help is side-effect-free even if the guard is removed by a mutation.
    monkeypatch.setattr(cli, "HOME", Path.home() / ".hark")
    with pytest.raises(AssertionError, match="tests must not use the default HARK home"):
        cli.main(["--help"])


def test_cli_guard_protects_original_default_after_home_mock(monkeypatch, tmp_path):
    original_default = Path.home() / ".hark"
    fake_home = tmp_path / "fake-user"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.setattr(cli, "HOME", original_default)
    with pytest.raises(AssertionError, match="tests must not use the default HARK home"):
        cli.main(["--help"])


def test_cli_guard_refuses_default_environment_in_fake_home(tmp_path):
    fake_home = tmp_path / "fake-user"
    fake_home.mkdir()
    isolated = tmp_path / "child-hark"
    isolated.mkdir()
    suite = tmp_path / "suite"
    suite.mkdir()
    shutil.copy(ROOT / "tests" / "conftest.py", suite / "conftest.py")
    (suite / "test_guard.py").write_text('''from pathlib import Path
import pytest
from hark import cli

def test_environment_guard(monkeypatch):
    monkeypatch.setenv("HARK_DIR", str(Path.home() / ".hark"))
    with pytest.raises(AssertionError, match="tests must not use the default HARK_DIR"):
        cli.main(["--help"])
''')
    env = dict(os.environ, HOME=str(fake_home), HARK_DIR=str(isolated),
               PYTHONPATH=str(ROOT), GRADIUM_API_KEY="subprocess-local-only")
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", str(suite)],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (fake_home / ".hark").exists()


def test_subprocess_cli_import_uses_isolated_environment(tmp_path):
    home = tmp_path / "child-hark"
    home.mkdir()
    env = dict(os.environ, HARK_DIR=str(home), GRADIUM_API_KEY="subprocess-local-only")
    result = subprocess.run([sys.executable, "-c", "from hark import cli; print(cli.HOME)"],
                            cwd=ROOT, env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()) == home.resolve()


def smoke(tmp_path, *, default_home):
    fake_user = tmp_path / "fake-user"
    fake_user.mkdir()
    isolated = tmp_path / "child-hark"
    isolated.mkdir()
    audio = tmp_path / "quiet.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\0" * 3200)
    env = dict(os.environ, HOME=str(fake_user), HARK_DIR=str(isolated),
               GRADIUM_API_KEY="subprocess-local-only")
    command = [sys.executable, str(ROOT / "scripts" / "gradium-smoke.py"),
               "--file", str(audio), "--synthetic-cluster"]
    if default_home:
        command += ["--home", str(fake_user / ".hark")]
    # A broken guard can only run a quiet local-mock file in the fake user's home.
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=20)
    return result, fake_user


def test_smoke_refuses_default_home_in_sandbox(tmp_path):
    result, fake_user = smoke(tmp_path, default_home=True)
    assert result.returncode == 2, result.stderr
    assert "smoke checks must not use the default HARK home" in result.stderr
    assert not (fake_user / ".hark").exists()


def test_smoke_default_creates_isolated_home(tmp_path):
    result, fake_user = smoke(tmp_path, default_home=False)
    assert result.returncode == 0, result.stderr
    home = Path(json.loads(result.stdout.splitlines()[0])["home"])
    assert home != fake_user / ".hark"
    assert "# ended " in (home / "meeting.txt").read_text()
    assert not (home / "meeting.json").exists()


def test_device_capability_matches_actual_platform():
    assert cli._device_capture_available() == (sys.platform == "darwin")
