"""The mutation runner's physical write guard is tested only on a fake home."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def driver():
    path = Path(__file__).resolve().parents[1] / "scripts" / "gradium-mutations.py"
    spec = importlib.util.spec_from_file_location("gradium_mutations", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sandbox_profile_quotes_explicit_protected_root(driver, monkeypatch, tmp_path):
    protected = tmp_path / 'fake "quoted" user' / ".hark"
    monkeypatch.setattr(driver.sys, "platform", "darwin")
    monkeypatch.setattr(driver.os, "access", lambda path, mode: True)
    command = [sys.executable, "-c", "pass"]
    wrapped = driver.sandbox_command(command, protected)
    expected = json.dumps(str(protected.resolve()), ensure_ascii=False)
    assert wrapped == [driver.SANDBOX_EXEC, "-p",
                       f'(version 1)(allow default)(deny file-write* (subpath {expected}))', *command]


@pytest.mark.parametrize("platform,available", [("linux", True), ("darwin", False)])
def test_sandbox_fallback_keeps_command(driver, monkeypatch, tmp_path, platform, available):
    monkeypatch.setattr(driver.sys, "platform", platform)
    monkeypatch.setattr(driver.os, "access", lambda path, mode: available)
    command = [sys.executable, "-m", "pytest"]
    assert driver.sandbox_command(command, tmp_path / ".hark") == command


def test_runner_protects_captured_parent_default_after_home_changes(driver, monkeypatch, tmp_path):
    protected = driver.DEFAULT_HARK_HOME
    fake_user = tmp_path / "fake-user"
    fake_user.mkdir()
    monkeypatch.setenv("HOME", str(fake_user))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_user))
    monkeypatch.setattr(driver.sys, "platform", "darwin")
    monkeypatch.setattr(driver.os, "access", lambda path, mode: True)
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(driver.subprocess, "run", run)
    assert driver.run_tests(tmp_path, ["tests"])["returncode"] == 0
    command, options = calls[0]
    assert command[:2] == [driver.SANDBOX_EXEC, "-p"]
    assert json.dumps(str(protected), ensure_ascii=False) in command[2]
    assert command[3:] == [sys.executable, "-m", "pytest", "-q", "tests"]
    assert Path(options["env"]["HARK_DIR"]) != protected
    assert options["env"]["GRADIUM_API_KEY"] == "mutation-local-only"


@pytest.mark.skipif(sys.platform != "darwin" or not os.access("/usr/bin/sandbox-exec", os.X_OK),
                    reason="Darwin sandbox-exec is unavailable")
def test_physical_sandbox_blocks_only_fake_home_writes(driver, tmp_path):
    fake_user = tmp_path / "fake-user"
    protected = fake_user / ".hark"
    protected.mkdir(parents=True)
    record = protected / "meeting.json"
    record.write_bytes(b"fake meeting sentinel\n")
    isolated = tmp_path / "child-hark"
    isolated.mkdir()
    script = '''import os
from pathlib import Path

protected = Path.home() / ".hark"
for target in [protected / "meeting.json", protected / "new-file"]:
    try:
        target.write_text("forbidden write")
    except PermissionError:
        pass
    else:
        raise AssertionError("sandbox allowed a write to the fake protected home")
(Path(os.environ["HARK_DIR"]) / "allowed.txt").write_text("allowed write")
'''
    command = driver.sandbox_command([sys.executable, "-c", script], protected)
    env = dict(os.environ, HOME=str(fake_user), HARK_DIR=str(isolated),
               GRADIUM_API_KEY="sandbox-local-only")
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=10)
    if (command[0] == driver.SANDBOX_EXEC and result.returncode == 71
            and "sandbox_apply: Operation not permitted" in result.stderr):
        pytest.skip("macOS disallows applying a nested sandbox")
    assert result.returncode == 0, result.stderr
    assert record.read_bytes() == b"fake meeting sentinel\n"
    assert not (protected / "new-file").exists()
    assert (isolated / "allowed.txt").read_text() == "allowed write"
