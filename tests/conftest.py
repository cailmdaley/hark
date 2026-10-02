"""Tests isolate HARK state and never read real keys or use hosted Gradium."""

import os
from pathlib import Path
from urllib.parse import urlsplit

import pytest


FORBIDDEN_DEFAULT = (Path.home() / ".hark").resolve()


@pytest.fixture(autouse=True)
def local_gradium_only(monkeypatch, tmp_path):
    from hark import cli
    import hark.gradium as gradium

    home = tmp_path / "hark-home"
    monkeypatch.setenv("HARK_DIR", str(home))
    monkeypatch.setattr(cli, "HOME", home)
    real_main = cli.main

    def isolated_main(*args, **kwargs):
        if cli.HOME.resolve() == FORBIDDEN_DEFAULT:
            raise AssertionError("tests must not use the default HARK home")
        if Path(os.environ["HARK_DIR"]).expanduser().resolve() == FORBIDDEN_DEFAULT:
            raise AssertionError("tests must not use the default HARK_DIR")
        return real_main(*args, **kwargs)

    monkeypatch.setattr(cli, "main", isolated_main)
    monkeypatch.setenv("GRADIUM_API_KEY", "pytest-local-only")
    real_connect = gradium.connect
    def local_connect(url, *args, **kwargs):
        if urlsplit(url).hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise AssertionError("test attempted a hosted Gradium connection")
        return real_connect(url, *args, **kwargs)
    def no_meter(*args, **kwargs):
        raise AssertionError("test attempted hosted Gradium metering")
    monkeypatch.setattr(gradium, "connect", local_connect)
    monkeypatch.setattr(gradium, "urlopen", no_meter)
