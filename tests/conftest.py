"""Tests never read a real Gradium key or contact its hosted endpoints."""

from urllib.parse import urlsplit

import pytest


@pytest.fixture(autouse=True)
def local_gradium_only(monkeypatch):
    import hark.gradium as gradium

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
