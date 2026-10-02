"""Small checks for the test suite's home and hosted-network guards."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest

from hark import cli, gradium


def test_home_guard_refuses_default(monkeypatch):
    monkeypatch.setattr(cli, 'HOME', Path.home() / '.hark')
    with pytest.raises(AssertionError, match='default HARK home'):
        cli.main(['--help'])


def test_network_guard_refuses_hosted_connection_and_metering():
    with pytest.raises(AssertionError, match='hosted Gradium connection'):
        gradium.connect(gradium.URL)
    with pytest.raises(AssertionError, match='hosted Gradium metering'):
        gradium.urlopen(None)


def test_smoke_refuses_default_home_without_reading_real_keys():
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix='hk-', dir='/tmp') as home:
        result = subprocess.run(
            [sys.executable, str(root / 'scripts/gradium-smoke.py'), '--home', str(Path(home) / '.hark')],
            env=dict(os.environ, HOME=home, HARK_DIR=home, GRADIUM_API_KEY='local-only'),
            capture_output=True, text=True, timeout=10)
        assert result.returncode == 2
        assert 'must not use the default HARK home' in result.stderr
        assert not (Path(home) / '.hark').exists()
