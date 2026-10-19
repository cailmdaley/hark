"""An error on one track cannot discard the other's admitted PCM or AEC tail."""
import numpy as np
import pytest

from hark import cli
from test_cli import fake_live_capture, feed_chunks, read_wav, tone


@pytest.mark.parametrize("failure", ["system-drain", "mic-model"])
def test_independent_preservation_on_error(tmp_path, monkeypatch, failure):
    _, Source = fake_live_capture(monkeypatch, tmp_path)
    Mic, _ = feed_chunks(Source, "mic", [tone(440, 40000), tone(440, 8000)])
    System, _ = feed_chunks(Source, "system", [tone(220, 40000), tone(220, 8000)])
    heard = {"mic": [], "system": []}

    class FailedSystem(System):
        def drain(self, limit):
            if len(self.left) < 2:
                raise RuntimeError("system drain failed")
            return super().drain(limit)

    class Track(cli.Track):
        def feed(self, samples, **kwargs):
            if self.name == "mic" and failure == "mic-model" and samples.size:
                raise RuntimeError("mic model failed")
            heard[self.name].append(samples)

    monkeypatch.setattr(cli, "MicSource", Mic)
    monkeypatch.setattr(cli, "SystemSource", FailedSystem if failure == "system-drain" else System)
    monkeypatch.setattr(cli, "Track", Track)
    monkeypatch.setattr(cli, "_system_capture_available", lambda: True)
    out = tmp_path / "session.txt"
    # The controlled capture/model error ends main; no live loop is left waiting for a signal.
    with pytest.raises(RuntimeError, match="system drain failed|mic model failed"):
        cli.main(["--ear", "local", "--launch", "test", "--keep-audio", "-o", str(out)])
    assert len(read_wav(out.with_suffix(".mic.wav"))) == 48000
    expected_system = 40000 if failure == "system-drain" else 48000
    assert len(read_wav(out.with_suffix(".system.wav"))) == expected_system
    assert len(np.concatenate(heard["system"])) == expected_system
    assert out.read_text().splitlines()[-1].startswith("# ended")
