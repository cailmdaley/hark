import numpy as np
import pytest

from hark import cli
from hark.gradium import GradiumTrack


def test_abort_keeps_pending_phrase_audio_until_close():
    track = GradiumTrack('phone', key='mock', fixed_speaker='me')
    track.audio.append(0, np.ones(16000, np.float32))
    track.phrases[(1, 0)] = [('pending words', 0, 1, [(0, 1)])]
    track.abort()
    assert track.flush(force=True)[0].text == 'pending words'
    assert not track.audio.file.closed
    track.close()
    assert track.audio.file.closed


def test_primary_capture_error_survives_cleanup_and_other_track_flushes(monkeypatch, tmp_path):
    cli.HOME.mkdir(parents=True, exist_ok=True)
    primary = RuntimeError('primary capture failure')
    flushed = []
    class Source:
        name = 'mic'
        anchor = 0
        def __init__(self, *args): pass
        def start(self): pass
        def stop(self): pass
        def drain(self, limit): raise primary
    class System(Source): name = 'system'
    class Track:
        processed = 0
        def __init__(self, name, *args, **kwargs): self.name = name
        def flush(self, force=False):
            flushed.append(self.name)
            if self.name == 'mic': raise RuntimeError('cleanup failure')
            return []
    monkeypatch.setattr(cli, '_default_ear', lambda: 'local')
    monkeypatch.setattr(cli, '_device_capture_available', lambda: True)
    monkeypatch.setattr(cli, 'load_models', lambda _: (None, None))
    monkeypatch.setattr(cli, 'MicSource', Source)
    monkeypatch.setattr(cli, 'SystemSource', System)
    monkeypatch.setattr(cli, 'Track', Track)
    with pytest.raises(RuntimeError) as error:
        cli.main(['--ear', 'local', '--pause-for', 'none', '--no-save-audio', '-o', str(tmp_path / 'out.txt')])
    assert error.value is primary
    assert flushed == ['mic', 'system']
