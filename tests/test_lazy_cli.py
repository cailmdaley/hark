"""The local CLI doesn't require hosted-backend or optional audio decoders."""

import os
import subprocess
import sys
import tempfile


def test_local_help_and_capture_without_cloud_dependencies():
    code = r'''
import importlib.abc
import sys
import wave
from pathlib import Path
class Reject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'websockets', 'soundfile', 'hark.gradium'} or fullname == 'hark.gradium':
            raise AssertionError('optional import: ' + fullname)
sys.meta_path.insert(0, Reject())
from hark import cli
try:
    cli.main(['--help'])
except SystemExit as error:
    assert error.code == 0
cli._default_ear = lambda: 'local'
cli.load_models = lambda _: (None, None)
class Track:
    processed = 0
    def __init__(self, name, *args, **kwargs): self.name = name
    def feed(self, samples, final=False): pass
    def flush(self, force=False): return []
cli.Track = Track
home = cli.HOME
home.mkdir(parents=True, exist_ok=True)
audio = home / 'input.wav'
with wave.open(str(audio), 'wb') as wav:
    wav.setparams((1, 2, 16000, 0, 'NONE', 'none'))
    wav.writeframes(b'\0\0' * 1600)
assert cli.main(['--ear', 'local', '--file', str(audio), '-o', str(home / 'out.txt')]) is None
assert 'hark.gradium' not in sys.modules
'''
    with tempfile.TemporaryDirectory(prefix='hk-', dir='/tmp') as home:
        result = subprocess.run([sys.executable, '-c', code],
                                env=dict(os.environ, HARK_DIR=home), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
