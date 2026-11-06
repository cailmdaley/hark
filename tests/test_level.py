import json
import time

import numpy as np
import pytest

from hark.capture import BLOCK, LevelMeter, Source, dbfs


@pytest.mark.parametrize("peak, level", [(1.0, 0.0), (0.1, -20.0), (10 ** (-50 / 20), -50.0), (0.0, -90.3)])
def test_dbfs(peak, level):
    assert dbfs(peak) == pytest.approx(level, abs=0.05)


def wait_for(check):
    deadline = time.time() + 2
    while not check() and time.time() < deadline:
        time.sleep(0.01)
    return check()


def test_meter_writes_each_new_block_level_and_removes_the_file_on_stop(tmp_path):
    path = tmp_path / "level.json"
    mic = Source("mic")
    mic.last_audio = time.time()
    meter = LevelMeter(mic, path).start()
    time.sleep(0.15)
    assert not path.exists()  # no block heard yet

    block = np.zeros(BLOCK, np.float32)
    block[7] = -0.1
    mic._deliver(block)
    assert wait_for(path.exists)
    assert json.loads(path.read_text()) == {"mic": -20.0, "t": mic.last_audio}

    mic._deliver(block * 0.1)
    assert wait_for(lambda: json.loads(path.read_text())["mic"] == -40.0)
    assert json.loads(path.read_text())["t"] == mic.last_audio

    meter.stop()
    assert list(tmp_path.iterdir()) == []
