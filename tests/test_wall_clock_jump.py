import time

import numpy as np

from hark.capture import JUMP_PAD, SAMPLE_RATE, Source


class _Idle(Source):
    def _open(self):
        pass


def _source(slept):
    src = _Idle("mic")
    src.start()
    src.anchor -= slept  # the process was suspended for `slept` seconds
    return src


def test_a_suspend_gap_is_skipped_not_padded():
    src = _source(4700)
    before = src.anchor
    out = src.drain(limit=float("inf"))
    assert out.size < (JUMP_PAD + 2) * SAMPLE_RATE
    assert abs(src.take_jump() - (4700 - JUMP_PAD)) < 1
    assert src.anchor > before
    assert src.take_jump() == 0.0


def test_ordinary_lateness_is_still_padded():
    src = _source(5)
    out = src.drain(limit=float("inf"))
    assert out.size > 3 * SAMPLE_RATE
    assert src.take_jump() == 0.0


def test_drain_is_quiet_after_the_skip():
    src = _source(4700)
    src.drain(limit=float("inf"))
    time.sleep(0.05)
    assert src.drain(limit=float("inf")).size < SAMPLE_RATE
