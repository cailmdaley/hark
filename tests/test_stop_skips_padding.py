import threading

import numpy as np

from hark.transcript import Track


class _Counting:
    def __init__(self):
        self.fed = 0

    def feed(self, samples, final=False):
        self.fed += samples.size
        return []


def _track(stop):
    track = Track.__new__(Track)
    track.stop = stop
    track.session = _Counting()
    track.audio = None
    track.audio_samples = 0
    track.pending = {}
    return track


def test_silent_backlog_is_fed_until_stop_then_skipped():
    stop = threading.Event()
    track = _track(stop)
    silence = np.zeros(16000 * 60, np.float32)
    track.feed(silence)
    assert track.session.fed == silence.size
    stop.set()
    track.feed(silence)
    assert track.session.fed == silence.size


def test_real_audio_is_still_fed_after_stop():
    stop = threading.Event()
    stop.set()
    track = _track(stop)
    track.feed(np.full(16000, 0.1, np.float32))
    assert track.session.fed == 16000
