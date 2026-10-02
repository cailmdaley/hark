"""Recognition outages are anchored to waiting source audio, not quiet sockets."""

from datetime import datetime
import time

import numpy as np

from gradium_mock import MockGradium
from test_gradium import finish, speech, track, wait


ANCHOR = datetime(2026, 10, 2, 12)


def test_failed_startup_is_quiet_until_speech_waits_and_progress_alone_recovers():
    with MockGradium(error=('no workers', 1011), plans=[[('late text', 0, 1000)]]) as mock:
        t = track(mock, preroll=0, backoff=.03)
        t.t0 = ANCHOR
        t.start()
        try:
            wait(lambda: len(mock.connections) >= 2)
            assert not t.degraded
            assert t.take_notices() == []
            t.feed(np.zeros(20 * 16000, np.float32))
            assert t.take_notices() == []
            t.feed(speech(.16))
            wait(lambda: t.degraded)
            assert t.take_notices() == ['gradium lost at 12:00:20']
            mock.error = None
            wait(lambda: not t.degraded)
            assert t.take_notices() == ['gradium back at 12:00:20']
            assert t.flush(force=True) == []  # Recovery doesn't depend on finalized text.
            t.feed(speech(.16))
            time.sleep(.08)
            assert t.take_notices() == []
            assert finish(t) == []
        finally:
            t.close()


def test_acknowledged_progress_drop_is_not_loss_but_new_audio_during_reconnect_is():
    with MockGradium(plans=[[('late text', 0, 1000)]], drop_first_at=1.04) as mock:
        t = track(mock, backoff=.3, preroll=0)
        t.t0 = ANCHOR
        t.start()
        try:
            t.feed(speech(1.04))
            wait(lambda: len(mock.connections) > 1 and mock.connections[1]['samples'] == 16640)
            time.sleep(.05)
            assert t.take_notices() == []
            assert not t.degraded
            t.feed(speech(.16))  # Queued while the retry is sleeping.
            wait(lambda: t.degraded)
            assert t.take_notices() == ['gradium lost at 12:00:01']
            wait(lambda: not t.degraded)
            assert t.take_notices() == ['gradium back at 12:00:01']
            assert t.flush(force=True) == []
            assert finish(t) == []
            assert t.take_notices() == []
        finally:
            t.close()


def test_unacknowledged_suffix_loss_uses_source_anchor_and_back_is_once_on_step():
    with MockGradium(plans=[[('late text', 0, 1000)]], drop_first_at=.4,
                     steps=False) as mock:
        t = track(mock, backoff=.2, preroll=0)
        t.t0 = ANCHOR
        t.start()
        try:
            t.feed(np.zeros(round(10.4 * 16000), np.float32))
            t.feed(speech(1.2))
            wait(lambda: t.degraded)
            assert t.take_notices() == ['gradium lost at 12:00:10']
            mock.steps = True
            wait(lambda: not t.degraded)
            assert t.take_notices() == ['gradium back at 12:00:10']
            assert t.flush(force=True) == []
            t.feed(speech(.16))
            time.sleep(.08)
            assert t.take_notices() == []
            finish(t)
        finally:
            t.close()


def test_quiet_failed_startup_terminates_promptly_even_in_long_backoff():
    with MockGradium(error=('no workers', 1011)) as mock:
        t = track(mock, backoff=30)
        t.start()
        try:
            assert not t.degraded
            began = time.monotonic()
            assert finish(t) == []
            assert time.monotonic() - began < .5
            assert not t.thread.is_alive()
            assert t.take_notices() == []
        finally:
            t.close()


def test_failed_speech_flush_with_all_audio_acknowledged_does_not_claim_loss():
    with MockGradium(plans=[[('word', 0, 1)]], stalled_flush=True) as mock:
        t = track(mock, timeout=.08, backoff=.2)
        t.start()
        try:
            t.feed(speech(1.04))
            t.feed(np.zeros(16000, np.float32))
            wait(lambda: len(mock.connections) > 1 and any(
                m['type'] == 'flush' for m in mock.connections[1]['messages']))
            time.sleep(.15)
            assert not t.degraded
            assert t.take_notices() == []
            finish(t)
        finally:
            t.close()
