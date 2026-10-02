"""Quiet request retirement follows source time without opening empty sockets."""

import time

import numpy as np
import pytest

from gradium_mock import MockGradium
from hark.gradium import GradiumTrack
from hark.transcript import flush_tracks
from test_gradium import finish, speech, track, wait


def test_quiet_timeout_default_is_sixty_source_seconds_and_can_be_disabled():
    t = GradiumTrack('phone', key='mock-key')
    disabled = GradiumTrack('phone', key='mock-key', quiet_timeout=None)
    try:
        assert t.quiet_timeout == 60 * 16000
        assert disabled.quiet_timeout is None
    finally:
        t.close()
        disabled.close()


def test_source_quiet_retires_request_without_periodic_empty_connections():
    with MockGradium(plans=[[('first', 0, 2)], [('second', .32, 2.32)]]) as mock:
        t = track(mock)
        t.start()
        try:
            t.feed(speech(2))
            first_request = t.request
            t.feed(np.zeros(40 * 16000, np.float32))
            assert t.request is first_request and not first_request.done
            t.feed(np.zeros(20 * 16000, np.float32))
            assert first_request.done and t.request is None
            wait(lambda: t.backlog_samples == 0)
            first = flush_tracks([t])
            t.feed(np.zeros(480 * 16000, np.float32))
            time.sleep(.1)
            assert len(mock.connections) == 2
            assert t.sent_seconds == pytest.approx(2.8)
            t.feed(speech(2))
            second = finish(t)
            assert [u.text for u in first + second] == ['first', 'second']
            assert second[0].speech == [(542, 544)]
            assert t.take_notices() == []
        finally:
            t.close()


def test_proactive_eos_precedes_real_provider_idle_timeout():
    with MockGradium(plans=[[('first', 0, 1)]], idle_timeout=.25) as mock:
        t = track(mock, quiet_timeout=1)
        t.start()
        try:
            t.feed(speech(1.04))
            t.feed(np.zeros(2 * 16000, np.float32))
            wait(lambda: t.backlog_samples == 0)
            time.sleep(.35)
            assert not mock.connections[1].get('idle_closed')
            assert len(mock.connections) == 2
            assert any(m['type'] == 'end_of_stream' for m in mock.connections[1]['messages'])
            assert [u.text for u in finish(t)] == ['first']
            assert t.take_notices() == []
        finally:
            t.close()


def test_speech_during_eos_and_connect_keeps_mapping_and_final_partial_frame():
    plans = [[('first', 0, 1)], [('queued', .24, 1.24625)]]
    with MockGradium(plans=plans, finish_delay=.12, stall_setup={2: .08},
                     dangling_last=True) as mock:
        t = track(mock, quiet_timeout=1, realtime=False, backlog_seconds=4)
        t.start()
        try:
            t.feed(speech(1.04))
            t.feed(np.zeros(round(1.04 * 16000), np.float32))
            wait(lambda: len(mock.connections) > 1 and any(
                m['type'] == 'end_of_stream' for m in mock.connections[1]['messages']))
            live = flush_tracks([t])
            assert [u.text for u in live] == ['first']
            t.feed(speech(1.00625))
            assert t.backlog_samples <= t.backlog_limit
            rest = finish(t)
            assert [u.text for u in rest] == ['queued']
            assert rest[0].speech == [(2.08, 3.08625)]
            assert rest[0].end == t.processed
            assert len(mock.connections) == 3
            assert t.backlog_samples == 0 and t.missed is None
        finally:
            t.close()


def test_tiny_quiet_timeout_does_not_create_silent_hangover_requests():
    with MockGradium(plans=[[('first', 0, .16)]]) as mock:
        t = track(mock, quiet_timeout=.16)
        t.start()
        try:
            t.feed(speech(.16))
            t.feed(np.zeros(60 * 16000, np.float32))
            wait(lambda: t.backlog_samples == 0)
            assert t.request is None and not t.active
            assert len(mock.connections) == 2
            assert t.sent_seconds == pytest.approx(.32)
            assert [u.text for u in finish(t)] == ['first']
        finally:
            t.close()
