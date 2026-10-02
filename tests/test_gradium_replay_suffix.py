"""Dropped connections retain only the uncommitted suffix on the source clock."""

import base64
import time

import numpy as np
import pytest

from gradium_mock import MockGradium
from hark.capture import to_pcm16
from hark.transcript import flush_tracks
from test_gradium import finish, speech, track, wait


def audio(connection):
    return np.concatenate([np.frombuffer(base64.b64decode(m['audio']), dtype='<i2')
                           for m in connection['messages'] if m['type'] == 'audio'])


def test_idle_drop_after_acknowledged_flush_does_not_replay_or_report_outage():
    with MockGradium(plans=[[('first', 0, 2)], [('second', .32, 2.32)]],
                     idle_timeout=.15) as mock:
        t = track(mock)
        t.start()
        try:
            t.feed(speech(2))
            t.feed(np.zeros(16000, np.float32))
            wait(lambda: t.results.qsize() >= 2)
            first = flush_tracks([t])
            wait(lambda: mock.connections[1].get('idle_closed'))
            time.sleep(.25)
            assert len(mock.connections) == 2  # No empty retry socket or old audio.
            assert t.sent_seconds == pytest.approx(2.8)
            assert t.take_notices() == []
            t.feed(np.zeros(10 * 16000, np.float32))
            t.feed(speech(2))
            rest = finish(t)
            assert [u.text for u in first + rest] == ['first', 'second']
            assert rest[0].speech == [(12.96, 14.96)]
            assert len(mock.connections) == 3
            assert t.backlog_samples == 0
        finally:
            t.close()


def test_retry_trims_inside_frame_and_rebases_provider_clock_once():
    end = 16050 / 16000
    plans = [[('committed', 0, end)], [('suffix', 0, .24)]]
    with MockGradium(plans=plans, drop_first_at=1.12, steps=False) as mock:
        t = track(mock, backoff=.2, realtime=False)
        samples = np.linspace(.002, .02, 19200, dtype=np.float32)
        t.start()
        try:
            t.feed(np.zeros(round(3.04 * 16000), np.float32))
            # Disable preroll to give the dropped request an exact source anchor.
            t.pre.clear()
            t.feed(samples)
            output = finish(t)
            assert [u.text for u in output] == ['committed', 'suffix']
            assert output[0].speech == [(3.04, 3.04 + end)]
            assert output[1].speech == [(3.04 + end, 4.24)]
            pcm = audio(mock.connections[2])
            expected = to_pcm16(samples[16050:16640])
            assert np.array_equal(pcm[:expected.size], expected)
            assert not np.any(pcm[expected.size:1280])
            assert np.array_equal(pcm[1280:], to_pcm16(samples[16640:]))
            assert len(pcm) == 3 * 1280
            assert t.backlog_samples == 0
        finally:
            t.close()


def test_stream_with_earlier_commit_keeps_its_pending_suffix():
    plans = [[('long', 0, 1.04, 0), ('short', 0, .56, 1)],
             [('short tail', 0, .48, 1), ('long tail', .48, .64, 0)]]
    with MockGradium(plans=plans, drop_first_at=1.12, steps=False) as mock:
        t = track(mock, realtime=False)
        t.start()
        try:
            t.feed(speech(1.2))
            output = finish(t)
            assert [u.text for u in output] == ['short', 'long', 'short tail', 'long tail']
            assert output[2].speech == [(.56, 1.04)]
            assert output[3].speech == [(1.04, 1.2)]
            assert mock.connections[2]['samples'] == 8 * 1280
        finally:
            t.close()


def test_replay_keeps_speech_queued_during_backoff_and_discarded_source_gap():
    plans = [[('committed', 0, 1.04)], [('tail', 0, .16), ('new', 1.28, 2.08)]]
    with MockGradium(plans=plans, drop_first_at=1.12, steps=False) as mock:
        t = track(mock, backoff=.2, realtime=False)
        t.start()
        try:
            t.feed(speech(1.2))
            wait(lambda: t.degraded)
            t.feed(np.zeros(4 * 16000, np.float32))
            t.feed(speech(.8))
            t.feed(np.zeros(16000, np.float32))
            output = finish(t)
            assert [u.text for u in output] == ['committed', 'tail', 'new']
            assert [u.speech for u in output] == [[(0, 1.04)], [(1.04, 1.2)], [(5.2, 6)]]
            assert mock.connections[2]['samples'] == round(2.88 * 16000)
            assert t.backlog_samples == 0
        finally:
            t.close()


def test_partial_final_frame_in_replay_excludes_both_padding_regions():
    plans = [[('committed', 0, .5)], [('final', .1, .2)]]
    with MockGradium(plans=plans, drop_first_at=.72, dangling_last=True) as mock:
        # Only the replay's final word dangles; the original word must be finalized.
        mock.dangling_words = {1}
        mock.dangling_last = False
        plans[1].append(('tail', .2, .3))
        t = track(mock, realtime=False)
        t.start()
        try:
            t.feed(speech(1.00625))
            output = finish(t)
            assert [u.text for u in output] == ['committed', 'final tail']
            assert output[1].speech == [(.58, 1.00625)]
            assert output[1].end == t.processed
            assert mock.connections[2]['samples'] == 7 * 1280
            assert t.backlog_samples == 0
        finally:
            t.close()


def test_replayed_stream_overlap_is_suppressed_without_discarding_other_stream(capsys):
    plans = [[('long', 0, 1.04, 0), ('short', 0, .56, 1)],
             [('duplicate long', 0, .48, 0), ('short tail', 0, .48, 1)]]
    with MockGradium(plans=plans, drop_first_at=1.12, steps=False) as mock:
        t = track(mock, realtime=False)
        t.start()
        try:
            t.feed(speech(1.2))
            output = finish(t)
            assert [u.text for u in output] == ['short', 'long', 'short tail']
            assert 'replay skips segment' in capsys.readouterr().err
            assert output[-1].speech == [(.56, 1.04)]
        finally:
            t.close()


def test_progress_without_final_text_does_not_discard_recognition_on_retry():
    with MockGradium(plans=[[('word', 0, 2)]], drop_first_at=1.04) as mock:
        t = track(mock, realtime=False)
        t.start()
        try:
            t.feed(speech(2))
            output = finish(t)
            assert [u.text for u in output] == ['word']
            assert mock.connections[2]['samples'] == 2 * 16000
        finally:
            t.close()
