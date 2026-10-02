"""Unended text on a dropped request must not commit the undecoded audio tail."""

from gradium_mock import MockGradium
from hark.gradium import AuthenticationError
from hark.transcript import flush_tracks
from test_gradium import finish, speech, track, wait


def reset_observed(t):
    # A false clean retirement is also an observable reset in the defective code.
    return t.request.done or t.unavailable


def test_dangling_reset_replays_undecoded_tail_and_keeps_actual_following_word():
    plans = [[('prefix', .2, .8)], [('prefix', .2, .8), ('following', 1.2, 1.9)]]
    with MockGradium(plans=plans, dangling_last=True, drop_first_at=1.12,
                     steps=False, delay=.005) as mock:
        t = track(mock, realtime=False, backoff=.15, phrase_seconds=.5)
        t.start()
        try:
            t.feed(speech(2))
            wait(lambda: t.sent_seconds >= 2 and reset_observed(t))
            frontier = t.request.replay_start
            waiting = t.degraded
            interim = flush_tracks([t])
            output = interim + finish(t)
            # The old prefix isn't a finalized replacement for the following word.
            assert [u.text for u in output] == ['prefix', 'following']
            assert interim == []
            assert frontier == 0 and waiting
            assert [u.speech for u in output] == [[(.2, .8)], [(1.2, 2)]]
            assert mock.connections[1]['samples'] == round(1.12 * 16000)
            assert mock.connections[2]['samples'] == 2 * 16000
            assert t.backlog_samples == 0
            notices = t.take_notices()
            assert sum('lost at' in n for n in notices) == 1
            assert sum('back at' in n for n in notices) == 1
        finally:
            t.close()


def test_unexpected_eos_does_not_commit_pending_text_over_the_undecoded_tail():
    plans = [[], [('prefix', 0, .8), ('following', 1.2, 1.9)]]
    anomalies = [{'type': 'text', 'text': 'prefix', 'start_s': 0}, {'type': 'end_of_stream'}]
    with MockGradium(plans=plans, anomalies=anomalies, dangling_last=True,
                     steps=False, delay=.005) as mock:
        t = track(mock, realtime=False, backoff=.15, phrase_seconds=.5)
        t.start()
        try:
            t.feed(speech(2))
            wait(lambda: t.sent_seconds >= 2 and reset_observed(t))
            interim = flush_tracks([t])
            output = interim + finish(t)
            assert [u.text for u in output] == ['prefix', 'following']
            assert interim == []
            assert mock.connections[2]['samples'] == 2 * 16000
        finally:
            t.close()


def test_pending_reset_is_not_finalized_during_backoff_but_survives_final_abort():
    with MockGradium(plans=[[('prefix', .2, .8)]], dangling_last=True,
                     drop_first_at=1.12, steps=False, delay=.005) as mock:
        t = track(mock, backoff=30)
        t.start()
        try:
            t.feed(speech(2))
            wait(lambda: t.sent_seconds >= 2 and reset_observed(t))
            assert t.flush(force=True) == []  # Replay is still possible.
            assert t.request.replay_start == 0
            assert t.degraded
            t.abort()
            words = t.flush(force=True)
            assert [u.text for u in words] == ['prefix']
            assert words[0].speech == [(.2, 2)]
            assert t.flush(force=True) == []
            assert t.audio.slice(0, 2).size == 2 * 16000
        finally:
            t.close()


def test_reset_hypotheses_are_bounded_without_committing_any_stream_audio():
    words = [(f'word{i}', i * .08, (i + 1) * .08, i) for i in range(65)]
    with MockGradium(plans=[words], dangling_words=range(65), drop_first_at=5.28,
                     steps=False, delay=.005) as mock:
        t = track(mock, backoff=30)
        t.start()
        try:
            t.feed(speech(6))
            wait(lambda: t.sent_seconds >= 6 and reset_observed(t))
            assert len(t.reset_tails) == t.results.maxsize == 64
            assert t.request.replay_start == 0
            assert all(horizon == 0 for horizon in t.request.horizons.values())
            assert t.flush(force=True) == []
            assert t.degraded
            t.abort()
            assert len(t.flush(force=True)) == 64
            assert not t.reset_tails
        finally:
            t.close()


def test_retained_pending_word_survives_terminal_authentication_on_retry():
    with MockGradium(plans=[[('prefix', .2, .8)]], dangling_last=True,
                     drop_first_at=1.12, steps=False, delay=.005,
                     error=('authentication refused', 401), error_from=2) as mock:
        t = track(mock)
        t.start()
        try:
            t.feed(speech(2))
            wait(lambda: t.finished.is_set())
            assert isinstance(t.error, AuthenticationError)
            words = t.flush(force=True)
            assert [u.text for u in words] == ['prefix']
            assert words[0].speech == [(.2, 2)]
            assert t.flush(force=True) == []
            assert len(mock.connections) == 3
            assert t.audio.slice(0, 2).size == 2 * 16000
        finally:
            t.close()
