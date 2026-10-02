import numpy as np

from gradium_mock import MockGradium
from test_gradium import track, speech, finish, wait
from hark.transcript import flush_tracks


def test_late_end_after_flush_and_new_word_does_not_finalize_or_drop_new_word():
    with MockGradium(plans=[[('before', 1.6, 2), ('next', 3.12, 5.12)]], late_end_after_flush=True) as mock:
        t = track(mock, phrase_seconds=4)
        t.start()
        try:
            t.feed(speech(2))
            t.feed(np.zeros(4 * 16000, np.float32))
            wait(lambda: t.results.qsize() >= 2)
            first = flush_tracks([t])
            assert [u.text for u in first] == ['before']
            assert first[0].speech == [(1.6, 2.8)]
            request = t.request
            t.feed(np.zeros(600 * 16000, np.float32))
            assert t.request is request and not request.done
            t.feed(speech(2))
            t.feed(np.zeros(16000, np.float32))
            rest = finish(t)
            assert [u.text for u in rest] == ['next']
            assert rest[0].speech == [(606, 608)]
            assert len(mock.connections) == 2
            assert mock.connections[1]['samples'] == 94720
        finally:
            t.close()
