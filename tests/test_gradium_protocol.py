import pytest

from gradium_mock import MockGradium
from test_gradium import track, speech, finish


@pytest.mark.parametrize('anomalies,expected', [
    ([{'type': 'text', 'text': 'clamped', 'start_s': -1}, {'type': 'end_text', 'stop_s': 99}], ['clamped next']),
    ([{'type': 'end_text', 'stop_s': .08}], ['next']),
    ([{'type': 'text', 'text': 'pending', 'start_s': 0}], ['pending next']),
    ([{'type': 'end_of_stream'}], ['next']),
    (['not json'], ['next']),
    ([{'type': 'error', 'message': 'concurrency limit', 'code': 1008}], ['next']),
])
def test_protocol_anomaly_recovers_next_valid_segment(anomalies, expected):
    with MockGradium(plans=[[('next', .08, 1)]], anomalies=anomalies) as mock:
        t = track(mock, phrase_seconds=4)
        t.start()
        try:
            t.feed(speech(1.04))
            output = finish(t)
            assert [u.text for u in output] == expected
            assert all(0 <= u.start <= u.end <= t.processed for u in output)
            assert t.error is None
        finally:
            t.close()


def test_text_flood_rotates_request_with_bounded_results():
    anomalies = [{'type': 'text', 'text': 'x' * 10001, 'start_s': 0}]
    with MockGradium(plans=[[('next', 0, 1)]], anomalies=anomalies) as mock:
        t = track(mock)
        t.start()
        try:
            t.feed(speech(1.04))
            finish(t)
            assert t.results.qsize() <= 64
            assert t.error is None
        finally:
            t.close()
