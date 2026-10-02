"""Phrase limits use the union of mapped speech, with a separate source wall ceiling."""

import numpy as np
import pytest

from gradium_mock import MockGradium
from hark.transcript import flush_tracks
from test_gradium import finish, speech, wait
from test_gradium_grouping import default_track


@pytest.mark.parametrize("gap,live", [(4, False), (8, True)])
def test_cross_gap_phrase_counts_only_submitted_intervals_but_caps_source_wall(gap, live):
    with MockGradium(plans=[[("bridge", 1.6, 3.92)]]) as mock:
        t = default_track(mock)
        t.cluster.embedder = lambda _: pytest.fail("dropped quiet became speaker samples")
        t.start()
        try:
            t.feed(speech(2))
            t.feed(np.zeros(gap * 16000, np.float32))
            t.feed(speech(1.04))
            wait(lambda: t.results.qsize() >= 2)  # First flush marker and completed bridge.
            output = flush_tracks([t])
            assert bool(output) is live
            if not live:
                output = finish(t)
            else:
                assert finish(t) == []
            assert [u.text for u in output] == ["bridge"]
            assert output[0].speech == [(1.6, 2.8), (gap + 1.68, gap + 2.8)]
            assert sum(b - a for a, b in output[0].speech) == pytest.approx(2.32)
            assert t.cluster.counts == []
            assert len(mock.connections) == 2
        finally:
            t.close()
