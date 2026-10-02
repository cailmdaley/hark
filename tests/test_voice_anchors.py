"""Confident enrolled identities stabilize slots without relaxing anonymous clustering."""

import json

import numpy as np
import pytest

from gradium_mock import MockGradium
from hark import cli, gradium
from hark.capture import WavRecorder
from hark.voice import OnlineCluster
from test_gradium import speech


def phonetic_windows():
    a, b, similarity = .716515, .651075, .522158
    first = np.array([a, np.sqrt(1 - a * a), 0])
    y = (similarity - a * b) / first[1]
    second = np.array([b, y, np.sqrt(1 - b * b - y * y)])
    assert np.dot(first, second) == pytest.approx(.522158)
    assert first[0] > .65 and second[0] > .65
    return first, second


def cluster_for(vectors, **kwargs):
    heard = iter(vectors)
    return OnlineCluster(lambda _: next(heard), minimum_duration=4, **kwargs)


@pytest.mark.parametrize('enrolled,expected', [(True, ['S1', 'S1']), (False, ['S1', 'S2'])])
def test_same_identity_reuses_slot_despite_below_threshold_phonetic_windows(enrolled, expected):
    first, second = phonetic_windows()
    bank = {'me': np.array([1., 0., 0.])} if enrolled else {}
    c = cluster_for([first, second], voices=bank)
    audio = speech(4)
    assert [c(audio), c(audio)] == expected
    assert c.threshold == .55
    assert c.counts == ([2] if enrolled else [1, 1])
    if enrolled:
        np.testing.assert_allclose(c.centroids[0], (first + second) / 2, atol=1e-7)
    # A short clip inherits without consuming an embedding or updating a centroid.
    assert c(speech(1)) == expected[-1]


@pytest.mark.parametrize('bank', [
    {'me': np.array([1., 0., 0.]), 'other': np.array([1., 0., 0.])},
    {'weak': np.array([0., -1., 0.])},
])
def test_ambiguous_or_weak_bank_claim_leaves_anonymous_split_unchanged(bank):
    first, second = phonetic_windows()
    c = cluster_for([first, second], voices=bank)
    assert [c(speech(4)), c(speech(4))] == ['S1', 'S2']


def test_conflicting_confident_identities_do_not_merge_above_centroid_threshold():
    first, second = np.array([1., 0.]), np.array([.6, .8])
    c = cluster_for([first, second, first, second], voices={'ada': first, 'bob': second})
    assert [c(speech(4)) for _ in range(4)] == ['S1', 'S2', 'S1', 'S2']
    assert c.counts == [2, 2]
    anonymous = cluster_for([first, second])
    assert [anonymous(speech(4)), anonymous(speech(4))] == ['S1', 'S1']


def test_ambiguous_clip_keeps_generic_behavior_even_with_anchored_slots():
    first, second = np.array([1., 0.]), np.array([0., 1.])
    ambiguous = np.array([1., 1.]) / np.sqrt(2)
    c = cluster_for([first, second, ambiguous], voices={'ada': first, 'bob': second})
    assert [c(speech(4)) for _ in range(3)] == ['S1', 'S2', 'S1']
    assert c.counts == [2, 1]


@pytest.mark.parametrize('enrolled,expected', [(True, ['S1', 'S1']), (False, ['S1', 'S2'])])
def test_cli_passes_voice_bank_into_default_gradium_cluster(tmp_path, monkeypatch, enrolled, expected):
    bank_vector = np.array([1., 0., 0.])
    if enrolled:
        voices = cli.HOME / 'voices'
        voices.mkdir()
        np.save(voices / 'me.npy', bank_vector)
    heard = iter([*phonetic_windows(), bank_vector])
    monkeypatch.setattr('hark.voice.Embedder', lambda: lambda audio: (
        next(heard) if audio.any() else bank_vector))
    made = []
    real_track = gradium.GradiumTrack
    with MockGradium(plans=[[('first', 0, 4)], [('second', .32, 5.04)]]) as mock:
        def make_track(name, **kwargs):
            t = real_track(name, **kwargs, url=mock.url, timeout=.5, shutdown_timeout=5)
            made.append(t)
            return t
        monkeypatch.setattr(gradium, 'GradiumTrack', make_track)
        monkeypatch.setattr(cli.sys, 'platform', 'linux')
        path = tmp_path / 'phonetic.wav'
        recorder = WavRecorder(path)
        recorder.write(np.concatenate([speech(4), np.zeros(60 * 16000, np.float32), speech(4.72)]))
        recorder.close()
        out = tmp_path / 'transcript.txt'
        assert cli.main(['--ear', 'gradium', '--no-gradium-metering', '--file', str(path),
                         '-o', str(out)]) is None
        records = [json.loads(line) for line in out.with_suffix('.jsonl').read_text().splitlines()]
        words = [r for r in records if 'text' in r]
        assert [r['speaker'] for r in words] == expected
        assert [(r['start'], r['end']) for r in words] == [(0, 4), (64, 68.72)]
        assert made[0].cluster.threshold == .55
        assert made[0].cluster.counts == ([2] if enrolled else [1, 1])
        assert not (cli.HOME / 'meeting.json').exists()
