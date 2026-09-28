import numpy as np

from hark.masking import MaskPolicy


P = np.array([[0.9, 0.6, 0.1],   # two over threshold: shared by default
              [0.3, 0.95, 0.92],  # a near-tie
              [0.4, 0.2, 0.1]])   # nobody


def test_shared_is_the_plain_threshold():
    assert (MaskPolicy()(P) == (P > 0.5)).all()


def test_exclusive_gives_each_frame_to_the_most_probable_speaker():
    assert MaskPolicy.parse("exclusive")(P).tolist() == [[True, False, False],
                                                         [False, True, False],
                                                         [False, False, False]]


def test_overlap_threshold_lets_a_confident_second_speaker_share():
    assert MaskPolicy.parse("overlap=0.9")(P).tolist() == [[True, False, False],
                                                           [False, True, True],
                                                           [False, False, False]]
