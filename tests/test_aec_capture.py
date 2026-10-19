"""Software AEC alignment, real native processing, and raw-device selection."""

from types import SimpleNamespace

import numpy as np
import pytest

from hark import capture
from hark.aec import EchoCanceller, FRAME, HISTORY, LOOKAHEAD, estimate_delay


def test_mic_is_raw_with_explicit_device(monkeypatch):
    opened = []
    monkeypatch.setattr(capture.MicSource, "_stream", lambda self: opened.append(self.device))
    monkeypatch.setattr(capture.MicSource, "_watch", lambda self: None)
    mic = capture.MicSource("USB")
    mic.start()
    mic.stop()
    assert opened == ["USB"]
    assert not hasattr(mic, "helper")


@pytest.mark.parametrize("lag", [-16000, -3200, 0, 4200, 16000])
def test_alignment_measures_positive_and_negative_timeline_offsets(lag):
    rng = np.random.default_rng(71)
    reference = rng.normal(0, .1, 100000).astype(np.float32)
    mic = reference[30000 - lag:50000 - lag] * .2
    delay, confidence = estimate_delay(mic, reference[6000:74000])
    assert abs(delay - lag) <= 4
    assert confidence > .99


def test_silent_reference_has_no_fft_roundoff_matches():
    mic = np.random.default_rng(1).normal(0, .001, 16000).astype(np.float32)
    reference = np.zeros(64000, np.float32)
    reference[40000:50000] = .1
    _, confidence = estimate_delay(mic, reference)
    assert 0 <= confidence < .15
    assert estimate_delay(np.zeros(16000, np.float32), reference) is None


class Identity:
    def __init__(self): self.events = []
    def set_stream_delay_ms(self, delay): assert delay == 50
    def process_reverse_stream(self, frame): self.events.append("reverse")
    def process_stream(self, frame): self.events.append("mic")


def test_chunking_length_pcm_grid_order_and_final_partial():
    raw = capture.on_pcm16_grid(np.random.default_rng(2).normal(0, .02, 70017).astype(np.float32))
    processor = Identity()
    aec = EchoCanceller(processor=processor)
    output = []
    for i in range(0, len(raw), 1137):
        output.append(aec.feed(raw[i:i + 1137], np.zeros(1137, np.float32)))
    output.append(aec.feed(final=True))
    np.testing.assert_array_equal(np.concatenate(output), raw)
    assert processor.events == ["reverse", "mic"] * ((len(raw) + FRAME - 1) // FRAME)
    assert len(aec.mic) <= HISTORY


def test_lookahead_and_independent_reference_chunks():
    aec = EchoCanceller(system_offset=100, processor=Identity())
    assert not aec.feed(np.ones(LOOKAHEAD, np.float32)).size
    assert not aec.feed(system=np.ones(50000, np.float32)).size
    assert aec.feed(np.ones(160, np.float32)).size == 160
    assert aec.feed(final=True).size == LOOKAHEAD


def test_muted_samples_stay_silent_with_filter_tail():
    class Tail(Identity):
        def process_stream(self, frame):
            frame._data = bytearray(frame._data)
            np.asarray(frame.data)[:] = 123
    aec = EchoCanceller(processor=Tail())
    raw = np.r_[np.ones(173, np.float32) * .1, np.zeros(334, np.float32)]
    clean = aec.feed(raw, final=True)
    assert clean[:173].any()
    assert not clean[173:].any()


def test_real_webrtc_cancels_echo_and_preserves_double_talk():
    # Band-limited echo plus a pulsed near-end tone exercises the native library.
    # Intelligible double-talk is measured separately with speech clips.
    from scipy.signal import butter, sosfilt

    rng = np.random.default_rng(91)
    n = 12 * 16000
    far = sosfilt(butter(3, [150, 3000], fs=16000, btype="bandpass", output="sos"),
                  rng.normal(0, .08, n)).astype(np.float32)
    t = np.arange(n) / 16000
    near = (.04 * np.sin(2 * np.pi * (230 * t + 40 * np.sin(t * 6)))
            * np.maximum(0, np.sin(2 * np.pi * 3 * t))).astype(np.float32)
    near[:8 * 16000] = 0
    echo = np.r_[np.zeros(4000), far[:-4000] * .5].astype(np.float32)
    raw = echo + near
    aec = EchoCanceller()
    output = [aec.feed(raw[i:i + 1600], far[i:i + 1600]) for i in range(0, n, 1600)]
    output.append(aec.feed(final=True))
    clean = np.concatenate(output)
    steady = slice(4 * 16000, 7 * 16000)
    suppression = 20 * np.log10(np.linalg.norm(raw[steady]) / np.linalg.norm(clean[steady]))
    assert suppression > 15
    # WebRTC's capture filter has a small algorithmic delay; compare correlations.
    from scipy.signal import correlate
    mixed = slice(9 * 16000, 11 * 16000)
    correlation = correlate(clean[mixed], near[mixed], mode="full", method="fft")
    similarity = max(correlation) / (np.linalg.norm(clean[mixed]) * np.linalg.norm(near[mixed]))
    assert similarity > .5
    assert np.linalg.norm(clean[mixed]) / np.linalg.norm(near[mixed]) > .4
    assert abs(aec.delay - 4000) < 320
