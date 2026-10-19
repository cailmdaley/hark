"""Software echo cancellation on the tracks' sample timelines; no audio devices.

A bounded lookahead admits a late system reference before the corresponding mic
frame. Waveform correlation removes coarse capture/padding offsets; WebRTC AEC3
tracks the remaining acoustic delay and clock drift in 10 ms frames.
"""

import numpy as np
from scipy.signal import correlate

from .capture import SAMPLE_RATE, to_pcm16

FRAME = SAMPLE_RATE // 100
SEARCH = int(1.5 * SAMPLE_RATE)
LOOKAHEAD = 2 * SAMPLE_RATE
HISTORY = 4 * SAMPLE_RATE


def aligned_slice(audio, base, start, size):
    """Read a sample-indexed window, with silence outside the available track."""
    out = np.zeros(size, np.float32)
    lo, hi = max(start, base), min(start + size, base + len(audio))
    if hi > lo:
        out[lo - start:hi - start] = audio[lo - base:hi - base]
    return out


def estimate_delay(mic, reference, max_delay=SEARCH):
    """Return (mic lag in samples, normalized correlation), or None for silence.

    Reference surrounds mic by max_delay samples on each side. Downsampling to
    4 kHz keeps this coarse alignment inexpensive; AEC3 handles sub-frame delay.
    """
    step = 4
    mic, reference = mic[::step], reference[::step]
    energy = float(mic @ mic)
    if energy < 1e-8 or len(reference) < len(mic):
        return None
    cross = correlate(reference, mic, mode="valid", method="fft")
    powers = np.r_[0., np.cumsum(reference.astype(np.float64) ** 2)]
    window_power = powers[len(mic):] - powers[:-len(mic)]
    norms = np.sqrt(np.maximum(1e-20, energy * window_power))
    # FFT roundoff over silent reference windows must not look like a match.
    scores = np.where(window_power > 1e-6, np.minimum(1., np.abs(cross) / norms), 0.)
    index = int(np.argmax(scores))
    return max_delay - index * step, float(scores[index])


class EchoCanceller:
    """Accept independently chunked PCM tracks; return only processed mic samples.

    system_offset is the system track's start relative to the mic, in samples.
    feed() preserves the mic's length and indices, including a short final frame.
    A two-second mic hold covers reference arrivals and ±1.5 s timeline offsets.
    Buffers retain only four seconds of history plus the pending input.
    """

    def __init__(self, system_offset=0, *, processor=None, report=None):
        from livekit.rtc import AudioFrame, AudioProcessingModule

        self.frame = AudioFrame
        self.processor = processor or AudioProcessingModule(echo_cancellation=True)
        self.processor.set_stream_delay_ms(50)
        self.report = report or (lambda _: None)
        self.mic = np.zeros(0, np.float32)
        self.reference = np.zeros(0, np.float32)
        self.mic_base, self.reference_base = 0, system_offset
        self.position = 0
        self.delay = 0
        self.next_estimate = 0

    def feed(self, mic=(), system=(), *, final=False):
        self.mic = np.concatenate((self.mic, np.asarray(mic, np.float32)))
        self.reference = np.concatenate((self.reference, np.asarray(system, np.float32)))
        end = self.mic_base + len(self.mic)
        reference_end = self.reference_base + len(self.reference)
        ready = end if final else max(self.position, min(end - LOOKAHEAD,
                                                        reference_end - SEARCH - 800))
        output = []
        while self.position < ready:
            if self.position >= self.next_estimate:
                self._align(end)
                self.next_estimate = self.position + SAMPLE_RATE
            size = min(FRAME, end - self.position)
            if size < FRAME and not final:
                break
            raw = aligned_slice(self.mic, self.mic_base, self.position, FRAME)
            # Present the reference 50 ms ahead of its echo, within AEC3's search.
            reference = aligned_slice(self.reference, self.reference_base,
                                      self.position - self.delay + 800, FRAME)
            reverse = self.frame(to_pcm16(reference).tobytes(), SAMPLE_RATE, 1, FRAME)
            near = self.frame(to_pcm16(raw).tobytes(), SAMPLE_RATE, 1, FRAME)
            self.processor.process_reverse_stream(reverse)
            self.processor.process_stream(near)
            clean = np.array(near.data, dtype=np.float32) / 32768
            # Muted/padded input stays silent, even if the filter has a tail.
            clean[raw == 0] = 0
            output.append(clean[:size])
            self.position += size
        self.mic, self.mic_base = self._trim(self.mic, self.mic_base)
        self.reference, self.reference_base = self._trim(self.reference, self.reference_base)
        return np.concatenate(output) if output else np.zeros(0, np.float32)

    def _align(self, end):
        start = max(self.mic_base, self.position - 2 * SAMPLE_RATE)
        stop = min(end, self.position + SAMPLE_RATE // 2)
        if stop - start < SAMPLE_RATE // 2:
            return
        mic = aligned_slice(self.mic, self.mic_base, start, stop - start)
        reference = aligned_slice(self.reference, self.reference_base,
                                  start - SEARCH, stop - start + 2 * SEARCH)
        estimate = estimate_delay(mic, reference)
        if estimate is None:
            return
        delay, confidence = estimate
        # Don't chase near-end speech or jitter; AEC3 handles the fine adjustment.
        if confidence >= 0.15 and abs(delay - self.delay) >= 320:
            self.delay = delay
            self.report(f"aec: mic/reference lag {delay / SAMPLE_RATE:+.3f} s "
                        f"(correlation {confidence:.3f})")

    def _trim(self, audio, base):
        drop = min(len(audio), max(0, self.position - HISTORY - base))
        return audio[drop:], base + drop
