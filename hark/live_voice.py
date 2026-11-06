"""Low-latency mic track with an inexpensive idle gate and bounded clock mapping."""

from collections import deque
from copy import copy
from datetime import timedelta

import numpy as np

from .capture import SAMPLE_RATE
from .transcript import Track, _Pending

VOICE_GAP = 0.5
VOICE_MAX_LEN = 6.0
VOICE_LATENCY = "ultra_low"


class VoiceTrack(Track):
    """Keep the recognizer's state, but don't run either model on an idle mic.

    A -55 dBFS RMS gate uses 100 ms frames, 300 ms pre-roll and a three-second
    tail to drain delayed tokens. The recognizer sees a compressed audio clock;
    token timestamps are translated back to the mic clock before turn detection.
    Speaker caches and identities survive idle periods.
    """

    frame = SAMPLE_RATE // 10
    preroll = 3 * frame
    tail = 3 * SAMPLE_RATE
    threshold = 10 ** (-55 / 20)

    def __init__(self, *args, gate=True, **kwargs):
        kwargs.setdefault("gap", VOICE_GAP)
        kwargs.setdefault("max_len", VOICE_MAX_LEN)
        super().__init__(*args, **kwargs)
        self.gate = gate
        self._held = np.zeros(0, np.float32)
        self._tail_left = 0
        self._sent = 0
        self._clock = deque([(0, 0)])  # model sample index -> skipped mic samples
        self._wall_clock = deque([(0.0, 0.0)])  # mic seconds -> wall-clock offset after sleep

    def feed(self, samples, final=False):
        # The idle gate skips model work without removing time from a final drain.
        step = SAMPLE_RATE // 2
        for i in range(0, samples.size, step):
            self._step(samples[i:i + step])
        if final:
            self._step(samples[:0], final=True)

    def add_wall_gap(self, at, origin):
        """Anchor resumed mic samples without moving speech buffered before sleep."""
        if not self._tail_left and not any(p.tokens for p in self.pending.values()):
            self._wall_clock = deque([self._wall_clock[-1]])
        self._wall_clock.append((at, origin - self.t0.timestamp()))

    def _utterance(self, speaker, tokens):
        utterance = super()._utterance(speaker, tokens)
        if utterance:
            offset = self._wall_clock[0][1]
            for boundary, value in self._wall_clock:
                if boundary > utterance.start:
                    break
                offset = value
            utterance.wall += timedelta(seconds=offset)
        return utterance

    def _flush_plan(self, tracks, force, wall_times):
        cuts = super()._flush_plan(tracks, force, wall_times)
        for speaker, pending in self.pending.items():
            for boundary, _ in list(self._wall_clock)[1:]:
                before = sum(token.start < boundary for token in pending.tokens)
                if before and self.processed >= boundary:
                    cuts[speaker] = min(cuts.get(speaker, before), before)
                    break
        return cuts

    def _apply_flush(self, cuts):
        out = super()._apply_flush(cuts)
        safe = self._mic_time(super().processed) - 1
        safe = min((token.start for p in self.pending.values() for token in p.tokens), default=safe)
        while len(self._wall_clock) > 1 and self._wall_clock[1][0] < safe:
            self._wall_clock.popleft()
        return out

    def _mic_time(self, seconds, *, end=False):
        sample = round(seconds * SAMPLE_RATE)
        offset = self._clock[0][1]
        for boundary, skipped in self._clock:
            if boundary > sample or (end and boundary == sample):
                break
            offset = skipped
        return seconds + offset / SAMPLE_RATE

    @property
    def processed(self):
        if self.gate and not self._tail_left:
            return (self.audio_samples - self._held.size) / SAMPLE_RATE
        return self._mic_time(super().processed)

    def _submit(self, samples, start, final=False):
        skipped = start - self._sent
        if samples.size and skipped != self._clock[-1][1]:
            self._clock.append((self._sent, skipped))
        self._sent += samples.size
        for delta in self.session.feed(samples, final=final):
            pending = self.pending.setdefault(delta.speaker, _Pending())
            for original in delta.tokens:
                token = copy(original)
                token.start = self._mic_time(original.start)
                token.end = self._mic_time(original.end, end=True)
                pending.tokens.append(token)
        # New tokens cannot precede the model's committed feature clock.
        committed = self.session._mel_offset * self.hop - SAMPLE_RATE
        while len(self._clock) > 1 and self._clock[1][0] < committed:
            self._clock.popleft()

    def _step(self, samples, final=False):
        start = self.audio_samples
        if self.audio is not None and samples.size:
            self.audio.append(start / SAMPLE_RATE, samples)
        self.audio_samples += samples.size
        if not self.gate:
            self._submit(samples, start, final=final)
            return
        for i in range(0, samples.size, self.frame):
            chunk = samples[i:i + self.frame]
            at = start + i
            loud = float(np.mean(chunk * chunk)) >= self.threshold ** 2
            if loud:
                self._tail_left = self.tail
                if self._held.size:
                    self._submit(self._held, at - self._held.size)
                    self._held = np.zeros(0, np.float32)
            if self._tail_left:
                self._submit(chunk, at)
                self._tail_left = max(0, self._tail_left - chunk.size)
            else:
                self._held = np.concatenate((self._held, chunk))[-self.preroll:]
        if final:
            if self._held.size:
                self._submit(self._held, self.audio_samples - self._held.size)
                self._held = np.zeros(0, np.float32)
            self._submit(samples[:0], self.audio_samples, final=True)
