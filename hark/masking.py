"""Speaker masks for the per-speaker ASR streams.

mlx-audio's `SpeakerStreamingSession` opens speaker k's ASR stream for an 80 ms
frame whenever k's mean diarizer probability over the frame exceeds 0.5. The
mask is temporal: when two speakers' frames are open at once, both decoders hear
the same mixed audio and both transcribe the louder voice. `MaskPolicy` decides
which frames each stream may hear; `GatedSession` is the session with that
decision made by the policy instead of the fixed threshold.

    overlap  a frame is shared with a second open speaker only when that speaker's
             probability is at least `overlap` (1.0: the most probable speaker
             alone hears the frame; None: every speaker over the threshold does)
"""

from dataclasses import dataclass

import mlx.core as mx
import numpy as np
from mlx_audio.stt.models.nemotron_asr.speaker_streaming import (
    SpeakerStreamingSession, SpeakerTranscript, mask_features)
from mlx_audio.stt.models.nemotron_asr.streaming import ConformerStreamingState
from mlx_audio.stt.models.nemotron_asr.rnnt import GreedyDecoderState


@dataclass(frozen=True)
class MaskPolicy:
    threshold: float = 0.5
    overlap: float | None = None

    def __call__(self, probs):
        """(frames, speakers) mean probabilities at the ASR stride → boolean activity."""
        p = np.asarray(probs)
        active = p > self.threshold
        if self.overlap is not None and p.size:
            top = p.argmax(1)
            dominant = np.zeros_like(active)
            dominant[np.arange(len(p)), top] = True
            active &= dominant | (p >= self.overlap)
        return active

    @classmethod
    def parse(cls, spec):
        """'0.5' | 'exclusive' | 'overlap=0.9' | 'threshold=0.6,overlap=0.9'"""
        if spec in (None, "", "default"):
            return cls()
        if spec == "exclusive":
            return cls(overlap=1.0)
        kw = dict(part.split("=") for part in spec.split(","))
        return cls(**{k: float(v) for k, v in kw.items()})


class GatedSession(SpeakerStreamingSession):
    """`SpeakerStreamingSession` whose per-frame speaker activity comes from a `MaskPolicy`."""

    def __init__(self, model, diarization_model, *, policy=MaskPolicy(), **kwargs):
        super().__init__(model, diarization_model, threshold=policy.threshold, **kwargs)
        self.policy = policy

    def _activity(self, probs):
        pad = -probs.shape[0] % self.factor
        mean = mx.pad(probs, [(0, pad), (0, 0)]).reshape(-1, self.factor, probs.shape[1]).mean(axis=1)
        return mx.array(self.policy(np.array(mean)))

    def _push_features(self, mel, probs, *, final=False):
        # SpeakerStreamingSession._push_features with the activity line swapped for the policy
        self._mel = mx.concatenate([self._mel, mel], axis=1)
        self._probs = mx.concatenate([self._probs, probs], axis=0)
        if final:
            missing = self._mel.shape[1] - self._probs.shape[0]
            if missing < 0:
                raise ValueError("Diarization timeline extends beyond ASR features")
            self._probs = mx.pad(self._probs, [(0, missing), (0, 0)])
        updates = []
        while self._mel.shape[1]:
            count = min(self.chunk_mel, self._mel.shape[1])
            if not final and (count < self.chunk_mel or self._probs.shape[0] < count):
                break
            last = final and count == self._mel.shape[1]
            activity = self._activity(self._probs[:count])
            self._history.append(mx.any(activity, axis=0))
            active = (mx.any(mx.stack(list(self._history)), axis=0) if self.cache_gating
                      else mx.ones((self.num_speakers,), dtype=mx.bool_))
            for speaker in np.flatnonzero(np.array(active)):
                speaker = int(speaker)
                if speaker not in self._encoders:
                    self._encoders[speaker] = ConformerStreamingState(
                        self.model.encoder, att_context_size=self.att_context_size)
                    self._decoders[speaker] = GreedyDecoderState(self.model)
                encoder = self._encoders[speaker]
                masked = mask_features(self._mel[:, :count], activity[:, speaker], self.factor)
                start_frame = self._mel_offset // self.factor
                for encoded in encoder.push(masked, final=last):
                    prompted = self.model.apply_prompt(encoded, self.language)
                    encoder.materialize(prompted)
                    tokens = self._decoders[speaker].decode(prompted, start_frame)
                    start_frame += prompted.shape[1]
                    if tokens:
                        updates.append(SpeakerTranscript(f"speaker_{speaker}", tokens))
            self._mel_offset += count
            self._mel = self._mel[:, count:]
            self._probs = self._probs[count:]
        mx.eval(self._mel, self._probs, *self._history)
        return updates
