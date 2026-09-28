"""Speaker masks for the per-speaker ASR streams.

mlx-audio's `SpeakerStreamingSession` opens speaker k's ASR stream for an 80 ms
frame whenever k's mean diarizer probability over the frame exceeds 0.5. The
mask is temporal: when two speakers' frames are open at once, both decoders hear
the same mixed audio and both transcribe the louder voice. `MaskPolicy` decides
which frames each stream may hear (`hark.session.GatedSession` applies it).

    overlap  a frame is shared with a second open speaker only when that speaker's
             probability is at least `overlap` (1.0: the most probable speaker
             alone hears the frame; None: every speaker over the threshold does)
"""

from dataclasses import dataclass

import numpy as np


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
        """'default' | 'exclusive' | 'overlap=0.9' | 'threshold=0.6,overlap=0.9'"""
        if spec in (None, "", "default"):
            return cls()
        if spec == "exclusive":
            return cls(overlap=1.0)
        kw = dict(part.split("=") for part in spec.split(","))
        return cls(**{k: float(v) for k, v in kw.items()})
