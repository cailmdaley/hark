"""Streaming speaker-attributed ASR → finished utterances → an append-only file.

Each track (mic, system audio, a file) owns one mlx-audio SpeakerStreamingSession:
Nemotron-3-Diarization gates Nemotron 3.5 streaming ASR, with an independent
decoder per speaker, so tokens arrive already tagged. Tokens accumulate per
speaker until that speaker has been quiet for `gap` seconds of audio; the
utterance is then written as one line.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .capture import SAMPLE_RATE

ASR_MODEL = "mlx-community/nemotron-3.5-asr-streaming-0.6b"
DIAR_MODEL = "mlx-community/Nemotron-3-Diarization"


def load_models(diar_preset="low"):
    from mlx_audio.stt import load as load_asr
    from mlx_audio.vad import load as load_diarization

    diar = load_diarization(DIAR_MODEL, strict=True)
    diar.set_streaming_config(diar_preset)
    asr = load_asr(ASR_MODEL)
    return asr, diar


@dataclass
class Utterance:
    track: str
    speaker: str
    start: float  # seconds since the track started
    end: float
    wall: datetime
    text: str

    def line(self):
        return f"{self.wall:%H:%M:%S} {self.speaker:<4} {self.text}"

    def record(self):
        return {"wall": self.wall.isoformat(timespec="seconds"), "track": self.track,
                "speaker": self.speaker, "start": round(self.start, 2),
                "end": round(self.end, 2), "text": self.text}


@dataclass
class _Pending:
    tokens: list = field(default_factory=list)


class Track:
    """One audio stream through its own speaker-streaming session."""

    def __init__(self, name, asr, diar, *, speaker_label, language=None, gap=1.0, max_len=30.0,
                 t0=None):
        self.name = name
        self.session = asr.create_speaker_streaming_session(diar, language=language)
        self.speaker_label = speaker_label  # "speaker_3" -> "S4" or "me"
        self.gap = gap
        self.max_len = max_len
        self.hop = asr.preprocessor_config.hop_length
        self.t0 = t0 or datetime.now()  # a file's transcript counts from midnight: 00:01:23
        self.pending = {}

    @property
    def processed(self):
        """Seconds of audio the models have fully consumed (lags input by ~2 s)."""
        return self.session._mel_offset * self.hop / SAMPLE_RATE

    def feed(self, samples, final=False):
        out = []
        step = SAMPLE_RATE // 2  # flush decisions every 0.5 s even when fed a backlog
        for i in range(0, samples.size, step):
            out += self._step(samples[i : i + step])
        if final:
            out += self._step(samples[:0], final=True)
        return out

    def _step(self, samples, final=False):
        for delta in self.session.feed(samples, final=final):
            self.pending.setdefault(delta.speaker, _Pending()).tokens.extend(delta.tokens)
        return self._flush(force=final)

    def _flush(self, force=False):
        out = []
        for speaker, p in list(self.pending.items()):
            if not p.tokens:
                continue
            quiet = self.processed - p.tokens[-1].end
            span = p.tokens[-1].end - p.tokens[0].start
            if force or quiet >= self.gap or span >= self.max_len:
                cut = len(p.tokens)
                if not force and quiet < self.gap:
                    # long monologue: break before the last word, which may be unfinished
                    cut = max((i for i, t in enumerate(p.tokens) if t.text.startswith(" ")), default=0) or cut
                u = self._utterance(speaker, p.tokens[:cut])
                p.tokens = p.tokens[cut:]
                if u:
                    out.append(u)
        return sorted(out, key=lambda u: u.start)

    def _utterance(self, speaker, tokens):
        # RNN-T often emits the closing punctuation of a turn late, at the start of the next
        text = re.sub(r"\s+", " ", "".join(t.text for t in tokens)).strip().lstrip(".,;:?! ")
        if not text:
            return None
        start = tokens[0].start
        return Utterance(self.name, self.speaker_label(speaker), start, tokens[-1].end,
                         self.t0 + timedelta(seconds=start), text)


def numbered(speaker):
    """speaker_0 -> S1 (arrival order within the track)."""
    return f"S{int(speaker.rsplit('_', 1)[1]) + 1}"


class Sink:
    """The session's text file (for people and agents) plus a JSONL sidecar."""

    def __init__(self, txt_path, header):
        self.txt = open(txt_path, "a", buffering=1)
        self.jsonl = open(txt_path.with_suffix(".jsonl"), "a", buffering=1)
        self.txt.write(f"# {header}\n")

    def write(self, u):
        self.txt.write(u.line() + "\n")
        self.jsonl.write(json.dumps(u.record(), ensure_ascii=False) + "\n")

    def close(self, footer):
        self.txt.write(f"# {footer}\n")
        self.txt.close()
        self.jsonl.close()
