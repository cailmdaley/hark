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

from .capture import SAMPLE_RATE, log

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

    name: str | None = None

    @property
    def wall_span(self):
        return self.wall.timestamp(), self.wall.timestamp() + self.end - self.start

    def line(self):
        return f"{self.wall:%H:%M:%S} {self.name or self.speaker:<16} {self.text}"

    def record(self):
        record = {"wall": self.wall.isoformat(timespec="seconds"), "track": self.track,
                  "speaker": self.speaker, "start": round(self.start, 2),
                  "end": round(self.end, 2), "text": self.text}
        if self.name:
            record["name"] = self.name
        return record


@dataclass
class _Pending:
    tokens: list = field(default_factory=list)


class Track:
    """One audio stream through its own speaker-streaming session."""

    def __init__(self, name, asr, diar, *, speaker_label, language=None, gap=1.5, max_len=30.0):
        self.name = name
        self.session = asr.create_speaker_streaming_session(diar, language=language)
        self.speaker_label = speaker_label  # "speaker_3" -> "S4" or "me"
        self.gap = gap
        self.max_len = max_len
        self.hop = asr.preprocessor_config.hop_length
        self.t0 = datetime.now()  # wall time of sample 0; the caller sets it when capture starts
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
                    cut = _monologue_cut(p.tokens)
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


def _monologue_cut(tokens):
    boundaries = [i for i, token in enumerate(tokens) if token.text.startswith(" ")]
    midpoint = (tokens[0].start + tokens[-1].end) / 2
    pauses = [(tokens[i].start - tokens[i - 1].end, i) for i in boundaries
              if i and tokens[i].start >= midpoint]
    return max(pauses)[1] if pauses else (boundaries[-1] if boundaries else len(tokens))


def _words(text):
    return re.findall(r"\w+", text.casefold())


def numbered(speaker):
    """speaker_0 -> S1 (arrival order within the track)."""
    return f"S{int(speaker.rsplit('_', 1)[1]) + 1}"


class EchoGate:
    """Hold mic utterances until system audio has passed them by one second."""

    def __init__(self, threshold=0.6, margin=1.0):
        self.threshold, self.margin = threshold, margin
        self.pending, self.system = [], []

    def push(self, utterance):
        if utterance.track == "system":
            self.system.append(utterance)
            return [utterance]
        self.pending.append(utterance)
        return []

    def release(self, watermark, final=False):
        ready, waiting = [], []
        for me in self.pending:
            if not final and watermark < me.wall.timestamp() + (me.end - me.start) + self.margin:
                waiting.append(me)
                continue
            start, end = me.wall_span
            overlapping = [s for s in self.system if s.wall_span[0] <= end and s.wall_span[1] >= start]
            mine, theirs = _words(me.text), set(_words(" ".join(s.text for s in overlapping)))
            if mine and sum(word in theirs for word in mine) / len(mine) >= self.threshold:
                log(f'echo: dropped me "{me.text}"')
            else:
                ready.append(me)
        self.pending = waiting
        return ready


class Sink:
    """The session's text file and JSONL sidecar, with append-only name mappings."""

    def __init__(self, txt_path, header):
        self.path = txt_path
        self.txt = open(txt_path, "a+", buffering=1)
        self.jsonl = open(txt_path.with_suffix(".jsonl"), "a", buffering=1)
        self.names = {}
        self.txt.seek(0, 2)
        self.offset = self.txt.tell()
        self.txt.write(f"# {header}\n")
        self.offset = self.txt.tell()

    def name(self, speaker, name):
        self.names[speaker] = name
        self.txt.seek(0, 2)
        self.txt.write(f"# {speaker} = {name}\n")
        self.offset = self.txt.tell()
        self.jsonl.write(json.dumps({"wall": datetime.now().isoformat(timespec="seconds"),
                                     "name": {"speaker": speaker, "as": name}}, ensure_ascii=False) + "\n")

    def poll_names(self):
        self.txt.seek(self.offset)
        lines = self.txt.readlines()
        self.offset = self.txt.tell()
        for line in lines:
            if match := re.fullmatch(r"# (S\d+) = (.+)\n?", line):
                self.names[match[1]] = match[2]

    def write(self, u):
        self.poll_names()
        u.name = self.names.get(u.speaker)
        self.txt.seek(0, 2)
        self.txt.write(u.line() + "\n")
        self.offset = self.txt.tell()
        self.jsonl.write(json.dumps(u.record(), ensure_ascii=False) + "\n")

    def close(self, footer):
        self.txt.write(f"# {footer}\n")
        self.txt.close()
        self.jsonl.close()
