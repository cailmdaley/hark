"""Streaming speaker-attributed ASR → finished utterances → an append-only file.

Each track (mic, system audio, a file) owns one mlx-audio SpeakerStreamingSession:
Nemotron-3-Diarization gates Nemotron 3.5 streaming ASR, with an independent
decoder per speaker, so tokens arrive already tagged. Tokens accumulate by
speaker until another sustained speaker takes the turn, the speaker is quiet
for `gap`, or a monologue reaches its maximum length.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .capture import SAMPLE_RATE, log

ASR_MODEL = "mlx-community/nemotron-3.5-asr-streaming-0.6b"
DIAR_MODEL = "mlx-community/Nemotron-3-Diarization"


def load_models(diar_preset="low"):
    import mlx.core as mx
    from mlx_audio.stt import load as load_asr
    from mlx_audio.vad import load as load_diarization

    mx.set_cache_limit(512 * 1024**2)
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
    speech: list[tuple[float, float]] = field(default_factory=list)

    @property
    def wall_span(self):
        return self.wall.timestamp(), self.wall.timestamp() + self.end - self.start

    def line(self):
        label = self.name or f"{self.speaker:<4}"
        return f"{self.wall:%H:%M:%S} {label} {self.text}"

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

    def __init__(self, name, asr, diar, *, speaker_label, language=None, gap=3.0, max_len=30.0):
        self.name = name
        self.session = asr.create_speaker_streaming_session(diar, language=language)
        self.speaker_label = speaker_label  # "speaker_3" -> "S4" or "me"
        self.gap = gap
        self.max_len = max_len
        self.hop = asr.preprocessor_config.hop_length
        self.t0 = datetime.now()  # wall time of sample 0; the caller sets it when capture starts
        self.pending = {}
        self.echo_gate = None
        from .voice import AudioBuffer

        self.audio = AudioBuffer()
        self.audio_samples = 0

    @property
    def processed(self):
        """Seconds of audio the models have fully consumed (lags input by ~2 s)."""
        return self.session._mel_offset * self.hop / SAMPLE_RATE

    def feed(self, samples, final=False):
        step = SAMPLE_RATE // 2  # feed in bounded chunks even when a source has a backlog
        for i in range(0, samples.size, step):
            self._step(samples[i : i + step])
        if final:
            self._step(samples[:0], final=True)

    def _step(self, samples, final=False):
        if samples.size:
            if self.audio is not None:
                self.audio.append(self.audio_samples / SAMPLE_RATE, samples)
            self.audio_samples += samples.size
        for delta in self.session.feed(samples, final=final):
            if self.echo_gate:
                self.echo_gate.hold(delta.speaker, delta.tokens)
            else:
                self.pending.setdefault(delta.speaker, _Pending()).tokens.extend(delta.tokens)

    def _flush_plan(self, tracks, force):
        cuts = {}
        for speaker, pending in self.pending.items():
            if not pending.tokens:
                continue
            quiet = self.processed - pending.tokens[-1].end
            span = pending.tokens[-1].end - pending.tokens[0].start
            turn_cuts = [cut for track in tracks for other_speaker in track.pending
                         if (cut := track._turn_cut(self, speaker, other_speaker)) is not None]
            turn_cut = min(turn_cuts, default=None)
            if force:
                cuts[speaker] = len(pending.tokens)
            elif turn_cut is not None:
                cuts[speaker] = turn_cut
            elif span >= self.max_len:
                cuts[speaker] = _monologue_cut(pending.tokens)
            elif quiet >= self.gap:
                cuts[speaker] = len(pending.tokens)
        return cuts

    def _apply_flush(self, cuts):
        out = []
        for speaker, cut in cuts.items():
            pending = self.pending[speaker]
            finished = pending.tokens[:cut]
            u = self._utterance(speaker, finished)
            pending.tokens = pending.tokens[cut:]
            if gate := getattr(self, "echo_gate", None):
                gate.forget(finished)
            if u:
                out.append(u)
        return out

    def _turn_cut(self, own_track, own_speaker, other_speaker):
        if self is own_track and other_speaker == own_speaker:
            return None
        own = own_track.pending[own_speaker].tokens
        other = self.pending[other_speaker].tokens
        for i, first in enumerate(other):
            start = self._wall_time(first.start)
            run = other[i:]
            sustained = next((j for j, token in enumerate(run[1:], 1)
                              if self._wall_time(token.end) - start >= 0.8), None)
            if sustained is None:
                continue
            end = self._wall_time(run[sustained].end)
            own_times = [(own_track._wall_time(token.start), own_track._wall_time(token.end))
                         for token in own]
            if any(a < end and b > start for a, b in own_times):
                continue
            cut = next((j for j, token in enumerate(own)
                        if own_track._wall_time(token.start) >= start), len(own))
            if cut:
                return cut
        return None

    def _wall_time(self, seconds):
        return self.t0.timestamp() + seconds

    def _utterance(self, speaker, tokens):
        # RNN-T often emits the closing punctuation of a turn late, at the start of the next
        text = re.sub(r"\s+", " ", "".join(t.text for t in tokens)).strip().lstrip(".,;:?! ")
        if not text:
            return None
        start = tokens[0].start
        return Utterance(self.name, self.speaker_label(speaker), start, tokens[-1].end,
                         self.t0 + timedelta(seconds=start), text,
                         speech=[(token.start, token.end) for token in tokens])


def flush_tracks(tracks, force=False):
    plans = [track._flush_plan(tracks, False) for track in tracks]
    out = [utterance for track, cuts in zip(tracks, plans)
           for utterance in track._apply_flush(cuts)]
    if force:
        plans = [track._flush_plan(tracks, True) for track in tracks]
        out.extend(utterance for track, cuts in zip(tracks, plans)
                   for utterance in track._apply_flush(cuts))
    return sorted(out, key=lambda u: u.wall)


def _monologue_cut(tokens):
    boundaries = [i for i, token in enumerate(tokens) if token.text.startswith(" ")]
    midpoint = (tokens[0].start + tokens[-1].end) / 2
    pauses = [(tokens[i].start - tokens[i - 1].end, i) for i in boundaries
              if i and tokens[i].start >= midpoint]
    cut = max(pauses)[1] if pauses else (boundaries[-1] if boundaries else len(tokens))
    return cut or len(tokens)


def _words(text):
    return re.findall(r"\w+", text.casefold())


def numbered(speaker):
    """speaker_0 -> S1 (arrival order within the track)."""
    return f"S{int(speaker.rsplit('_', 1)[1]) + 1}"


class EchoGate:
    """Hold mic tokens until system ASR has caught up, then remove echoed runs."""

    def __init__(self, lag=1.0, window=1.5):
        self.lag, self.window = lag, window
        self.pending, self.system, self.seen = [], [], set()
        self.mic_seen, self.run = set(), []

    def hold(self, speaker, tokens):
        fresh = [token for token in tokens if id(token) not in self.mic_seen]
        self.pending.extend((speaker, token) for token in fresh)
        self.mic_seen.update(id(token) for token in fresh)

    def forget(self, tokens):
        self.mic_seen.difference_update(id(token) for token in tokens)

    def capture(self, mic, system):
        for token in (token for pending in system.pending.values() for token in pending.tokens):
            if id(token) not in self.seen:
                self.system.append((system, token))
                self.seen.add(id(token))
        for speaker, pending in mic.pending.items():
            fresh = [(speaker, token) for token in pending.tokens if id(token) not in self.mic_seen]
            self.pending.extend(fresh)
            self.mic_seen.update(id(token) for _, token in fresh)
            if fresh:
                fresh_ids = {id(token) for _, token in fresh}
                pending.tokens[:] = [token for token in pending.tokens if id(token) not in fresh_ids]

    def release(self, mic, watermark, final=False):
        ready, waiting = [], []
        for item in self.pending:
            speaker, token = item
            if not final and watermark < mic._wall_time(token.end) + self.lag:
                waiting.append(item)
            else:
                ready.append(item)
        self.pending = waiting
        times = [(track._wall_time(token.start), track._wall_time(token.end), _words(token.text))
                 for track, token in self.system]
        keep, dropped = [], []
        for item in ready:
            speaker, token = item
            if _echo_match(token, times, mic, self.window):
                self.run.append(item)
                continue
            if len(self.run) == 1:
                keep.extend(self.run)
            elif self.run:
                dropped.extend(token for _, token in self.run)
            self.run.clear()
            keep.append(item)
        if final:
            if len(self.run) == 1:
                keep.extend(self.run)
            else:
                dropped.extend(token for _, token in self.run)
            self.run.clear()
        for speaker, token in keep:
            mic.pending.setdefault(speaker, _Pending()).tokens.append(token)
        self.forget(dropped)
        if dropped:
            log(f"echo: dropped {len(dropped)} mic tokens ({' '.join(t.text for t in dropped).strip()!r})")
        cutoff = watermark - self.window
        retained = [(track, token) for track, token in self.system
                    if track._wall_time(token.end) >= cutoff]
        self.seen = {id(token) for _, token in retained}
        self.system = retained
        return [token for _, token in keep]


def _echo_match(token, system, mic, window):
    midpoint = (mic._wall_time(token.start) + mic._wall_time(token.end)) / 2
    words = _words(token.text)
    return bool(words) and any(word == other and abs(midpoint - (start + end) / 2) <= window
                               for start, end, theirs in system for word in words for other in theirs)


class Sink:
    """The session's text file and JSONL sidecar, with append-only name mappings."""

    def __init__(self, txt_path, header):
        self.path = txt_path
        self.txt = open(txt_path, "a+", buffering=1)
        self.jsonl = open(txt_path.with_suffix(".jsonl"), "a", buffering=1)
        self.names = {}
        self.txt.seek(0, 2)
        self.txt.write(f"# {header}\n")
        self.offset = self.path.stat().st_size

    def name(self, speaker, name):
        self.names[speaker] = name
        self.txt.seek(0, 2)
        self.txt.write(f"# {speaker} = {name}\n")
        self.jsonl.write(json.dumps({"wall": datetime.now().isoformat(timespec="seconds"),
                                     "name": {"speaker": speaker, "as": name}}, ensure_ascii=False) + "\n")

    def poll_names(self):
        with self.path.open("rb") as transcript:
            transcript.seek(self.offset)
            data = transcript.read()
        complete = data.rfind(b"\n") + 1
        lines = data[:complete].decode("utf-8").splitlines()
        self.offset += complete
        for line in lines:
            if match := re.fullmatch(r"# (S\d+) = (.+)", line):
                self.names[match[1]] = match[2]

    def write(self, u):
        self.poll_names()
        u.name = self.names.get(u.speaker)
        self.txt.seek(0, 2)
        self.txt.write(u.line() + "\n")
        self.jsonl.write(json.dumps(u.record(), ensure_ascii=False) + "\n")

    def close(self, footer):
        self.txt.write(f"# {footer}\n")
        self.txt.close()
        self.jsonl.close()
