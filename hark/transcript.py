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

from .capture import SAMPLE_RATE

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

    def __init__(self, name, asr, diar, *, speaker_label, language=None, gap=3.0, max_len=30.0,
                 mask=None):
        self.name = name
        if mask:
            from .session import GatedSession

            self.session = GatedSession(asr, diar, language=language, policy=mask)
        else:
            self.session = asr.create_speaker_streaming_session(diar, language=language)
        self.speaker_label = speaker_label  # "speaker_3" -> "S4" or "me"
        self.gap = gap
        self.max_len = max_len
        self.hop = asr.preprocessor_config.hop_length
        self.t0 = datetime.now()  # wall time of sample 0; the caller sets it when capture starts
        self.pending = {}
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
            self.pending.setdefault(delta.speaker, _Pending()).tokens.extend(delta.tokens)

    def _flush_plan(self, tracks, force, wall_times):
        cuts = {}
        for speaker, pending in self.pending.items():
            if not pending.tokens:
                continue
            quiet = self.processed - pending.tokens[-1].end
            span = pending.tokens[-1].end - pending.tokens[0].start
            turn_cuts = [cut for track in tracks for other_speaker in track.pending
                         if (cut := track._turn_cut(self, speaker, other_speaker, wall_times)) is not None]
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
            if u:
                out.append(u)
        return out

    def _turn_cut(self, own_track, own_speaker, other_speaker, wall_times):
        if self is own_track and other_speaker == own_speaker:
            return None
        own = own_track.pending[own_speaker].tokens
        other = self.pending[other_speaker].tokens
        own_times = [wall_times[id(own_track), id(token)] for token in own]
        other_times = [wall_times[id(self), id(token)] for token in other]
        for i, (start, _) in enumerate(other_times):
            sustained = next((j for j, (_, end) in enumerate(other_times[i + 1 :], 1)
                              if end - start >= 0.8), None)
            if sustained is None:
                continue
            end = other_times[i + sustained][1]
            if any(a < end and b > start for a, b in own_times):
                continue
            cut = next((j for j, (token_start, _) in enumerate(own_times)
                        if token_start >= start), len(own))
            if cut:
                return cut
        return None

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
    wall_times = {}
    for track in tracks:
        origin = track.t0.timestamp()
        wall_times.update({(id(track), id(token)): (origin + token.start, origin + token.end)
                           for pending in track.pending.values() for token in pending.tokens})
    plans = [track._flush_plan(tracks, False, wall_times) for track in tracks]
    out = []
    while force and any(plans):
        out.extend(utterance for track, cuts in zip(tracks, plans)
                   for utterance in track._apply_flush(cuts))
        plans = [track._flush_plan(tracks, False, wall_times) for track in tracks]
    if not force:
        out.extend(utterance for track, cuts in zip(tracks, plans)
                   for utterance in track._apply_flush(cuts))
    if force:
        plans = [track._flush_plan(tracks, True, wall_times) for track in tracks]
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


def numbered(speaker):
    """speaker_0 -> S1 (arrival order within the track)."""
    return f"S{int(speaker.rsplit('_', 1)[1]) + 1}"


class Sink:
    """The session's text file and JSONL sidecar, with append-only name mappings.

    `# S2 = Mike` names a slot; `# S2 = S2` returns it to its anonymous label.
    """

    def __init__(self, txt_path, header):
        self.path = txt_path
        self.txt = open(txt_path, "a+", buffering=1)
        self.jsonl = open(txt_path.with_suffix(".jsonl"), "a", buffering=1)
        self.names = {}
        self.txt.seek(0, 2)
        self.txt.write(f"# {header}\n")
        self.offset = self.path.stat().st_size

    def _map(self, speaker, name):
        if name == speaker:
            self.names.pop(speaker, None)
        else:
            self.names[speaker] = name

    def name(self, speaker, name):
        self._map(speaker, name)
        self.txt.seek(0, 2)
        self.txt.write(f"# {speaker} = {name}\n")
        self.jsonl.write(json.dumps({"wall": datetime.now().isoformat(timespec="seconds"),
                                     "name": {"speaker": speaker, "as": None if name == speaker else name}},
                                    ensure_ascii=False) + "\n")

    def poll_names(self):
        with self.path.open("rb") as transcript:
            transcript.seek(self.offset)
            data = transcript.read()
        complete = data.rfind(b"\n") + 1
        lines = data[:complete].decode("utf-8").splitlines()
        self.offset += complete
        for line in lines:
            if match := re.fullmatch(r"# (S\d+) = (.+)", line):
                self._map(match[1], match[2])

    def write(self, u):
        self.poll_names()
        u.name = self.names.get(u.speaker)
        self.txt.seek(0, 2)
        self.txt.write(u.line() + "\n")
        self.jsonl.write(json.dumps(u.record(), ensure_ascii=False) + "\n")

    def source(self, event):
        """Mark a live source going quiet (`# mic lost at …`) or coming back."""
        who, what = ("system audio", "the call") if event.track == "system" else (event.track, "the mic")
        since = datetime.fromtimestamp(event.since)
        record = {"track": event.track, "state": event.state,
                  "since": since.isoformat(timespec="seconds"), "cause": event.cause}
        if event.state == "silent":
            why = (f"no signal from the {'tap' if event.track == 'system' else 'device'}; "
                   f"nothing from {what} is being transcribed" if event.cause == "no signal"
                   else f"only silence while others speak; {what} may not be captured")
            line = f"# {who} lost at {since:%H:%M:%S} — {why}"
        else:
            back = datetime.fromtimestamp(event.at)
            record["back"] = back.isoformat(timespec="seconds")
            line = f"# {who} back at {back:%H:%M:%S} after {_duration(event.at - event.since)} lost"
        self.txt.seek(0, 2)
        self.txt.write(line + "\n")
        self.jsonl.write(json.dumps({"wall": datetime.now().isoformat(timespec="seconds"),
                                     "source": record}) + "\n")

    def close(self, footer):
        self.txt.write(f"# {footer}\n")
        self.txt.close()
        self.jsonl.close()


def _duration(seconds):
    """14m15s, 1h02m03s, 45s."""
    h, rest = divmod(round(seconds), 3600)
    m, s = divmod(rest, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s" if m else f"{s}s"
