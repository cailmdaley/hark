"""Low-latency mic track with an inexpensive idle gate and bounded clock mapping."""

from collections import deque
from copy import copy
from datetime import datetime, timedelta
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time

import numpy as np

from .capture import SAMPLE_RATE
from .transcript import Track, Utterance, _Pending

VOICE_GAP = 5.0
VOICE_MAX_LEN = 75.0
VOICE_WORDS = 200
VOICE_MIN_WORDS = 12
VOICE_LATENCY = "ultra_low"
CUT_DRAIN = 1.0  # delayed words before a route switch land within this much model audio
EAR_STALL = 10.0  # a failed call this slow is a dead link, not a refused run
EAR_REST = 60.0
VOICE_END = re.compile(r"\bsend[.!?]*\s*$", re.IGNORECASE)
VOICE_END_HOLD = .7  # seconds after a closing word, so "send it to Bob" keeps going
VOICE_NAME = re.compile(r"\b(?:claude|cloud|clawed|claw|klaud|clot|clock|clod|claud)\b", re.IGNORECASE)


class EarResting(Exception):
    """The ear is resting after a stall; this run keeps its raw words."""


class RestingEar:
    """Rest a dead link across every track, since each call blocks the capture loop."""

    def __init__(self, recognize):
        self.recognize = recognize
        self.until = 0.0

    def __call__(self, samples):
        asked = time.monotonic()
        if asked < self.until:
            raise EarResting
        try:
            return self.recognize(samples)
        except Exception:
            if time.monotonic() - asked >= EAR_STALL:
                self.until = time.monotonic() + EAR_REST
            raise


def load_refiner(language=None):
    """Load an operator-owned external ear once, before capture starts.

    The module exports create(language=None), returning a callable that accepts
    mono float32 16 kHz PCM and returns finished text. Private credentials and
    service protocols belong to that module, not this repository.
    """
    path = Path(os.environ.get("HARK_VOICE_EAR", "~/.hark/voice_ear.py")).expanduser()
    try:
        spec = importlib.util.spec_from_file_location("hark_external_voice_ear", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return RestingEar(module.create(language=language))
    except Exception:
        raise RuntimeError("voice ear unavailable; check HARK_VOICE_EAR or ~/.hark/voice_ear.py") from None


def _normalized(word):
    return re.sub(r"[^\w']", "", word.casefold().replace("’", "'"))


def _word_anchors(pending):
    words = []
    for speaker, part in pending.items():
        text, owners = "", []
        for token in part.tokens:
            text += token.text
            owners.extend([token] * len(token.text))
        for match in re.finditer(r"\S+", text):
            tokens = list({id(t): t for t in owners[match.start():match.end()]}.values())
            if _normalized(match.group()):
                words.append((speaker, tokens, match.group()))
    return sorted(words, key=lambda word: word[1][0].start)


def _align(text, anchors):
    """Align final words to streaming word anchors without inventing timestamps."""
    final = text.split()
    raw = [_normalized(word[2]) for word in anchors]
    rows = [list(range(len(final) + 1))]
    paths = []
    for i, a in enumerate(raw, 1):
        row, path = [i], [1]
        for j, b in enumerate(final, 1):
            cost, operation = min((rows[-1][j - 1] + (a != _normalized(b)), 0),
                                  (rows[-1][j] + 1, 1), (row[-1] + 1, 2))
            row.append(cost)
            path.append(operation)
        rows.append(row)
        paths.append(path)
    i, j = len(raw), len(final)
    indices = []
    while j:
        operation = paths[i - 1][j] if i else 2
        if operation == 0:
            indices.append(i - 1)
            i, j = i - 1, j - 1
        elif operation == 1:
            i -= 1
        else:
            indices.append(max(0, i - 1))
            j -= 1
    return [(anchors[index], word) for index, word in zip(reversed(indices), final)]


def quiet_boundary(audio, at):
    """Place an acoustic cut in a quiet 40 ms frame near an RNNT timestamp."""
    frame = SAMPLE_RATE // 25
    lo = max(audio.start, at - .6)
    samples = audio.slice(lo, at + .2)
    count = len(samples) // frame
    if not count:
        return at
    energy = np.mean(samples[:count * frame].reshape(count, frame) ** 2, axis=1)
    if energy.min() > 10 ** (-55 / 10):
        return at
    quiet = np.flatnonzero(energy <= max(energy.min() * 2, 1e-7))
    centers = lo + (quiet + .5) * frame / SAMPLE_RATE
    return float(centers[np.argmin(abs(centers - at))])


class RunQuiet:
    """Reload a launcher's quiet setting when its JSON file changes."""

    def __init__(self, path):
        self.path = Path(path).expanduser() if path else None
        self.stamp = None
        self.seconds = VOICE_GAP

    def read(self):
        if self.path is None:
            return self.seconds
        try:
            stamp = self.path.stat().st_mtime_ns
            if stamp == self.stamp:
                return self.seconds
            value = json.loads(self.path.read_text()).get("quiet_seconds", VOICE_GAP)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("invalid quiet setting")
        except (OSError, ValueError, AttributeError, OverflowError):
            stamp, value = None, VOICE_GAP
        self.stamp, self.seconds = stamp, float(value)
        return self.seconds


class RouteCuts:
    """New switch times from the launcher's route journal, an append-only JSONL.

    Each row's `t` is an ISO timestamp. A journal that shrinks is read from its
    new start.
    """

    def __init__(self, path):
        self.path = Path(path).expanduser() if path else None
        self.offset = self._size()

    def _size(self):
        try:
            return self.path.stat().st_size if self.path else 0
        except OSError:
            return 0

    def read(self):
        size = self._size()
        if size == self.offset:
            return []
        self.offset = min(self.offset, size) if size >= self.offset else 0
        with open(self.path, "rb") as stream:
            stream.seek(self.offset)
            lines = stream.read().splitlines(keepends=True)
        cuts = []
        for line in lines:
            if not line.endswith(b"\n"):
                break
            self.offset += len(line)
            try:
                cuts.append(datetime.fromisoformat(json.loads(line)["t"]).timestamp())
            except (ValueError, KeyError, TypeError):
                pass
        return cuts


class Partial:
    """Atomically replaced, private display text; never part of the transcript."""

    def __init__(self, transcript):
        self.path = Path(transcript).with_suffix(".partial")
        self.text = None
        self.write("")

    def write(self, text):
        if text == self.text:
            return
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                         prefix=self.path.name + ".", delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(text)
                stream.close()
                os.replace(temporary, self.path)
            finally:
                temporary.unlink(missing_ok=True)
        self.text = text


class VoiceTrack(Track):
    """Keep the recognizer's state, but don't run either model on an idle mic.

    A -50 dBFS RMS gate uses 100 ms frames, 300 ms pre-roll and a three-second
    tail to drain delayed tokens. The recognizer sees a compressed audio clock;
    token timestamps are translated back to the mic clock before turn detection.
    Speaker caches and identities survive idle periods.

    A run closes on quiet since its last word, and reaches the external ear only
    if the local recognizer heard words in it. Room noise and residual call audio
    stay above the acoustic gate, so the gate can't say when speech has stopped.
    """

    ordered_flush = True  # External word order survives overlapping diarizer anchors.
    frame = SAMPLE_RATE // 10
    preroll = 3 * frame
    tail = 3 * SAMPLE_RATE
    threshold = 10 ** (-50 / 20)

    def __init__(self, *args, gate=True, refiner=None, report=None, words=VOICE_WORDS, **kwargs):
        if refiner is not None and not gate:
            raise ValueError("an external voice ear requires the acoustic gate")
        kwargs.setdefault("gap", VOICE_GAP)
        kwargs.setdefault("max_len", VOICE_MAX_LEN)
        super().__init__(*args, **kwargs)
        self.gate = gate
        self.refiner = refiner
        self.words = words
        self.report = report or (lambda message: print(message, file=sys.stderr))
        self._run_start = None
        self._run_end = 0.0
        self._cut = None
        self._last_loud = 0.0
        self._ready = []
        if refiner is not None:
            from .voice import AudioBuffer

            self.audio = AudioBuffer(seconds=self.max_len + 15)
            self.flush = self._flush_voice
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
            if self.refiner is not None:
                self._ready.extend(self._flush_run())
        if final:
            self._step(samples[:0], final=True)
            if self.refiner is not None:
                self._ready.extend(self._flush_run(force=True))

    def cut(self, at):
        """Close the running speech at wall time `at`, so each side keeps its words."""
        self._cut = at - self.t0.timestamp() - self._wall_clock[-1][1]

    def add_wall_gap(self, at, origin):
        """Anchor resumed mic samples without moving speech buffered before sleep."""
        if not self._tail_left and not any(p.tokens for p in self.pending.values()):
            self._wall_clock = deque([self._wall_clock[-1]])
        self._wall_clock.append((at, origin - self.t0.timestamp()))

    @property
    def partial(self):
        """Raw words for display only; never a finalized transcript line."""
        lines = []
        for speaker, pending in self.pending.items():
            utterance = Track._utterance(self, speaker, pending.tokens) if pending.tokens else None
            if utterance:
                lines.append((utterance.start, utterance.text))
        return "\n".join(text for _, text in sorted(lines))

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

    def _flush_voice(self, force=False):
        out, self._ready = self._ready, []
        while self._run_start is not None:
            before = self._run_start
            out.extend(self._flush_run(force=force))
            if not force or self._run_start == before:
                break
        return out

    def _flush_run(self, force=False):
        if self._run_start is None:
            return []
        start = self._run_start
        clock = self.audio_samples / SAMPLE_RATE
        ends = [t.end for p in self.pending.values() for t in p.tokens]
        word_count = sum(len("".join(t.text for t in p.tokens).split()) for p in self.pending.values())
        # A run without words may be waiting on delayed ones, which the tail drains.
        # A short run waits longer, so the ear hears enough context to be accurate.
        last, wait = ((max(ends), self.gap if word_count >= VOICE_MIN_WORDS else 2 * self.gap) if ends
                      else (start, max(self.gap, self.tail / SAMPLE_RATE)))
        quiet = clock - last >= wait and self.processed >= last + .5
        preview = self.partial
        ended = VOICE_END.search(preview) and clock - last >= VOICE_END_HOLD
        early = word_count >= self.words or VOICE_NAME.search(preview) or ended
        if self._cut is not None and self._cut <= start:
            self._cut = None
        cutting = self._cut is not None and self.processed >= self._cut + CUT_DRAIN
        if not force and not quiet and not early and not cutting and self.processed - start < self.max_len:
            return []
        end = min(clock, self.processed if not force else clock, start + self.max_len)
        end = min(end, next((b for b, _ in self._wall_clock if start < b <= end), end))
        if cutting:
            end, self._cut = min(end, self._cut), None
        if end < clock and end - start >= self.max_len - .01:
            end = min(end, max(start, quiet_boundary(self.audio, end)))
        # Delayed RNNT anchors can overshoot the physical PCM endpoint. When a
        # run closes, all its anchors belong to it, including those future times.
        if not any(p.tokens for p in self.pending.values()):
            end = max(start, end - 2)  # Delayed tokens for this window's tail still count.
        complete = end >= self._last_loud
        chosen = {speaker: _Pending([t for t in part.tokens if complete or t.start < end])
                  for speaker, part in self.pending.items()}
        anchors = _word_anchors(chosen)
        samples = self.audio.slice(start, end)
        # Preroll can straddle a sleep boundary; an empty prefix must still advance.
        speech = anchors and samples.size and float(np.max(samples * samples)) >= self.threshold ** 2
        try:
            text = self.refiner(samples).strip() if speech else ""
            if not text:
                raise ValueError("empty finished text")
        except Exception as error:
            # External adapters may carry secrets in exception messages or URLs.
            if speech and not isinstance(error, EarResting):
                self.report(f"voice ear failed ({type(error).__name__}); writing raw recognition")
            output = [u for speaker, part in chosen.items()
                      if part.tokens and (u := self._utterance(speaker, part.tokens))]
        else:
            output, groups = [], []
            for (speaker, tokens, _), word in _align(text, anchors):
                if not groups or groups[-1][0] != speaker:
                    groups.append((speaker, [], []))
                groups[-1][1].extend(tokens)
                groups[-1][2].append(word)
            for speaker, tokens, words in groups:
                unique = sorted({id(t): t for t in tokens}.values(), key=lambda t: t.start)
                utterance = self._utterance(speaker, unique)
                utterance.text = " ".join(words)
                output.append(utterance)
        for part in self.pending.values():
            part.tokens = [] if complete else [t for t in part.tokens if t.start >= end]
        self._run_end = end
        self._run_start = None if complete else end
        self._prune_wall_clock()
        return output

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
        self._prune_wall_clock()
        return out

    def _prune_wall_clock(self):
        safe = self._mic_time(super().processed) - 1
        safe = min((token.start for p in self.pending.values() for token in p.tokens), default=safe)
        while len(self._wall_clock) > 1 and self._wall_clock[1][0] < safe:
            self._wall_clock.popleft()

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
                if self.refiner is None or (self._run_start is not None and token.start >= self._run_end):
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
                if self._run_start is None:
                    if self.refiner is not None:
                        onset = max(0, (at - self.preroll) / SAMPLE_RATE)
                        self._run_start = max(self._run_end, self.audio.start, onset)
                self._last_loud = (at + chunk.size) / SAMPLE_RATE
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
