"""Cost-aware gated Gradium ASR with bounded request replay on the source clock."""

import asyncio
import base64
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import math
import os
from pathlib import Path
import queue
import stat
import threading
import time
from urllib.request import Request, urlopen

import numpy as np
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus

from .capture import SAMPLE_RATE, log, to_pcm16
from .transcript import Utterance
from .voice import AudioArchive, OnlineCluster

URL = "wss://api.gradium.ai/api/speech/asr"
CREDITS_URL = "https://api.gradium.ai/api/usages/credits"
FRAME = 1280


class GradiumError(RuntimeError):
    pass


class AuthenticationError(GradiumError):
    pass


def api_key(path=None):
    """Read an environment key or an owner-only one-line configuration file."""
    key = os.environ.get("GRADIUM_API_KEY")
    if key is None:
        path = Path(path or Path.home() / ".config/hark/gradium.key")
        try:
            with path.open() as file:
                info = os.fstat(file.fileno())
                if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
                    raise GradiumError(f"key file must have mode 600: chmod 600 {path}")
                if info.st_uid != os.getuid():
                    raise GradiumError(f"key file must belong to the current user: {path}")
                key = file.read()
        except FileNotFoundError as error:
            raise GradiumError(f"missing API key: set GRADIUM_API_KEY or create {path} (mode 600)") from error
        except OSError as error:
            raise GradiumError(f"cannot read key file {path}: {error.strerror}") from error
    key = key.strip()
    if not key or any(c.isspace() for c in key):
        raise GradiumError("API key must be one non-empty line without whitespace")
    return key


def credits_left(key, url=CREDITS_URL):
    """Read the documented CreditsSummary.remaining_credits field; metering is optional."""
    try:
        with urlopen(Request(url, headers={"x-api-key": key}), timeout=3) as response:
            value = json.load(response)["remaining_credits"]
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError("remaining_credits is not an integer")
        return value
    except Exception as error:
        log(f"gradium: credits unavailable ({type(error).__name__})")
        return None


def _union(spans):
    """Count each source sample once while keeping omitted gaps disjoint."""
    merged = []
    for a, b in sorted(spans):
        if b <= a:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return merged


@dataclass
class _Mapping:
    cloud: int
    source: int
    count: int


@dataclass
class _Request:
    start: int
    frames: list = field(default_factory=list)  # None marks a speech flush, not EOS.
    runs: list = field(default_factory=list)
    samples: int = 0
    wire_samples: int = 0
    done: bool = False
    rotate: bool = False
    horizons: dict = field(default_factory=dict)  # Committed cloud-clock ends by stream.

    def append(self, source, frame, valid):
        cloud = self.wire_samples
        if (self.runs and self.runs[-1].cloud + self.runs[-1].count == cloud
                and self.runs[-1].source + self.runs[-1].count == source):
            self.runs[-1].count += valid
        else:
            self.runs.append(_Mapping(cloud, source, valid))
        self.samples += valid
        self.wire_samples += FRAME
        # Publish mapping before making the frame available to the sender.
        self.frames.append(frame)

    def project(self, start, end):
        """Intersect a cloud interval with valid samples, excluding omitted source gaps."""
        lo, hi = round(start * SAMPLE_RATE), round(end * SAMPLE_RATE)
        spans = []
        point = self.start
        for run in self.runs:
            if lo > run.cloud:
                point = run.source + min(lo - run.cloud, run.count)
            a, b = max(lo, run.cloud), min(hi, run.cloud + run.count)
            if b > a:
                spans.append(((run.source + a - run.cloud) / SAMPLE_RATE,
                              (run.source + b - run.cloud) / SAMPLE_RATE))
        if spans:
            return spans[0][0], spans[-1][1], spans
        return point / SAMPLE_RATE, point / SAMPLE_RATE, []


class GradiumTrack:
    """One alternate track in hark's shared capture and sink lifecycle.

    Short dialogue gaps share one request without uploading the omitted quiet.
    Piecewise integer-sample mappings project finalized cloud words onto the
    full source clock. Replay suppresses each stream's committed cloud horizon.
    """

    def __init__(self, name, *, key, language=None, cluster=None, fixed_speaker=None, url=URL,
                 rms=0.001, preroll=0.32, hangover=0.8, max_duration=58,
                 queue_size=None, backlog_seconds=120, retries=2, backoff=0.5, timeout=8,
                 shutdown_timeout=30, realtime=True, phrase_seconds=4.0):
        self.name, self.key, self.url = name, key, url
        self.fixed_speaker = fixed_speaker
        self.language = (language or "any").lower().split("-")[0]
        if self.language not in {"en", "fr", "any"}:
            raise GradiumError("language must be en, fr or any")
        self.cluster = cluster if cluster is not None else OnlineCluster(minimum_duration=4)
        self.rms, self.hangover = rms, max(1, round(hangover * SAMPLE_RATE / FRAME))
        # Reserve decoder-tail headroom in approximately minute-long requests.
        self.max_frames = max(1, int(max_duration * SAMPLE_RATE / FRAME))
        self.pre = deque(maxlen=round(preroll * SAMPLE_RATE / FRAME))
        self.jobs = queue.Queue(maxsize=queue_size or math.ceil(backlog_seconds * SAMPLE_RATE / FRAME))
        self.results = queue.Queue(maxsize=64)
        self.backlog_limit = round(backlog_seconds * SAMPLE_RATE)
        self.backlog_samples = 0
        self.backlog_lock = threading.Lock()
        self.last_progress = time.monotonic()
        self.cluster_failed = False
        self.last_speaker = "S1"
        self.phrase_seconds = phrase_seconds
        self.phrases = {}
        self.retries, self.backoff, self.timeout = retries, backoff, timeout
        self.shutdown_timeout, self.realtime = shutdown_timeout, realtime
        self.audio = AudioArchive()
        self.t0 = datetime.now()
        self.audio_samples = self.position = 0
        self.tail = np.zeros(0, np.float32)
        self.request = None
        self.active = False
        self.quiet = 0
        self.sent_seconds = 0.0
        self.error = None
        self.degraded = False
        self.notices = queue.Queue(maxsize=64)
        self.missed = None
        self.ready = threading.Event()
        self.stopping = threading.Event()
        self.cancel = threading.Event()
        self.finished = threading.Event()
        self.thread = None
        self.completed = []

    @property
    def processed(self):
        return self.audio_samples / SAMPLE_RATE

    def start(self, stop=None):
        if not self.fixed_speaker and hasattr(self.cluster, "warm"):
            try:
                self.cluster.warm()
            except Exception as error:
                self.cluster_failed = True
                log(f"gradium: speaker warmup unavailable; keeping S1 ({type(error).__name__})")
        if stop and stop.is_set():
            return
        self.thread = threading.Thread(target=self._run, name=f"gradium-{self.name}", daemon=True)
        self.thread.start()
        deadline = time.monotonic() + (self.retries + 1) * (2 * self.timeout + self.backoff * 4)
        while not self.ready.wait(0.05):
            if stop and stop.is_set():
                self.abort()
                return
            if time.monotonic() >= deadline:
                self.abort()
                raise GradiumError("startup timed out")
        self.check()

    def check(self):
        if self.error:
            raise self.error

    def feed(self, samples, final=False):
        self.check()
        if samples.size:
            self.audio.append(self.audio_samples / SAMPLE_RATE, samples)
            self.audio_samples += samples.size
            self.tail = np.concatenate((self.tail, samples))
            while self.tail.size >= FRAME:
                frame, self.tail = self.tail[:FRAME].copy(), self.tail[FRAME:]
                self._frame(frame)
        if final:
            if self.tail.size:
                self._frame(np.pad(self.tail, (0, FRAME - self.tail.size)), valid=self.tail.size)
                self.tail = self.tail[:0]
            self._end_request()
            self.stopping.set()
            self._report_missed()
            if self.degraded:
                self.abort()
            self.last_progress = time.monotonic()
            finish_started = self.last_progress
            while not self.finished.is_set():
                self._collect()
                if ((self.degraded and time.monotonic() - finish_started > 0.5)
                        or time.monotonic() - self.last_progress > self.shutdown_timeout):
                    break
                time.sleep(0.01)
            if not self.finished.is_set():
                log("gradium: stop abandons uncommitted recognition; saved audio is retained")
                self.abort()
            if self.thread:
                self.thread.join()
            self._collect()
            self.check()

    def _frame(self, samples, valid=FRAME):
        self.check()
        voiced = float(np.sqrt(np.mean(samples * samples))) >= self.rms
        if self.request and self.request.rotate:
            self._end_request()
        self.quiet = 0 if voiced else self.quiet + FRAME
        self.active = self.active or voiced
        if self.active:
            if self.request is None:
                # Reserve one frame for the current input even with a tiny request cap.
                pre = list(self.pre)[-(self.max_frames - 1):] if self.max_frames > 1 else []
            else:
                pre = list(self.pre)
            self.pre.clear()
            for position, frame, count in pre:
                self._append(position, frame, count)
            self._append(self.position, samples, valid)
            if not voiced and self.quiet >= self.hangover * FRAME:
                self._speech_flush()
                self.active = False
        else:
            # Only discarded quiet is eligible for the next speech's preroll.
            self.pre.append((self.position, samples, valid))
        self.position += FRAME
        if self.request and self.request.wire_samples >= self.max_frames * FRAME:
            self._end_request()

    def _append(self, position, samples, valid):
        if self.request and self.request.wire_samples >= self.max_frames * FRAME:
            self._end_request()
        if not self._reserve(FRAME):
            self._end_request()
            self._miss(position, valid)
            return
        if self.request is None:
            request = _Request(position)
            if not self._enqueue(request):
                with self.backlog_lock:
                    self.backlog_samples -= FRAME
                self._miss(position, valid)
                return
            self.request = request
        self._report_missed()
        self.request.append(position, samples, valid)

    def _speech_flush(self):
        if self.request and self.request.frames and self.request.frames[-1] is not None:
            self.request.frames.append(None)

    def _end_request(self):
        if self.request:
            self._speech_flush()
            self.request.done = True
            self.request = None

    def _reserve(self, count):
        while True:
            self.check()
            with self.backlog_lock:
                if self.backlog_samples + count <= self.backlog_limit:
                    self.backlog_samples += count
                    return True
            self._collect()
            if self.degraded or self.realtime or self.cancel.is_set() or time.monotonic() - self.last_progress > self.shutdown_timeout:
                return False
            time.sleep(0.01)

    def _enqueue(self, burst):
        while True:
            self.check()
            self._collect()
            try:
                self.jobs.put_nowait(burst)
                return True
            except queue.Full:
                if self.degraded or self.realtime or self.cancel.is_set() or time.monotonic() - self.last_progress > self.shutdown_timeout:
                    return False
                time.sleep(0.01)

    def _notice(self, text):
        try:
            self.notices.put_nowait(text)
        except queue.Full:
            pass  # Status history is bounded independently of audio and accepted text.

    def take_notices(self):
        out = []
        while True:
            try:
                out.append(self.notices.get_nowait())
            except queue.Empty:
                return out

    def _degrade(self, lost):
        if self.degraded != lost:
            self.degraded = lost
            self._notice(f"gradium {'lost' if lost else 'back'} at {datetime.now():%H:%M:%S}")

    def _miss(self, position, valid):
        if self.missed is None:
            log("gradium: STT backlog full; omitting recognition only, source audio is retained")
            self.missed = [position, position + valid]
        else:
            self.missed[1] = position + valid

    def _report_missed(self):
        if self.missed is not None:
            start, end = self.missed
            text = f"gradium missed recognition {start / SAMPLE_RATE:.2f}–{end / SAMPLE_RATE:.2f} s on {self.name}"
            log(text)
            self._notice(text)
            self.missed = None

    def _collect(self):
        while True:
            try:
                burst, segments = self.results.get_nowait()
            except queue.Empty:
                return
            if segments is None:
                for key in list(self.phrases):
                    if key[0] == id(burst):
                        self._phrase(key)
                continue
            for text, start, end, stream in segments:
                start, end, spans = burst.project(start, end)
                text = text.strip()
                if not text:
                    continue
                key = id(burst), stream
                words = self.phrases.setdefault(key, [])
                if words and start - words[-1][2] >= 0.8:
                    self._phrase(key)
                    words = self.phrases.setdefault(key, [])
                words.append((text, start, end, spans))
                intervals = _union(span for _, _, _, spans in words for span in spans)
                if (sum(b - a for a, b in intervals) >= self.phrase_seconds
                        or max(w[2] for w in words) - min(w[1] for w in words) >= 8):
                    self._phrase(key)

    def _phrase(self, key):
        words = self.phrases.pop(key, [])
        if not words:
            return
        spans = _union(span for _, _, _, intervals in words for span in intervals)
        start = spans[0][0] if spans else min(w[1] for w in words)
        end = spans[-1][1] if spans else max(w[2] for w in words)
        clips = [self.audio.slice(a, b) for a, b in spans]
        samples = np.concatenate(clips) if clips else np.zeros(0, np.float32)
        slot = self.fixed_speaker or self.last_speaker
        if not self.fixed_speaker and not self.cluster_failed:
            try:
                slot = self.cluster.assign(samples)
            except Exception as error:
                self.cluster_failed = True
                log(f"gradium: speaker clustering disabled; keeping {slot} ({type(error).__name__})")
        self.last_speaker = slot
        text = ""
        for word, _, _, _ in words:
            text += (" " if text and word[0] not in ".,;:!?" else "") + word
        self.completed.append(Utterance(self.name, slot, start, end,
                                        self.t0 + timedelta(seconds=start), text, speech=spans))

    def flush(self, force=False):
        self._collect()
        if force:
            for key in list(self.phrases):
                self._phrase(key)
        out, self.completed = self.completed, []
        return out

    def abort(self):
        # Make room for the receiver's pending word before cancelling its bounded collector.
        self._collect()
        self.cancel.set()
        if self.thread:
            self.thread.join(timeout=self.timeout + 2)
        self._collect()

    def close(self):
        self.abort()
        self.audio.close()

    def _run(self):
        try:
            asyncio.run(self._supervise())
        except asyncio.CancelledError:
            pass
        except BaseException as error:
            cls = AuthenticationError if isinstance(error, AuthenticationError) else GradiumError
            self.error = cls(str(error).replace(self.key, "[redacted]"))
        finally:
            self.ready.set()
            self.finished.set()

    async def _supervise(self):
        worker = asyncio.create_task(self._worker())
        try:
            while not worker.done():
                if self.cancel.is_set():
                    worker.cancel()
                    break
                await asyncio.sleep(0.01)
            await worker
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

    async def _open(self):
        ws = await connect(self.url, additional_headers={"x-api-key": self.key},
                           open_timeout=self.timeout, close_timeout=1, max_size=65536,
                           max_queue=16, proxy=None)
        try:
            await ws.send(json.dumps({"type": "setup", "model_name": "default",
                                      "input_format": "pcm_16000",
                                      "json_config": {"language": self.language}}))
            msg = json.loads(await asyncio.wait_for(ws.recv(), self.timeout))
            self._error(msg)
            if (msg.get("type") != "ready" or not isinstance(msg.get("sample_rate"), int)
                    or msg["sample_rate"] <= 0 or not isinstance(msg.get("frame_size"), int)
                    or msg["frame_size"] <= 0):
                raise GradiumError("invalid ready sample_rate/frame_size")
            self.last_progress = time.monotonic()
            log(f"gradium: ready (delay {msg.get('delay_in_frames')} frames)")
            return ws
        except BaseException:
            await ws.close()
            raise

    def _error(self, msg):
        if msg.get("type") == "error":
            reason = str(msg.get("message", "")).lower()
            auth = msg.get("code") in {401, 403} or (
                msg.get("code") == 1008 and any(word in reason for word in ("authentication", "api key", "api_key", "unauthorized")))
            cls = AuthenticationError if auth else GradiumError
            raise cls(f"{msg.get('message', 'server error')} (code {msg.get('code')})")

    async def _retry(self, operation):
        attempt = 0
        while not self.cancel.is_set():
            try:
                return await operation()
            except AuthenticationError:
                raise
            except InvalidStatus as error:
                if error.response.status_code in {401, 403}:
                    raise AuthenticationError(f"authentication refused (HTTP {error.response.status_code})") from error
                why = f"HTTP {error.response.status_code}"
            except Exception as error:
                why = str(error) or type(error).__name__
            self._degrade(True)
            self.ready.set()  # Capture starts after the first failure, not after service recovery.
            if self.stopping.is_set() and attempt >= self.retries:
                raise asyncio.CancelledError
            delay = min(30.0, self.backoff * 2 ** min(attempt, 16))
            if self.stopping.is_set():
                delay = min(delay, 0.05)
            why = why.replace(self.key, "[redacted]")
            log(f"gradium: reconnect in {delay:.2f} s ({why})")
            attempt += 1
            await asyncio.sleep(delay)  # The supervisor cancels this wait promptly on stop.
        raise asyncio.CancelledError

    async def _worker(self):
        # Startup validates authentication even when the entire meeting is silent.
        ws = await self._retry(self._open)
        self._degrade(False)
        await ws.close()
        ws = None
        self.ready.set()
        try:
            while not self.cancel.is_set():
                try:
                    burst = self.jobs.get_nowait()
                except queue.Empty:
                    if self.stopping.is_set():
                        break
                    await asyncio.sleep(0.01)
                    continue
                async def transcribe():
                    nonlocal ws
                    if ws is None:
                        ws = await self._open()
                    try:
                        return await self._session(ws, burst)
                    finally:
                        await ws.close()
                        ws = None
                await self._retry(transcribe)
                while not self.cancel.is_set():
                    try:
                        self.results.put_nowait((burst, None))
                        break
                    except queue.Full:
                        await asyncio.sleep(0.01)
                with self.backlog_lock:
                    self.backlog_samples -= burst.wire_samples
                burst.frames.clear()
                self.last_progress = time.monotonic()
        finally:
            if ws:
                await ws.close()

    async def _session(self, ws, burst):
        flushed = asyncio.Event()
        sent_eos = asyncio.Event()
        pending = {}
        inferred = {}  # Published flush tails awaiting a possible late end_text, by stream.
        limited = False
        characters = messages = 0
        replay_horizons = burst.horizons.copy()
        processed = 0.0
        sent_wire = submitted = settled = 0
        flush_id = 0
        pending_flush = None
        advanced = time.monotonic()

        def waiting():
            return pending_flush is not None or sent_eos.is_set() or sent_wire > settled

        def progress():
            nonlocal advanced
            advanced = self.last_progress = time.monotonic()

        async def send(message):
            await asyncio.wait_for(ws.send(json.dumps(message)), self.timeout)

        async def boundary():
            while not self.cancel.is_set():
                try:
                    self.results.put_nowait((burst, None))
                    return
                except queue.Full:
                    await asyncio.sleep(0.01)

        async def sender():
            nonlocal sent_wire, submitted, flush_id, pending_flush
            i = 0
            while not self.cancel.is_set():
                if i < len(burst.frames):
                    frame = burst.frames[i]
                    if not waiting():
                        # A new input/flush gets a fresh budget after legitimate idle.
                        progress()
                    if frame is None:
                        flush_id += 1
                        pending_flush = flush_id
                        flushed.clear()
                        await send({"type": "flush", "flush_id": flush_id})
                        await flushed.wait()
                    else:
                        sent_wire += FRAME
                        submitted = min(sent_wire, burst.samples)
                        await send({"type": "audio", "audio": base64.b64encode(to_pcm16(frame).tobytes()).decode()})
                        self.sent_seconds += FRAME / SAMPLE_RATE
                    i += 1
                    await asyncio.sleep(0)
                elif burst.done:
                    break
                else:
                    await asyncio.sleep(0.01)
            if self.cancel.is_set():
                raise GradiumError("cancelled")
            progress()
            sent_eos.set()
            await send({"type": "end_of_stream"})

        async def publish(text, start, end, stream, healthy=True):
            duration = submitted / SAMPLE_RATE
            if not (math.isfinite(start) and math.isfinite(end)
                    and 0 <= start <= end <= duration + 0.081):
                log("gradium: segment timestamps clamped to submitted audio")
            start = min(max(start if math.isfinite(start) else 0, 0), duration)
            end = min(max(end if math.isfinite(end) else duration, start), duration)
            progress()
            if start >= replay_horizons.get(stream, 0) - 1 / SAMPLE_RATE:
                while True:
                    try:
                        self.results.put_nowait((burst, [(text, start, end, stream)]))
                        burst.horizons[stream] = max(end, burst.horizons.get(stream, 0))
                        if healthy:
                            self._degrade(False)
                        return
                    except queue.Full:
                        if self.cancel.is_set():
                            return
                        await asyncio.sleep(0.01)
            else:
                log(f"gradium: replay skips segment {start:.2f}–{end:.2f} before committed horizon "
                    f"{replay_horizons[stream]:.2f}")

        async def receiver():
            nonlocal characters, messages, processed, settled, pending_flush, limited
            while True:
                # The semantic watchdog, not socket traffic, bounds outstanding work.
                msg = json.loads(await ws.recv())
                self._error(msg)
                kind, stream = msg.get("type"), msg.get("stream_id", 0)
                if kind == "text":
                    if limited:
                        continue
                    characters += len(msg["text"])
                    messages += 1
                    if characters >= 1200:
                        burst.rotate = True
                    if characters > 10000 or messages > 2000:
                        log("gradium: transcript bound reached; closing this request")
                        limited = True
                        burst.rotate = burst.done = True
                        for slot, (text, start) in list(pending.items()):
                            await publish(text, start, submitted / SAMPLE_RATE, slot)
                        pending.clear()
                        continue
                    progress()
                    next_start = float(msg["start_s"])
                    if stream in pending:
                        text, start = pending[stream]
                        if next_start == start:
                            pending[stream] = (text + msg["text"], start)
                            continue
                        await publish(text, start, next_start, stream)
                    pending[stream] = (msg["text"], next_start)
                elif kind == "end_text":
                    if stream not in pending:
                        inferred.pop(stream, None)
                        log("gradium: end_text without pending text ignored")
                        continue
                    stop = float(msg["stop_s"])
                    if stream in inferred and stop <= pending[stream][1]:
                        log("gradium: late end_text for an inferred flush tail ignored")
                        inferred.pop(stream, None)
                        continue
                    inferred.pop(stream, None)
                    text, start = pending.pop(stream)
                    await publish(text, start, stop, stream)
                    if characters >= 1200:
                        burst.rotate = True
                elif kind == "step":
                    duration = float(msg.get("total_duration_s", 0))
                    if duration > processed:
                        processed = duration
                        settled = max(settled, min(sent_wire, round(duration * SAMPLE_RATE)))
                        progress()
                elif kind == "flushed" and pending_flush is not None and msg.get("flush_id") == pending_flush:
                    progress()
                    settled = sent_wire
                    self._degrade(False)
                    for slot, (text, start) in list(pending.items()):
                        log("gradium: pending text end inferred at speech flush")
                        await publish(text, start, submitted / SAMPLE_RATE, slot)
                        inferred[slot] = start
                    pending.clear()
                    await boundary()
                    pending_flush = None
                    flushed.set()
                elif kind == "end_of_stream":
                    if not sent_eos.is_set():
                        for slot, (text, start) in list(pending.items()):
                            await publish(text, start, submitted / SAMPLE_RATE, slot)
                        pending.clear()
                        raise GradiumError("unexpected EOS; restarting request")
                    duration = submitted / SAMPLE_RATE
                    for stream, (text, start) in list(pending.items()):
                        log("gradium: final text end inferred from submitted duration")
                        await publish(text, start, duration, stream)
                    pending.clear()
                    return
                # step carries semantic VAD, not a phrase boundary or a source offset.

        async def watchdog():
            while not consumer.done():
                if self.cancel.is_set():
                    raise GradiumError("cancelled")
                if waiting() and time.monotonic() - advanced > self.timeout:
                    raise TimeoutError("no ASR progress")
                await asyncio.sleep(0.01)

        producer, consumer = asyncio.create_task(sender()), asyncio.create_task(receiver())
        monitor = asyncio.create_task(watchdog())
        try:
            await asyncio.gather(producer, consumer, monitor)
            return consumer.result()
        finally:
            for task in (producer, consumer, monitor):
                task.cancel()
            await asyncio.gather(producer, consumer, monitor, return_exceptions=True)
            for stream, (text, start) in list(pending.items()):
                log("gradium: pending text end inferred at request reset")
                await publish(text, start, submitted / SAMPLE_RATE, stream, healthy=False)
