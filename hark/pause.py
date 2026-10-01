"""Mute the mic track while a dictation app records or the user has run `hark pause`.

A watcher thread polls CoreAudio's per-process input flag (through ctypes) and the manual
pause flag, and turns each pause into a wall-clock mute interval. The mic is held back by
MIC_LOOKBACK_SEC, so an interval opened at detection still reaches the audio captured
just before it. Muted samples become zeros, so the track and its WAV keep their timeline.
"""

import ctypes as C
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .capture import SAMPLE_RATE, log

MIC_LOOKBACK_SEC = 0.300
AUTO_RESUME_TAIL_SEC = 0.300
PAUSE_POLL_SEC = 0.075
DEFAULT_PAUSE_FOR = "aquavoice"


def fourcc(value):
    return int.from_bytes(value.encode("ascii"), "big")


class PropertyAddress(C.Structure):
    _fields_ = [("selector", C.c_uint32), ("scope", C.c_uint32), ("element", C.c_uint32)]


@dataclass(frozen=True)
class AudioProcess:
    object_id: int
    pid: int
    bundle_id: str
    running_input: bool


class CoreAudioProcesses:
    """Read HAL process objects, including idle ones, using CoreAudio/CoreFoundation ctypes."""

    def __init__(self):
        if sys.platform != "darwin":
            raise OSError("CoreAudio process detection requires macOS")
        self.audio = C.CDLL("/System/Library/Frameworks/CoreAudio.framework/CoreAudio")
        self.cf = C.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
        self.audio.AudioObjectGetPropertyDataSize.argtypes = [
            C.c_uint32, C.POINTER(PropertyAddress), C.c_uint32, C.c_void_p, C.POINTER(C.c_uint32)]
        self.audio.AudioObjectGetPropertyDataSize.restype = C.c_int32
        self.audio.AudioObjectGetPropertyData.argtypes = [
            C.c_uint32, C.POINTER(PropertyAddress), C.c_uint32, C.c_void_p,
            C.POINTER(C.c_uint32), C.c_void_p]
        self.audio.AudioObjectGetPropertyData.restype = C.c_int32
        self.cf.CFStringGetLength.argtypes = [C.c_void_p]
        self.cf.CFStringGetLength.restype = C.c_long
        self.cf.CFStringGetMaximumSizeForEncoding.argtypes = [C.c_long, C.c_uint32]
        self.cf.CFStringGetMaximumSizeForEncoding.restype = C.c_long
        self.cf.CFStringGetCString.argtypes = [C.c_void_p, C.c_void_p, C.c_long, C.c_uint32]
        self.cf.CFStringGetCString.restype = C.c_bool
        self.cf.CFRelease.argtypes = [C.c_void_p]
        self.cf.CFRelease.restype = None

    @staticmethod
    def _check(status):
        if status:
            raise OSError(f"CoreAudio property read failed (OSStatus {status})")

    def _address(self, selector):
        return PropertyAddress(fourcc(selector), fourcc("glob"), 0)

    def _get(self, object_id, selector, value):
        address = self._address(selector)
        size = C.c_uint32(C.sizeof(value))
        self._check(self.audio.AudioObjectGetPropertyData(
            object_id, C.byref(address), 0, None, C.byref(size), C.byref(value)))
        return value

    def object_ids(self):
        address = self._address("prs#")  # kAudioHardwarePropertyProcessObjectList
        # The HAL list can grow between the size and data calls.
        for attempt in range(3):
            size = C.c_uint32()
            self._check(self.audio.AudioObjectGetPropertyDataSize(
                1, C.byref(address), 0, None, C.byref(size)))
            objects = (C.c_uint32 * (size.value // C.sizeof(C.c_uint32)))()
            try:
                self._get(1, "prs#", objects)
            except OSError:
                if attempt == 2:
                    raise
                continue
            # A shrinking list leaves unused slots in the allocated array.
            return [int(obj) for obj in objects if obj]

    def bundle_id(self, object_id):
        value = self._get(object_id, "pbid", C.c_void_p()).value
        if not value:
            return ""
        try:
            encoding = 0x08000100  # kCFStringEncodingUTF8
            size = self.cf.CFStringGetMaximumSizeForEncoding(
                self.cf.CFStringGetLength(value), encoding) + 1
            buffer = C.create_string_buffer(size)
            if not self.cf.CFStringGetCString(value, buffer, size, encoding):
                raise OSError("CoreAudio bundle ID is not a UTF-8 CFString")
            return buffer.value.decode("utf-8")
        finally:
            self.cf.CFRelease(value)

    def read(self, idle=True):
        """Every process object, or with `idle=False` only those capturing input."""
        processes = []
        for object_id in self.object_ids():
            try:
                running = bool(self._get(object_id, "piri", C.c_uint32()).value)
                if not (running or idle):
                    continue
                pid = self._get(object_id, "ppid", C.c_int32()).value
                bundle = self.bundle_id(object_id)
            except OSError:  # a process may disappear while its properties are being read
                continue
            processes.append(AudioProcess(object_id, pid, bundle, running))
        return processes


def parse_patterns(value):
    patterns = tuple(part.strip().casefold() for part in value.split(",") if part.strip())
    if "none" in patterns:
        if len(patterns) != 1:
            raise ValueError("none must be used alone in --pause-for")
        return ()
    if not patterns:
        raise ValueError("--pause-for needs a bundle ID pattern or none")
    return patterns


def watched_bundles(processes, patterns, pid=None):
    """Case-insensitive substring matching; CoreSpeech needs an explicit corespeech pattern."""
    pid = os.getpid() if pid is None else pid
    return tuple(sorted({p.bundle_id for p in processes
                         if p.pid != pid and p.running_input and p.bundle_id
                         and any(pattern in p.bundle_id.casefold()
                                 and ("corespeech" not in p.bundle_id.casefold()
                                      or "corespeech" in pattern)
                                 for pattern in patterns)}))


class ManualPause:
    """`hark pause` creates a flag file that the live session polls; `hark resume` removes it."""

    def __init__(self, home):
        self.path = Path(home) / "paused"

    def read(self):
        return self.path.exists()

    def set(self, paused):
        if paused:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.touch()
        else:
            self.path.unlink(missing_ok=True)


@dataclass(frozen=True)
class PauseEvent:
    reason: str  # manual or automatic
    paused: bool
    at: float
    since: float
    bundles: tuple[str, ...] = ()


@dataclass
class MuteInterval:
    start: float
    end: float | None = None


class PauseState:
    """Pure transition/interval logic; manual and automatic reasons independently compose."""

    def __init__(self):
        self.active = {}
        self.intervals = []

    def update(self, now, manual, automatic=()):
        events = []
        for reason, enabled, bundles in (("manual", manual, ()),
                                         ("automatic", bool(automatic), tuple(automatic))):
            previous = self.active.get(reason)
            if enabled and previous is None:
                interval = MuteInterval(now - MIC_LOOKBACK_SEC)
                self.intervals.append(interval)
                self.active[reason] = now, interval, bundles
                events.append(PauseEvent(reason, True, now, now, bundles))
            elif not enabled and previous is not None:
                since, interval, previous_bundles = self.active.pop(reason)
                interval.end = now + (AUTO_RESUME_TAIL_SEC if reason == "automatic" else 0)
                events.append(PauseEvent(reason, False, now, since, previous_bundles))
            elif enabled:
                since, interval, _ = previous
                self.active[reason] = since, interval, bundles
        return events

    def gate(self, samples, start):
        """Preserve the PCM grid and timeline, replacing samples in mute intervals with zeros."""
        out = samples.copy()
        for interval in self.intervals:
            first = max(0, math.ceil((interval.start - start) * SAMPLE_RATE - 1e-6))
            last = (out.size if interval.end is None else
                    min(out.size, math.ceil((interval.end - start) * SAMPLE_RATE - 1e-6)))
            if first < last:
                out[first:last] = 0
        end = start + samples.size / SAMPLE_RATE
        self.intervals = [i for i in self.intervals if i.end is None or i.end > end]
        return out


class MicGate:
    """Hold the mic's newest 300 ms, timed by when the device delivered it, then emit every
    sample once (on the final flush too), zeroed inside mute intervals.

    A sample's wall time comes from `source.heard`, not from its index since `anchor`:
    the device runs up to `LATE_OK` behind the padded timeline (0.1 s from the start,
    0.5-1 s after any padding), which would otherwise shift every mute interval later.
    """

    def __init__(self, source):
        self.source = source
        self.emitted = 0
        self.pending = np.zeros(0, np.float32)

    def feed(self, samples, now, state, final=False):
        self.pending = np.concatenate((self.pending, samples))
        index, arrived = self.source.heard
        start = arrived - (index - self.emitted) / SAMPLE_RATE  # wall time of pending[0]
        ready = math.floor((now - MIC_LOOKBACK_SEC - start) * SAMPLE_RATE + 1e-6)
        count = self.pending.size if final else min(self.pending.size, max(0, ready))
        out = state.gate(self.pending[:count], start)
        self.pending = self.pending[count:]
        self.emitted += count
        return out


class PauseMonitor:
    """Poll on its own thread, so a pause is timed when it happens even while the main loop
    is busy with inference."""

    def __init__(self, home, patterns):
        self.manual = ManualPause(home)
        self.patterns = patterns
        self.reader = None
        self.state = PauseState()
        self.events = []
        self.automatic = ()
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.thread = None
        self.warned = set()

    def _warn(self, kind, error):
        if kind not in self.warned:
            self.warned.add(kind)
            log(f"pause: {kind}: {error}")

    def start(self):
        """A session starts unpaused: a manual pause left by an earlier session is cleared."""
        if self.manual.read():
            self.manual.set(False)
            log("pause: cleared a manual pause left from an earlier session")
        if self.patterns:
            try:
                self.reader = CoreAudioProcesses()
            except OSError as error:
                self._warn("automatic detection unavailable", error)
        self.poll()
        self.thread = threading.Thread(target=self._watch, name="hark-pause", daemon=True)
        self.thread.start()

    def poll(self):
        try:
            manual = self.manual.read()
        except OSError as error:
            self._warn("cannot read the manual pause flag", error)
            manual = "manual" in self.state.active
        if self.reader:
            try:
                self.automatic = watched_bundles(self.reader.read(idle=False), self.patterns)
            except Exception as error:  # noqa: BLE001 — keep the last observation, never leak
                self._warn("automatic detection failed", error)
        with self.lock:
            self.events.extend(self.state.update(time.time(), manual, self.automatic))

    def _watch(self):
        while not self.stopping.wait(PAUSE_POLL_SEC):
            try:
                self.poll()
            except Exception as error:  # noqa: BLE001 — the watcher must outlive any one poll
                self._warn("poll failed", error)

    def take_events(self):
        with self.lock:
            events, self.events = self.events, []
            return events

    def feed(self, gate, samples, now, final=False):
        with self.lock:
            return gate.feed(samples, now, self.state, final=final)

    def stop(self):
        """Join the watcher, then poll once more so a pause that began in the last poll
        interval still mutes the held mic audio before it is flushed."""
        if self.stopping.is_set():
            return
        self.stopping.set()
        if self.thread:
            self.thread.join()
        self.poll()
