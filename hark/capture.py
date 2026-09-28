"""Audio sources: 16 kHz mono float32 on the 16-bit PCM grid (k / 32768), drained by the main loop.

Every live sample is exactly representable as s16le PCM, so a track saved by `WavRecorder`
and replayed with `--file` feeds the models the same numbers they heard live.

Mic via PortAudio (sounddevice); system audio via `audiotee`, a Core Audio
process tap that writes s16le PCM to stdout (built by scripts/build-audiotee.sh).

Live sources are held to the wall clock: `drain` pads with silence whatever
the device failed to deliver (a stalled tap, a vanished mic, the laptop asleep),
so transcript times stay true and pending utterances still get flushed.
"""

import json
import os
import queue
import select
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
BLOCK = SAMPLE_RATE // 10  # 100 ms
AUDIOTEE = Path(os.environ.get("HARK_AUDIOTEE", Path(__file__).parent.parent / "bin" / "audiotee"))
STALL_SEC = 5.0
LATE_OK = 1.0  # a device may run this far behind the wall clock before we pad


def to_pcm16(samples):
    return np.clip(np.rint(samples * 32768.0), -32768, 32767).astype("<i2")


def on_pcm16_grid(samples):
    """Round float samples to the nearest s16le value, as `WavRecorder` and `--file` see them."""
    return to_pcm16(samples).astype(np.float32) / 32768.0


def log(msg):
    print(f"[hark] {msg}", file=sys.stderr, flush=True)


class Source:
    live = True

    def __init__(self, name):
        self.name = name
        self.queue = queue.Queue()
        self.anchor = None  # wall-clock time of sample 0
        self.delivered = 0  # samples handed to the caller, padding included
        self.last_audio = None  # wall-clock time the device last delivered

    def start(self):
        self.anchor = self.last_audio = time.time()
        self._open()

    def drain(self, limit=2 * SAMPLE_RATE):
        """Samples since the last call, up to about `limit`, padded to the wall clock."""
        blocks = []
        while sum(b.size for b in blocks) < limit:
            try:
                blocks.append(self.queue.get_nowait())
            except queue.Empty:
                break
        if self.live and self.anchor is not None:
            got = sum(b.size for b in blocks)
            if self.queue.empty():
                behind = (time.time() - self.anchor) * SAMPLE_RATE - self.delivered - got
            else:
                behind = 0
            if behind > LATE_OK * SAMPLE_RATE:
                blocks.append(np.zeros(int(behind - LATE_OK * SAMPLE_RATE / 2), np.float32))
        out = np.concatenate(blocks) if blocks else np.zeros(0, np.float32)
        self.delivered += out.size
        return out

    def _open(self):
        raise NotImplementedError

    def stop(self):
        pass


class MicSource(Source):
    def __init__(self, device=None):
        super().__init__("mic")
        self.device = device
        self.stream = None
        self.stopping = threading.Event()

    def _open(self):
        self._stream()
        threading.Thread(target=self._watch, daemon=True).start()

    def _stream(self):
        import sounddevice as sd

        def callback(indata, frames, t, status):
            if status:
                log(f"mic: {status}")
            self.last_audio = time.time()
            self.queue.put(on_pcm16_grid(indata[:, 0]))

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            blocksize=BLOCK, device=self.device, callback=callback,
        )
        self.stream.start()

    def _watch(self):
        """Reopen the input if it goes quiet (a headset unplugged, AirPods switched)."""
        import sounddevice as sd

        while not self.stopping.wait(1.0):
            if time.time() - self.last_audio < STALL_SEC:
                continue
            log("mic: no input for 5 s, reopening")
            self.last_audio = time.time()
            try:
                self._close()
                sd._terminate()
                sd._initialize()  # refresh PortAudio's device list
                self._stream()
            except Exception as e:  # noqa: BLE001 — keep trying; drain() pads the gap
                log(f"mic: reopen failed ({e})")

    def _close(self):
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:  # noqa: BLE001
                pass
            self.stream = None

    def stop(self):
        self.stopping.set()
        self._close()


class SystemSource(Source):
    """Everything the Mac plays (Zoom, Meet, a video), via a supervised audiotee."""

    def __init__(self):
        super().__init__("system")
        if not AUDIOTEE.exists():
            raise SystemExit(f"audiotee not found at {AUDIOTEE} — run scripts/build-audiotee.sh")
        self.proc = None
        self.stopping = threading.Event()
        self.heard_sound = False

    def _open(self):
        threading.Thread(target=self._supervise, daemon=True).start()

    def stop(self):
        self.stopping.set()
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()

    def _supervise(self):
        failures = 0
        while not self.stopping.is_set():
            self.proc = subprocess.Popen(
                [str(AUDIOTEE), "--sample-rate", str(SAMPLE_RATE)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                start_new_session=True,  # a terminal Ctrl-C reaches hark, not the tap
            )
            threading.Thread(target=self._stderr, args=(self.proc,), daemon=True).start()
            got_audio = self._pump(self.proc)
            if self.proc.poll() is None:
                self.proc.kill()
            self.proc.wait()
            if self.stopping.is_set():
                break
            failures = 0 if got_audio else failures + 1
            delay = min(30.0, 2.0 ** failures)
            if failures >= 3:
                log(f"system audio: tap keeps failing ({failures}×), retrying in {delay:.0f} s")
            self.stopping.wait(delay)

    def _pump(self, proc):
        """Forward PCM until the tap exits or stalls; True if any audio arrived."""
        fd = proc.stdout.fileno()
        pending = b""
        last = time.monotonic()
        got = False
        while not self.stopping.is_set():
            ready, _, _ = select.select([fd], [], [], 0.5)
            if not ready:
                if time.monotonic() - last > STALL_SEC:
                    return got
                continue
            chunk = os.read(fd, 2 * BLOCK)
            if not chunk:
                return got
            last = time.monotonic()
            self.last_audio = time.time()
            got = True
            pending += chunk
            n = len(pending) // 2 * 2
            samples = np.frombuffer(pending[:n], dtype="<i2").astype(np.float32) / 32768.0
            pending = pending[n:]
            if not self.heard_sound:
                if samples.size and np.abs(samples).max() > 1e-4:
                    self.heard_sound = True
                elif time.time() - self.anchor > 20:
                    log("system audio is pure silence after 20 s. If something is playing, grant "
                        "your terminal System Settings → Privacy & Security → Screen & System "
                        "Audio Recording → 'System Audio Recording Only', then restart the terminal.")
                    self.heard_sound = True  # warn once
            self.queue.put(samples)
        return got

    def _stderr(self, proc):
        for line in proc.stderr:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("message_type") == "error":
                log(f"audiotee: {rec.get('data', rec)}")


class WavRecorder:
    """One track's samples as 16 kHz mono 16-bit PCM WAV, appended as they are drained.

    It receives exactly the samples the track is fed, from the track's first sample on, so
    second `t` of the file is second `t` of the track: an utterance's `start`/`end` index it
    directly. Live samples sit on the 16-bit grid, so the file is lossless and `--file`
    replays it bit for bit. `wave` patches the header on every write, so the file is a valid
    WAV at any moment and complete once `close` returns.
    """

    def __init__(self, path):
        import wave

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.wav = wave.open(str(self.path), "wb")
        self.wav.setnchannels(1)
        self.wav.setsampwidth(2)
        self.wav.setframerate(SAMPLE_RATE)

    def write(self, samples):
        if samples.size:
            self.wav.writeframes(to_pcm16(samples).tobytes())

    def close(self):
        self.wav.close()


class FileSource(Source):
    """Replays an audio file, as fast as possible or at real-time pace."""

    live = False

    def __init__(self, path, realtime=False):
        super().__init__(Path(path).stem)
        self.path = path
        self.realtime = realtime
        self.done = threading.Event()
        self.stopping = threading.Event()

    def _open(self):
        from mlx_audio.stt.utils import load_audio

        audio = np.array(load_audio(str(self.path), sr=SAMPLE_RATE), dtype=np.float32)

        def feed():
            for i in range(0, audio.size, BLOCK):
                if self.stopping.is_set():
                    break
                self.queue.put(audio[i : i + BLOCK])
                if self.realtime:
                    time.sleep(BLOCK / SAMPLE_RATE)
            self.done.set()

        threading.Thread(target=feed, daemon=True).start()

    def stop(self):
        """Abandon the unplayed rest of the file."""
        self.stopping.set()
        self.done.wait()
        while not self.queue.empty():
            self.queue.get_nowait()
