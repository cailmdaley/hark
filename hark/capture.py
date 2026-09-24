"""Audio sources: 16 kHz mono float32 blocks delivered on a queue.

Mic via PortAudio (sounddevice); system audio via `audiotee`, a Core Audio
process tap that writes s16le PCM to stdout (built by scripts/build-audiotee.sh).
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


def log(msg):
    print(f"[hark] {msg}", file=sys.stderr, flush=True)


class Source:
    def __init__(self, name):
        self.name = name
        self.queue = queue.Queue()

    def drain(self):
        """All samples captured since the last call (possibly empty)."""
        blocks = []
        while True:
            try:
                blocks.append(self.queue.get_nowait())
            except queue.Empty:
                break
        return np.concatenate(blocks) if blocks else np.zeros(0, np.float32)

    def start(self):
        raise NotImplementedError

    def stop(self):
        pass


class MicSource(Source):
    def __init__(self, device=None):
        super().__init__("mic")
        self.device = device
        self.stream = None

    def start(self):
        import sounddevice as sd

        def callback(indata, frames, t, status):
            if status:
                log(f"mic: {status}")
            self.queue.put(indata[:, 0].copy())

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            blocksize=BLOCK, device=self.device, callback=callback,
        )
        self.stream.start()

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()


class SystemSource(Source):
    """Everything the Mac plays (Zoom, Meet, a video), via a supervised audiotee."""

    def __init__(self):
        super().__init__("system")
        self.proc = None
        self.stopping = threading.Event()
        self.heard_sound = False
        self.last_data = None  # monotonic time of the last sample received

    def start(self):
        if not AUDIOTEE.exists():
            raise SystemExit(f"audiotee not found at {AUDIOTEE} — run scripts/build-audiotee.sh")
        threading.Thread(target=self._supervise, daemon=True).start()

    def stop(self):
        self.stopping.set()
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()

    def _supervise(self):
        started = time.monotonic()
        while not self.stopping.is_set():
            self.proc = subprocess.Popen(
                [str(AUDIOTEE), "--sample-rate", str(SAMPLE_RATE)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            threading.Thread(target=self._stderr, args=(self.proc,), daemon=True).start()
            self._pump(self.proc, started)
            if self.proc.poll() is None:
                self.proc.kill()
            if not self.stopping.is_set():
                log("system audio: tap ended or stalled, restarting")
                time.sleep(1.0)

    def _pump(self, proc, started):
        fd = proc.stdout.fileno()
        pending = b""
        last = time.monotonic()
        while not self.stopping.is_set():
            ready, _, _ = select.select([fd], [], [], 0.5)
            if not ready:
                if time.monotonic() - last > STALL_SEC:
                    return
                continue
            chunk = os.read(fd, 2 * BLOCK)
            if not chunk:
                return
            last = time.monotonic()
            if self.last_data is not None and last - self.last_data > 0.5:
                # the tap was restarted: pad the gap so transcript times stay on the wall clock
                self.queue.put(np.zeros(int((last - self.last_data - 0.2) * SAMPLE_RATE), np.float32))
            self.last_data = last
            pending += chunk
            n = len(pending) // 2 * 2
            samples = np.frombuffer(pending[:n], dtype="<i2").astype(np.float32) / 32768.0
            pending = pending[n:]
            if not self.heard_sound:
                if samples.size and np.abs(samples).max() > 1e-4:
                    self.heard_sound = True
                elif time.monotonic() - started > 20:
                    log("system audio is pure silence after 20 s. If something is playing, grant "
                        "your terminal System Settings → Privacy & Security → Screen & System "
                        "Audio Recording → 'System Audio Recording Only', then restart the terminal.")
                    self.heard_sound = True  # warn once
            self.queue.put(samples)

    def _stderr(self, proc):
        for line in proc.stderr:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("message_type") == "error":
                log(f"audiotee: {rec.get('data', rec)}")


class FileSource(Source):
    """Replays an audio file, as fast as possible or at real-time pace."""

    def __init__(self, path, realtime=False):
        super().__init__(Path(path).stem)
        self.path = path
        self.realtime = realtime
        self.done = threading.Event()

    def start(self):
        from mlx_audio.stt.utils import load_audio

        audio = np.array(load_audio(str(self.path), sr=SAMPLE_RATE), dtype=np.float32)

        def feed():
            for i in range(0, audio.size, BLOCK):
                self.queue.put(audio[i : i + BLOCK])
                if self.realtime:
                    time.sleep(BLOCK / SAMPLE_RATE)
            self.done.set()

        threading.Thread(target=feed, daemon=True).start()
