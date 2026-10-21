"""Audio sources: 16 kHz mono float32 on the 16-bit PCM grid (k / 32768), drained by the main loop.

Every live sample is exactly representable as s16le PCM, so a track saved by `WavRecorder`
and replayed with `--file` feeds the models the same numbers they heard live.

Raw mic via PortAudio (sounddevice); system audio via `audiotee`, a Core Audio
process tap that writes s16le PCM to stdout (built by scripts/build-audiotee.sh);
a phone (or any remote mic) as s16le PCM written into a Unix socket hark listens on.

Live sources are held to the wall clock: `drain` pads with silence whatever
the device failed to deliver (a stalled tap, a vanished mic, the laptop asleep),
so transcript times stay true and pending utterances still get flushed.
Each live source notes when the device last delivered (`last_audio`) and when
it last delivered sound (`last_sound`); padding never counts as either. `heard` pins the
newest device sample drained so far to the wall time it arrived, because a device can run
up to `LATE_OK` behind the padded timeline.
"""

import json
import os
import queue
import select
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
BLOCK = SAMPLE_RATE // 10  # 100 ms
AUDIOTEE = Path(os.environ.get("HARK_AUDIOTEE", Path(__file__).parent.parent / "bin" / "audiotee"))
STALL_SEC = 5.0
JUMP_SEC = 30.0  # a wall-clock deficit this large is a suspend or stall, not jitter: skip it
JUMP_PAD = 5.0  # silence still fed after a skip, so pending speech closes before the timeline moves
LATE_OK = 1.0  # a device may run this far behind the wall clock before we pad
SOUND = 1e-4  # peak amplitude above digital silence


def to_pcm16(samples):
    return np.clip(np.rint(samples * 32768.0), -32768, 32767).astype("<i2")


def on_pcm16_grid(samples):
    """Round float samples to the nearest s16le value, as `WavRecorder` and `--file` see them."""
    return to_pcm16(samples).astype(np.float32) / 32768.0


_log_file = None


def log(msg, file_only=False):
    """Report on stderr and, once `log_to` has opened one, in the session log."""
    if not file_only:
        print(f"[hark] {msg}", file=sys.stderr, flush=True)
    if _log_file:
        _log_file.write(f"{datetime.now():%H:%M:%S} {msg}\n")


def log_to(path):
    """Append every later `log` line, timestamped, to `path`; None closes the file."""
    global _log_file
    previous, _log_file = _log_file, open(path, "a", buffering=1) if path else None
    if previous:
        previous.close()


def _no_stats():
    return {"device": 0, "padded": 0, "peak": 0.0}


class Source:
    live = True

    def __init__(self, name):
        self.name = name
        self.queue = queue.Queue()
        self.queue_lock = threading.Lock()
        self.queued_samples = 0
        self.sample_limit = None
        self.anchor = None  # wall-clock time of sample 0
        self.delivered = 0  # samples handed to the caller, padding included
        self.last_audio = None  # wall-clock time the device last delivered
        self.last_sound = None  # ... and last delivered a block above digital silence
        self.has_sound = False  # whether any device block exceeded SOUND this session
        self.heard = None  # (samples drained, wall time the last of them arrived)
        self.stats = _no_stats()
        self.error = None
        self.jump = 0.0  # seconds of wall time skipped since the last `take_jump`

    def start(self):
        self.anchor = self.last_audio = self.last_sound = time.time()
        self.has_sound = False
        self.heard = 0, self.anchor
        self._open()

    def _deliver(self, samples):
        """Queue samples the device delivered, noting when and whether they carry sound."""
        now = self.last_audio = time.time()
        peak = float(np.abs(samples).max()) if samples.size else 0.0
        if peak > SOUND:
            self.last_sound = now
            self.has_sound = True
        self.stats["device"] += samples.size
        self.stats["peak"] = max(self.stats["peak"], peak)
        with self.queue_lock:
            if self.error:
                return
            try:
                if self.sample_limit is not None and self.queued_samples + samples.size > self.sample_limit:
                    raise queue.Full
                self.queue.put_nowait((now, samples))
                self.queued_samples += samples.size
            except queue.Full:
                self.error = RuntimeError(f"{self.name}: capture backlog full; audio cannot be preserved")

    def take_stats(self):
        """Samples delivered by the device and padded by `drain`, and peak, since the last call."""
        stats, self.stats = self.stats, _no_stats()
        return stats

    def take_jump(self):
        """Seconds of wall time `drain` skipped since the last call."""
        jump, self.jump = self.jump, 0.0
        return jump

    def drain(self, limit=2 * SAMPLE_RATE):
        """Samples since the last call, up to about `limit`, padded to the wall clock."""
        if self.error and self.queue.empty():
            raise self.error
        blocks, got = [], 0
        while got < limit:
            try:
                with self.queue_lock:
                    arrived, block = self.queue.get_nowait()
                    self.queued_samples = max(0, self.queued_samples - block.size)
            except queue.Empty:
                break
            blocks.append(block)
            got += block.size
            if arrived is not None:
                self.heard = self.delivered + got, arrived
        if self.live and self.anchor is not None and not self.error:
            if self.queue.empty():
                behind = (time.time() - self.anchor) * SAMPLE_RATE - self.delivered - got
            else:
                behind = 0
            if behind > JUMP_SEC * SAMPLE_RATE:
                skipped = behind - JUMP_PAD * SAMPLE_RATE
                self.anchor += skipped / SAMPLE_RATE
                self.jump += skipped / SAMPLE_RATE
                behind -= skipped
                log(f"{self.name}: wall clock ran {skipped / SAMPLE_RATE:.0f} s ahead of the audio "
                    "(system sleep?); skipping the gap")
            if behind > LATE_OK * SAMPLE_RATE:
                blocks.append(np.zeros(int(behind - LATE_OK * SAMPLE_RATE / 2), np.float32))
                self.stats["padded"] += blocks[-1].size
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
            self._deliver(on_pcm16_grid(indata[:, 0]))

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            blocksize=BLOCK, device=self.device, callback=callback,
        )
        self.stream.start()

    def _watch(self):
        """Reopen the input if it goes quiet (a headset unplugged, AirPods switched)."""
        import sounddevice as sd

        reopened = 0.0
        while not self.stopping.wait(1.0):
            if time.time() - max(self.last_audio, reopened) < STALL_SEC:
                continue
            log("mic: no input for 5 s, reopening")
            reopened = time.time()
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


class PCMProcessSource(Source):
    """Supervise a helper emitting mono 16 kHz s16le PCM."""

    def __init__(self, name, command):
        super().__init__(name)
        self.command = command
        self.proc = None
        self.supervisor = None
        self.delivery_lock = threading.Lock()
        self.stopping = threading.Event()

    def _open(self):
        self.supervisor = threading.Thread(target=self._supervise, daemon=True)
        self.supervisor.start()

    def stop(self):
        """Fence delivery before the caller's final drain, then reap and join the helper."""
        with self.delivery_lock:
            self.stopping.set()
        proc = self.proc
        if proc:
            try:
                if proc.poll() is None:
                    proc.terminate()
                try:
                    proc.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=0.5)
            except ProcessLookupError:
                pass  # The supervisor reaped it concurrently.
            except subprocess.TimeoutExpired:
                log(f"{self.name}: helper did not exit after kill; delivery is stopped")
        if self.supervisor and self.supervisor is not threading.current_thread():
            self.supervisor.join(timeout=1.0)
            if self.supervisor.is_alive():
                log(f"{self.name}: supervisor shutdown timed out; delivery is stopped")

    def _supervise(self):
        failures = 0
        while not self.stopping.is_set():
            got_audio, code = False, None
            self.proc = None
            try:
                self.proc = subprocess.Popen(
                    self.command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    start_new_session=True,  # Ctrl-C reaches hark, not the helper
                )
                threading.Thread(target=self._stderr, args=(self.proc,), daemon=True).start()
                got_audio, why = self._pump(self.proc)
            except Exception as error:  # noqa: BLE001 — retry while drain pads the gap
                why = f"failed ({error}); capture unavailable"
            finally:
                if self.proc:
                    if self.proc.poll() is None:
                        self.proc.kill()
                    code = self.proc.wait()
                    self.proc.stdout.close()

            if self.stopping.is_set():
                break
            failures = 0 if got_audio else failures + 1
            delay = min(30.0, 2.0 ** min(failures, 5))
            log(f"{self.name}: helper {why} (exit code {code}); capture unavailable until recovery"
                + (f", keeps failing ({failures}× without audio)" if failures >= 3 else "")
                + f", restarting in {delay:.0f} s")
            self.stopping.wait(delay)

    def _pump(self, proc):
        """Forward PCM until the helper exits or stalls, returning whether audio arrived and why."""
        fd = proc.stdout.fileno()
        pending = b""
        last = time.monotonic()
        got = False
        while not self.stopping.is_set():
            ready, _, _ = select.select([fd], [], [], 0.5)
            if not ready:
                if time.monotonic() - last > STALL_SEC:
                    return got, f"stalled ({STALL_SEC:.0f} s without data)"
                continue
            chunk = os.read(fd, 2 * BLOCK)
            if not chunk:
                return got, "hit EOF"
            last = time.monotonic()
            got = True
            pending += chunk
            n = len(pending) // 2 * 2
            samples = np.frombuffer(pending[:n], dtype="<i2").astype(np.float32) / 32768.0
            pending = pending[n:]
            if n:
                with self.delivery_lock:
                    if self.stopping.is_set():
                        return got, "stopped"
                    self._deliver(samples)
                    self._check_silence()
        return got, "stopped"

    def _check_silence(self):
        pass

    def _stderr(self, proc):
        for line in proc.stderr:
            text = line.decode("utf-8", errors="replace").strip()
            if text:
                log(f"{self.name}: {text}")
        proc.stderr.close()


class SystemSource(PCMProcessSource):
    """Everything the Mac plays (Zoom, Meet, a video), via a supervised audiotee."""

    def __init__(self):
        if not AUDIOTEE.exists():
            raise SystemExit(f"audiotee not found at {AUDIOTEE} — run scripts/build-audiotee.sh")
        super().__init__("system", [str(AUDIOTEE), "--sample-rate", str(SAMPLE_RATE)])
        self.warned = False

    def _check_silence(self):
        if not self.warned and not self.has_sound and time.time() - self.anchor > 20:
            log("system audio has been silent for 20 s; this is expected if nothing is playing. "
                "If playback is missing, check your terminal's permission in System Settings → "
                "Privacy & Security → Screen & System Audio Recording → "
                "'System Audio Recording Only'; after changing it, restart the terminal.")
            self.warned = True

    def _stderr(self, proc):
        for line in proc.stderr:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("message_type") == "error":
                log(f"audiotee: {rec.get('data', rec)}")
        proc.stderr.close()


class PhoneSource(Source):
    """A remote microphone streaming s16le 16 kHz mono PCM into a Unix socket.

    hark listens at `path`; whoever relays the phone (the Shuttle board's phone page)
    connects and writes raw PCM. One sender at a time: a new connection (the page
    reloaded, the phone reconnected) replaces the old one. Between senders `drain` pads
    the gap with silence, and a long one reaches the transcript as `# phone lost at …`.
    """

    def __init__(self, path):
        super().__init__("phone")
        self.path = Path(path)
        self.sample_limit = 600 * SAMPLE_RATE  # Ten minutes of samples, independent of packet size.
        self.server = None
        self.conn = None
        self.lock = threading.Lock()
        self.stopping = threading.Event()

    def _open(self):
        import socket

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.unlink(missing_ok=True)
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(str(self.path))
        self.server.listen(2)
        self.server.settimeout(0.5)
        threading.Thread(target=self._accept, daemon=True).start()
        log(f"phone: waiting for audio at {self.path}")

    def _accept(self):
        import socket

        while not self.stopping.is_set():
            try:
                conn, _ = self.server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with self.lock:
                previous, self.conn = self.conn, conn
            if previous:
                log("phone: a new sender replaced the previous one")
                previous.close()
            threading.Thread(target=self._pump, args=(conn,), daemon=True).start()

    def _pump(self, conn):
        log("phone: connected")
        pending = b""
        try:
            while not self.stopping.is_set():
                chunk = conn.recv(4 * BLOCK)
                if not chunk:
                    break
                with self.lock:
                    if self.conn is not conn:
                        return  # replaced; the new sender owns the track
                pending += chunk
                n = len(pending) // 2 * 2
                if n:
                    self._deliver(np.frombuffer(pending[:n], dtype="<i2").astype(np.float32) / 32768.0)
                    pending = pending[n:]
        except OSError:
            pass
        with self.lock:
            if self.conn is conn:
                self.conn = None
                if not self.stopping.is_set():
                    log("phone: disconnected; waiting for it to come back")
        conn.close()

    def stop(self):
        self.stopping.set()
        with self.lock:
            conn, self.conn = self.conn, None
        for s in (conn, self.server):
            if s:
                try:
                    s.close()
                except OSError:
                    pass
        self.path.unlink(missing_ok=True)


class WavRecorder:
    """One track's samples as 16 kHz mono 16-bit PCM WAV, appended as they are drained.

    It receives exactly the samples the track is fed, from the track's first sample on, so
    second `t` of the file is second `t` of the track: an utterance's `start`/`end` index it
    directly. Live samples sit on the 16-bit grid, so the file is lossless and `--file`
    replays it bit for bit. `wave` patches the header on every write, so the file is a valid
    WAV at any moment and complete once `close` returns. The audio is a by-product: an I/O
    error (a full disk) stops this recording, logged once, and never the meeting.
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
        if samples.size and self.wav:
            self._guard(self.wav.writeframes, to_pcm16(samples).tobytes())

    def close(self):
        if self.wav:
            self._guard(self.wav.close)
            self.wav = None

    def _guard(self, call, *args):
        try:
            call(*args)
        except OSError as error:
            log(f"audio: stopped saving {self.path.name}: {error}")
            self.wav = None


def load_audio(path, sr=SAMPLE_RATE):
    """Decode mono audio on the PCM grid without importing MLX."""
    import math
    import wave

    try:
        with wave.open(str(path), "rb") as wav:
            rate, channels, width = wav.getframerate(), wav.getnchannels(), wav.getsampwidth()
            raw = wav.readframes(wav.getnframes())
        if width != 2:
            raise wave.Error("use portable decoder for this sample width")
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32).reshape(-1, channels).mean(axis=1) / 32768
    except (wave.Error, EOFError):
        import soundfile as sf

        try:
            audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
            audio = audio.mean(axis=1)
        except sf.LibsndfileError:
            # ffmpeg supplies formats libsndfile doesn't decode, e.g. m4a.
            try:
                decoded = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "s16le",
                                          "-ar", str(sr), "-ac", "1", "pipe:1"],
                                         check=True, capture_output=True)
            except FileNotFoundError as error:
                raise ValueError("this audio format requires ffmpeg on PATH") from error
            return np.frombuffer(decoded.stdout, dtype="<i2").astype(np.float32) / 32768
    if rate != sr:
        from scipy.signal import resample_poly

        divisor = math.gcd(rate, sr)
        audio = resample_poly(audio, sr // divisor, rate // divisor)
    return on_pcm16_grid(np.asarray(audio, dtype=np.float32))


class FileSource(Source):
    """Replays an audio file, as fast as possible or at real-time pace."""

    live = False

    def __init__(self, path, realtime=False):
        super().__init__(Path(path).stem)
        self.path = path
        self.realtime = realtime
        self.done = threading.Event()
        self.stopping = threading.Event()
        self.queue = queue.Queue(maxsize=20)

    def _open(self):
        audio = load_audio(self.path)

        def feed():
            for i in range(0, audio.size, BLOCK):
                if self.stopping.is_set():
                    break
                while not self.stopping.is_set():
                    try:
                        self.queue.put((None, audio[i : i + BLOCK]), timeout=0.1)
                        break
                    except queue.Full:
                        pass
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
