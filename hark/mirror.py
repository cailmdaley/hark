"""Resume an append-only transcript mirror over a long-lived SSH pipe."""

import shlex
import subprocess
import threading
import time
from pathlib import Path

from .capture import log


def _remote_path(path):
    if path == "~":
        return '"$HOME"'
    if path.startswith("~/"):
        return f'"$HOME"/{shlex.quote(path[2:])}'
    return shlex.quote(path)


def ssh_commands(host, path):
    target = _remote_path(path)
    stream = f'mkdir -p -- "$(dirname -- {target})" && cat >> {target}'
    size = f'if [ -e {target} ]; then wc -c < {target}; else printf \'0\\n\'; fi'
    return ["ssh", host, stream], ["ssh", host, size]


class MirrorError(RuntimeError):
    pass


class TranscriptMirror:
    """Copy appended transcript bytes and resume at the remote file's byte count."""

    def __init__(self, local_path, host, remote_path, *, command=None, size_command=None,
                 popen=subprocess.Popen, run=subprocess.run, backoff=0.25, max_backoff=5.0,
                 poll_interval=0.05):
        self.path = Path(local_path)
        self.host = host
        self.remote_path = remote_path
        default_command, default_size = ssh_commands(host, remote_path)
        self.command = command if command is not None else default_command
        self.size_command = size_command if size_command is not None else default_size
        self.popen = popen
        self.run = run
        self.backoff = backoff
        self.max_backoff = max_backoff
        self.poll_interval = poll_interval
        self.thread = None
        self.process = None
        self.target_size = None
        self.failure = None
        self.completed = False
        self.stopping = threading.Event()
        self.wake = threading.Event()

    def start(self):
        if self.thread is not None:
            raise RuntimeError("mirror already started")
        self.path.stat()
        self.thread = threading.Thread(target=self._run, name="hark-mirror", daemon=True)
        self.thread.start()

    def check(self):
        if self.failure:
            raise MirrorError(f"mirror failed: {self.failure}") from self.failure

    def finish(self, timeout=30):
        if self.thread is None:
            raise RuntimeError("mirror was not started")
        if self.target_size is None:
            self.target_size = self.path.stat().st_size
        self.wake.set()
        self.thread.join(timeout)
        if self.thread.is_alive():
            self.stopping.set()
            if self.process and self.process.poll() is None:
                self.process.kill()
            self.wake.set()
            self.thread.join(1)
            log(f"mirror: timed out after {timeout:g} s; the remote # ended footer may be incomplete")
            return False
        self.check()
        return self.completed

    def _remote_size(self):
        result = self.run(self.size_command, capture_output=True, text=True, check=True, timeout=10)
        return int(result.stdout.strip())

    def _new_command(self):
        return self.command() if callable(self.command) else self.command

    def _drop(self, process):
        if process is None:
            return
        if process.stdin:
            try:
                process.stdin.close()
            except OSError:
                pass
        if process.poll() is None:
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        self.process = None

    def _send(self, process, data):
        remaining = memoryview(data)
        while remaining:
            written = process.stdin.write(remaining)
            if not written:
                raise BrokenPipeError("SSH mirror accepted no bytes")
            remaining = remaining[written:]

    def _run(self):
        process = None
        position = 0
        connected = False
        failures = 0
        try:
            while not self.stopping.is_set():
                if process is None:
                    try:
                        offset = self._remote_size()
                    except (OSError, subprocess.SubprocessError) as error:
                        failures += 1
                        delay = min(self.max_backoff, self.backoff * 2 ** (failures - 1))
                        log(f"mirror: remote size check failed ({error}); retrying in {delay:g} s")
                        self.stopping.wait(delay)
                        continue
                    local_size = self.path.stat().st_size
                    if offset > local_size:
                        raise MirrorError(f"remote mirror is {offset} bytes but local transcript is only {local_size}")
                    if not connected and offset:
                        raise MirrorError(f"remote mirror target already contains {offset} bytes")
                    position = offset
                    try:
                        process = self.popen(self._new_command(), stdin=subprocess.PIPE, bufsize=0)
                    except OSError as error:
                        failures += 1
                        delay = min(self.max_backoff, self.backoff * 2 ** (failures - 1))
                        log(f"mirror: SSH start failed ({error}); retrying in {delay:g} s")
                        self.stopping.wait(delay)
                        continue
                    self.process = process
                    connected = True
                    failures = 0
                    log(f"mirror: streaming to {self.host}:{self.remote_path}")

                returncode = process.poll()
                if returncode is not None:
                    self._drop(process)
                    process = None
                    failures += 1
                    delay = min(self.max_backoff, self.backoff * 2 ** (failures - 1))
                    log(f"mirror: SSH exited {returncode}; reconnecting in {delay:g} s")
                    self.stopping.wait(delay)
                    continue

                local_size = self.path.stat().st_size
                if self.target_size is not None and local_size < self.target_size:
                    raise MirrorError("local transcript shrank while it was being mirrored")
                available = min(local_size, self.target_size) if self.target_size is not None else local_size
                if position < available:
                    with self.path.open("rb") as transcript:
                        transcript.seek(position)
                        data = transcript.read(min(65536, available - position))
                    if data:
                        try:
                            self._send(process, data)
                        except (BrokenPipeError, OSError) as error:
                            self._drop(process)
                            process = None
                            failures += 1
                            delay = min(self.max_backoff, self.backoff * 2 ** (failures - 1))
                            log(f"mirror: SSH write failed ({error}); reconnecting in {delay:g} s")
                            self.stopping.wait(delay)
                            continue
                        position += len(data)
                        continue

                if self.target_size is not None and position >= self.target_size:
                    self._drop(process)
                    process = None
                    try:
                        offset = self._remote_size()
                    except (OSError, subprocess.SubprocessError) as error:
                        failures += 1
                        delay = min(self.max_backoff, self.backoff * 2 ** (failures - 1))
                        log(f"mirror: final size check failed ({error}); retrying in {delay:g} s")
                        self.stopping.wait(delay)
                        continue
                    if offset == self.target_size:
                        self.completed = True
                        log(f"mirror: delivered {offset} bytes, including # ended")
                        return
                    if offset > self.target_size:
                        raise MirrorError(f"remote mirror is {offset} bytes, beyond final local size {self.target_size}")
                    log(f"mirror: remote has {offset}/{self.target_size} bytes; resuming")
                    failures += 1
                    delay = min(self.max_backoff, self.backoff * 2 ** (failures - 1))
                    self.stopping.wait(delay)
                    continue

                self.wake.wait(self.poll_interval)
                self.wake.clear()
        except Exception as error:
            self.failure = error
            log(f"mirror: failed: {error}")
        finally:
            self._drop(process)
