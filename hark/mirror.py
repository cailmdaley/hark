"""Resume an append-only transcript mirror over a long-lived SSH pipe."""

import shlex
import subprocess
import threading
import time
from pathlib import Path

from .capture import log

SSH_OPTIONS = [
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=10",
    "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=3",
]
EOF_TIMEOUT = 3.0


def _remote_path(path):
    if path == "~":
        return '"$HOME"'
    if path.startswith("~/"):
        return f'"$HOME"/{shlex.quote(path[2:])}'
    return shlex.quote(path)


def ssh_commands(host, path, offset=0, *, reset=False):
    target = _remote_path(path)
    clear = f"truncate -s 0 -- {target} && " if reset else ""
    stream = (f'mkdir -p -- "$(dirname -- {target})" && {clear}'
              f"dd of={target} bs=64k seek={offset} oflag=seek_bytes conv=notrunc status=none")
    size = f"if [ -e {target} ]; then wc -c < {target}; else printf '0\\n'; fi"
    ssh = ["ssh", *SSH_OPTIONS, host]
    return [*ssh, stream], [*ssh, size]


class MirrorError(RuntimeError):
    pass


class TranscriptMirror:
    """Copy transcript bytes at fixed offsets so reconnects can safely overlap."""

    def __init__(self, local_path, host, remote_path, *, command=None, size_command=None,
                 popen=subprocess.Popen, run=subprocess.run, backoff=0.25, max_backoff=5.0,
                 poll_interval=0.05, eof_timeout=EOF_TIMEOUT, resume=False):
        self.path = Path(local_path)
        self.host = host
        self.remote_path = remote_path
        self.command = command
        _, default_size = ssh_commands(host, remote_path)
        self.size_command = size_command if size_command is not None else default_size
        self.popen = popen
        self.run = run
        self.backoff = backoff
        self.max_backoff = max_backoff
        self.poll_interval = poll_interval
        self.eof_timeout = eof_timeout
        self.resume = resume
        self.thread = None
        self.process = None
        self.target_size = None
        self.failure = None
        self.completed = False
        self.stopping = threading.Event()
        self.wake = threading.Event()
        self.recovery_logged = False

    @property
    def resume_command(self):
        return shlex.join(["hark", "mirror", "--resume", str(self.path),
                           f"{self.host}:{self.remote_path}"])

    def start(self):
        if self.thread is not None:
            raise RuntimeError("mirror already started")
        try:
            self.path.stat()
            self.thread = threading.Thread(target=self._run, name="hark-mirror", daemon=True)
            self.thread.start()
        except Exception as error:
            self._stop(error)
            return False
        return True

    def finish(self, timeout=30):
        try:
            return self._finish(timeout)
        except Exception as error:
            self.stopping.set()
            self.wake.set()
            self._stop(error)
            self._log_recovery()
            return False

    def _finish(self, timeout):
        if self.thread is None:
            if self.failure is None:
                self._stop(MirrorError("mirror was not started"))
            self._log_recovery()
            return False
        if self.target_size is None:
            try:
                self.target_size = self.path.stat().st_size
            except OSError as error:
                self._stop(error)
                self.stopping.set()
                self.wake.set()
        self.wake.set()
        self.thread.join(timeout)
        if self.thread.is_alive():
            self.stopping.set()
            process = self.process
            if process and process.poll() is None:
                try:
                    process.kill()
                except OSError:
                    pass
            self.wake.set()
            self.thread.join(self.eof_timeout + 1)
            log(f"mirror: timed out after {timeout:g} s; capture is local and mirroring has stopped")
        if not self.completed:
            self._log_recovery()
        return self.completed

    def _remote_size(self):
        result = self.run(self.size_command, capture_output=True, text=True, check=True,
                          timeout=10, start_new_session=True)
        return int(result.stdout.strip())

    @staticmethod
    def _retryable(error):
        if isinstance(error, FileNotFoundError):
            return False
        return not isinstance(error, subprocess.CalledProcessError) or error.returncode == 255

    def _new_command(self, offset, *, reset=False):
        if self.command is None:
            return ssh_commands(self.host, self.remote_path, offset, reset=reset)[0]
        if callable(self.command):
            return self.command(offset, reset=reset)
        return self.command

    def _stop(self, error):
        if self.failure is None:
            self.failure = error
            log(f"mirror: stopped: {error}; capture continues locally and the transcript is safe")

    def _log_recovery(self):
        if not self.recovery_logged:
            log(f"mirror: resume after capture with: {self.resume_command}")
            self.recovery_logged = True

    def _drop(self, process):
        if process is None:
            return None
        if process.stdin:
            try:
                process.stdin.close()
            except Exception:
                pass
        try:
            process.wait(timeout=self.eof_timeout)
        except subprocess.TimeoutExpired:
            log(f"mirror: SSH did not exit {self.eof_timeout:g} s after EOF; terminating the old channel")
            try:
                process.kill()
                process.wait(timeout=1)
            except Exception:
                pass
        except Exception as error:
            log(f"mirror: SSH channel cleanup failed ({error}); terminating the old channel")
            try:
                process.kill()
            except Exception:
                pass
        finally:
            if self.process is process:
                self.process = None
        try:
            return process.poll()
        except Exception:
            return None

    def _send(self, process, data):
        remaining = memoryview(data)
        while remaining:
            written = process.stdin.write(remaining)
            if not written:
                raise BrokenPipeError("SSH mirror accepted no bytes")
            remaining = remaining[written:]

    def _delay(self, failures):
        if self.backoff <= 0:
            return 0.0
        delay = self.backoff
        for _ in range(failures - 1):
            if delay >= self.max_backoff:
                break
            delay = min(self.max_backoff, delay * 2)
        return delay

    def _retry(self, failures, message):
        failures += 1
        delay = self._delay(failures)
        log(f"mirror: {message}; retrying in {delay:g} s")
        self.stopping.wait(delay)
        return failures

    def _run(self):
        process = None
        position = 0
        connected = False
        failures = 0
        force_reset = False
        try:
            while not self.stopping.is_set():
                if process is None:
                    try:
                        offset = self._remote_size()
                    except (OSError, subprocess.SubprocessError) as error:
                        if not self._retryable(error):
                            raise MirrorError(f"remote size check failed ({error})") from error
                        failures = self._retry(failures, f"remote size check failed ({error})")
                        continue
                    local_size = self.path.stat().st_size
                    reset = force_reset or offset > local_size
                    force_reset = False
                    if reset:
                        log(f"mirror: remote has {offset} bytes but local transcript has {local_size}; rewriting from byte 0")
                        position = 0
                    elif offset and not connected and not self.resume:
                        raise MirrorError(f"remote target already contains {offset} bytes; use hark mirror --resume")
                    else:
                        position = offset
                    try:
                        process = self.popen(self._new_command(position, reset=reset),
                                             stdin=subprocess.PIPE, bufsize=0,
                                             start_new_session=True)
                    except OSError:
                        raise
                    self.process = process
                    connected = True
                    failures = 0
                    log(f"mirror: streaming to {self.host}:{self.remote_path}")

                returncode = process.poll()
                if returncode is not None:
                    self._drop(process)
                    process = None
                    if returncode not in (0, 255):
                        raise MirrorError(f"SSH exited {returncode}")
                    failures = self._retry(failures, f"SSH exited {returncode}")
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
                            returncode = self._drop(process)
                            process = None
                            if returncode not in (None, 0, 255):
                                raise MirrorError(f"SSH write failed ({error}); SSH exited {returncode}") from error
                            failures = self._retry(failures, f"SSH write failed ({error})")
                            continue
                        position += len(data)
                        continue

                if self.target_size is not None and position >= self.target_size:
                    self._drop(process)
                    process = None
                    try:
                        offset = self._remote_size()
                    except (OSError, subprocess.SubprocessError) as error:
                        if not self._retryable(error):
                            raise MirrorError(f"final size check failed ({error})") from error
                        failures = self._retry(failures, f"final size check failed ({error})")
                        continue
                    if offset == self.target_size:
                        self.completed = True
                        log(f"mirror: delivered {offset} bytes, including # ended")
                        return
                    if offset > self.target_size:
                        force_reset = True
                    else:
                        log(f"mirror: remote has {offset}/{self.target_size} bytes; resuming")
                    failures += 1
                    self.stopping.wait(self._delay(failures))
                    continue

                self.wake.wait(self.poll_interval)
                self.wake.clear()
        except Exception as error:
            self._stop(error)
        finally:
            self._drop(process)
