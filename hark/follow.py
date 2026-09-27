"""hark follow — batch a live transcript's new lines for an agent to read.

Everything already in the file at startup is the first batch, printed at once.
After that, new lines are held pending until something flushes them: a line
that addresses the agent, the pending word count crossing a threshold, a
timeout since the first pending line arrived, or the transcript's `# ended`
footer, which flushes and ends the follow.
"""

import re
import time
from pathlib import Path

DEFAULT_NAMES = ("claude", "cloud", "clawed", "klaud")

_TIMESTAMP = re.compile(r"^\d{2}:\d{2}:\d{2}\s+(.*)$")


class Batcher:
    """Pure batching logic: feed lines and clock ticks, get back batches to flush."""

    def __init__(self, *, words=150, seconds=15.0, names=DEFAULT_NAMES):
        self.words = words
        self.seconds = seconds
        self.pattern = re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\b", re.IGNORECASE)
        self.pending = []
        self.word_count = 0
        self.first_arrival = None
        self.ended = False

    def add(self, line, now):
        """Add one line arriving at `now`; return a batch to flush, or None."""
        self.pending.append(line)
        if self.first_arrival is None:
            self.first_arrival = now
        if line.startswith("# ended"):
            self.ended = True
            return self._flush()
        if not line.startswith("#"):
            text = match[1] if (match := _TIMESTAMP.match(line)) else line
            self.word_count += len(text.split())
            if self.pattern.search(text) or self.word_count >= self.words:
                return self._flush()
        return None

    def tick(self, now):
        """Called periodically between lines; return a batch if the time threshold elapsed."""
        if self.pending and now - self.first_arrival >= self.seconds:
            return self._flush()
        return None

    def _flush(self):
        batch, self.pending = self.pending, []
        self.word_count = 0
        self.first_arrival = None
        return batch


def _print_flushed(line):
    print(line, flush=True)


def _emit_batch(lines, emit):
    for line in lines:
        emit(line)
    emit("")


def follow(path, batcher, *, poll_interval=1.0, sleep=time.sleep, clock=time.monotonic, emit=_print_flushed):
    """Poll `path` for new lines, batching them through `batcher` until it ends.

    The file may not exist yet (a fresh meeting hasn't been started); wait for it.
    Everything present at the first successful read is the startup batch, flushed
    unconditionally. A shrunken file (truncated, or rewritten from scratch) is
    read again from byte 0.
    """
    path = Path(path)
    while not path.exists():
        sleep(poll_interval)
    offset = 0
    started = False
    while True:
        size = path.stat().st_size
        if size < offset:
            offset = 0
        with path.open("rb") as transcript:
            transcript.seek(offset)
            data = transcript.read()
        complete = data.rfind(b"\n") + 1
        if complete:
            lines = data[:complete].decode("utf-8", errors="replace").splitlines()
            offset += complete
            if not started:
                started = True
                _emit_batch(lines, emit)
                if any(line.startswith("# ended") for line in lines):
                    return
            else:
                for line in lines:
                    batch = batcher.add(line, clock())
                    if batch is not None:
                        _emit_batch(batch, emit)
                        if batcher.ended:
                            return
        if started:
            batch = batcher.tick(clock())
            if batch is not None:
                _emit_batch(batch, emit)
        sleep(poll_interval)
