"""Read Aqua's accepted dictation lifecycle without treating an open mic as speech."""

from datetime import datetime
from pathlib import Path
import logging
import os
import re
import time

log = logging.getLogger(__name__).warning

MAX_LEASE_SECONDS = 15 * 60
MAX_LOG_BYTES = 1024 * 1024
_EVENT = re.compile(
    r"^\[([^]]+)\].*\(DictationOrchestrator\)\s+dictation\.([a-z_]+)\b(.*)$")
_LEASE = re.compile(r"(?:^|\s)lease_id=(\d+)(?=\s|$)")
_STARTS = {"start_accepted", "lock_accepted"}
_ENDS = {"completed", "completed_canceled", "completed_offline_edit_suppressed",
         "completed_stale_paste_suppressed", "cancel_accepted", "start_failed",
         "owner_lost_during_session", "stop_requested", "stop_ready"}


class AquaDictation:
    """Follow complete lifecycle records in Aqua Voice's current and rotated logs.

    ``read`` takes CoreAudio processes including idle ones and returns their Aqua
    bundle IDs only during an accepted lease. Microphone ownership never activates
    a lease. Missing/unreadable logs fail open, with one warning: dictation may then
    reach hark, but a bridge left open cannot permanently silence it.

    Default info logs end a lease at completion, including finalization time.
    An unmatched lease expires after 15 minutes, measured from its logged start;
    dictation lasting longer than that is deliberately no longer muted.
    """

    def __init__(self, path=None):
        self.path = (Path(path) if path is not None else
                     Path.home() / "Library/Logs/Aqua Voice/main.log")
        self.old_path = self.path.with_name(self.path.stem + ".old" + self.path.suffix)
        self._owners = {}
        self._lease = None
        self._identity = None
        self._offset = 0
        self._signature = b""
        self._warned = set()

    def _warn(self, kind, detail):
        if kind not in self._warned:
            self._warned.add(kind)
            log(f"pause: Aqua {detail}")

    def _consume(self, data):
        for line in data.decode("utf-8", errors="replace").splitlines():
            match = _EVENT.match(line)
            if not match or not (lease := _LEASE.search(match[3])):
                continue
            try:
                at = datetime.fromisoformat(match[1].replace("Z", "+00:00")).timestamp()
            except ValueError:
                continue
            event, identity = match[2], lease[1]
            if event in _STARTS:
                if (self._lease is None or at >= self._lease[1]
                        and (event == "start_accepted" or identity != self._lease[0])):
                    self._lease = identity, at
            elif (event in _ENDS and self._lease is not None
                  and identity == self._lease[0] and at >= self._lease[1]):
                self._lease = None

    def _scan(self, stream, offset=0):
        """Consume at most one log's size, leaving a partial final line unread."""
        size = stream.seek(0, 2)
        start = max(offset, size - MAX_LOG_BYTES)
        stream.seek(start)
        data = stream.read(MAX_LOG_BYTES)
        if start > offset:
            # A tail-only scan starts inside an unknown record.
            first = data.find(b"\n") + 1
            start += first
            data = data[first:]
        complete = data.rfind(b"\n") + 1
        self._consume(data[:complete])
        offset = start + complete
        stream.seek(max(0, offset - 64))
        return offset, stream.read(min(offset, 64))

    def _changes(self):
        with self.path.open("rb") as current:
            stat = os.fstat(current.fileno())
            identity = stat.st_dev, stat.st_ino
            if self._identity is None:
                try:
                    with self.old_path.open("rb") as old:
                        self._scan(old)
                except FileNotFoundError:
                    pass
            elif identity != self._identity:
                try:
                    with self.old_path.open("rb") as old:
                        previous = os.fstat(old.fileno())
                        if (previous.st_dev, previous.st_ino) == self._identity:
                            self._scan(old, self._offset)
                        else:
                            self._lease = None
                except FileNotFoundError:
                    self._lease = None
                self._offset = 0
                self._signature = b""
            else:
                current.seek(max(0, self._offset - 64))
                if (stat.st_size < self._offset
                        or current.read(min(self._offset, 64)) != self._signature):
                    self._lease = None
                    self._offset = 0
            self._offset, self._signature = self._scan(current, self._offset)
            self._identity = identity

    def read(self, processes, now=None):
        now = time.time() if now is None else now
        aqua = [process for process in processes
                if any(name in process.bundle_id.casefold() for name in ("aquavoice", "aqua-voice"))]
        owners = {process.bundle_id: process.pid for process in aqua
                  if "helper" not in process.bundle_id.casefold()}
        # CoreAudio can omit an individual object during a poll. New helpers and
        # missing objects don't end a lease; an observed owner PID replacement does.
        if any(bundle in self._owners and self._owners[bundle] != pid
               for bundle, pid in owners.items()):
            self._lease = None
        self._owners.update(owners)
        try:
            self._changes()
        except OSError as error:
            self._lease = None
            self._warn("unavailable", f"dictation state unavailable ({error}); mic remains unmuted")
        if not aqua:
            self._lease = None
            self._owners.clear()
        if self._lease is None:
            return ()
        age = now - self._lease[1]
        if not 0 <= age < MAX_LEASE_SECONDS:
            self._warn("expired", "dictation lease exceeded its 15-minute limit or has a future "
                       "timestamp; mic remains unmuted")
            return ()
        return tuple(sorted({process.bundle_id for process in aqua}))
