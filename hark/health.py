"""Transcript markers for live sources that stop delivering device samples.

A source is lost after 90 seconds without samples and back when samples resume,
including digital silence. Each episode yields one loss and at most one recovery.
"""

from dataclasses import dataclass

QUIET_SEC = 90.0


@dataclass
class Quiet:
    track: str
    state: str  # "silent" or "back"
    since: float  # wall time of the last device samples before the gap
    cause: str = "no signal"
    at: float | None = None  # wall time samples resumed, for "back"


class QuietWatch:
    def __init__(self, quiet=QUIET_SEC):
        self.quiet = quiet
        self.open = {}  # track -> (since, cause)

    def check(self, now, sources):
        """Return new events for `sources`: {track: last device delivery wall time}."""
        events = []
        for track, audio in sources.items():
            if track in self.open:
                since, cause = self.open[track]
                if audio > since:
                    del self.open[track]
                    events.append(Quiet(track, "back", since, cause, audio))
            elif now - audio >= self.quiet:
                self.open[track] = audio, "no signal"
                events.append(Quiet(track, "silent", audio))
        return events
