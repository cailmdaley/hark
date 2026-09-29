"""When a live source goes quiet and when it comes back, for markers in the transcript.

A source is quiet after QUIET_SEC with either
  - no signal: the device delivered nothing (a stalled or restarting tap, a vanished mic); or
  - silence: it delivered only digital silence while another track produced an utterance
    within the last QUIET_SEC, so a quiet room or a lull in the meeting never counts.
It is back once it delivers sound again. Each episode yields one "silent" and at most one
"back" event; an episode still open when the session ends stays open.
"""

from dataclasses import dataclass

QUIET_SEC = 90.0


@dataclass
class Quiet:
    track: str
    state: str  # "silent" or "back"
    since: float  # wall time the source went quiet
    cause: str  # "no signal" or "silence"
    at: float | None = None  # wall time sound returned, for "back"


class QuietWatch:
    def __init__(self, quiet=QUIET_SEC):
        self.quiet = quiet
        self.open = {}  # track -> (since, cause)

    def check(self, now, sources, spoken):
        """`sources`: {track: (last_audio, last_sound)}; `spoken`: {track: wall end of its last
        utterance}. Returns the events since the previous call."""
        events = []
        for track, (audio, sound) in sources.items():
            if track in self.open:
                since, cause = self.open[track]
                if sound > since:
                    del self.open[track]
                    events.append(Quiet(track, "back", since, cause, sound))
                continue
            others = any(end > now - self.quiet for other, end in spoken.items() if other != track)
            if now - audio >= self.quiet:
                self.open[track] = audio, "no signal"
            elif now - sound >= self.quiet and others:
                self.open[track] = sound, "silence"
            else:
                continue
            events.append(Quiet(track, "silent", *self.open[track]))
        return events
