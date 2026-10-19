"""Conservative text-only suppression of concurrent mic copies of system speech."""

from difflib import SequenceMatcher
import re


def echo_match(mic, system):
    """Match substantial, nearly identical text only when wall spans overlap.

    Short acknowledgments and added near-end content are deliberately retained.
    """
    if mic.track != "mic" or system.track != "system":
        return False
    a, b = mic.wall_span, system.wall_span
    if min(a[1], b[1]) <= max(a[0], b[0]):
        return False
    left, right = (re.findall(r"\w+", u.text.casefold()) for u in (mic, system))
    if len(left) < 3 or len(left) != len(right):
        return False
    # Fuzz spelling, not missing/added words or semantic substitutions.
    return all(a == b or (min(len(a), len(b)) >= 6
                         and SequenceMatcher(None, a, b, autojunk=False).ratio() >= .9
                         and not a.endswith(b) and not b.endswith(a))
               for a, b in zip(left, right))


class EchoDeduper:
    """Hold mic lines briefly so either track can finish its copy first.

    System lines are emitted immediately. Mic lines wait three seconds after
    finalization; the recent system cache covers the diarizer's 30 s turn limit.
    """

    def __init__(self, report=None):
        self.system = []
        self.pending = []
        self.report = report or (lambda _: None)

    def feed(self, utterances, now, *, final=False):
        self.system.extend(u for u in utterances if u.track == "system")
        output = [u for u in utterances if u.track != "mic"]
        self.pending.extend((now, u) for u in utterances if u.track == "mic")
        held = []
        for at, mic in self.pending:
            match = next((u for u in self.system if echo_match(mic, u)), None)
            if match:
                self.report(f"dedupe: dropped mic echo {mic.wall:%H:%M:%S}: {mic.text!r} "
                            f"(system {match.wall:%H:%M:%S}: {match.text!r})")
            elif final or now - at >= 3:
                output.append(mic)
            else:
                held.append((at, mic))
        self.pending = held
        self.system = [u for u in self.system if u.wall_span[1] >= now - 60]
        return sorted(output, key=lambda u: u.wall_span[1])
