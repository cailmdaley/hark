"""Speech said into a manually paused mic, held on disk until it is recovered or the session ends.

While `hark pause` mutes the mic or phone track, the gate hands `HeldSpeech.add` the real
samples it zeroes, with their index on the track's timeline and the wall time of the first.
Only audio around speech is kept, through the idle gate `VoiceTrack` uses (-50 dBFS RMS over
100 ms frames): 300 ms before a loud frame and one second after the last. A span is a run
of kept samples contiguous on the track's timeline; its wall-clock start is taken once,
when it opens, and a route switch (`cut`) ends it: the next span starts at the first
sample at or after the switch. Spans are
appended, as s16le 16 kHz mono, to ten-minute segment files in `<stem>.held/` beside the
transcript, indexed by `<stem>.held/index.jsonl`: one `{"t", "segment", "at"}` row per span,
`t` its wall-clock start (Unix seconds) and `at` its first sample in that segment.

The hold keeps the newest `CAP_SEC` of held speech: past it, the oldest segment not being
replayed is dropped, and the transcript says what span of time went. That is the one way held speech
goes before the recording process stops.

Each session registers in `~/.hark/held/<pid>.json`: `{"pid", "started", "launch",
"transcript", "audio", "seconds", "since", "recovering"}`, how many seconds of muted speech
it holds, when the oldest was said, and whether a recovery is running. `hark recover`
creates `~/.hark/held/<pid>.recover`; a request while a recovery runs is dropped. A
recovery closes the current segment and takes every span held until then, replaying them
in batches of at most `BATCH_SEC`, one after another, each through a separate process
(`python -m hark.held`) that builds its replay and runs `hark --file` on it, so the live
loop neither copies audio nor pauses its models. The replay separates spans by their real
silence, or by `GAP_SEC` when they were said that far apart or across a route switch, so
remarks said apart never join. Each batch's words go into the transcript as recovered
(`Sink.recovered`) at their original wall-clock times and its spans are dropped; once
every batch is done, or one fails or writes nothing for `STALL_SEC`, a single `recovery`
row closes the recovery, as does stopping the session in the middle of one. A failed
batch keeps its speech, and so does everything after it; an aborted batch's process is
reaped and its files removed, so a retry starts clean.
Speech held during a recovery waits for the next. A replay process exits when the session
that started it does.

The hold is a by-product, like a saved WAV: a failure to write held audio stops new
holding, logged once, and never capture; what is already held stays recoverable, and an
I/O error in a recovery step ends only that recovery. The hold ends with the recording
process: stopping deletes it, and a session that died without stopping is deleted by the
next live session, which tells a reused pid from the original by the process's start time.
"""

import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

from .capture import SAMPLE_RATE, log, to_pcm16
from .live_voice import VOICE_GAP, VoiceTrack

FRAME, PREROLL, THRESHOLD = VoiceTrack.frame, VoiceTrack.preroll, VoiceTrack.threshold
TAIL_SEC = 1.0  # kept after the last loud frame
GAP_SEC = 2 * VOICE_GAP + 5  # past the longest quiet any replay track waits before ending a turn
LEAD_SEC = 0.5  # replayed silence before a batch's first span
SEGMENT_SEC = 600
BATCH_SEC = 1200  # replayed audio, silences included, per replay process
CAP_SEC = 6 * 3600
STALL_SEC = 120
PUBLISH_SEC = 1.0


def registry(home):
    return Path(home) / "held"


def started(pid):
    """The process's start time as `ps` reports it, or None when no such process runs."""
    try:
        out = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True, text=True).stdout
    except OSError:
        return None
    return out.strip() or None


def live(status):
    """Whether a registered session's process still runs: its pid, started when it registered."""
    return bool(status.get("started")) and started(status.get("pid")) == status["started"]


def _say(message):
    """Log, where a full disk can fail the log too."""
    try:
        log(message)
    except OSError:
        pass


def transcribe_file(ear="local", refined=False, language=None):
    """The replay command: this module's entry point, which builds the replay and runs `hark --file`."""
    def start(manifest, out):
        argv = [sys.executable, "-m", "hark.held", str(manifest), str(os.getpid()),
                "--file", str(Path(manifest).with_suffix(".wav")), "-o", str(out), "--ear", ear]
        if ear == "gradium":
            argv.append("--no-gradium-metering")
        if refined:
            argv.append("--external-ear")
        if language:
            argv += ["--lang", language]
        return subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, start_new_session=True)
    return start


def build_replay(manifest):
    """Write the replay WAV beside a manifest: each piece's silence, then its held samples."""
    manifest = Path(manifest)
    audio = manifest.with_suffix(".wav")
    with wave.open(str(audio), "wb") as replay:
        replay.setnchannels(1)
        replay.setsampwidth(2)
        replay.setframerate(SAMPLE_RATE)
        for segment, at, count, silence in json.loads(manifest.read_text())["pieces"]:
            replay.writeframes(bytes(silence * 2))
            with open(segment, "rb") as held:
                held.seek(at * 2)
                replay.writeframes(held.read(count * 2))
    return audio


class HeldSpeech:
    def __init__(self, transcript, home, transcribe, launch=None, track="mic"):
        self.transcript = Path(transcript)
        self.dir = self.transcript.with_suffix(".held")
        self.registry = registry(home)
        self.status = self.registry / f"{os.getpid()}.json"
        self.request = self.registry / f"{os.getpid()}.recover"
        self.transcribe, self.launch, self.track = transcribe, launch, track
        self.started = started(os.getpid())
        self.spans = []  # [wall start, segment, sample in segment, samples, track index, after a cut, id]
        self.ids = 0
        self.segment = -1
        self.file = None  # the open segment, while holding
        self.next_index = None
        self.quiet = []  # recent (track index, wall time, frame) below the gate: the next span's preroll
        self.tail = 0
        self.cuts = []  # route switch times the held audio has not reached
        self.run = None  # a recovery: the last span it takes, and what it has recovered
        self.job = None  # the batch being replayed
        self.published = None
        self.publish_at = 0.0
        self.ready = False
        try:
            self._open()
            self.ready = True
        except OSError as error:
            self._stop(error)

    def _open(self):
        self.registry.mkdir(parents=True, exist_ok=True)
        self.sweep()
        shutil.rmtree(self.dir, ignore_errors=True)
        self.dir.mkdir(parents=True)
        self.request.unlink(missing_ok=True)
        self._rotate()
        self.publish()

    def _stop(self, error):
        """Stop new holding after an I/O error; what is held stays recoverable."""
        if self.file:
            _say(f"held: stopped holding new muted speech ({error})")
            try:
                self.file.close()
            except OSError:
                pass
        self.file = None

    def sweep(self):
        """Delete what sessions that died without stopping still hold."""
        for path in self.registry.glob("*.json*"):
            try:
                left = json.loads(path.read_text())
                int(left["pid"])
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if left["pid"] != os.getpid() and live(left):
                continue
            shutil.rmtree(left.get("audio", ""), ignore_errors=True)
            for leftover in (".json", ".json.tmp", ".recover"):
                (self.registry / f"{left['pid']}{leftover}").unlink(missing_ok=True)
            _say(f"held: deleted muted speech a stopped session left at {left.get('audio')}")

    @property
    def samples(self):
        return sum(span[3] for span in self.spans)

    @property
    def seconds(self):
        return self.samples / SAMPLE_RATE

    def cut(self, at):
        """A route switch at wall time `at`: speech on either side of it is never one span."""
        self.cuts.append(at)

    def add(self, index, t, samples):
        """Keep the speech among muted samples starting at track index `index`, heard at wall time `t`."""
        if self.file:
            try:
                self._add(index, t, samples)
            except OSError as error:
                self._stop(error)

    def _add(self, index, t, samples):
        if index != self.next_index:
            self.quiet, self.tail = [], 0
        self.next_index = index + samples.size
        for i in range(0, samples.size, FRAME):
            chunk = samples[i:i + FRAME]
            if float(np.mean(chunk * chunk)) >= THRESHOLD ** 2:
                for quiet in self.quiet:
                    self._write(*quiet)
                self.quiet = []
                self._write(index + i, t + i / SAMPLE_RATE, chunk)
                self.tail = round(TAIL_SEC * SAMPLE_RATE)
            elif self.tail:
                self._write(index + i, t + i / SAMPLE_RATE, chunk)
                self.tail = max(0, self.tail - chunk.size)
            else:
                self.quiet.append((index + i, t + i / SAMPLE_RATE, chunk))
                while sum(q[2].size for q in self.quiet) - self.quiet[0][2].size >= PREROLL:
                    self.quiet.pop(0)

    def _write(self, index, t, samples, cut=False):
        """Hold `samples`; a route switch among them ends the span at the first sample at or after it."""
        last = self.spans[-1] if self.spans else None
        follows = last is not None and index == last[4] + last[3]
        if follows:
            t = last[0] + last[3] / SAMPLE_RATE  # one span, one clock
        while self.cuts and self.cuts[0] <= t:
            self.cuts.pop(0)
            cut = True
        if self.cuts:
            split = math.ceil((self.cuts[0] - t) * SAMPLE_RATE)
            if split < samples.size:
                self._write(index, t, samples[:split], cut)
                self.cuts.pop(0)
                return self._write(index + split, t + split / SAMPLE_RATE, samples[split:], True)
        at = self.file.tell() // 2
        self.file.write(to_pcm16(samples).tobytes())
        self.file.flush()
        if follows and not cut and last[1] == self.segment:
            last[3] += samples.size
        else:
            self.ids += 1
            self.spans.append([t, self.segment, at, samples.size, index, cut, self.ids])
            with open(self.dir / "index.jsonl", "a") as rows:
                rows.write(json.dumps({"t": t, "segment": self.segment, "at": at}) + "\n")
        if at + samples.size >= SEGMENT_SEC * SAMPLE_RATE:
            self._rotate()

    def _segment(self, segment):
        return self.dir / f"{segment:06d}.pcm"

    def _rotate(self):
        if self.file:
            self.file.close()
        self.segment += 1
        self.file = open(self._segment(self.segment), "ab")

    def _forget(self, ids):
        """Drop spans, and every closed segment no span is left in."""
        self.spans = [span for span in self.spans if span[6] not in ids]
        kept = {span[1] for span in self.spans} | ({self.segment} if self.file else set())
        for path in self.dir.glob("*.pcm"):
            if int(path.stem) not in kept:
                path.unlink(missing_ok=True)

    def poll(self, sink):
        """Advance a recovery, take up a request, keep the cap, and publish what is held."""
        if not self.ready:
            return
        try:
            self._poll(sink)
        except OSError as error:
            # Holding goes on: only a failure to write held audio stops it.
            _say(f"held: a recovery step failed ({error})")
            self._abort()
            if self.run:
                self.run["failed"] = True
                try:
                    self._end(sink)
                except OSError:
                    self.run = None

    def _abort(self):
        """Stop the batch being replayed, reap its process and clear its files, so a retry starts clean."""
        if self.job:
            self.job["process"].kill()
            self.job["process"].wait()
            self.job = None
        for path in self.dir.glob("replay.*"):
            path.unlink(missing_ok=True)

    def _wrote(self):
        """When the replay last wrote; a file it renames away mid-look counts for nothing."""
        latest = 0.0
        for path in self.dir.glob("replay.*"):
            try:
                latest = max(latest, path.stat().st_mtime)
            except FileNotFoundError:
                pass
        return latest

    def _poll(self, sink):
        if self.run:
            self.request.unlink(missing_ok=True)  # one recovery at a time, however it ends
        if self.job:
            code = self.job["process"].poll()
            if code is None and time.monotonic() - self.job["checked"] >= 1:
                self.job["checked"] = time.monotonic()
                if time.time() - max(self._wrote(), self.job["since"]) > STALL_SEC:
                    self._abort()
                    _say(f"held: a replay wrote nothing for {STALL_SEC} s; its muted speech stays held")
                    self.run["failed"] = True
                    self._end(sink)
            elif code is not None:
                self._finish(sink, code)
        if self.run:
            if not self.job:
                self._next(sink)
        elif self.request.exists():
            self.request.unlink()
            if self.spans:
                if self.file:
                    self._rotate()
                self.run = {"until": self.ids, "utterances": [], "seconds": 0.0, "failed": False}
                self._next(sink)
        self._cap(sink)
        self.publish()

    def _cap(self, sink):
        replaying = {span[1] for span in self.job["spans"]} if self.job else set()
        while self.seconds > CAP_SEC:
            closed = sorted({span[1] for span in self.spans} - replaying - ({self.segment} if self.file else set()))
            if not closed:
                return
            dropped = [span for span in self.spans if span[1] == closed[0]]
            self._forget({span[6] for span in dropped})
            first = datetime.fromtimestamp(dropped[0][0])
            last = datetime.fromtimestamp(dropped[-1][0] + dropped[-1][3] / SAMPLE_RATE)
            sink.comment(f"muted speech said {first:%H:%M:%S}–{last:%H:%M:%S} dropped: over the "
                         f"{CAP_SEC / 3600:g} h of muted speech a session holds")

    def _next(self, sink):
        """Start the recovery's next batch, or end the recovery when none is left."""
        spans, length = [], LEAD_SEC
        for span in self.spans:
            if span[6] > self.run["until"]:
                break
            silence = 0.0 if not spans else GAP_SEC if span[5] else min(GAP_SEC, max(0.0, span[0] - spans[-1][0] - spans[-1][3] / SAMPLE_RATE))
            if spans and length + silence + span[3] / SAMPLE_RATE > BATCH_SEC:
                break
            spans.append(span)
            length += silence + span[3] / SAMPLE_RATE
        if not spans:
            return self._end(sink)
        pieces, mapping, position, previous = [], [], 0.0, None
        for t, segment, at, count, _, cut, _ in spans:
            silence = (LEAD_SEC if previous is None else
                       GAP_SEC if cut else min(GAP_SEC, max(0.0, t - previous)))
            position += round(silence * SAMPLE_RATE) / SAMPLE_RATE
            pieces.append([str(self._segment(segment)), at, count, round(silence * SAMPLE_RATE)])
            mapping.append((position, position + count / SAMPLE_RATE, t))
            position += count / SAMPLE_RATE
            previous = t + count / SAMPLE_RATE
        manifest = self.dir / "replay.json"
        seconds = sum(span[3] for span in spans) / SAMPLE_RATE
        try:
            for path in self.dir.glob("replay.*"):
                path.unlink(missing_ok=True)
            manifest.write_text(json.dumps({"pieces": pieces}))
            process = self.transcribe(manifest, self.dir / "replay.txt")
        except OSError as error:
            _say(f"held: could not start a replay ({error})")
            self.run["failed"] = True
            return self._end(sink)
        self.job = {"process": process, "spans": spans, "seconds": seconds, "mapping": mapping,
                    "since": time.time(), "checked": time.monotonic()}
        _say(f"held: recovering {seconds:.0f} s of muted speech")

    @staticmethod
    def _wall(mapping, offset):
        """The wall time of replay second `offset`, read in the span it is nearest."""
        start, end, t = min(mapping, key=lambda span: max(span[0] - offset, offset - span[1], 0))
        return t + min(max(offset - start, 0.0), end - start)

    def _finish(self, sink, code):
        from .transcript import Utterance

        job, self.job = self.job, None
        rows = []
        try:
            with open(self.dir / "replay.jsonl") as records:
                rows = [row for line in records if "text" in (row := json.loads(line))]
        except (OSError, ValueError) as error:
            code = code or error
        for path in self.dir.glob("replay.*"):
            path.unlink(missing_ok=True)
        if code != 0:
            _say(f"held: a replay failed ({code}); its muted speech stays held")
            self.run["failed"] = True
            return self._end(sink)
        utterances = sorted((Utterance(self.track, self.track, 0.0, row["end"] - row["start"],
                                       datetime.fromtimestamp(self._wall(job["mapping"], row["start"])).astimezone(),
                                       row["text"], name=row.get("name"))
                             for row in rows), key=lambda u: u.wall)
        sink.recovered(utterances, job["seconds"])
        self._forget({span[6] for span in job["spans"]})
        self.run["utterances"] += utterances
        self.run["seconds"] += job["seconds"]

    def _end(self, sink):
        run, self.run = self.run, None
        left = sum(span[3] for span in self.spans if span[6] <= run["until"]) / SAMPLE_RATE
        if run["failed"]:
            sink.comment(f"recovery from mute stopped; {left:.0f} s of muted speech still held")
        sink.recovery(run["utterances"], run["seconds"], left if run["failed"] else None)
        _say(f"held: recovered {len(run['utterances'])} utterances from {run['seconds']:.0f} s of muted speech"
             + (f"; {left:.0f} s still held" if run["failed"] else ""))

    def publish(self):
        state = (round(self.seconds, 2), self.run is not None)
        if state == self.published:
            return
        moved = self.published is None or state[1] != self.published[1] or (state[0] > 0) != (self.published[0] > 0)
        if not moved and time.monotonic() - self.publish_at < PUBLISH_SEC:
            return
        since = self.spans[0][0] if self.spans else None
        temporary = self.status.with_name(self.status.name + ".tmp")
        temporary.write_text(json.dumps({
            "pid": os.getpid(), "started": self.started, "launch": self.launch,
            "transcript": str(self.transcript), "audio": str(self.dir),
            "seconds": state[0],
            "since": datetime.fromtimestamp(since).astimezone().isoformat(timespec="seconds") if since else None,
            "recovering": state[1]}))
        os.replace(temporary, self.status)
        self.published, self.publish_at = state, time.monotonic()

    def close(self, sink=None):
        """End the hold: stop any recovery, closing it in `sink` with what it already recovered,
        and delete the held audio, its status and any request."""
        try:
            self._abort()
        except OSError:
            self.job = None
        if self.run and sink:
            self.run["failed"] = True
            try:
                self._end(sink)
            except OSError:
                pass
        self.run = None
        if self.file:
            try:
                self.file.close()
            except OSError:
                pass
            self.file = None
        shutil.rmtree(self.dir, ignore_errors=True)
        for path in (self.status, self.status.with_name(self.status.name + ".tmp"), self.request):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


def _replay(argv):
    """`python -m hark.held MANIFEST SESSION-PID CLI-ARGS…`: build the replay, then run `hark` on
    it while the session that started it lives."""
    parent = int(argv[1])

    def watch():
        while os.getppid() == parent:
            time.sleep(1)
        os._exit(1)

    threading.Thread(target=watch, daemon=True).start()
    build_replay(argv[0])
    from .cli import main

    return main(argv[2:])


if __name__ == "__main__":
    sys.exit(_replay(sys.argv[1:]))
