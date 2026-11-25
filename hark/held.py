"""Speech said into a manually paused mic, held on disk until it is recovered or the session ends.

While `hark pause` mutes the mic or phone track, the gate hands `HeldSpeech.add` the real
samples it zeroes, with their index on the track's timeline and the wall time of the first.
Only audio around speech is kept, through the idle gate `VoiceTrack` uses (-50 dBFS RMS over
100 ms frames): 300 ms before a loud frame and one second after the last. A span is a run
of kept samples contiguous on the track's timeline; its wall-clock start is taken once,
when it opens, and a route switch (`cut`) closes it. Spans are appended, as s16le 16 kHz
mono, to ten-minute segment files in `<stem>.held/` beside the transcript, indexed by
`<stem>.held/index.jsonl`: one `{"t", "segment", "at"}` row per span, `t` its wall-clock
start (Unix seconds) and `at` its first sample in that segment. Past `CAP_SEC` of held
speech the oldest segments are dropped.

Each session registers in `~/.hark/held/<pid>.json`: `{"pid", "started", "launch",
"transcript", "audio", "seconds", "since", "recovering"}`, how many seconds of muted speech
it holds, when the oldest was said, and whether a recovery is running. `hark recover`
creates `~/.hark/held/<pid>.recover` for the session; a request while one is running is
dropped. The live loop then closes the current segment and hands the closed ones to a
separate process (`python -m hark.held`), which builds the replay and runs `hark --file`
on it, so the live loop neither copies audio nor pauses its models. The replay separates
spans by their real silence, or by `GAP_SEC` when they were said that far apart or across
a route switch, so remarks said apart never join. Once it exits, the words go into the
transcript as recovered (`Sink.recovered`) at their original wall-clock times and the
replayed segments are deleted; speech held meanwhile waits in the newer segments. A replay
that fails, or writes nothing for `STALL_SEC`, keeps everything held. A replay process
exits when its session does.

The hold is a by-product, like a saved WAV: an I/O error stops it, logged once, and never
capture. It ends with the recording process: stopping deletes it, and a session that died
without stopping is deleted by the next live session, which tells a reused pid from the
original by the process's start time.
"""

import json
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
LEAD_SEC = 0.5  # replayed silence before the first span
SEGMENT_SEC = 600
CAP_SEC = 6 * 3600
STALL_SEC = 120


def registry(home):
    return Path(home) / "held"


def started(pid):
    """The process's start time as `ps` reports it, or None when no such process runs."""
    try:
        out = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True, text=True).stdout
    except OSError:
        return None
    return out.strip() or None


def transcribe_file(ear="local", refined=False, language=None):
    """The replay command: this module's entry point, which builds the replay and runs `hark --file`."""
    def start(manifest, out):
        argv = [sys.executable, "-m", "hark.held", str(manifest),
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
        self.spans = []  # [wall start, segment, sample in segment, samples, track index, after a cut]
        self.segments = {}  # segment -> samples
        self.segment = -1
        self.file = None
        self.next_index = None
        self.quiet = []  # recent (track index, wall time, frame) below the gate: the next span's preroll
        self.tail = 0
        self.cuts = []  # route switch times the held audio has not reached
        self.job = None
        self.published = None
        self._guard(self._open)

    def _open(self):
        self.registry.mkdir(parents=True, exist_ok=True)
        self.sweep()
        shutil.rmtree(self.dir, ignore_errors=True)
        self.dir.mkdir(parents=True)
        self.request.unlink(missing_ok=True)
        self._rotate()
        self.publish()

    def _guard(self, operation, *args):
        """Run a step of the hold; an I/O error ends the hold, never the capture."""
        try:
            return operation(*args)
        except OSError as error:
            log(f"held: stopped holding muted speech ({error})")
            if self.file:
                try:
                    self.file.close()
                except OSError:
                    pass
            self.file = None

    def sweep(self):
        """Delete what sessions that died without stopping still hold."""
        for status in self.registry.glob("*.json"):
            try:
                left = json.loads(status.read_text())
                pid = int(left["pid"])
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if pid != os.getpid() and left.get("started") and started(pid) == left["started"]:
                continue
            shutil.rmtree(left.get("audio", ""), ignore_errors=True)
            status.unlink(missing_ok=True)
            status.with_suffix(".recover").unlink(missing_ok=True)
            log(f"held: deleted muted speech a stopped session left at {left.get('audio')}")

    @property
    def samples(self):
        return sum(self.segments.values())

    @property
    def seconds(self):
        return self.samples / SAMPLE_RATE

    def cut(self, at):
        """A route switch at wall time `at`: speech on either side of it is never one span."""
        self.cuts.append(at)

    def add(self, index, t, samples):
        """Keep the speech among muted samples starting at track index `index`, heard at wall time `t`."""
        if self.file:
            self._guard(self._add, index, t, samples)

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

    def _write(self, index, t, samples):
        last = self.spans[-1] if self.spans else None
        follows = last is not None and index == last[4] + last[3]
        if follows:
            t = last[0] + last[3] / SAMPLE_RATE  # one span, one clock
        cut = bool(self.cuts) and self.cuts[0] <= t
        while self.cuts and self.cuts[0] <= t:
            self.cuts.pop(0)
        if follows and not cut and last[1] == self.segment:
            last[3] += samples.size
        else:
            self.spans.append([t, self.segment, self.segments[self.segment], samples.size, index, cut])
            with open(self.dir / "index.jsonl", "a") as rows:
                rows.write(json.dumps({"t": t, "segment": self.segment, "at": self.segments[self.segment]}) + "\n")
        self.file.write(to_pcm16(samples).tobytes())
        self.file.flush()
        self.segments[self.segment] += samples.size
        if self.segments[self.segment] >= SEGMENT_SEC * SAMPLE_RATE:
            self._rotate()

    def _segment(self, segment):
        return self.dir / f"{segment:06d}.pcm"

    def _rotate(self):
        if self.file:
            self.file.close()
        self.segment += 1
        self.segments[self.segment] = 0
        self.file = open(self._segment(self.segment), "ab")

    def _drop(self, segments):
        for segment in segments:
            self._segment(segment).unlink(missing_ok=True)
            del self.segments[segment]
        self.spans = [span for span in self.spans if span[1] not in segments]

    def poll(self, sink):
        """Start a requested recovery, finish one that has ended, and publish what is held."""
        if self.file:
            self._guard(self._poll, sink)

    def _poll(self, sink):
        if self.job:
            self.request.unlink(missing_ok=True)  # one recovery at a time
            code = self.job["process"].poll()
            if code is None and time.monotonic() - self.job["checked"] >= 1:
                self.job["checked"] = time.monotonic()
                wrote = max((p.stat().st_mtime for p in self.dir.glob("replay.*")), default=0)
                if time.time() - max(wrote, self.job["since"]) > STALL_SEC:
                    self.job["process"].kill()
                    self.job["process"].wait()
                    code = f"no output for {STALL_SEC} s"
            if code is not None:
                self._finish(sink, code)
        elif self.request.exists():
            self.request.unlink()
            if self.spans:
                self._start(sink)
        else:
            old = sorted(self.segments)[:-1]
            while old and self.seconds > CAP_SEC:
                self._drop([old.pop(0)])
        self.publish()

    def _start(self, sink):
        self._rotate()
        boundary = self.segment
        pieces, mapping, position, previous = [], [], 0.0, None
        for t, segment, at, count, _, cut in self.spans:
            if segment >= boundary:
                break
            silence = (LEAD_SEC if previous is None else
                       GAP_SEC if cut else min(GAP_SEC, max(0.0, t - previous)))
            position += round(silence * SAMPLE_RATE) / SAMPLE_RATE
            pieces.append([str(self._segment(segment)), at, count, round(silence * SAMPLE_RATE)])
            mapping.append((position, position + count / SAMPLE_RATE, t))
            position += count / SAMPLE_RATE
            previous = t + count / SAMPLE_RATE
        manifest = self.dir / "replay.json"
        manifest.write_text(json.dumps({"pieces": pieces}))
        seconds = sum(piece[2] for piece in pieces) / SAMPLE_RATE
        try:
            process = self.transcribe(manifest, self.dir / "replay.txt")
        except OSError as error:
            sink.comment(f"recovery from mute failed; {seconds:.0f} s of muted speech still held")
            log(f"held: could not start the replay ({error}); the muted speech stays held")
            return
        self.job = {"process": process, "boundary": boundary, "seconds": seconds, "mapping": mapping,
                    "since": time.time(), "checked": time.monotonic()}
        log(f"held: recovering {seconds:.0f} s of muted speech")

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
        if code == 0:
            utterances = sorted((Utterance(self.track, self.track, 0.0, row["end"] - row["start"],
                                           datetime.fromtimestamp(self._wall(job["mapping"], row["start"])).astimezone(),
                                           row["text"], name=row.get("name"))
                                 for row in rows), key=lambda u: u.wall)
            sink.recovered(utterances, job["seconds"])
            self._drop([segment for segment in self.segments if segment < job["boundary"]])
            log(f"held: recovered {len(utterances)} utterances from {job['seconds']:.0f} s of muted speech")
        else:
            sink.comment(f"recovery from mute failed; {job['seconds']:.0f} s of muted speech still held")
            log(f"held: recovery failed ({code}); the muted speech stays held")
        for path in self.dir.glob("replay.*"):
            path.unlink(missing_ok=True)

    def publish(self):
        state = (self.samples > 0, int(self.seconds), self.job is not None)
        if state == self.published:
            return
        since = self.spans[0][0] if self.spans else None
        temporary = self.status.with_name(self.status.name + ".tmp")
        temporary.write_text(json.dumps({
            "pid": os.getpid(), "started": self.started, "launch": self.launch,
            "transcript": str(self.transcript), "audio": str(self.dir),
            "seconds": round(self.seconds, 1),
            "since": datetime.fromtimestamp(since).astimezone().isoformat(timespec="seconds") if since else None,
            "recovering": self.job is not None}))
        os.replace(temporary, self.status)
        self.published = state

    def close(self):
        """End the hold: stop any recovery and delete the held audio, its status and any request."""
        if self.job:
            try:
                self.job["process"].kill()
                self.job["process"].wait()
            except OSError:
                pass
            self.job = None
        if self.file:
            try:
                self.file.close()
            except OSError:
                pass
            self.file = None
        shutil.rmtree(self.dir, ignore_errors=True)
        for path in (self.status, self.request):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


def _replay(argv):
    """`python -m hark.held MANIFEST CLI-ARGS…`: build the replay, then run `hark` on it while
    the session that started it lives."""
    parent = os.getppid()

    def watch():
        while os.getppid() == parent:
            time.sleep(1)
        os._exit(1)

    threading.Thread(target=watch, daemon=True).start()
    build_replay(argv[0])
    from .cli import main

    return main(argv[1:])


if __name__ == "__main__":
    sys.exit(_replay(sys.argv[1:]))
