"""Speech said into a manually paused mic, held on disk until it is recovered or the session ends.

While `hark pause` mutes the mic or phone track, the gate hands the real samples it
zeroes to `HeldSpeech.add`. Only audio around speech is kept, through the idle gate
`VoiceTrack` uses (-50 dBFS RMS over 100 ms frames): 300 ms before a loud frame and one
second after the last. Kept audio is appended to `<stem>.held.pcm` beside the transcript
(s16le, 16 kHz, mono), indexed by `<stem>.held.jsonl`: one `{"t", "at"}` row per
contiguous span, `t` its first sample's wall-clock time (Unix seconds) and `at` that
sample's index in the PCM file; a span runs to the next row's `at` or the end of the file.

`~/.hark/held.json` publishes `{"pid", "launch", "audio", "seconds", "since",
"recovering"}`: how many seconds of muted speech the session holds, when the oldest of
it was said, and whether a recovery is running.

`hark recover` creates `~/.hark/recover`. The live loop then replays the held spans,
separated by their real silences (up to `GAP_SEC`), through a separate `hark --file`
process, so the live models never pause; once it exits, the words go into the
transcript as recovered (`Sink.recovered`) at their original wall-clock times, and the
recovered spans are dropped. Speech held meanwhile waits for the next request.

The held audio is deleted when the session stops; a session that died without stopping
leaves `held.json` naming its audio, which the next live session deletes.
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from .capture import SAMPLE_RATE, log, to_pcm16
from .live_voice import VoiceTrack

FRAME, PREROLL, THRESHOLD = VoiceTrack.frame, VoiceTrack.preroll, VoiceTrack.threshold
TAIL_SEC = 1.0  # kept after the last loud frame
GAP_SEC = 6.0  # longest silence replayed between spans, past every turn gap


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _remove(audio):
    """Delete a held store and any recovery replay beside it."""
    audio = Path(audio)
    stem = audio.name.removesuffix(".held.pcm")
    for path in (audio, audio.with_name(stem + ".held.jsonl"),
                 *audio.parent.glob(stem + ".recovering.*")):
        path.unlink(missing_ok=True)


def transcribe_file(ear="local", refined=False, language=None):
    """The replay command: a separate `hark --file` run, so live capture keeps its models."""
    def start(audio, out):
        argv = [sys.executable, "-m", "hark.cli", "--file", str(audio), "-o", str(out), "--ear", ear]
        if ear == "gradium":
            argv.append("--no-gradium-metering")
        if refined:
            argv.append("--external-ear")
        if language:
            argv += ["--lang", language]
        return subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, start_new_session=True)
    return start


class HeldSpeech:
    def __init__(self, transcript, home, transcribe, launch=None, track="mic"):
        transcript = Path(transcript)
        self.audio = transcript.with_suffix(".held.pcm")
        self.index = transcript.with_suffix(".held.jsonl")
        self.replay = transcript.with_suffix(".recovering.wav")
        self.status = Path(home) / "held.json"
        self.request = Path(home) / "recover"
        self.transcribe, self.launch, self.track = transcribe, launch, track
        self.sweep()
        self.request.unlink(missing_ok=True)
        _remove(self.audio)
        self.file = open(self.audio, "ab")
        self.spans = []  # [wall start, sample index, samples], oldest first
        self.samples = 0
        self.quiet = []  # recent (t, frame) below the gate, the next span's preroll
        self.tail = 0
        self.next_t = None
        self.job = None
        self.published = None
        self.publish()

    def sweep(self):
        """Delete what a session that died without stopping still holds."""
        try:
            left = json.loads(self.status.read_text())
            pid, audio = int(left["pid"]), left["audio"]
        except (OSError, ValueError, KeyError, TypeError):
            return
        if pid != os.getpid() and _alive(pid):
            return
        _remove(audio)
        self.status.unlink(missing_ok=True)
        log(f"held: deleted muted speech a stopped session left at {audio}")

    @property
    def seconds(self):
        return self.samples / SAMPLE_RATE

    def add(self, t, samples):
        """Keep the speech among muted samples whose first one was heard at wall time `t`."""
        if self.file is None:
            return
        if self.next_t is None or abs(t - self.next_t) * SAMPLE_RATE >= 1:
            self.quiet, self.tail = [], 0
        self.next_t = t + samples.size / SAMPLE_RATE
        for i in range(0, samples.size, FRAME):
            chunk = samples[i:i + FRAME]
            at = t + i / SAMPLE_RATE
            if float(np.mean(chunk * chunk)) >= THRESHOLD ** 2:
                for quiet_at, quiet in self.quiet:
                    self._write(quiet_at, quiet)
                self.quiet = []
                self._write(at, chunk)
                self.tail = round(TAIL_SEC * SAMPLE_RATE)
            elif self.tail:
                self._write(at, chunk)
                self.tail = max(0, self.tail - chunk.size)
            else:
                self.quiet.append((at, chunk))
                while sum(q.size for _, q in self.quiet) - self.quiet[0][1].size >= PREROLL:
                    self.quiet.pop(0)

    def _write(self, t, samples):
        last = self.spans[-1] if self.spans else None
        if last and abs(t - (last[0] + last[2] / SAMPLE_RATE)) * SAMPLE_RATE < 1:
            last[2] += samples.size
        else:
            self.spans.append([t, self.samples, samples.size])
            with open(self.index, "a") as index:
                index.write(json.dumps({"t": t, "at": self.samples}) + "\n")
        self.file.write(to_pcm16(samples).tobytes())
        self.file.flush()
        self.samples += samples.size

    def poll(self, sink):
        """Start a requested recovery, or finish one whose replay has exited."""
        if self.job and self.job["process"].poll() is not None:
            self._finish(sink)
        if not self.job and self.request.exists():
            self.request.unlink(missing_ok=True)
            if self.spans:
                self._start()
        self.publish()

    def _start(self):
        import wave

        spans = list(self.spans)
        mapping, position, previous = [], 0.5, None
        with open(self.audio, "rb") as held, wave.open(str(self.replay), "wb") as replay:
            replay.setnchannels(1)
            replay.setsampwidth(2)
            replay.setframerate(SAMPLE_RATE)
            replay.writeframes(bytes(round(position * SAMPLE_RATE) * 2))
            for t, at, count in spans:
                if previous is not None:
                    silence = min(GAP_SEC, max(0.0, t - previous))
                    replay.writeframes(bytes(round(silence * SAMPLE_RATE) * 2))
                    position += round(silence * SAMPLE_RATE) / SAMPLE_RATE
                held.seek(at * 2)
                replay.writeframes(held.read(count * 2))
                mapping.append((position, t))
                position += count / SAMPLE_RATE
                previous = t + count / SAMPLE_RATE
        out = self.replay.with_suffix(".txt")
        self.job = {"process": self.transcribe(self.replay, out), "spans": len(spans),
                    "seconds": sum(count for *_, count in spans) / SAMPLE_RATE,
                    "mapping": mapping, "out": out}
        log(f"held: recovering {self.job['seconds']:.0f} s of muted speech")

    def _wall(self, mapping, offset):
        start, t = mapping[0]
        for start_, t_ in mapping:
            if start_ > offset:
                break
            start, t = start_, t_
        return t + offset - start

    def _finish(self, sink):
        from .transcript import Utterance

        job, self.job = self.job, None
        code = job["process"].poll()
        rows = []
        try:
            with open(job["out"].with_suffix(".jsonl")) as records:
                rows = [row for line in records if "text" in (row := json.loads(line))]
        except (OSError, ValueError) as error:
            code = code or error
        if code == 0:
            utterances = [Utterance(self.track, row["speaker"], 0.0, row["end"] - row["start"],
                                    datetime.fromtimestamp(self._wall(job["mapping"], row["start"])).astimezone(),
                                    row["text"], name=row.get("name"))
                          for row in sorted(rows, key=lambda row: row["start"])]
            sink.recovered(utterances, job["seconds"])
            self._drop(job["spans"])
            log(f"held: recovered {len(utterances)} utterances from {job['seconds']:.0f} s of muted speech")
        else:
            sink.comment(f"recovery from mute failed; {job['seconds']:.0f} s of muted speech still held")
            log(f"held: recovery failed ({code}); the muted speech stays held")
        for path in self.replay.parent.glob(self.replay.name.removesuffix(".wav") + ".*"):
            path.unlink(missing_ok=True)

    def _drop(self, count):
        """Forget the oldest `count` spans, rewriting the store with what was held since."""
        kept = self.spans[count:]
        start = kept[0][1] if kept else self.samples
        with open(self.audio, "rb") as held:
            held.seek(start * 2)
            rest = held.read()
        self.file.close()
        with open(self.audio, "wb") as held:
            held.write(rest)
        self.file = open(self.audio, "ab")
        self.spans = [[t, at - start, n] for t, at, n in kept]
        self.samples -= start
        with open(self.index, "w") as index:
            index.writelines(json.dumps({"t": t, "at": at}) + "\n" for t, at, _ in self.spans)

    def publish(self):
        state = (round(self.seconds), self.job is not None)
        if state == self.published:
            return
        since = self.spans[0][0] if self.spans else None
        temporary = self.status.with_name(self.status.name + ".tmp")
        temporary.write_text(json.dumps({
            "pid": os.getpid(), "launch": self.launch, "audio": str(self.audio),
            "seconds": round(self.seconds, 1),
            "since": datetime.fromtimestamp(since).astimezone().isoformat(timespec="seconds") if since else None,
            "recovering": self.job is not None}))
        os.replace(temporary, self.status)
        self.published = state

    def close(self):
        """End the hold: stop any recovery and delete the held audio and its status."""
        if self.file is None:
            return
        if self.job:
            self.job["process"].kill()
            self.job["process"].wait()
            self.job = None
        self.file.close()
        self.file = None
        _remove(self.audio)
        self.status.unlink(missing_ok=True)
