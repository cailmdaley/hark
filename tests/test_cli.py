from datetime import datetime
import json
import os
from pathlib import Path
import signal
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from hark.cli import _emit
from hark.transcript import Sink, Utterance


def test_voice_matching_failure_logs_once_and_preserves_lines(tmp_path, monkeypatch):
    sink = Sink(tmp_path / "meeting.txt", "session")
    messages = []
    monkeypatch.setattr("hark.cli.log", messages.append)

    class BrokenMatcher:
        calls = 0

        def finished(self, track, utterance):
            self.calls += 1
            raise ValueError("malformed embedding")

    matcher = BrokenMatcher()
    track = object()
    failed_slots = set()
    for start, text in [(0, "first line"), (2, "second line")]:
        _emit(Utterance("system", "S1", start, start + 1, datetime.now(), text),
              sink, matcher, {"system": track}, failed_slots)
    sink.close("ended")

    transcript = (tmp_path / "meeting.txt").read_text()
    assert "first line" in transcript and "second line" in transcript
    assert matcher.calls == 1
    assert messages == ["voice: disabled matching for system/S1 after error: malformed embedding"]


def test_remote_mirror_failure_does_not_abort_local_capture(tmp_path, monkeypatch, capsys):
    import hark.cli as cli
    from hark.mirror import TranscriptMirror

    class Source:
        name = "file"
        anchor = time.time()

        def __init__(self, *args, **kwargs):
            self.done = threading.Event()
            self.done.set()
            self.queue = SimpleNamespace(empty=lambda: True)

        def start(self):
            pass

        def stop(self):
            pass

        def drain(self, limit):
            return np.zeros(0, dtype=np.float32)

    class FakeTrack:
        def __init__(self, name, *args, **kwargs):
            self.name, self.processed, self.audio = name, 0.0, None

        def feed(self, *args, **kwargs):
            pass

    remote = tmp_path / "remote.txt"
    remote.write_text("stale remote bytes\n")
    real_mirror = TranscriptMirror
    monkeypatch.setattr(cli, "TranscriptMirror", lambda out, host, path: real_mirror(
        out, host, path, command=lambda offset, reset=False: ["python", "-c", "import sys; sys.stdin.buffer.read()"],
        size_command=["python", "-c", f"print({remote.stat().st_size})"], backoff=0.01))
    monkeypatch.setattr(cli, "FileSource", Source)
    monkeypatch.setattr(cli, "load_models", lambda latency: (None, None))
    monkeypatch.setattr(cli, "Track", FakeTrack)
    monkeypatch.setattr(cli, "flush_tracks", lambda tracks, force=False: [])
    monkeypatch.setattr(cli, "HOME", tmp_path)
    audio = tmp_path / "x.wav"
    audio.touch()

    cli.main(["--file", str(audio), "--mirror", f"h:{remote}", "-o", str(tmp_path / "out")])

    assert not (tmp_path / "meeting.json").exists()
    transcript = (tmp_path / "out.txt").read_text()
    assert transcript.startswith("# hark ")
    assert "# ended " in transcript
    output = capsys.readouterr().err
    assert "mirror: stopped:" in output
    assert "hark mirror --resume" in output


def test_mirror_resume_verb_reopens_an_existing_target(tmp_path, monkeypatch):
    import hark.cli as cli

    seen = {}

    class FakeMirror:
        def __init__(self, local, host, remote_path, *, resume):
            seen.update(local=local, host=host, remote_path=remote_path, resume=resume)

        def start(self):
            return True

        def finish(self):
            return True

    monkeypatch.setattr(cli, "TranscriptMirror", FakeMirror)
    assert cli.main(["mirror", "--resume", str(tmp_path / "local.txt"), "candide:~/notes.txt"]) == 0
    assert seen == {"local": tmp_path / "local.txt", "host": "candide",
                    "remote_path": "~/notes.txt", "resume": True}


def fake_live_capture(monkeypatch, tmp_path, *, load_models=None):
    import hark.cli as cli

    from hark.capture import Source as LiveSource

    class Source(LiveSource):
        def __init__(self, *args, **kwargs):
            super().__init__("mic")
            self.starts = 0

        def _open(self):
            self.starts += 1

        def stop(self):
            pass

        def drain(self, limit):
            return np.zeros(0, dtype=np.float32)

    class FakeTrack:
        def __init__(self, name, *args, **kwargs):
            self.name, self.processed, self.audio = name, 0.0, None

        def feed(self, *args, **kwargs):
            pass

    monkeypatch.setattr(cli, "HOME", tmp_path)
    monkeypatch.setattr(cli, "MicSource", Source)
    monkeypatch.setattr(cli, "Track", FakeTrack)
    monkeypatch.setattr(cli, "load_models", load_models or (lambda latency: (None, None)))
    monkeypatch.setattr(cli, "flush_tracks", lambda tracks, force=False: [])
    return cli, Source


def record_lifecycle_phases(monkeypatch, cli, tmp_path):
    phases = []
    replace = cli.os.replace

    def record_replace(source, target):
        if Path(target) == tmp_path / "meeting.json":
            phases.append(json.loads(Path(source).read_text()))
        replace(source, target)

    monkeypatch.setattr(cli.os, "replace", record_replace)
    return phases


def signal_when_live(tmp_path):
    state_path = tmp_path / "meeting.json"
    observations = []

    def send_signal():
        signal.pthread_sigmask(signal.SIG_BLOCK,
                               {signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGUSR1})
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if state_path.exists() and json.loads(state_path.read_text())["phase"] == "live":
                sent = time.monotonic()
                os.kill(os.getpid(), signal.SIGINT)
                deadline = time.monotonic() + 1
                while time.monotonic() < deadline:
                    if json.loads(state_path.read_text())["phase"] == "stopping":
                        observations.append(time.monotonic() - sent)
                        return
                    time.sleep(0.005)
                observations.append(None)
                return
            time.sleep(0.005)
        observations.append(None)

    thread = threading.Thread(target=send_signal)
    thread.start()
    return thread, observations


def test_live_capture_ends_cleanly_and_records_lifecycle(tmp_path, monkeypatch):
    cli, _ = fake_live_capture(monkeypatch, tmp_path)
    phases = record_lifecycle_phases(monkeypatch, cli, tmp_path)
    signaller, latency = signal_when_live(tmp_path)
    out = tmp_path / "sessions" / "normal.txt"

    cli.main(["--room", "-o", str(out), "--title", "Planning"])
    signaller.join(timeout=2)

    state = json.loads((tmp_path / "meeting.json").read_text())
    assert [snapshot["phase"] for snapshot in phases] == ["loading", "live", "stopping", "ended"]
    assert set(state) == {"pid", "phase", "title", "started", "transcript", "mirror", "launch", "error"}
    assert state["launch"] is None
    assert state["pid"] == os.getpid()
    assert state["title"] == "Planning"
    assert datetime.fromisoformat(state["started"]).tzinfo is not None
    assert state["transcript"] == str(out)
    assert state["mirror"] is None and state["error"] is None
    assert latency and latency[0] is not None and latency[0] < 1
    assert "# ended " in out.read_text()
    assert (tmp_path / "current.txt").resolve() == out
    assert (tmp_path / "current.jsonl").resolve() == out.with_suffix(".jsonl")


def test_signal_during_model_loading_stops_without_starting_audio(tmp_path, monkeypatch):
    import hark.cli as cli

    source_instances = []

    def load_models(latency):
        os.kill(os.getpid(), signal.SIGINT)
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            state = json.loads((tmp_path / "meeting.json").read_text())
            if state["phase"] == "stopping":
                return None, None
            time.sleep(0.005)
        pytest.fail("stopping was not written promptly during model loading")

    cli, Source = fake_live_capture(monkeypatch, tmp_path, load_models=load_models)
    original_start = Source.start

    def record_start(self):
        source_instances.append(self)
        original_start(self)

    monkeypatch.setattr(Source, "start", record_start)
    phases = record_lifecycle_phases(monkeypatch, cli, tmp_path)

    cli.main(["--room", "--title", "Loading stop"])

    state = json.loads((tmp_path / "meeting.json").read_text())
    assert [snapshot["phase"] for snapshot in phases] == ["loading", "stopping", "ended"]
    assert state["phase"] == "ended"
    assert state["title"] == "Loading stop"
    transcript = Path(state["transcript"])
    assert transcript.is_absolute() and transcript.parent == tmp_path / "sessions"
    assert source_instances == []
    assert "# ended " in transcript.read_text()


def test_signal_during_capture_finishes_mirror_after_closing_transcript(tmp_path, monkeypatch):
    import hark.cli as cli

    cli, _ = fake_live_capture(monkeypatch, tmp_path)
    phases = record_lifecycle_phases(monkeypatch, cli, tmp_path)
    out = tmp_path / "meetings" / "capture.txt"
    finished = []

    class FakeMirror:
        def __init__(self, local, host, remote_path):
            self.local, self.host, self.remote_path = local, host, remote_path

        def start(self):
            assert json.loads((tmp_path / "meeting.json").read_text())["phase"] == "live"

        def finish(self, timeout):
            assert "# ended " in self.local.read_text()
            assert json.loads((tmp_path / "meeting.json").read_text())["phase"] == "stopping"
            finished.append((self.host, self.remote_path))
            return True

    monkeypatch.setattr(cli, "TranscriptMirror", FakeMirror)
    signaller, latency = signal_when_live(tmp_path)
    cli.main(["--room", "-o", str(out), "--mirror", "candide:~/.hark/meetings/capture.txt"])
    signaller.join(timeout=2)

    state = json.loads((tmp_path / "meeting.json").read_text())
    assert [snapshot["phase"] for snapshot in phases] == ["loading", "live", "stopping", "ended"]
    assert state["mirror"] == "candide:~/.hark/meetings/capture.txt"
    assert state["phase"] == "ended"
    assert finished == [("candide", "~/.hark/meetings/capture.txt")]
    assert latency and latency[0] is not None and latency[0] < 1
    assert (tmp_path / "current.txt").resolve() == out


def test_live_capture_exception_records_one_line_failure(tmp_path, monkeypatch):
    def fail_loading(latency):
        state = json.loads((tmp_path / "meeting.json").read_text())
        assert state["phase"] == "loading"
        assert Path(state["transcript"]).is_absolute()
        raise RuntimeError("model loading failed\nwith details")

    cli, _ = fake_live_capture(monkeypatch, tmp_path, load_models=fail_loading)
    phases = record_lifecycle_phases(monkeypatch, cli, tmp_path)

    with pytest.raises(RuntimeError, match="model loading failed"):
        cli.main(["--room", "--title", "Broken"])

    state = json.loads((tmp_path / "meeting.json").read_text())
    assert [snapshot["phase"] for snapshot in phases] == ["loading", "failed"]
    assert set(state) == {"pid", "phase", "title", "started", "transcript", "mirror", "launch", "error"}
    assert state["launch"] is None
    assert state["phase"] == "failed"
    assert state["error"] == "model loading failed with details"
    assert "\n" not in state["error"]


def test_file_replay_does_not_write_lifecycle_or_change_current(tmp_path, monkeypatch):
    import hark.cli as cli

    class Source:
        name = "recording"
        anchor = 1.0

        def __init__(self, *args, **kwargs):
            self.done = threading.Event()
            self.queue = SimpleNamespace(empty=lambda: True)

        def start(self):
            self.done.set()

        def stop(self):
            pass

        def drain(self, limit):
            return np.zeros(0, dtype=np.float32)

    class FakeTrack:
        def __init__(self, name, *args, **kwargs):
            self.name, self.processed, self.audio = name, 0.0, None

        def feed(self, *args, **kwargs):
            pass

    monkeypatch.setattr(cli, "HOME", tmp_path)
    monkeypatch.setattr(cli, "FileSource", Source)
    monkeypatch.setattr(cli, "Track", FakeTrack)
    monkeypatch.setattr(cli, "load_models", lambda latency: (None, None))
    monkeypatch.setattr(cli, "flush_tracks", lambda tracks, force=False: [])
    audio = tmp_path / "recording.wav"
    audio.touch()
    previous = tmp_path / "previous.txt"
    previous.write_text("existing transcript\n")
    previous.with_suffix(".jsonl").write_text("existing records\n")
    (tmp_path / "current.txt").symlink_to(previous)
    (tmp_path / "current.jsonl").symlink_to(previous.with_suffix(".jsonl"))

    cli.main(["--file", str(audio), "-o", str(tmp_path / "replay.txt")])

    assert not (tmp_path / "meeting.json").exists()
    assert (tmp_path / "current.txt").resolve() == previous
    assert (tmp_path / "current.txt").read_text() == "existing transcript\n"
    assert (tmp_path / "current.jsonl").resolve() == previous.with_suffix(".jsonl")
    assert (tmp_path / "current.jsonl").read_text() == "existing records\n"
    assert "# ended " in (tmp_path / "replay.txt").read_text()


def test_meeting_is_not_a_hark_command():
    import hark.cli as cli

    with pytest.raises(SystemExit):
        cli.main(["meeting"])


def test_launch_id_and_unfinished_mirror_are_recorded(tmp_path, monkeypatch):
    cli, _ = fake_live_capture(monkeypatch, tmp_path)

    class Mirror:
        def __init__(self, *args):
            pass

        def start(self):
            pass

        def finish(self, timeout=30):
            return False

    monkeypatch.setattr(cli, "TranscriptMirror", Mirror)
    signaller, _ = signal_when_live(tmp_path)
    out = tmp_path / "meetings" / "call.txt"

    cli.main(["--room", "-o", str(out), "--launch", "L42", "--mirror", "remote:~/.hark/meetings/call.txt"])
    signaller.join(timeout=2)

    state = json.loads((tmp_path / "meeting.json").read_text())
    assert state["phase"] == "ended" and state["launch"] == "L42"
    assert state["error"].startswith("mirror incomplete; resume with: hark mirror --resume ")
    assert state["error"].endswith("'remote:~/.hark/meetings/call.txt'")


def read_wav(path):
    import wave

    with wave.open(str(path), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16000)
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768


def feed_chunks(Source, name, chunks):
    """Make `Source` a live track called `name` that drains `chunks`, recording what it hands out."""
    fed = []

    class Track(Source):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.name, self.left = name, list(chunks)

        def drain(self, limit):
            out = self.left.pop(0) if self.left else np.zeros(0, dtype=np.float32)
            fed.append(out)
            return out

    return Track, fed


def tone(freq, n):
    from hark.capture import on_pcm16_grid

    return on_pcm16_grid((0.25 * np.sin(2 * np.pi * freq * np.arange(n) / 16000)).astype(np.float32))


def test_room_saves_the_mic_track_by_default_sample_for_sample(tmp_path, monkeypatch):
    cli, Source = fake_live_capture(monkeypatch, tmp_path)
    mic = tone(440, 8000)
    Mic, fed = feed_chunks(Source, "mic", [mic[:5000], mic[5000:]])
    monkeypatch.setattr(cli, "MicSource", Mic)
    signaller, _ = signal_when_live(tmp_path)
    out = tmp_path / "sessions" / "phrase.txt"

    cli.main(["--room", "-o", str(out)])
    signaller.join(timeout=2)

    assert sorted(p.name for p in out.parent.glob("*.wav")) == ["phrase.mic.wav"]
    assert np.array_equal(read_wav(out.with_suffix(".mic.wav")), np.concatenate(fed))
    assert np.concatenate(fed).size == mic.size
    assert json.loads((tmp_path / "meeting.json").read_text())["phase"] == "ended"


def test_call_saves_mic_and_system_tracks_each_with_its_own_samples(tmp_path, monkeypatch):
    cli, Source = fake_live_capture(monkeypatch, tmp_path)
    Mic, mic_fed = feed_chunks(Source, "mic", [tone(440, 3000)])
    System, system_fed = feed_chunks(Source, "system", [tone(220, 4000), tone(330, 1000)])
    monkeypatch.setattr(cli, "MicSource", Mic)
    monkeypatch.setattr(cli, "SystemSource", System)
    signaller, _ = signal_when_live(tmp_path)
    out = tmp_path / "meetings" / "call.txt"

    cli.main(["-o", str(out)])
    signaller.join(timeout=2)

    assert np.array_equal(read_wav(out.with_suffix(".mic.wav")), np.concatenate(mic_fed))
    assert np.array_equal(read_wav(out.with_suffix(".system.wav")), np.concatenate(system_fed))
    assert read_wav(out.with_suffix(".system.wav")).size == 5000


def test_no_save_audio_opts_out_and_a_file_is_never_rerecorded(tmp_path, monkeypatch):
    cli, _ = fake_live_capture(monkeypatch, tmp_path)
    signaller, _ = signal_when_live(tmp_path)
    out = tmp_path / "sessions" / "quiet.txt"
    cli.main(["--room", "--no-save-audio", "-o", str(out)])
    signaller.join(timeout=2)
    assert not list(out.parent.glob("*.wav"))
    with pytest.raises(SystemExit):
        cli.main(["--room", "--save-audio", "-o", str(out)])


def test_saved_audio_replays_bit_for_bit_through_the_file_loader(tmp_path):
    from mlx_audio.stt.utils import load_audio

    from hark.capture import WavRecorder, on_pcm16_grid

    heard = on_pcm16_grid(np.random.default_rng(0).uniform(-1.2, 1.2, 16000).astype(np.float32))
    assert np.array_equal(on_pcm16_grid(heard), heard)
    recorder = WavRecorder(tmp_path / "x.mic.wav")
    recorder.write(heard[:7000])
    recorder.write(heard[7000:])
    recorder.close()
    assert np.array_equal(np.array(load_audio(str(recorder.path), sr=16000)), heard)


def test_expire_audio_deletes_only_old_wavs_under_the_given_roots(tmp_path, monkeypatch):
    import hark.cli as cli

    messages = []
    monkeypatch.setattr(cli, "log", messages.append)
    now = time.time()
    day = 86400
    files = {name: tmp_path / name for name in [
        "meetings/old.mic.wav", "meetings/old.system.wav", "meetings/old.txt", "meetings/old.jsonl",
        "meetings/recent.mic.wav", "sessions/sub/old.mic.wav", "voices/me.wav"]}
    for name, path in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        age = 3 * day if name.startswith("meetings/recent") else 15 * day
        os.utime(path, (now - age, now - age))

    removed = cli.expire_audio([tmp_path / "meetings", tmp_path / "sessions", tmp_path / "missing"], now=now)

    gone = {"meetings/old.mic.wav", "meetings/old.system.wav", "sessions/sub/old.mic.wav"}
    assert set(removed) == {files[name] for name in gone}
    assert {name for name, path in files.items() if not path.exists()} == gone
    assert len(messages) == 3 and all("older than 14 days" in m for m in messages)
    assert cli.expire_audio([tmp_path / "meetings"], days=2, now=now) == [files["meetings/recent.mic.wav"]]


def test_live_start_expires_old_audio_under_meetings_and_sessions(tmp_path, monkeypatch):
    cli, _ = fake_live_capture(monkeypatch, tmp_path)
    roots = []
    monkeypatch.setattr(cli, "expire_audio", lambda r: roots.append(r))
    signaller, _ = signal_when_live(tmp_path)
    cli.main(["--room", "-o", str(tmp_path / "sessions" / "x.txt")])
    signaller.join(timeout=2)
    assert roots == [[tmp_path / "meetings", tmp_path / "sessions"]]


def test_wav_recorder_io_error_stops_recording_not_the_meeting(tmp_path):
    import numpy as np
    from hark.capture import WavRecorder

    recorder = WavRecorder(tmp_path / "t.mic.wav")
    def full(_):
        raise OSError(28, "No space left on device")
    recorder.wav.writeframes = full
    recorder.write(np.ones(160, dtype=np.float32))
    recorder.write(np.ones(160, dtype=np.float32))
    recorder.close()
    assert recorder.wav is None


def test_expire_audio_skips_a_file_it_cannot_remove(tmp_path, monkeypatch):
    import os
    from pathlib import Path
    from hark import cli

    meetings = tmp_path / "meetings"; meetings.mkdir()
    for name in ("a.wav", "b.wav"):
        (meetings / name).write_bytes(b"")
        os.utime(meetings / name, (0, 0))
    real = Path.unlink
    def flaky(self, *a, **k):
        if self.name == "a.wav":
            raise FileNotFoundError(self)
        return real(self, *a, **k)
    monkeypatch.setattr(Path, "unlink", flaky)
    cli.expire_audio([meetings])
    assert not (meetings / "b.wav").exists()
