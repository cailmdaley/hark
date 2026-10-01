import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest

from hark.mirror import TranscriptMirror, ssh_commands


def wait_for(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    pytest.fail("timed out waiting for mirror")


def positioned_command(remote, offset, *, reset=False):
    code = """import os, sys
path, offset, reset = sys.argv[1], int(sys.argv[2]), sys.argv[3] == '1'
fd = os.open(path, os.O_CREAT | os.O_WRONLY, 0o666)
if reset:
    os.ftruncate(fd, 0)
while data := os.read(0, 65536):
    os.pwrite(fd, data, offset)
    offset += len(data)
os.close(fd)
"""
    return [sys.executable, "-c", code, str(remote), str(offset), "1" if reset else "0"]


def remote_size(remote):
    code = "import os, sys; print(os.path.getsize(sys.argv[1]) if os.path.exists(sys.argv[1]) else 0)"
    return [sys.executable, "-c", code, str(remote)]


def test_streams_new_bytes_in_order_and_remote_stays_a_prefix(tmp_path):
    local, remote = tmp_path / "local.txt", tmp_path / "remote.txt"
    local.write_bytes(b"# hark meeting\n")
    mirror = TranscriptMirror(local, "fake", str(remote),
                              command=lambda offset, reset=False: positioned_command(remote, offset, reset=reset),
                              size_command=remote_size(remote), backoff=0.01)
    mirror.start()
    for line in (b"10:00:00-10:00:01 S1   hello\n", b"10:00:01-10:00:02 me   yes\n"):
        with local.open("ab") as stream:
            stream.write(line)
        wait_for(lambda: remote.exists() and remote.read_bytes() == local.read_bytes())
        assert local.read_bytes().startswith(remote.read_bytes())
    assert mirror.finish(timeout=3)
    assert remote.read_bytes() == local.read_bytes()


@pytest.mark.parametrize("drop_at", [17, 39], ids=["mid-line", "line-boundary"])
def test_resumes_from_remote_byte_count_after_a_drop(tmp_path, drop_at):
    local, remote = tmp_path / "local.txt", tmp_path / "remote.txt"
    local.write_bytes(b"# hark\n10:00:00-10:00:01 S1   a transcript line\n# ended 10:00:01\n")
    launches = 0

    def command(offset, reset=False):
        nonlocal launches
        launches += 1
        if launches == 1:
            code = """import os, sys
path, offset, count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
data = sys.stdin.buffer.read(count)
fd = os.open(path, os.O_CREAT | os.O_WRONLY, 0o666)
os.pwrite(fd, data, offset)
os.close(fd)
"""
            return [sys.executable, "-c", code, str(remote), str(offset), str(drop_at)]
        return positioned_command(remote, offset, reset=reset)

    mirror = TranscriptMirror(local, "fake", str(remote), command=command,
                              size_command=remote_size(remote), backoff=0.01)
    mirror.start()
    assert mirror.finish(timeout=5)
    assert launches >= 2
    assert remote.read_bytes() == local.read_bytes()


def test_remote_larger_than_local_is_rewritten_from_the_local_transcript(tmp_path):
    local, remote = tmp_path / "local.txt", tmp_path / "remote.txt"
    local.write_bytes(b"# local\n# ended\n")
    remote.write_bytes(b"stale remote data longer than the local transcript")
    mirror = TranscriptMirror(local, "fake", str(remote),
                              command=lambda offset, reset=False: positioned_command(remote, offset, reset=reset),
                              size_command=remote_size(remote), backoff=0.01)
    mirror.start()
    assert mirror.finish(timeout=3)
    assert remote.read_bytes() == local.read_bytes()


def test_finish_waits_until_ended_footer_is_mirrored(tmp_path):
    local, remote = tmp_path / "local.txt", tmp_path / "remote.txt"
    local.write_bytes(b"# hark\n10:00:00-10:00:01 S1   goodbye\n")
    mirror = TranscriptMirror(local, "fake", str(remote),
                              command=lambda offset, reset=False: positioned_command(remote, offset, reset=reset),
                              size_command=remote_size(remote), backoff=0.01)
    mirror.start()
    wait_for(lambda: remote.exists() and remote.read_bytes() == local.read_bytes())
    with local.open("ab") as stream:
        stream.write(b"\n# ended 10:00:05\n")
    assert mirror.finish(timeout=3)
    assert remote.read_bytes() == local.read_bytes()
    assert remote.read_bytes().endswith(b"# ended 10:00:05\n")


@pytest.mark.parametrize("rel", [
    "~/.hark/meetings/a b.txt",
    "~/x/it's $HOME `id` é☃.txt",
    "~/.hark/-dash.txt",
])
def test_ssh_command_strings_quote_paths(tmp_path, rel):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_dd = fake_bin / "dd"
    fake_dd.write_text("""#!/usr/bin/env python3
import os, sys
args = sys.argv[1:]
path = next(arg[3:] for arg in args if arg.startswith('of='))
offset = int(next(arg[5:] for arg in args if arg.startswith('seek=')))
data = sys.stdin.buffer.read()
fd = os.open(path, os.O_CREAT | os.O_WRONLY, 0o666)
os.pwrite(fd, data, offset)
os.close(fd)
""")
    fake_dd.chmod(0o755)
    stream, size = ssh_commands("h", rel)
    env = {**os.environ, "HOME": str(tmp_path), "PATH": f"{fake_bin}:{os.environ['PATH']}"}
    subprocess.run(["sh", "-c", stream[-1]], input=b"abc\n", env=env, check=True)
    target = tmp_path / rel[2:]
    assert target.read_bytes() == b"abc\n"
    assert subprocess.run(["sh", "-c", size[-1]], env=env, capture_output=True,
                          text=True, check=True).stdout.strip() == "4"
    assert "oflag=seek_bytes conv=notrunc" in stream[-1]
    assert all(option in stream for option in (
        "BatchMode=yes", "ConnectTimeout=10", "ServerAliveInterval=15", "ServerAliveCountMax=3"))


CHILD = textwrap.dedent("""
    import sys, signal, time
    from pathlib import Path
    from hark.mirror import TranscriptMirror
    local, remote = Path(sys.argv[1]), Path(sys.argv[2])
    launches = []
    def command(offset, reset=False):
        launches.append(1)
        code = \"import os,sys; p=sys.argv[1]; o=int(sys.argv[2]); d=sys.stdin.buffer.read(); f=os.open(p,os.O_CREAT|os.O_WRONLY,0o666); os.pwrite(f,d,o); os.close(f)\"
        return [sys.executable, \"-c\", code, str(remote), str(offset)]
    stop = False
    def on_signal(*_):
        global stop; stop = True
    signal.signal(signal.SIGINT, on_signal)
    m = TranscriptMirror(local, \"fake\", str(remote), command=command,
                         size_command=[sys.executable, \"-c\", \"import os,sys; print(os.path.getsize(sys.argv[1]) if os.path.exists(sys.argv[1]) else 0)\", str(remote)],
                         backoff=0.01)
    m.start()
    while not stop:
        time.sleep(0.02)
    with local.open(\"a\") as f:
        f.write(\"# ended\\n\")
    ok = m.finish(timeout=5)
    print(f\"launches={len(launches)} ok={ok}\", flush=True)
""")


def test_ctrl_c_does_not_kill_the_mirror_pipe(tmp_path):
    local, remote = tmp_path / "l.txt", tmp_path / "r.txt"
    local.write_text("# hark\n")
    child = subprocess.Popen([sys.executable, "-c", CHILD, str(local), str(remote)],
                             start_new_session=True, stdout=subprocess.PIPE, text=True,
                             cwd=os.path.dirname(os.path.dirname(__file__)))
    time.sleep(0.5)
    os.killpg(child.pid, signal.SIGINT)
    out, _ = child.communicate(timeout=10)
    assert remote.read_bytes() == local.read_bytes()
    assert "launches=1 ok=True" in out, out.strip()


STALE = textwrap.dedent("""
    import os, sys, time
    path, offset = sys.argv[1], int(sys.argv[2])
    data = sys.stdin.buffer.read()
    if os.fork() == 0:
        time.sleep(1.0)
        fd = os.open(path, os.O_CREAT | os.O_WRONLY, 0o666)
        os.pwrite(fd, data, offset)
        os.close(fd)
        os._exit(0)
    time.sleep(5)
""")


def test_late_bytes_from_an_abandoned_connection_do_not_duplicate(tmp_path):
    local, remote = tmp_path / "l.txt", tmp_path / "r.txt"
    local.write_bytes(b"# hark\n10:00 S1 hello\n# ended\n")
    launches = []

    def command(offset, reset=False):
        launches.append(1)
        if len(launches) == 1:
            return [sys.executable, "-c", STALE, str(remote), str(offset)]
        return positioned_command(remote, offset, reset=reset)

    mirror = TranscriptMirror(local, "fake", str(remote), command=command,
                              size_command=remote_size(remote), backoff=0.01)
    mirror.start()
    assert mirror.finish(timeout=5)
    time.sleep(1.2)
    assert remote.read_bytes() == local.read_bytes()
