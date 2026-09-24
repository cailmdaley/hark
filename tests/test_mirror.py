import shlex
import time

import pytest

from hark.mirror import TranscriptMirror


def wait_for(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    pytest.fail("timed out waiting for mirror")


def local_cat(remote):
    path = shlex.quote(str(remote))
    return ["sh", "-c", f"cat >> {path}"]


def remote_size(remote):
    path = shlex.quote(str(remote))
    return ["sh", "-c", f"if test -f {path}; then wc -c < {path}; else printf '0\\n'; fi"]


def assert_remote_is_prefix(local, remote):
    source = local.read_bytes()
    mirrored = remote.read_bytes() if remote.exists() else b""
    assert source.startswith(mirrored)


def test_streams_new_bytes_in_order_and_remote_stays_a_prefix(tmp_path):
    local, remote = tmp_path / "local.txt", tmp_path / "remote.txt"
    local.write_bytes(b"# hark meeting\n")
    mirror = TranscriptMirror(local, "fake", str(remote), command=local_cat(remote),
                              size_command=remote_size(remote), backoff=0.01)
    mirror.start()
    for line in (b"10:00:00 S1   hello\n", b"10:00:01 me   yes\n"):
        with local.open("ab") as stream:
            stream.write(line)
        wait_for(lambda: remote.exists() and remote.read_bytes() == local.read_bytes())
        assert_remote_is_prefix(local, remote)
    assert mirror.finish(timeout=3)
    assert remote.read_bytes() == local.read_bytes()


@pytest.mark.parametrize("drop_at", [17, 39], ids=["mid-line", "line-boundary"])
def test_resumes_from_remote_byte_count_after_a_drop(tmp_path, drop_at):
    local, remote = tmp_path / "local.txt", tmp_path / "remote.txt"
    local.write_bytes(b"# hark\n10:00:00 S1   a transcript line\n# ended 10:00:01\n")
    launches = 0

    def command():
        nonlocal launches
        launches += 1
        if launches == 1:
            return ["sh", "-c", f"head -c {drop_at} >> {shlex.quote(str(remote))}"]
        return local_cat(remote)

    mirror = TranscriptMirror(local, "fake", str(remote), command=command,
                              size_command=remote_size(remote), backoff=0.01)
    mirror.start()
    assert mirror.finish(timeout=5)
    assert launches >= 2
    assert remote.read_bytes() == local.read_bytes()


def test_finish_waits_until_ended_footer_is_mirrored(tmp_path):
    local, remote = tmp_path / "local.txt", tmp_path / "remote.txt"
    local.write_bytes(b"# hark\n10:00:00 S1   goodbye\n")
    mirror = TranscriptMirror(local, "fake", str(remote), command=local_cat(remote),
                              size_command=remote_size(remote), backoff=0.01)
    mirror.start()
    wait_for(lambda: remote.exists() and remote.read_bytes() == local.read_bytes())
    with local.open("ab") as stream:
        stream.write(b"# ended 10:00:05\n")
    assert mirror.finish(timeout=3)
    assert remote.read_bytes() == local.read_bytes()
    assert remote.read_bytes().endswith(b"# ended 10:00:05\n")
