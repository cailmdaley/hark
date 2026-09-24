import ast
import shlex
import subprocess
import textwrap

import pytest

from hark.meeting import build_remote_setup, prepare_remote_meeting, render_meeting_fiber


def test_meeting_constitution_rendering_carries_scribe_contract():
    body = render_meeting_fiber(
        title="Shear planning",
        when="2026-09-25 10:15 CEST",
        host="candide",
        transcript_path="~/.hark/meetings/2026-09-25_1015_shear-planning.txt",
    )
    assert body.startswith("A scribe fiber for **Shear planning**")
    assert "Cail is on the laptop" in body
    assert "live transcript is `~/.hark/meetings/2026-09-25_1015_shear-planning.txt`" in body
    assert "mirrored from hark on Cail's Mac" in body
    assert "`me` means Cail in call mode" in body
    assert "S1… are anonymous until a `# S2 = name` line appears" in body
    assert "[[roles/scribe]]" in body
    assert "Act when addressed as \"Claude\" (ASR may hear Cloud, Clawed or Klaud)" in body
    assert "notes, decisions, open questions and action items by name" in body
    assert "timestamp and speaker label" in body
    assert "Treat proposals as proposals; do not promote them" in body
    assert "When `# ended` appears, consolidate the record and close this fiber" in body
    assert body.endswith("## Status\n")


def build_script(*, title="Shear planning", under="tools/hark"):
    fiber_id = f"{under}/meetings/2026-09-25-1015-shear-planning"
    transcript = "~/.hark/meetings/2026-09-25_1015_shear-planning.txt"
    return build_remote_setup(
        host="candide",
        project="/automnt/n17data/cdaley/unions",
        store="~/loom",
        fiber_id=fiber_id,
        under=under,
        title=title,
        agent="claude-opus",
        transcript_path=transcript,
        body="A constitution\n\n## Desired State\n\nFollow quietly.\n\n## Status\n",
    )


def test_remote_setup_is_one_ssh_round_trip_with_safe_ordering():
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))

    command, script = build_script()
    assert command[:4] == ["ssh", "candide", "bash", "-c"]
    assert command[4] == shlex.quote(script)
    assert 'test -f "$store/.felt/$under/${under##*/}.md"' in script
    assert script.index("test -f") < script.index('felt -C "$store" sync')
    assert script.index('felt -C "$store" sync') < script.index('felt -C "$store" add')
    assert script.index('felt -C "$store" add') < script.index('touch -- "$transcript"')
    assert 'felt -C "$store" add --top-level --body "$body"' in script
    assert '-- "$fiber_id" "$title"' in script
    assert 'felt -C "$store" shuttle install "$fiber_id" --host "$host" --project-dir "$project" --model "$agent" </dev/null' in script
    assert 'felt -C "$store" shuttle assign "$fiber_id" --role scribe --collaborator "$agent" </dev/null' in script
    assert 'felt -C "$store" edit "$fiber_id" --status active </dev/null' in script
    assert 'felt -C "$store" sync --push </dev/null' in script
    assert 'felt -C "$store" shuttle dispatch "$fiber_id" </dev/null' in script
    assert script.index("shuttle install") < script.index("shuttle assign") < script.index("sync --push") < script.index("shuttle dispatch")
    assert "mkdir -p -- \"$(dirname -- \"$transcript\")\"" in script
    assert "base64 -d" in script
    assert "A constitution" not in script

    result = prepare_remote_meeting(
        host="candide",
        project="/automnt/n17data/cdaley/unions",
        store="~/loom",
        fiber_id="tools/hark/meetings/2026-09-25-1015-shear-planning",
        under="tools/hark",
        title="Shear planning",
        agent="claude-opus",
        transcript_path="~/.hark/meetings/2026-09-25_1015_shear-planning.txt",
        body="A constitution\n",
        runner=fake_run,
    )
    assert result == ("tools/hark/meetings/2026-09-25-1015-shear-planning",
                      "~/.hark/meetings/2026-09-25_1015_shear-planning.txt")
    assert len(calls) == 1
    assert calls[0][0][:4] == ["ssh", "candide", "bash", "-c"]
    assert calls[0][1] == {"stdin": subprocess.DEVNULL, "text": True, "check": True}


def run_setup(tmp_path, title, *, fail=None, eat_stdin=None, under="proj/sub", parent=True):
    home, bin_ = tmp_path / "home", tmp_path / "bin"
    project = home / "proj"
    store = home / "loom" / ".felt"
    project.mkdir(parents=True, exist_ok=True)
    store.mkdir(parents=True, exist_ok=True)
    if parent:
        parent_file = store / under / f"{under.split('/')[-1]}.md"
        parent_file.parent.mkdir(parents=True, exist_ok=True)
        parent_file.touch()
    bin_.mkdir(exist_ok=True)
    log = tmp_path / "felt.log"
    stub = bin_ / "felt"
    stub.write_text(textwrap.dedent(f"""\
        #!/usr/bin/env python3
        import sys
        args = sys.argv[1:]
        open({str(log)!r}, "a").write(repr(args) + "\\n")
        verb = args[2] + (" " + args[3] if args[2] == "shuttle" else "")
        if {eat_stdin!r} and verb.startswith({eat_stdin!r}):
            sys.stdin.read()
        if {fail!r} and verb.startswith({fail!r}):
            sys.exit(3)
        """))
    stub.chmod(0o755)
    fiber = f"{under}/meetings/2026-09-25-1015-x"
    command, script = build_remote_setup(
        host="candide", project="~/proj", store="~/loom", fiber_id=fiber, under=under,
        title=title, agent="claude-opus", transcript_path="~/.hark/meetings/t.txt",
        body="body $HOME `id`\n")
    env = {"HOME": str(home), "PATH": f"{bin_}:/usr/bin:/bin"}
    remote_command = " ".join(command[2:])
    result = subprocess.run(["bash", "-c", remote_command], stdin=subprocess.DEVNULL,
                            text=True, env=env, capture_output=True)
    calls = [ast.literal_eval(line) for line in log.read_text().splitlines()] if log.exists() else []
    return result, calls, home, script


@pytest.mark.parametrize("title", ["it's \"quoted\" $HOME `id` $(id) — réunion ☃", "-starts-with-dash"])
def test_title_reaches_felt_verbatim(tmp_path, title):
    result, calls, _, _ = run_setup(tmp_path, title)
    assert result.returncode == 0, result.stderr
    add = next(c for c in calls if c[2] == "add" and "meeting" in c)
    separator = add.index("--")
    assert add[separator + 1] == "proj/sub/meetings/2026-09-25-1015-x"
    assert add[separator + 2] == title
    assert add[add.index("--body") + 1] == "body $HOME `id`"


def test_setup_failure_after_fiber_creation_preserves_transcript(tmp_path):
    result, calls, home, _ = run_setup(tmp_path, "Standup", fail="shuttle dispatch")
    assert result.returncode != 0
    assert (home / ".hark/meetings/t.txt").exists()
    assert any(c[2:4] == ["shuttle", "dispatch"] for c in calls)


def test_failed_felt_add_does_not_create_transcript(tmp_path):
    result, calls, home, _ = run_setup(tmp_path, "Standup", fail="add")
    assert result.returncode != 0
    assert not (home / ".hark/meetings/t.txt").exists()
    assert any(c[2] == "add" for c in calls)


def test_missing_under_fiber_fails_before_any_changes(tmp_path):
    result, calls, home, _ = run_setup(tmp_path, "Standup", parent=False)
    assert result.returncode != 0
    assert "parent fiber not found" in result.stderr
    assert calls == []
    assert not (home / ".hark/meetings/t.txt").exists()


def test_commands_cannot_consume_the_rest_of_the_setup_script(tmp_path):
    result, calls, _, _ = run_setup(tmp_path, "Standup", eat_stdin="sync")
    verbs = [" ".join(c[2:4]) for c in calls]
    assert result.returncode == 0, result.stderr
    assert "shuttle dispatch" in verbs
