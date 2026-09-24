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


def test_remote_setup_is_one_ssh_round_trip_with_ordered_lifecycle():
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))

    fiber_id = "tools/hark/meetings/2026-09-25-shear-planning"
    transcript = "~/.hark/meetings/2026-09-25_1015_shear-planning.txt"
    command, script = build_remote_setup(
        host="candide",
        project="/automnt/n17data/cdaley/unions",
        store="~/loom",
        fiber_id=fiber_id,
        under="tools/hark",
        title="Shear planning",
        agent="claude-opus",
        transcript_path=transcript,
        body="A constitution\n\n## Desired State\n\nFollow quietly.\n\n## Status\n",
    )
    assert command == ["ssh", "candide", "bash -s"]
    assert "felt -C \"$store\" sync" in script
    assert script.index('felt -C "$store" sync') < script.index('felt -C "$store" add')
    assert 'felt -C "$store" shuttle install "$fiber_id" --host "$host" --project-dir "$project" --model "$agent"' in script
    assert 'felt -C "$store" shuttle assign "$fiber_id" --role scribe --collaborator "$agent"' in script
    assert 'felt -C "$store" edit "$fiber_id" --status active' in script
    assert 'felt -C "$store" sync --push' in script
    assert 'felt -C "$store" shuttle dispatch "$fiber_id"' in script
    assert script.index('shuttle install') < script.index('shuttle assign') < script.index('sync --push') < script.index('shuttle dispatch')
    assert 'mkdir -p "$parent_dir"' in script
    assert 'touch "$transcript"' in script
    assert "base64 -d" in script
    assert "A constitution" not in script

    result = prepare_remote_meeting(
        host="candide",
        project="/automnt/n17data/cdaley/unions",
        store="~/loom",
        fiber_id=fiber_id,
        under="tools/hark",
        title="Shear planning",
        agent="claude-opus",
        transcript_path=transcript,
        body="A constitution\n\n## Desired State\n\nFollow quietly.\n\n## Status\n",
        runner=fake_run,
    )
    assert result == (fiber_id, transcript)
    assert len(calls) == 1
    assert calls[0][0] == ["ssh", "candide", "bash -s"]
    assert calls[0][1]["check"] is True
    assert calls[0][1]["text"] is True
    assert calls[0][1]["input"] == script
