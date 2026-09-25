"""Render and prepare the host-side scribe fiber before capture starts."""

import base64
import re
import shlex
import subprocess
from pathlib import Path
from string import Template

from .mirror import _remote_path

TEMPLATE = Path(__file__).with_name("meeting.md")


def render_meeting_fiber(*, title, when, host, transcript_path, mirrored=True,
                         template_path=TEMPLATE):
    transcript_note = " and is mirrored from hark on Cail's Mac" if mirrored else ""
    return Template(Path(template_path).read_text()).substitute(
        title=title, when=when, host=host, transcript_path=transcript_path,
        transcript_note=transcript_note,
    )


def _build_setup(*, project, store, fiber_id, under, title, agent, transcript_path, body):
    parent_parts = under.split("/")
    fiber_parts = fiber_id.split("/")
    if not under or under.startswith("/") or any(part in {"", ".", ".."} for part in parent_parts):
        raise ValueError(f"invalid parent fiber path: {under}")
    if (not fiber_id or fiber_id.startswith("/") or len(fiber_parts) < 2
            or any(part in {"", ".", ".."} for part in fiber_parts)
            or fiber_parts[:-2] != parent_parts or fiber_parts[-2] != "meetings"):
        raise ValueError(f"invalid meeting fiber path: {fiber_id}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", agent):
        raise ValueError(f"invalid Shuttle agent id: {agent}")
    if not project or not store:
        raise ValueError("project and store must be non-empty")

    encoded_body = base64.b64encode(body.encode()).decode()
    return f"""set -eu
store={_remote_path(store)}
project={_remote_path(project)}
fiber_id={shlex.quote(fiber_id)}
under={shlex.quote(under)}
title={shlex.quote(title)}
agent={shlex.quote(agent)}
transcript={_remote_path(transcript_path)}

cd "$project"
test -f "$store/.felt/$under/${{under##*/}}.md" || {{ printf 'parent fiber not found: %s\\n' "$store/.felt/$under/${{under##*/}}.md" >&2; exit 1; }}
test ! -e "$transcript" || {{ echo "transcript already exists: $transcript" >&2; exit 1; }}
felt -C "$store" sync </dev/null
body=$(printf '%s' {shlex.quote(encoded_body)} | base64 -d)
felt -C "$store" add --top-level --body "$body" --outcome 'Follow the live transcript and close after # ended.' --tag meeting -- "$fiber_id" "$title" </dev/null
mkdir -p -- "$(dirname -- "$transcript")"
touch -- "$transcript"
if [ ! -f "$store/.felt/roles/scribe/$agent/$agent.md" ]; then
  felt -C "$store" add --top-level --body 'Model-specific continuity for the scribe role.' -- "roles/scribe/$agent" "$agent · scribe" </dev/null
fi
felt -C "$store" shuttle install "$fiber_id" --project-dir "$project" --model "$agent" </dev/null
felt -C "$store" shuttle assign "$fiber_id" --role scribe --collaborator "$agent" </dev/null
felt -C "$store" edit "$fiber_id" --status active </dev/null
felt -C "$store" sync --push </dev/null
felt -C "$store" shuttle dispatch "$fiber_id" </dev/null
printf 'Meeting fiber: %s\\nTranscript: %s\\n' "$fiber_id" "$transcript"
"""


def build_remote_setup(*, host, project, store, fiber_id, under, title, agent,
                       transcript_path, body):
    if not host:
        raise ValueError("host must be non-empty")
    script = _build_setup(
        project=project, store=store, fiber_id=fiber_id, under=under, title=title,
        agent=agent, transcript_path=transcript_path, body=body,
    )
    return ["ssh", host, "bash", "-c", shlex.quote(script)], script


def build_local_setup(*, project, store, fiber_id, under, title, agent,
                      transcript_path, body):
    script = _build_setup(
        project=project, store=store, fiber_id=fiber_id, under=under, title=title,
        agent=agent, transcript_path=transcript_path, body=body,
    )
    return ["bash", "-c", script], script


def prepare_remote_meeting(*, host, project, store, fiber_id, under, title, agent,
                           transcript_path, body, runner=None):
    command, _ = build_remote_setup(
        host=host, project=project, store=store, fiber_id=fiber_id, under=under,
        title=title, agent=agent, transcript_path=transcript_path, body=body,
    )
    (runner or subprocess.run)(command, stdin=subprocess.DEVNULL, text=True, check=True)
    return fiber_id, transcript_path


def prepare_local_meeting(*, project, store, fiber_id, under, title, agent,
                          transcript_path, body, runner=None):
    command, _ = build_local_setup(
        project=project, store=store, fiber_id=fiber_id, under=under, title=title,
        agent=agent, transcript_path=transcript_path, body=body,
    )
    (runner or subprocess.run)(command, stdin=subprocess.DEVNULL, text=True, check=True)
    return fiber_id, transcript_path
