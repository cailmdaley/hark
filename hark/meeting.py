"""Render and prepare the host-side scribe fiber before capture starts."""

import base64
import re
import shlex
import subprocess
from pathlib import Path
from string import Template

from .mirror import _remote_path

TEMPLATE = Path(__file__).with_name("meeting.md")


def render_meeting_fiber(*, title, when, host, transcript_path, template_path=TEMPLATE):
    return Template(Path(template_path).read_text()).substitute(
        title=title, when=when, host=host, transcript_path=transcript_path,
    )


def build_remote_setup(*, host, project, store, fiber_id, under, title, agent,
                       transcript_path, body):
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
    if not host or not project or not store:
        raise ValueError("host, project, and store must be non-empty")

    encoded_body = base64.b64encode(body.encode()).decode()
    script = f"""set -eu
host={shlex.quote(host)}
store={_remote_path(store)}
project={_remote_path(project)}
fiber_id={shlex.quote(fiber_id)}
under={shlex.quote(under)}
title={shlex.quote(title)}
agent={shlex.quote(agent)}
transcript={_remote_path(transcript_path)}

cd "$project"
felt -C "$store" sync
parent_dir="$store/.felt/$under/meetings"
mkdir -p "$parent_dir" "$(dirname -- "$transcript")"
test ! -e "$transcript" || {{ echo "transcript already exists: $transcript" >&2; exit 1; }}
touch "$transcript"
body=$(printf '%s' {shlex.quote(encoded_body)} | base64 -d)
felt -C "$store" add "$fiber_id" "$title" --body "$body" --outcome 'Follow the live transcript and close after # ended.' --tag meeting
if [ ! -f "$store/.felt/roles/scribe/$agent/$agent.md" ]; then
  felt -C "$store" add "roles/scribe/$agent" "$agent · scribe" --body 'Model-specific continuity for the scribe role.'
fi
felt -C "$store" shuttle install "$fiber_id" --host "$host" --project-dir "$project" --model "$agent"
felt -C "$store" shuttle assign "$fiber_id" --role scribe --collaborator "$agent"
felt -C "$store" edit "$fiber_id" --status active
felt -C "$store" sync --push
felt -C "$store" shuttle dispatch "$fiber_id"
printf 'Meeting fiber: %s\\nTranscript: %s\\n' "$fiber_id" "$transcript"
"""
    return ["ssh", host, "bash -s"], script


def prepare_remote_meeting(*, host, project, store, fiber_id, under, title, agent,
                           transcript_path, body, runner=None):
    command, script = build_remote_setup(
        host=host, project=project, store=store, fiber_id=fiber_id, under=under,
        title=title, agent=agent, transcript_path=transcript_path, body=body,
    )
    (runner or subprocess.run)(command, input=script, text=True, check=True)
    return fiber_id, transcript_path
