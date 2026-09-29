# Transcript format

hark writes two files per session, side by side: `<stem>.txt` for people and agents that read text, and `<stem>.jsonl` for programs. Both are append-only. Nothing already written is ever rewritten, so a reader can remember its position (a line count or a byte offset) and resume from there.

By default `<stem>` is `~/.hark/sessions/<YYYY-MM-DD>_<HHMMSS>[_title]`; `-o PATH` chooses another. `~/.hark/current.txt` and `~/.hark/current.jsonl` are symlinks to the live session's files; replays (`--file`) don't move them. Set `HARK_DIR` to move `~/.hark`.

## The text file

Utterance lines are `HH:MM:SS <speaker> <text>`, with the wall-clock time the turn began (for `--file`, the offset into the file, starting at `00:00:00`):

```
19:34:17 S1   Okay, let's get started.
19:34:23 me   Sure, I reran the pipeline last night with the new masks
```

Speaker labels:

- `me`: the microphone in call mode (it isn't diarized)
- `S1` … `S8`: diarized speakers, numbered in order of first appearance, per session
- a name, once a slot has been named (see below)

Every other line starts with `# `:

| Line | Meaning |
|---|---|
| `# hark 2026-09-24 19:34 — call: me = mic, S1… = system audio` | Header: start time and mode. Room mode says `room: mic diarized as S1…`, a replay `file <path>` |
| `# S2 = Ada` | From here on, slot S2 is Ada. The latest line for a slot wins. Apply it to earlier lines from that slot too |
| `# S2 = S2` | S2 is anonymous again |
| `# system audio lost at 16:37:55 — no signal from the tap; nothing from the call is being transcribed` | A live source stopped delivering audio. The time is when it went quiet |
| `# mic lost at 16:37:55 — only silence while others speak; the mic may not be captured` | A source delivers only digital silence while other tracks have speech |
| `# system audio back at 16:52:10 after 14m15s lost` | The source recovered |
| `# ended 19:35:19` | The session is over; nothing follows |

Lines from different speakers can arrive slightly out of time order, because each speaker's turn closes on its own schedule. Sort by timestamp if order matters.

## The JSONL file

One JSON object per line. Every record has `wall`, an ISO-8601 local timestamp. There are three kinds.

**Utterance**

```json
{"wall": "2026-09-24T19:34:17", "track": "system", "speaker": "S1", "start": 12.4, "end": 18.9, "text": "Okay, let's get started.", "name": "Ada"}
```

- `track`: `mic` or `system` live; for `--file`, the file's name without extension
- `speaker`: the stable slot (`me`, `S1`…), even when a name is shown in the text file
- `start`, `end`: seconds into the track's saved audio file (`<stem>.<track>.wav`), so an utterance can be cut out of the recording exactly
- `name`: present only when the slot is named

**Naming**

```json
{"wall": "2026-09-24T19:40:02", "name": {"speaker": "S2", "as": "Ada"}}
```

`"as": null` returns the slot to its anonymous label.

**Source health**

```json
{"wall": "...", "source": {"track": "system", "state": "silent", "since": "2026-09-24T16:37:55", "cause": "no signal"}}
{"wall": "...", "source": {"track": "system", "state": "back", "since": "2026-09-24T16:37:55", "cause": "no signal", "back": "2026-09-24T16:52:10"}}
```

`cause` is `no signal` (the device delivered nothing) or `silence` (it delivered only digital silence while another track had speech).

## Reading it from an agent

- **Stream**: `tail -n +1 -F ~/.hark/current.txt`, for example under Claude Code's `Monitor` tool, so each line arrives as an event.
- **Poll**: keep a line count `n`; each pass, read `tail -n +$((n+1)) <file>` and add what you read.
- **Resolve names** by scanning for `# Sx = name` lines, and apply the latest one to every line from that slot.
- **Stop** at `# ended`.
- **Name a speaker** by running `hark name S2 "Ada"`, which appends the naming line to the current session (`--session PATH` for another one). It accepts only diarized slots (`S<n>`) and refuses a session that has ended. Appending the line to the file by hand works the same way: hark reads naming lines back from the transcript before writing each utterance.
