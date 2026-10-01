# Transcript format

hark writes two files per session, side by side: `<stem>.txt` for people and agents that read text, and `<stem>.jsonl` for programs. Both are append-only. Nothing already written is ever rewritten, so a reader can remember its position (a line count or a byte offset) and resume from there.

By default `<stem>` is `~/.hark/sessions/<YYYY-MM-DD>_<HHMMSS>[_title]`; `-o PATH` chooses another. `~/.hark/current.txt` and `~/.hark/current.jsonl` are symlinks to the live session's files; replays (`--file`) don't move them. Set `HARK_DIR` to move `~/.hark`.

## The text file

Utterance lines are `HH:MM:SS-HH:MM:SS <speaker> <text>`.
The first full 24-hour timestamp is the start and the second is the end; both use `HH:MM:SS`, an ASCII hyphen, and no shared-hour elision.
For live sessions these are wall-clock times.
For `--file`, they are offsets into the file, starting at `00:00:00`.

Utterance lines are appended as turns finish, in end-time order rather than start-time order.
A short interjection can therefore appear before a longer turn that started earlier, while its two timestamps keep the intervals clear.

```
19:34:12-19:34:18 me   Sure, I reran the pipeline last night with the new masks
19:34:00-19:34:30 S1   Okay, let's get started.
19:34:29-19:34:38 S2   Did anyone check whether the redshift distributions changed?
```

Speaker labels:

- `me`: the microphone in call mode (it isn't diarized)
- `S1` … `S8`: diarized speakers, numbered in order of first appearance, per session
- a name, once a slot has been named (see below)

Every other line starts with `# `:

| Line | Meaning |
|---|---|
| `# hark 2026-09-24 19:34 — call: me = mic, S1… = system audio` | Header: start time and mode. Room mode says `room: mic diarized as S1…`, phone mode `phone: the phone's mic diarized as S1…`, a replay `file <path>` |
| `# S2 = Ada` | From here on, slot S2 is Ada. The latest line for a slot wins. Apply it to earlier lines from that slot too |
| `# S2 = S2` | S2 is anonymous again |
| `# system audio lost at 16:37:55 — no signal from the tap; nothing from the call is being transcribed` | A live source stopped delivering audio. The time is when it went quiet |
| `# mic lost at 16:37:55 — only silence while others speak; the mic may not be captured` | A source delivers only digital silence while other tracks have speech |
| `# phone lost at 10:02:13 — no signal from the device; nothing from the phone is being transcribed` | The phone disconnected (screen locked, tab closed, network gone) |
| `# system audio back at 16:52:10 after 14m15s lost` | The source recovered |
| `# ended 19:35:19` | The session is over; nothing follows |

## The JSONL file

One JSON object per line.
Every record has `wall`, an ISO-8601 local timestamp (to the millisecond on utterances, to the second elsewhere).
There are three kinds.

**Utterance**

```json
{"wall": "2026-09-24T19:34:00.412", "track": "system", "speaker": "S1", "start": 12.4, "end": 42.4, "text": "Okay, let's get started."}
```

- `track`: `mic`, `system` or `phone` live; for `--file`, the file's name without extension
- `speaker`: the stable slot (`me`, `S1`…), even when a name is shown in the text file
- `start`, `end`: seconds into the track's saved audio file (`<stem>.<track>.wav`), so an utterance can be cut out of the recording exactly.
  The text line's end time is `wall + (end - start)`, formatted to whole seconds.
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
