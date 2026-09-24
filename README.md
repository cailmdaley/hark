# hark

A live, local meeting transcript with speaker labels, written to a plain text
file that any agent can follow.

```
# hark 2026-09-24 19:34 — call: me = mic, S1… = system audio
19:34:17 S1   Okay, let's get started. I want to go over the cosmic shear covariance
19:34:23 me   Sure, I reran the pipeline last night with the new masks
19:34:29 S2   Did anyone check whether the redshift distributions changed?
# ended 19:35:19
```

Everything runs on the Mac (Apple Silicon, MLX): Nemotron 3.5 streaming ASR
(`mlx-community/nemotron-3.5-asr-streaming-0.6b`) gated by
Nemotron-3-Diarization (`mlx-community/Nemotron-3-Diarization`, up to 8
speakers, 1.04 s buffer) through mlx-audio's `SpeakerStreamingSession`. That
session keeps one ASR decoder per speaker, so words arrive already attributed.
Labels are anonymous, numbered in order of arrival. Nothing leaves the machine.

## Setup

```bash
scripts/build-audiotee.sh   # system-audio tap (Swift, pinned commit) → bin/audiotee
uv sync
```

macOS permissions for the terminal that runs hark: **Microphone**, and
**Screen & System Audio Recording → System Audio Recording Only**. Restart the
terminal after granting them.

## Use

```bash
uv run hark                 # a call: mic = "me", system audio (Zoom…) diarized S1…S8
uv run hark --room          # in person: the mic alone, diarized
uv run hark --file x.m4a    # a recording, through the same streaming path (~0.15× real time)
uv run hark --title "shear telecon"   # names the session file
```

### Meetings on a project host

Start capture on the Mac and dispatch a scribe beside the project:

```bash
uv run hark meeting --host candide --project /path/to/project --under tools/hark --title "shear telecon"
```

The command syncs the host's felt store, creates a meeting fiber, assigns the
scribe role, and dispatches it before capturing audio.
The transcript stays on the Mac and its `.txt` file streams to
`~/.hark/meetings/` on the host; `--room` selects mic-only capture.
For rehearsals, pass `--file recording.wav --realtime`.
Use `--agent` to select the Shuttle model and `--store` to select the host's felt store.
Direct capture can also mirror a transcript with `--mirror HOST:PATH`.

Each line is a conversational turn, not a pause-delimited fragment: brief
silences keep accumulating, a sustained reply (at least 1 s of speech) ends the
turn, and a 3 s silence ends it when nobody takes over. Short backchannels do
not end another speaker's turn. A 30 s monologue is split at its longest late
pause.

The live session is `~/.hark/current.txt`, a symlink to
`~/.hark/sessions/<date>_<time>[_title].txt`. In the absence of a sustained
reply, a line appears after 3 seconds of silence by default (`--gap`).
Lines from different speakers can land slightly out of time order. The
`# ended` footer marks a finished session. Beside the text file is a `.jsonl`
with `wall, track, speaker, start, end, text` per utterance, plus name-mapping
records when labels are resolved.

## Plugging into an agent

The file is the interface.

- **Claude Code**: ask it to watch the meeting. It runs the `Monitor` tool on
  `tail -n0 -F ~/.hark/current.txt`, which delivers each new line as an event.
  `tail -F` follows the symlink, so it also picks up the next session.
- **Codex, pi, anything else**: read the file and remember the line count, then
  reread from there (`tail -n +$((n+1)) ~/.hark/current.txt`) whenever you want
  to catch up, or in a sleep loop.
- **Name speakers**: `uv run hark name S2 "Mike Hudson"` appends a mapping to
  the current session. The line `# S2 = Mike Hudson` records that mapping; future
  utterances use the name and JSONL retains the stable `speaker` slot plus `name`.
- **Resolve labels**: agents should scan the transcript for `# Sx = name` lines
  and use those mappings for speaker labels, including for earlier utterances.
- **Enroll a voice**: `uv run hark enroll me --seconds 30` records from the microphone; `--file x.wav` enrolls from audio (the first 30 seconds by default). Voiceprints live in `~/.hark/voices/` (`HARK_DIR` relocates the directory). During diarized tracks, hark names a slot after enough finished speech matches an enrolled voice; the same `# Sx = name` line is used as manual naming, and a manual name is never replaced.
- **After the meeting**: the session file is the transcript.

Latency presets: `--latency very_low` (0.64 s) or `ultra_low` (0.32 s) trade
diarization accuracy for speed. `--lang fr-FR` etc. pins the ASR language
(default: auto).
