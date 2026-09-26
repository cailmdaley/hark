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

Audio processing runs on the Mac (Apple Silicon, MLX): Nemotron 3.5 streaming ASR
(`mlx-community/nemotron-3.5-asr-streaming-0.6b`) gated by
Nemotron-3-Diarization (`mlx-community/Nemotron-3-Diarization`, up to 8
speakers, 1.04 s buffer) through mlx-audio's `SpeakerStreamingSession`.
That session keeps one ASR decoder per speaker, so words arrive already attributed.
Labels are anonymous, numbered in order of arrival.

## Setup

```bash
scripts/build-audiotee.sh   # system-audio tap (Swift, pinned commit) → bin/audiotee
uv sync
ln -s "$PWD/.venv/bin/hark" ~/.local/bin/hark
```

The symlink puts the venv's `hark` executable on the Shuttle daemon's `PATH`.

macOS permissions for the terminal that runs hark: **Microphone**, and
**Screen & System Audio Recording → System Audio Recording Only**. Restart the
terminal after granting them.

## Use

```bash
uv run hark                 # a call: mic = "me", system audio (Zoom…) diarized S1…S8
uv run hark --room          # in person: the mic alone, diarized
uv run hark --file x.m4a    # a recording, through the same streaming path (~0.15× real time)
uv run hark --title "shear telecon"   # names the session file
uv run hark --room --save-audio       # also keep the mic track beside the transcript (<stem>.wav)
```

`--save-audio` writes the mic track as 16 kHz mono 16-bit PCM WAV next to the
transcript, appended as it is captured and complete when hark stops. Live input
is padded to the wall clock, so an utterance's JSONL `start`/`end` are seconds
into that file.

### Meetings with a scribe

Start meeting mode from Shuttle's Capture form and choose Call or Room.
Shuttle starts hark on the board daemon and launches a capture agent on the project's host to create the meeting fiber and follow the transcript as scribe.

From a terminal, record locally and mirror the transcript with:

```bash
uv run hark --mirror candide:~/.hark/meetings/x.txt -o ~/.hark/meetings/x.txt --title "shear telecon"
```

A scribe can be pointed at that file by hand.
The lifecycle file at `$HARK_DIR/meeting.json` (`~/.hark/meeting.json` by default) records the process, phase (loading, live, stopping, ended or failed), title, start time, transcript, mirror, the launcher's `--launch` id, and any error.
An `ended` recording whose mirror didn't finish carries the `hark mirror --resume …` command in `error`.
Send one SIGINT to its `pid` to stop a recording cleanly; a second one quits without flushing.

Each line is a conversational turn, not a pause-delimited fragment: brief
silences keep accumulating, a sustained reply (at least 1 s of speech) ends the
turn, and a 3 s silence ends it when nobody takes over. Short backchannels do
not end another speaker's turn. A 30 s monologue is split at its longest late
pause.

The live transcript is `~/.hark/current.txt`, a symlink to the active transcript; by default, that is
`~/.hark/sessions/<date>_<time>[_title].txt`, and `-o` can choose another path.
In the absence of a sustained
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
