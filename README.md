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
uv run hark --no-save-audio # don't keep the audio
```

Live capture keeps every track next to the transcript as 16 kHz mono 16-bit PCM
WAV: `<stem>.mic.wav` (the mic: `me` in a call, the diarized track in a room) and,
in a call, `<stem>.system.wav` (the diarized system audio). Each file holds exactly
the samples its track fed the models, from the track's first sample, appended as
captured and complete when hark stops. Live input is padded to the wall clock, so
an utterance's JSONL `start`/`end` are seconds into its track's file. The mic is
rounded to the 16-bit grid before the models see it, so the file is lossless:
`hark --file x.system.wav` replays a track through the same pipeline with the same
samples (fed in different chunk sizes).

Saved audio is temporary: each live start deletes `.wav` files under
`~/.hark/meetings/` and `~/.hark/sessions/` last modified more than 14 days ago
(`AUDIO_RETENTION_DAYS` in `hark/cli.py`), logging each one. Transcripts are
never deleted.

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

- **Watch it live**: `tail -F ~/.hark/current.txt` shows each line as it
  lands. For an agent, `felt shuttle follow <transcript>` (felt's CLI, on every
  host a Shuttle scribe runs on) prints the transcript in batches: everything
  already written, then new lines held until one addresses the agent by name
  (or a mishearing of it), 150 words pile up, 15 s pass, or `# ended` arrives,
  which flushes and exits. Run it under Claude Code's `Monitor` tool so each
  batch arrives as an event; elsewhere, read its stdout batch by batch.
- **Name speakers**: `uv run hark name S2 "Mike Hudson"` appends a mapping to
  the current session. The line `# S2 = Mike Hudson` records that mapping; future
  utterances use the name and JSONL retains the stable `speaker` slot plus `name`.
  `# S2 = S2` (`hark name S2 S2`) returns the slot to its anonymous label; its
  JSONL record is `{"name": {"speaker": "S2", "as": null}}`.
- **Resolve labels**: agents should scan the transcript for `# Sx = name` lines
  and use those mappings for speaker labels, including for earlier utterances;
  the latest line for a slot wins.
- **Enroll a voice**: `uv run hark enroll me --seconds 30` records from the microphone; `--file x.wav` enrolls from audio (the first 30 seconds by default). Voiceprints live in `~/.hark/voices/` (`HARK_DIR` relocates the directory). During diarized tracks, hark names a slot after enough finished speech matches an enrolled voice: cosine ≥ 0.55 (above the worst impostor clips seen, about 0.5) and 0.21 ahead of both the next enrolled voice and any other slot that clears the same bar for that voice. A name belongs to one slot at a time; if a clearly stronger slot turns up, hark writes `# S1 = S1` for the old holder before naming the new one. The same `# Sx = name` line is used as manual naming; a slot a human named or renamed is never touched, and its name is never given to another slot.
- **After the meeting**: the session file is the transcript.

Latency presets: `--latency very_low` (0.64 s) or `ultra_low` (0.32 s) trade
diarization accuracy for speed. `--lang fr-FR` etc. pins the ASR language
(default: auto).

Speaker masks: each speaker's ASR stream hears only the 80 ms frames the
diarizer gives that speaker. The mask is temporal, not a voice separator, so a
frame two speakers hold (a backchannel under someone's sentence, a turn's
overlapping edges) carries the louder voice into both streams, and both
transcribe it. hark therefore gives each frame to the more probable speaker
only (`--speaker-mask exclusive`, the default). On AMI and Zoom recordings this
cut words copied into the wrong speaker's line by 7–25× while word recall moved
by under a point. `--speaker-mask shared` restores the upstream behaviour: every
speaker over 0.5 hears the frame.

## Evaluating diarization

`uv run python -m hark.replay AUDIO OUTDIR [--mask exclusive]` runs a file through
the live pipeline and dumps the diarizer's per-frame probabilities, every
per-speaker token and the turns. `uv run scripts/diar_eval.py OUTDIR` scores a
dump: DER, cross-slot duplicate words, and, against AMI word alignments (an
`<meeting>.rttm` beside the audio, with `words/` and `corpusResources/`) or a Zoom
`.transcript.vtt`, which words landed in a slot whose speaker didn't say them.
