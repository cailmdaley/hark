# Usage

## Modes

```bash
hark                  # call: mic = "me", system audio diarized as S1…S8
hark --room           # in person: the mic alone, diarized
hark --file x.m4a     # a recording through the same streaming path (about 0.15× real time)
hark --file x.wav --realtime   # replay at real-time pace, e.g. to test an agent against it
```

Call mode assumes headphones: the mic hears only you, and everything the Mac plays (Zoom, Meet, a video) is the other side. On speakers the call leaks into the mic.

Ctrl-C ends a session cleanly: hark flushes open turns, finishes writing the audio, and writes `# ended`. SIGTERM and SIGHUP do the same, so closing the terminal doesn't lose the end. A second signal quits at once without flushing.

## Options

| Option | Default | |
|---|---|---|
| `--title TEXT` | | appended to the session filename |
| `-o PATH` | `~/.hark/sessions/…` | write the transcript here instead |
| `--mic DEVICE` | system default | input device, by name or index |
| `--lang CODE` | auto | pin the ASR language, e.g. `en-US`, `fr-FR` |
| `--latency` | `low` | diarizer lookahead: `low` 1.04 s, `very_low` 0.64 s, `ultra_low` 0.32 s. Shorter means faster lines and worse speaker separation |
| `--gap SECONDS` | `3` | silence that ends a turn when nobody else takes over |
| `--speaker-mask` | `exclusive` | who hears a frame two speakers share; see [how it works](how-it-works.md#speaker-masks) |
| `--no-save-audio` | | don't keep the audio |
| `--mirror HOST:PATH` | | also append the transcript to a file on another machine over SSH |
| `--launch ID` | | a launcher's id, echoed into `meeting.json` |

## Subcommands

```bash
hark name S2 "Ada" [--session PATH]         # name a speaker in the current (or given) session
hark enroll NAME [--seconds 30] [--mic DEV] # record a voiceprint from the mic
hark enroll NAME --file x.wav               # or from audio (the first 30 s by default)
hark mirror --resume LOCAL HOST:PATH        # finish a mirror that didn't complete
```

Enrollment needs at least 5 s of audio; the voiceprint and the clip it came from (`<name>.wav`) live in `~/.hark/voices/`. With voices enrolled, hark names a diarized slot on its own once enough of that slot's speech matches one voice clearly; see [how it works](how-it-works.md#voice-matching). A slot someone named by hand is never renamed automatically.

## Saved audio

Live sessions keep each track beside the transcript as 16 kHz mono 16-bit WAV: `<stem>.mic.wav`, and in call mode `<stem>.system.wav`. Each file holds exactly the samples the models heard, starting from the track's first sample. JSONL `start`/`end` times are seconds into these files. Live input is padded to hold the wall clock, so the files stay aligned with real time even when a device stalls.

The mic is rounded to 16-bit before the models see it, so `hark --file <stem>.mic.wav` replays a track through the same pipeline with identical samples. This is how diarization changes get tested against real meetings.

Audio is temporary. Each live start deletes `.wav` files under `~/.hark/sessions/` and `~/.hark/meetings/` that are older than 14 days (`AUDIO_RETENTION_DAYS` in `hark/cli.py`). Transcripts are never deleted. If writing audio fails (a full disk, say), the recording stops and the transcript carries on.

## Session log

`<stem>.log` keeps hark's own log, one timestamped line per message. It records every system-audio tap exit and restart, mic reopens and errors, plus a heartbeat once a minute for each live source: seconds the device delivered, seconds hark padded, peak amplitude, and utterances. It's the first place to look when a transcript has a gap.

## Lost sources

A live source is marked lost (see [format](format.md)) after 90 s in which the device either delivered no samples at all, or delivered only digital silence (peak ≤ 1e-4) while another track produced an utterance. A quiet room or a lull in the call therefore never counts. The rule is a heuristic, since a tap that runs but delivers zeros looks like a call where nobody else speaks. The heartbeat peaks in the log settle which one it was. Replays (`--file`) are never marked.

## Mirroring and launchers

To let an agent on another machine follow the meeting, mirror the transcript over SSH:

```bash
hark --mirror server:~/.hark/meetings/telecon.txt -o ~/.hark/meetings/telecon.txt --title telecon
```

Mirroring is best-effort: if the connection drops, local capture continues, and `hark mirror --resume` finishes the copy afterwards. The remote needs GNU `dd` (Linux; macOS's `dd` lacks `oflag=seek_bytes`), and hark refuses to mirror onto a remote file that already has content, except with `--resume`.

A program that starts hark (such as [Shuttle](https://github.com/cailmdaley/felt), which launches hark from its board and assigns an agent as scribe) can watch `$HARK_DIR/meeting.json`. It holds the process id, phase (`loading`, `live`, `stopping`, `ended` or `failed`), title, start time, transcript path, mirror target, the `--launch` id, and any error. An `ended` recording whose mirror didn't finish carries the `hark mirror --resume …` command in `error`. To stop the recording cleanly, send one SIGINT to `pid`.

## Environment

| Variable | |
|---|---|
| `HARK_DIR` | moves `~/.hark` (sessions, voices, `current.txt`, `meeting.json`) |
| `HARK_AUDIOTEE` | path to the audiotee binary, if not `bin/audiotee` in the checkout |
