# Usage

## Modes

```bash
hark                  # call: mic = "me", system audio diarized as S1…S8
hark --room           # in person: the mic alone, diarized
hark --phone          # in person, a phone as the mic: PCM arrives on a socket, diarized
hark --file x.m4a     # a recording through the same streaming path (about 0.15× real time)
hark --file x.wav --realtime   # replay at real-time pace, e.g. to test an agent against it
```

`--ear local|gradium` selects the recognition backend.
The default is local when MLX is importable, otherwise Gradium.
Linux supports phone and file input; call/room devices and microphone enrollment require macOS.
Explicit `--ear local` requires MLX.
Gradium can use Mac audio devices too; the call-mode mic keeps its fixed `me` label.

Call mode assumes headphones: the mic hears only you, and everything the Mac plays (Zoom, Meet, a video) is the other side.
On speakers the call leaks into the mic.

Phone mode suits a room where the laptop's mic is in the wrong place. hark listens on `~/.hark/phone.sock` for raw 16 kHz mono s16le PCM; a relay writes it there, such as the Shuttle board's phone page, which streams the phone's mic over the tailnet. Anything that produces that PCM works too, for example `ffmpeg -i talk.m4a -f s16le -ac 1 -ar 16000 - | nc -U ~/.hark/phone.sock`. One sender at a time: a new connection replaces the old one. While nothing is connected hark records silence, and after 90 s the transcript says `# phone lost at …`. The track is saved as `<stem>.phone.wav`.

Default standalone phone capture does not publish `meeting.json`. Daemon-launched (`--launch`) meetings advertise the socket in its `phone` field; explicit output beneath the HARK home's `meetings/` directory also owns that record.

Ctrl-C ends a session cleanly: hark flushes open turns, finishes writing the audio, and writes `# ended`. SIGTERM and SIGHUP do the same, so closing the terminal doesn't lose the end. A second signal quits at once without flushing.

## Gradium

```bash
hark --phone --ear gradium --lang en
hark --file recording.wav --ear gradium --lang fr
```

The key comes from `GRADIUM_API_KEY`, otherwise `~/.config/hark/gradium.key`.
The file must belong to you, have mode `600` and contain one non-empty key line.
Never put it in the checkout or a transcript.

Gradium receives gated audio over the internet and charges 3 credits per submitted second.
The free plan has 45,000 credits per month (an unrounded upper bound of 4 hours 10 minutes of STT audio), 3 concurrent streams, and a documented 1,500-character session limit whose STT applicability is ambiguous.
The [billing probes](../README.md#phone-meetings-on-linux) cost 495 credits: one 30-second clip cost 135; two 10-second clips separated by 20 seconds of no-audio idle cost 90; transmitting silence during that gap cost 135; three separate 5-second clips cost 135.
Charges fit 15-second rounding of the service progress clock, including an observed 1.04-second decoder tail; this is not a published guarantee.
Do not estimate the bill as simply 3 × submitted seconds.
hark groups speech bursts across short quiet periods without sending discarded audio.
Requests rotate after at most 58 submitted seconds, near 1,200 recognised characters, or 60 seconds of source-clock quiet.
A quiet request ends cleanly before the observed 120-second provider no-output limit; resumed speech opens a fresh one.
Logs record submitted seconds and observed credit balances; launcher-owned lifecycle files include the latest observation.
Balances can lag settled charges.
Background noise above the energy threshold still costs money.
`--no-gradium-metering` skips balance requests; hark does not enforce a monthly budget.

Words accumulate into four seconds of unique recognised audio for CPU speaker embeddings, then appear while the request is still open.
Phrases close at an eight-second source-wall span, an 800 ms gap, a speech-flush acknowledgement or the request's end too.
A last word without `end_text` uses the next word's start, or an inferred end at a speech-flush acknowledgement or EOS; 60 source seconds of quiet closes the request.
Less than four seconds inherits the preceding speaker.
There is only one speaker per phrase; short replies and overlap can be mislabelled, and slots can fragment.
The same enrolled-voice bank and manual names apply.
Confident bank identities also anchor Gradium clusters across gaps, using the naming similarity/margin rules before generic centroid matching; anonymous behavior is unchanged.

Missing keys and authentication failures close the transcript with `# ended`, write a `# gradium …` explanation and exit nonzero; an owned meeting is marked failed.
Other provider/network failures keep capture and WAV recording running.
Lost/back markers appear only while unacknowledged speech waits for unavailable recognition: loss starts at the first waiting source sample and recovery ends at the current capture horizon.
Cancellable reconnects use exponential backoff capped at 30 seconds for the meeting's duration, replay only uncommitted audio on its original source clock and suppress accepted segments.
Idle drops with all input committed open no retry socket and produce no outage markers.
A 120-second recognition backlog bounds retained upload/replay audio. Overflow omits further recognition with source-interval notices, not the saved recording.
Changed replay boundaries can lose an overlapping continuation; the log records the skipped interval.
See the [pipeline](how-it-works.md#gradium-recognition-and-speakers) for timing, recovery and resource bounds.

## Transcript timing

Each utterance line includes its full start and end time as `HH:MM:SS-HH:MM:SS`:

```
14:03:12-14:03:18 me   Sure, I reran the pipeline last night with the new masks
14:03:00-14:03:30 S1   Okay, let's get started.
14:03:29-14:03:38 S2   Did anyone check whether the redshift distributions changed?
```

Hark appends lines as turns finish, in end-time order; start times can move backward across speakers.
The [format contract](format.md) defines the text and JSONL records.

## Options

| Option | Default | |
|---|---|---|
| `--ear local\|gradium` | local if MLX imports, otherwise Gradium | recognition and speaker attribution backend |
| `--no-gradium-metering` | | skip credit-balance requests |
| `--title TEXT` | | appended to the session filename |
| `-o PATH` | `~/.hark/sessions/…` | write the transcript here instead |
| `--mic DEVICE` | system default | input device, by name or index |
| `--lang CODE` | auto | pin the ASR language, e.g. `en-US`, `fr-FR` |
| `--latency` | `low` | local-ear diarizer lookahead: `low` 1.04 s, `very_low` 0.64 s, `ultra_low` 0.32 s. Shorter means faster lines and worse speaker separation |
| `--gap SECONDS` | `3` | local-ear silence that ends a turn when nobody else takes over |
| `--speaker-mask` | `exclusive` | local-ear speakers that hear a frame two speakers share; see [how it works](how-it-works.md#speaker-masks) |
| `--no-save-audio` | | don't keep the audio |
| `--pause-for PATTERNS` | `aquavoice,aqua-voice` | pause the mic while a matching CoreAudio process captures input; give comma-separated bundle-ID substrings or `none` |
| `--mirror HOST:PATH` | | also append the transcript to a file on another machine over SSH |
| `--launch ID` | | a launcher's id, echoed into `meeting.json` |

## Subcommands

```bash
hark name S2 "Ada" [--session PATH]         # name a speaker in the current (or given) session
hark enroll NAME [--seconds 30] [--mic DEV] # record a voiceprint from the mic
hark enroll NAME --file x.wav               # or from audio (the first 30 s by default)
hark mirror --resume LOCAL HOST:PATH        # finish a mirror that didn't complete
hark pause                                 # manually pause mic intake
hark resume                                # resume mic intake
hark pause --status                        # show manual pause state
hark processes                             # list CoreAudio process objects and input flags
```

Enrollment needs at least 5 s of audio; the voiceprint and the clip it came from (`<name>.wav`) live in `~/.hark/voices/`.
With voices enrolled, hark names a diarized slot on its own once enough of that slot's speech matches one voice clearly; see [how it works](how-it-works.md#voice-matching).
A slot someone named by hand is never renamed automatically.

`hark pause` mutes the mic of the running session until `hark resume`; it creates `~/.hark/paused` (or `$HARK_DIR/paused`), which the session polls, and `hark pause --status` reports it.
A new session starts unpaused, clearing a pause an earlier session left behind.
Manual pauses add `# paused` and `# resumed at … after …` lines to the `.txt` transcript.
`--pause-for none` disables app detection but not manual pauses.

Automatic pause detection reads CoreAudio process objects and needs no extra permission.
The default match is `aquavoice,aqua-voice` (Aqua's audio bridge and its app); `--pause-for aqua,whisper` watches any process whose bundle ID contains either string.
`hark processes` prints each process object's PID, bundle ID and input-capture flag, idle apps included, so you can find what to match.
Dictation pauses appear in the session log and JSONL, not as `.txt` lines.

Pauses apply to the mic track only (not to `--phone`); system audio keeps flowing, so the call still reaches hark.
A 300 ms lookback mutes speech that began before the watcher noticed, and a 300 ms tail keeps out the last words after a dictation app stops.
The ASR and the saved mic WAV receive the same samples, with zeros for every pause.

## Saved audio

Live sessions keep each track beside the transcript as 16 kHz mono 16-bit WAV: `<stem>.mic.wav`, and in call mode `<stem>.system.wav`. Each file holds the full source track from its first sample.
The local models receive those samples directly; Gradium receives gated spans of them, while its utterance offsets still index the complete WAV. JSONL `start`/`end` times are seconds into these files. Live input is padded to hold the wall clock, so the files stay aligned with real time even when a device stalls.

The mic is rounded to 16-bit before the models see it, so `hark --file <stem>.mic.wav` replays a track through the same pipeline with identical samples.
Paused mic spans contain digital silence in the WAV; system audio remains untouched.
This is how diarization changes get tested against real meetings.

Audio is temporary. Each live start deletes `.wav` files under `~/.hark/sessions/` and `~/.hark/meetings/` that are older than 14 days (`AUDIO_RETENTION_DAYS` in `hark/cli.py`). Transcripts are never deleted. If writing audio fails (a full disk, say), the recording stops and the transcript carries on.

## Session log

`<stem>.log` keeps hark's own log, one timestamped line per message.
It records every system-audio tap exit and restart, mic reopens and errors, dictation-pause transitions, plus a heartbeat once a minute for each live source: seconds the device delivered, seconds hark padded, peak amplitude, and utterances.
It's the first place to look when a transcript has a gap.

## Lost sources

A live source is marked lost (see [format](format.md)) after 90 s in which the device either delivered no samples at all, or delivered only digital silence (peak ≤ 1e-4) while another track produced an utterance. A quiet room or a lull in the call therefore never counts. The rule is a heuristic, since a tap that runs but delivers zeros looks like a call where nobody else speaks. The heartbeat peaks in the log settle which one it was. Replays (`--file`) are never marked.

## Mirroring and launchers

To let an agent on another machine follow the meeting, mirror the transcript over SSH:

```bash
hark --mirror server:~/.hark/meetings/telecon.txt -o ~/.hark/meetings/telecon.txt --title telecon
```

Mirroring is best-effort: if the connection drops, local capture continues, and `hark mirror --resume` finishes the copy afterwards. The remote needs GNU `dd` (Linux; macOS's `dd` lacks `oflag=seek_bytes`), and hark refuses to mirror onto a remote file that already has content, except with `--resume`.

Launcher-owned live recordings publish `$HARK_DIR/meeting.json` (`~/.hark/meeting.json` when `HARK_DIR` is unset).
Ownership requires `--launch` or explicit `-o` physically beneath that home's `meetings/` directory.
Standalone capture without such a claim, arbitrary output paths and file transcription leave an existing record untouched.
A launcher such as [Shuttle](https://github.com/cailmdaley/felt) can watch its record rather than scraping output.
It holds the process id, phase (`loading`, `live`, `stopping`, `ended` or `failed`), title, start time, transcript path, mirror target, the `--launch` id, and any error.
A Gradium recording also includes `ear: {"name": "gradium", "seconds": 48.32, "credits_left": 44854}`.
`seconds` counts submitted audio, including reconnect replay; `credits_left` is the latest observed balance, or `null` when unavailable.
The initial and final balance are logged.
An `ended` recording whose mirror didn't finish carries the `hark mirror --resume …` command in `error`.
To stop the recording cleanly, send one SIGINT, SIGTERM or SIGHUP to `pid`.

## Environment

| Variable | |
|---|---|
| `HARK_DIR` | moves `~/.hark` (sessions, voices, manual mic-pause state, `current.txt`, `meeting.json`) |
| `HARK_AUDIOTEE` | path to the audiotee binary, if not `bin/audiotee` in the checkout |
| `GRADIUM_API_KEY` | Gradium API key; takes precedence over `~/.config/hark/gradium.key` |
