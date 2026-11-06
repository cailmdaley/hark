# Usage

## Modes

```bash
hark                  # desktop meeting, including in person: mic + system audio when available
hark --voice          # continuous local dictation, mic only
hark --mic-only      # transcribe the mic alone; keep system audio as the echo-canceller reference
hark --phone          # mobile meeting: phone PCM arrives on a socket, diarized
hark --file x.m4a     # a recording through the same streaming path (about 0.15× real time)
hark --file x.wav --realtime   # replay at real-time pace, e.g. to test an agent against it
```

`--ear local|gradium` selects the recognition backend.
The default is local when MLX is importable, otherwise Gradium.
Linux supports phone and file input; desktop audio devices and microphone enrollment require macOS.
Explicit `--ear local` requires MLX.
Gradium can use Mac audio devices too; both ears diarize every captured track.

Default live capture is for all desktop meetings, including in-person meetings.
It listens to the mic and everything the Mac plays (Zoom, Meet, a video).
The board uses Call capture for desktop meetings and Phone capture for mobile meetings.
Every speaker gets a numbered slot, you included, with unique slots across the two tracks.
Each track is diarized independently, so a voice heard on both tracks can occupy two slots.
Software acoustic echo cancellation (AEC) uses the captured system track to suppress playback picked up by the microphone, including an explicit `--mic` device.
It adds two seconds of mic buffering for alignment; residual mic copies of overlapping system utterances are dropped when their text closely matches.
AEC doesn't change device modes or output volume, but simultaneous near-end speech can still lose words; see the [measurements and limits](aec.md).
In an in-person meeting with nothing playing, the system track can stay silent for the whole session without being marked lost.
`--call` is a hidden compatibility alias for default live capture, accepted for launchers; it doesn't change attribution or require headphones.

System capture is available when macOS device capture is supported and the audiotee path is an executable file.
If not, default capture logs a warning and uses the mic alone; build the tap with `scripts/build-audiotee.sh` or set `HARK_AUDIOTEE` to an executable.
Permissions aren't probed by this check: a tap without System Audio Recording permission can still deliver silence.
Check the session log if playback is missing; digital silence doesn't produce a lost marker.
The one-time 20 s silence message notes that silence is expected when nothing is playing and suggests checking permissions only if playback is missing.
This diagnostic is conditional because silence alone doesn't establish a permission failure.

AEC is installed with hark through the LiveKit Python package; no mic helper needs building.
Phone capture, files, enrollment and mic-only capture have no AEC or cross-track deduplication.
The mic always uses raw PortAudio capture; processing happens inside hark before recognition and optional WAV saving.

Phone mode suits a room where the laptop's mic is in the wrong place. hark listens on `~/.hark/phone.sock` for raw 16 kHz mono s16le PCM; a relay writes it there, such as the Shuttle board's phone page, which streams the phone's mic over the tailnet. Anything that produces that PCM works too, for example `ffmpeg -i talk.m4a -f s16le -ac 1 -ar 16000 - | nc -U ~/.hark/phone.sock`. One sender at a time: a new connection replaces the old one. While nothing is connected hark records silence, and after 90 s the transcript says `# phone lost at …`. The track is saved as `<stem>.phone.wav`.

Default standalone phone capture does not publish `meeting.json`. Daemon-launched (`--launch`) meetings advertise the socket in its `phone` field; explicit output beneath the HARK home's `meetings/` directory also owns that record.

Ctrl-C ends a session cleanly: hark flushes open turns, finishes writing the audio, and writes `# ended`. SIGTERM and SIGHUP do the same, so closing the terminal doesn't lose the end. A second signal quits at once without flushing.

## Continuous voice

```bash
hark --voice                    # Apple Silicon, configured voice ear, microphone only
hark --voice -o ~/notes/voice    # voice.txt, voice.jsonl and voice.log
hark pause                      # mute the mic
hark resume                     # unmute it
```

Voice uses local Nemotron models for speaker labels, timing and raw preview, with `ultra_low` diarizer lookahead (0.32 s).
An operator-configured external ear supplies the final words, once per physical speech run, after five seconds of quiet and the streaming model's drain, about 200 raw words, or a hard 75 s run cap.
A raw spelling of Claude's name closes the run early.
`--gap` and `--latency` override quiet time and diarizer lookahead; desktop meetings keep their own defaults.
A cap can move to a nearby quiet acoustic frame without exceeding the hard boundary.
A 90 s rolling PCM buffer bounds retained audio.
A launcher can pass `--settings ~/.config/shuttle/voice.json` with a top-level `run_quiet_seconds` number.
Hark reloads that quiet setting while running; missing or invalid settings use five seconds, and an explicit `--gap` takes precedence.
Run finalization, external recognition time and the consumer's delivery batching all contribute to latency.
The transcript and JSONL contain only final text; `<stem>.partial` is a private, atomically replaced raw preview, never a delivery source.
It contains one `S1: unfinished words` line per pending speaker and is empty while muted or after clean shutdown.

Configure the external ear in `~/.hark/voice_ear.py`, or point `HARK_VOICE_EAR` at an operator-owned Python module.
It exports `create(language=None)`, which runs once at startup and returns a callable accepting mono float32 audio at 16 kHz and returning finished text.
Credentials, private service protocols and any network policy belong to that module, outside the repository.
There is no built-in private service adapter or local batch recognizer.
**The adapter may upload microphone audio.** Choose and authorize it before enabling voice; meeting capture's local default does not make an external voice ear local.
A failed or empty call appends the raw Nemotron text once and logs only the exception class, never the adapter's error message or URL.

Final words inherit speaker/timing anchors by alignment to Nemotron's raw words, so corrections do not duplicate overlapping slot windows.
External word order is preserved even when speaker anchors overlap or delayed timestamps extend beyond the submitted audio.
When Nemotron supplies no words for audible speech, the external ear still receives the run; recovered words carry the unknown label `S?` and the acoustic run's timing rather than an invented identity.
Diarization can still mistake one person for several slots, and a mono microphone cannot separate simultaneous voices cleanly.

`hark --external-ear` brings the same final words to a meeting. Every track runs as voice's speech runs, on `run_quiet_seconds` from `--settings`.
Next to system audio, a run reaches the ear only if Nemotron heard words in it, and it closes on quiet since the last word, because residual call audio keeps an echo-cancelled mic above the idle gate.
An adapter that can't load leaves the recording on local words and says so in the log.
**This uploads everyone's speech**: the room through the mic and the call through system audio. A paused track, and system audio under `--mic-only`, stay on the machine.

An ear call blocks the capture loop. A failed call that took 10 s or more rests the ear for 60 s on every track, with raw recognition in its place, so a dead link can't back up the meeting; quick failures, including empty replies, don't. Runs need at least 3 s of quiet to close, whatever `run_quiet_seconds` says.

Neither a system-audio tap nor AEC runs in this mode.
Videos on headphones can't address the transcript, but speech or playback audible in the room can.
Voice capture doesn't filter for an enrolled identity.
Speaker separation, naming and optional `--keep-audio` use the ordinary mic path.
On one 120 s, four-speaker AMI room-microphone replay, `low` and `ultra_low` both recovered 62.9% of reference words in the correct speaker slot; diarization error was 25.98% and 25.99% respectively.
Duplicated words increased from 1.64% to 4.26% with the faster lookahead, so lower latency has an attribution trade-off even when aggregate recall is unchanged.
This slice isn't a corpus-wide accuracy estimate; `--latency low` retains the larger lookahead when speaker separation matters more than response time.

A -55 dBFS RMS energy gate watches 100 ms frames, retains 300 ms of pre-roll and runs the models for three seconds after the last above-threshold frame to drain delayed words.
Both models stop running once the mic is idle; model weights and bounded streaming caches stay resident.
This is an energy check, not a speech classifier: steady noise above the threshold keeps inference active, and very quiet speech below it can be missed.
Skipped silence doesn't compress the transcript or saved WAV clock, and speaker state survives it.
`scripts/voice-bench.py` measures replay timing, memory and compute without opening a microphone or calling the external service; voice mode uses a constant text stub.
`scripts/voice-accuracy.py` builds synthetic technical-speech fixtures or evaluates a recorded manifest, measuring word edit rate and compute.
Its `--backend external` explicitly invokes the configured adapter and may upload the manifest's audio; other backends stay local.

Voice writes `~/.hark/meeting.json` even without `--launch`, with `"kind": "voice"`; owned meeting captures write `"kind": "meeting"`.
The lifecycle phases, `--launch`, SIGINT/TERM shutdown and final `# ended` marker are shared.
Voice utterance JSONL `wall` is the spoken-start timestamp with microseconds and a UTC offset, so a consumer can route it by when speech began, including across midnight or a target switch within one second.
Voice keeps a manual mic pause across restarts; a new meeting clears it.

For Aqua Voice, automatic voice-mode muting follows accepted dictation start/lock/completion events in `~/Library/Logs/Aqua Voice/main.log`, rather than the bridge's open microphone.
An idle bridge cannot mute voice capture.
Default Aqua logs expose completion rather than the precise end of recording, so muting can include finalization (about 1–1.8 s in the inspected Aqua 0.21.18 logs).
Missing state or a CoreAudio read failure leaves voice capture unmuted and logs a warning; an unmatched Aqua lease stops muting after 15 minutes.
Aqua dictation lasting longer than 15 minutes can therefore reach the voice transcript.
Use `hark pause` for a strict or longer mute, or `--pause-for none` to disable automatic detection.
Other configured apps and meeting capture use CoreAudio input activity as described below.

## Gradium

```bash
hark --phone --ear gradium --lang en
hark --file recording.wav --ear gradium --lang fr
```

The key comes from `GRADIUM_API_KEY`, otherwise `~/.config/hark/gradium.key`.
The file must belong to you, have mode `600` and contain one non-empty key line.
Never put it in the checkout or a transcript.
To create it: `mkdir -p ~/.config/hark && chmod 700 ~/.config/hark`, write the key with your editor, then `chmod 600 ~/.config/hark/gradium.key`.
On Linux, install with `uv tool install .` from a checkout; the CPU speaker model downloads from Hugging Face on first use, and formats other than WAV may need `ffmpeg` on `PATH`.

Gradium receives gated audio over the internet and charges 3 credits per submitted second.
The free plan has 45,000 credits per month (an unrounded upper bound of 4 hours 10 minutes of STT audio), 3 concurrent streams, and a documented 1,500-character session limit whose STT applicability is ambiguous.
Four isolated billing probes cost 495 credits:

| Request pattern | Uploaded audio (s) | Open-socket idle, no audio sent (s) | Requests | Credits |
|---|---:|---:|---:|---:|
| 30 s speech clip | 30 | 0 | 1 | 135 |
| 10 s speech, 20 s idle, 10 s speech | 20 | 20 | 1 | 90 |
| 10 s speech, 20 s submitted silence, 10 s speech | 40 | 0 | 1 | 135 |
| Three separate 5 s speech clips | 15.12 | 0 | 3 | 135 |

The last row includes 40 ms of frame padding per request.
Earlier, 59.76 seconds across 16 short requests cost 720 credits.
Charges fit 15-second rounding of the service progress clock, including an observed 1.04-second decoder tail; this is not a published guarantee.
Do not estimate the bill as simply 3 × submitted seconds.
The developer [request limit](https://docs.gradium.ai/guides/limits) is 3,000 seconds and the [pricing FAQ](https://gradium.ai/pricing) says 300; both exceed hark's request bounds. The [FAQ](https://docs.gradium.ai/guides/faq)'s 1,500-character session limit may or may not apply to STT; hark's ~1,200-character rotation stays below it in ordinary speech.
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
14:03:12-14:03:18 S1   Sure, I reran the pipeline last night with the new masks
14:03:00-14:03:30 S2   Okay, let's get started.
14:03:29-14:03:38 S3   Did anyone check whether the redshift distributions changed?
```

Hark appends lines as turns finish; mic lines in two-track capture wait another three seconds to catch system copies that finish later.
Time ranges, rather than file order, place lines across speakers.
The [format contract](format.md) defines the text and JSONL records.

## Options

| Option | Default | |
|---|---|---|
| `--phone` | | capture only PCM from the phone socket, diarized |
| `--voice` | | continuous mic dictation with a configured external final-text ear and a local idle gate |
| `--external-ear` | | final words for every track of a meeting from the configured external ear |
| `--ear local\|gradium` | local if MLX imports, otherwise Gradium | recognition and speaker attribution backend |
| `--no-gradium-metering` | | skip credit-balance requests |
| `--title TEXT` | | appended to the session filename |
| `-o PATH` | `~/.hark/sessions/…` | write the transcript here instead |
| `--mic DEVICE` | system default | input device, by name or index; software AEC stays on with system capture |
| `--lang CODE` | auto | pin the ASR language, e.g. `en-US`, `fr-FR` |
| `--latency` | meeting `low`, voice `ultra_low` | local-ear diarizer lookahead: `low` 1.04 s, `very_low` 0.64 s, `ultra_low` 0.32 s. Shorter means faster lines and worse speaker separation |
| `--gap SECONDS` | meeting `3`, voice `2` | silence that ends a meeting turn or a voice speech run, plus model drain time |
| `--speaker-mask` | `exclusive` | local-ear speakers that hear a frame two speakers share; see [how it works](how-it-works.md#speaker-masks) |
| `--keep-audio` | off | save each track as `<stem>.<track>.wav` |
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
hark pause --system                        # mute system audio in the running session
hark resume --system                       # restore system audio
hark processes                             # list CoreAudio process objects and input flags
```

Enrollment needs at least 5 s of audio; the voiceprint and the clip it came from (`<name>.wav`) live in `~/.hark/voices/`.
To name yourself on another host, copy `~/.hark/voices/me.npy` there.
There is no fixed `me` speaker: an enrolled `me` voice can name a mic slot only when ordinary matching is confident, and automatic naming never assigns `me` to system audio.
Phone and file voice matching use their enrolled banks as usual.
With voices enrolled, hark names a diarized slot on its own once enough of that slot's speech matches one voice clearly; see [how it works](how-it-works.md#voice-matching).
A slot someone named by hand is never renamed automatically.

`hark pause` mutes the mic of the running session until `hark resume`; it creates `~/.hark/paused` (or `$HARK_DIR/paused`), which the session polls, and `hark pause --status` reports it.
A new meeting starts unpaused, clearing a pause an earlier session left behind; `--voice` preserves the manual mic pause across restarts.
Manual pauses add `# paused` and `# resumed at … after …` lines to the `.txt` transcript.
`--pause-for none` disables app detection but not manual pauses.

`--mic-only` starts a desktop session with system audio muted from the beginning. The system tap still runs as the echo-canceller reference. Use `hark resume --system` to include call audio later; `hark pause --system` mutes it again. System pause state lives at `~/.hark/system-paused`, and a fresh session clears a pause left by an earlier session.

Automatic pause detection reads CoreAudio process objects and needs no extra permission.
The default match is `aquavoice,aqua-voice` (Aqua's audio bridge and its app); `--pause-for aqua,whisper` watches any process whose bundle ID contains either string.
`hark processes` prints each process object's PID, bundle ID and input-capture flag, idle apps included, so you can find what to match.
Dictation pauses appear in the session log and JSONL, not as `.txt` lines.

Pauses apply to the mic track only (not to `--phone`); system audio keeps flowing, so the call still reaches hark.
A 300 ms lookback mutes speech that began before the watcher noticed, and a 300 ms tail keeps out the last words after a dictation app stops.
The ASR and the saved mic WAV receive the same samples, with zeros for every pause.

## Saved audio

hark keeps no audio by default: transcripts are the record. With `--keep-audio`, a live session saves each captured track beside the transcript as 16 kHz mono 16-bit WAV: `<stem>.mic.wav` and, when system audio is enabled, `<stem>.system.wav`.
Mic-only fallback saves only the mic; `--phone` saves `<stem>.phone.wav` instead.
Each file holds the full source track from its first sample.
The local models receive those samples directly; Gradium receives gated spans of them, while its utterance offsets still index the complete WAV. JSONL `start`/`end` times are seconds into these files. Live input is padded to hold the wall clock, so the files stay aligned with real time even when a device stalls.

The saved mic is the echo-cancelled, pause-muted, 16-bit PCM that the models hear, so `hark --file <stem>.mic.wav` replays it with identical samples and no second AEC pass.
Paused mic spans contain digital silence in the WAV; system audio remains untouched.
This is how diarization changes get tested against real meetings.

Kept audio is temporary. Each live start deletes `.wav` files under `~/.hark/sessions/` and `~/.hark/meetings/` that are older than 14 days (`AUDIO_RETENTION_DAYS` in `hark/cli.py`). Transcripts are never deleted. If writing audio fails (a full disk, say), the recording stops and the transcript carries on.

## Session log

`<stem>.log` keeps hark's own log, one timestamped line per message.
It records system-tap exits and restarts, raw mic reopens and errors, AEC alignment changes, residual-echo drops, dictation-pause transitions, and a heartbeat once a minute for each live source.
Heartbeats report seconds the device delivered, seconds hark padded, raw peak amplitude, and emitted utterances.
It's the first place to look when a transcript has a gap.

## Lost sources

A live source is marked lost (see [format](format.md)) after 90 s without any device samples, even if it has never carried sound.
Padding generated by hark doesn't count as device samples.
Digital silence counts as device samples, regardless of speech on other tracks or prior sound.
Silence alone can't distinguish missing playback from an intentionally quiet source.
Each loss episode produces one marker; recovery occurs when device samples resume, including zeros.
A later loss can open a new episode.
The one-time 20 s pure-silence warning stays in the session log and doesn't produce a transcript marker.
Heartbeat peaks in the session log help diagnose missing playback but don't establish whether recording permissions are correct.
Replays (`--file`) are never marked.

## Mirroring and launchers

To let an agent on another machine follow the meeting, mirror the transcript over SSH:

```bash
hark --mirror server:~/.hark/meetings/telecon.txt -o ~/.hark/meetings/telecon.txt --title telecon
```

Mirroring uses your SSH configuration, including an existing ControlMaster connection for hosts that require interactive authentication.
Connection and keepalive timeouts, plus the mirror's progress watchdog, bound broken connections.
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
