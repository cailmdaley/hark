<p align="center">
  <img src="docs/assets/header.jpg" alt="hark" width="100%">
</p>

# hark

**A live meeting transcript with speaker labels, written to a plain text file that any agent can follow.**

```
# hark 2026-09-24 19:34 — call: me = mic, S1… = system audio
19:34:12-19:34:18 me   Sure, I reran the pipeline last night with the new masks
19:34:00-19:34:30 S1   Okay, let's get started.
19:34:29-19:34:38 S2   Did anyone check whether the redshift distributions changed?
# ended 19:35:19
```

hark captures audio, recognises speech, and appends speaker-labelled time ranges to a transcript and JSONL sidecar.
Your coding agent reads the file as it grows to take notes, answer questions, or pick up tasks mid-meeting.
There are two ears:

- **Local:** NVIDIA ASR and diarization through MLX on Apple Silicon; no audio leaves the machine.
- **Gradium:** streaming cloud STT with local WeSpeaker speaker embeddings on CPU; works on Linux, including phone meetings with the Mac closed.

`--ear local|gradium` chooses the backend.
The default is `local` when MLX is importable, otherwise `gradium`.
Both use the same transcript, source-health markers, mirroring, signals and meeting lifecycle.
Gradium receives speech audio over its API; it is not an offline or private-to-the-machine mode.

## Install

Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/) are required.
For Linux / Gradium, from a checkout:

```bash
uv tool install .
hark --help
```

Linux supports `--phone` and `--file`, without MLX or audio devices.
The CPU speaker model downloads from Hugging Face on first use.
WAV decoding needs no external executable; formats such as M4A may need `ffmpeg` on `PATH`.

Local capture needs Apple Silicon and macOS 14.2 or later, plus the Swift toolchain:

```bash
scripts/build-audiotee.sh   # system-audio tap → bin/audiotee
uv sync
uv run hark
```

The MLX models need about 3 GB of disk and download on first run.
Grant the terminal **Microphone** and **Screen & System Audio Recording → System Audio Recording Only** permissions, then restart it.

## Phone meetings on Linux

Put a Gradium API key in the `GRADIUM_API_KEY` environment variable or in `~/.config/hark/gradium.key`, one line, owned by you with mode `600`.
Keep it outside the checkout and never commit it.

```bash
mkdir -p ~/.config/hark
chmod 700 ~/.config/hark
# Write your key with your editor, then:
chmod 600 ~/.config/hark/gradium.key
hark --phone --ear gradium --lang en
```

hark listens on `~/.hark/phone.sock` for mono s16le PCM at 16 kHz.
The [Shuttle](https://github.com/cailmdaley/felt) board relays the phone browser's microphone to that socket and launches ordinary `hark --phone` on the selected host.
On a Linux host, the default ear is Gradium; no daemon configuration change is needed.
The phone must stay connected and keep microphone access; an always-on ear host does not make mobile browser screen-lock capture reliable.

Gradium advertises **3 credits per audio second**, **45,000 free credits/month** and **3 concurrent STT streams**.
Four isolated billing probes measured **495 credits** in total:

| Request pattern | Uploaded audio (s) | Open-socket idle, no audio sent (s) | Requests | Credits |
|---|---:|---:|---:|---:|
| 30 s speech clip | 30 | 0 | 1 | 135 |
| 10 s speech, 20 s idle, 10 s speech | 20 | 20 | 1 | 90 |
| 10 s speech, 20 s submitted silence, 10 s speech | 40 | 0 | 1 | 135 |
| Three separate 5 s speech clips | 15.12 | 0 | 3 | 135 |

The last row includes 40 ms of frame padding per request.
The service's final progress clock exceeded uploaded audio by about 1.04 seconds.
Charges fit rounding that clock up to 15-second units; this is a measured inference, **not a published billing guarantee**.
Idle without audio had no advancing server clock; sending silence cost more.
Earlier, 59.76 seconds across 16 short requests cost 720 credits.

hark gates silence with a conservative energy check, a 320 ms pre-roll and an 800 ms tail.
Speech bursts share a socket across short quiet periods without uploading the intervening quiet.
Requests rotate at **58 submitted seconds**, near **1,200 recognised characters**, or after **60 seconds of source-clock quiet**.
Quiet closes with EOS and drains the decoder before the observed 120-second provider no-output limit; the next speech opens a fresh request.
The duration leaves two seconds of headroom below a 60-second billing unit for the observed decoder tail; a 60-second input itself could spill into a 75-second bill.
Retries, shorter requests and gate tails still cost credits.
**3 × submitted seconds is not a reliable bill estimate**; the nominal 4 hours 10 minutes is an unrounded upper bound, not a guaranteed meeting allowance.
The log and `meeting.json` record submitted seconds, including retries, and the credit balance at startup and shutdown when metering succeeds.
`--no-gradium-metering` disables balance requests.
There is no automatic monthly spending cap in hark.

The [Gradium FAQ](https://docs.gradium.ai/guides/faq) also states a free-tier limit of **1,500 characters per session**, without distinguishing STT from TTS.
hark rotates requests after at most 58 seconds of input and near 1,200 recognised characters, below that character threshold in ordinary speech.
The developer [request limit](https://docs.gradium.ai/guides/limits) is 3,000 seconds; the [pricing FAQ](https://gradium.ai/pricing) says 300 seconds.
Both exceed hark's request limits.
The STT character ceiling remains unverified.

## Use

```bash
hark                          # local call: mic = me, system audio = S1…
hark --room                   # local room: microphone diarized
hark --phone                  # phone audio on the host's default ear
hark --file talk.wav           # recording, with file-relative timestamps
hark --ear gradium --file talk.wav --lang fr
hark --phone --title telecon
hark --phone --mirror host:~/notes.txt
hark pause                    # manually mute the Mac microphone
hark resume
hark name S2 Ada               # name a speaker during the session
hark enroll Ada --file ada.wav # create a voiceprint from at least 5 s of speech
```

Ctrl-C, SIGTERM and SIGHUP flush pending text and write `# ended`.
A missing key or rejected authentication writes a clear `# gradium …` comment and closes the transcript; an owned meeting is marked failed in `meeting.json`.
Other Gradium or network failures keep source capture and WAV recording running.
`# gradium lost at …` and `# gradium back at …` mark intervals when unacknowledged speech actually waits for unavailable recognition, not idle connection churn.
Recovery retries for the meeting's duration with cancellable exponential backoff capped at 30 seconds.
A 120-second recognition backlog includes replay audio; once full, further recognition is omitted with interval notices, while recording continues.
Only a live invocation with `--launch`, or explicit `-o` under the HARK home's `meetings/` directory, owns that lifecycle record; standalone capture does not overwrite it.
Reconnects preserve the source clock and upload only the uncommitted suffix, using a fresh connection's zero-based clock.
A drop after all input is committed causes neither replay nor outage markers; the next speech opens a new request.
Unended text on a reset stays uncommitted so it cannot conceal undecoded audio; a retained hypothesis can be finalized when recognition stops.
If the service changes segment boundaries on replay, a segment starting before the accepted horizon is skipped; its overlapping continuation can be lost.

On a Mac call, wear headphones: hark assumes the mic hears only you and system audio contains everyone else.
The mic pauses while Aqua Voice captures input; `--pause-for` selects other apps or `none` disables detection.
Manual pauses add `# paused` and `# resumed` markers.
Pausing does not mute the phone or system tracks.

Files go to `~/.hark/sessions/`; Shuttle uses `~/.hark/meetings/` through `-o`.
`~/.hark/current.txt` points at the live transcript.
Audio is saved beside it by default and expires after 14 days; transcripts are kept.
`--no-save-audio` opts out of the saved WAV, not cloud submission.

## Speakers and latency

The local ear uses NVIDIA's streaming diarizer, with one ASR decoder per speaker and up to eight speakers.
Gradium provides no diarization.
Its word-sized text spans accumulate into roughly **four seconds of unique recognised audio** before hark computes a WeSpeaker embedding and compares it with online speaker centroids at cosine similarity `0.55`.
Completed phrases appear while speech continues; hark does not wait for the request's end.
An eight-second source span, an 800 ms gap, a speech-flush acknowledgement or the request's end also closes a phrase.
A final word without `end_text` uses the next word's start, or an inferred end at a speech-flush acknowledgement or EOS; a long quiet period closes the request after 60 source seconds.
Phrases with less than four seconds of audio inherit the preceding speaker.
There is **one speaker per phrase**, not overlapping-speaker separation.
Brief replies can be mislabelled and speaker slots can fragment; this is weaker than the local ear's diarization.

Both ears use the same enrolled-voice naming rules.
Gradium also anchors clusters to confident enrolled identities: cosine at least `0.55`, leading the next bank candidate by `0.21`, reuses that identity's slot before centroid matching.
Weak or ambiguous bank matches use ordinary clustering; confidently different enrolled identities cannot share a slot.
This requires a voice bank and does not improve anonymous re-identification by lowering thresholds.
To name yourself on another host, copy `~/.hark/voices/me.npy` there.
Manual names take precedence.
Speaker numbers are per meeting.

## Following a meeting

```bash
tail -n +1 -F ~/.hark/current.txt
```

An agent can follow that stream, or poll by line count and remember its position.
The files are append-only; apply the latest `# S2 = Ada` mapping to earlier lines too, and stop at `# ended`.

- [Usage](docs/usage.md): options, phone input, saved audio, logs, mirroring and launchers.
- [Format](docs/format.md): text and JSONL contracts.
- [Pipeline](docs/how-it-works.md): local diarization, Gradium timing and recovery, voice matching and evaluation.

## Development

```bash
HARK_DIR="$(mktemp -d /tmp/hk-test.XXXXXX)" uv run pytest
```

Test fixtures isolate HARK state and reject the real default home.
Gradium tests use a local websocket protocol server and cannot contact the hosted API or read a real key.
Use a short temporary `HARK_DIR` for every test or smoke invocation, never the live home. In-process CLI helpers must also isolate `cli.HOME`.

## License

MIT; see [LICENSE](LICENSE).
Model weights and audiotee have their own licenses.
