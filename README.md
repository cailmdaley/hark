<p align="center">
  <img src="docs/assets/header.jpg" alt="hark" width="100%">
</p>

# hark

**hark turns a live conversation into a speaker-labelled transcript that grows as people talk, written to a plain text file any AI agent can read.**

```
# hark 2026-09-24 19:34 — call: me = mic, S1… = system audio
19:34:12-19:34:18 me   Sure, I reran the pipeline last night with the new masks
19:34:00-19:34:30 S1   Okay, let's get started.
19:34:29-19:34:38 S2   Did anyone check whether the redshift distributions changed?
# S2 = Ada
# ended 19:35:19
```

Start `hark` when a call or meeting begins.
A few seconds after each person finishes speaking, their words are appended to the file with a time range and a speaker label.
An agent running alongside you (Claude Code, Codex, or anything that can read a file) follows that file while the meeting is still going.
It can keep notes, record decisions, answer a question someone just asked, or start on a task the moment it's agreed.

## Why a file

Most transcription tools are apps: the transcript lives in their window, arrives after the meeting ends, or sits behind an API.
Dictation tools can't tell speakers apart and aren't built for an hour-long conversation.
hark takes the opposite approach.

- **The file is the interface.** hark only ever appends lines, and stops at an `# ended` line. Any agent harness can follow that, with `tail -F`, a file watcher, or by re-reading every so often and remembering how many lines it has seen. hark has no plugin system, server or SDK.
- **Live, not after the fact.** Lines land within seconds, so an agent can act during the meeting rather than summarise it afterwards.
- **Speakers are part of the record.** Every line says who spoke. Names can be assigned mid-meeting (`hark name S2 Ada`), or matched automatically against voices you've enrolled.
- **Local by default.** On Apple Silicon, recognition and speaker separation run entirely on the laptop. No audio leaves the machine, and nothing costs money per minute.
- **Honest about gaps.** If a microphone or the call audio goes silent, the transcript says so (`# system audio lost at 16:37:55`), so a reader doesn't take a gap for silence.

The format is a small, documented contract ([docs/format.md](docs/format.md)). Beside the text file sits a JSONL file carrying the same lines as structured records, with exact start and end times, speaker and audio source.

## How it works

hark listens to up to two audio sources and labels speakers within each.

- **On a call**, with headphones, the microphone is you (`me`) and the computer's own audio output (Zoom, Meet, a video) is everyone else. hark captures that output through a macOS system-audio tap ([audiotee](https://github.com/makeusabrew/audiotee)) and separates it into speakers `S1`…`S8`.
- **In a room**, the microphone alone is separated into speakers.
- **With a phone as the microphone**, raw audio arrives on a local socket and is separated the same way. This is useful when the laptop is in the wrong place.

Speech recognition and speaker separation ("diarization") are done by one of two back ends, which hark calls *ears*:

**The local ear** (the default on Apple Silicon) runs two NVIDIA models through [MLX](https://github.com/ml-explore/mlx), via [mlx-audio](https://github.com/Blaizzy/mlx-audio):

- [Nemotron-3-Diarization](https://huggingface.co/mlx-community/Nemotron-3-Diarization), a streaming Sortformer model. For every 80 ms of audio it estimates which of up to eight speakers is talking, committing after about one second of lookahead.
- [Nemotron 3.5 streaming ASR](https://huggingface.co/mlx-community/nemotron-3.5-asr-streaming-0.6b) (0.6 B parameters), run as **one decoder per speaker**.

They're combined the way NVIDIA's integration guide describes. The diarizer decides which frames belong to which speaker, and each speaker's decoder hears only its own frames, so words come out already attributed. No step after recognition guesses who said what. hark gives each frame to its single most likely speaker. On meeting recordings this cut words copied onto the wrong speaker's line by 7–25×, while word recall moved by less than a point. The models take about 3 GB and run comfortably in real time on an M-series laptop.

**The Gradium ear** sends gated speech to [Gradium](https://gradium.ai)'s streaming speech-to-text API. hark identifies speakers locally, by grouping [WeSpeaker](https://github.com/wenet-e2e/wespeaker) voice embeddings computed on the CPU. It runs on Linux machines without a GPU or audio devices, which makes it suitable for an always-on server receiving a phone's audio. Its speaker labels are coarser than the local ear's (one speaker per phrase), and audio leaves the machine.

Both ears share everything else: the transcript format, saved audio, voice enrollment and naming, lost-source markers, and mirroring the transcript to another machine over SSH.
[docs/how-it-works.md](docs/how-it-works.md) has the full pipeline, including how turns are formed, how voices are matched and how changes are evaluated against real meetings.

## Install and run

Local capture needs a Mac with Apple Silicon running macOS 14.2 or later, Python 3.11–3.13, [uv](https://docs.astral.sh/uv/) and the Swift toolchain (to build the system-audio tap):

```bash
git clone https://github.com/cailmdaley/hark && cd hark
scripts/build-audiotee.sh   # builds bin/audiotee
uv sync
uv run hark                 # models download on first run (~3 GB)
```

Grant your terminal the **Microphone** permission, plus **Screen & System Audio Recording → System Audio Recording Only**, then restart the terminal.
Without the second permission, the call side comes through as silence.

```bash
hark                          # a call: mic = me, system audio = S1…
hark --room                   # in person: the mic, diarized
hark --file talk.m4a          # an existing recording
hark name S2 Ada              # name a speaker in the live session
hark enroll Ada --file ada.wav  # enroll a voice (≥ 5 s) for automatic naming
```

Ctrl-C ends the session cleanly and writes `# ended`.
Transcripts go to `~/.hark/sessions/`, and `~/.hark/current.txt` always points at the live one.
Audio is kept beside the transcript for 14 days, so a meeting can be replayed through the pipeline later.

On Linux, or with the Gradium ear, see [docs/usage.md](docs/usage.md#gradium) for the API key, cost and limits.

## Following a meeting from an agent

```bash
tail -n +1 -F ~/.hark/current.txt
```

That is the whole integration.
Tell your agent to watch that stream, or to re-read the file from the last line it saw.
Then apply the latest `# S2 = Ada` naming line to every line from that slot, earlier ones included, and stop at `# ended`.
In Claude Code, the `Monitor` tool on that command delivers new lines to the agent as they arrive.

## hark in Shuttle

hark stands on its own, but it was built alongside [Shuttle](https://cailmdaley.github.io/felt/shuttle/), an orchestration app for running coding agents against written tasks, with a board for following their work ([source](https://github.com/cailmdaley/felt)).
In Shuttle, a meeting is just another way to start an agent:

- **Meetings from the board.** The board's Capture form has a Meeting toggle (Call, Room or Phone). A meeting can also be attached to an existing task's card. Shuttle starts hark on the Mac and launches an agent with the transcript path in its instructions. The card shows the meeting's duration and latest lines, with a Stop button.
- **A scribe.** The agent takes the *scribe* role. It files a note for the meeting where the project keeps them, keeps running notes of decisions and action items with timestamps and speakers, and keeps a live HTML report on the board that it rewrites as the meeting goes. When a meeting is attached to an existing task, that task's own agent follows it, so the conversation feeds straight into ongoing work.
- **The phone as a microphone.** The board has a phone page. Opened on a phone, it streams the phone's microphone over a private network ([Tailscale](https://tailscale.com)) to the socket of `hark --phone`. On a Linux host this uses the Gradium ear, so a room meeting can be transcribed with the laptop closed.
- **Agents on other machines.** `hark --mirror host:path` appends the transcript to a file on a remote machine over SSH, so the agent can run where the project lives, such as a computing cluster, while hark records on the laptop.
- **Lifecycle.** hark publishes its state (`loading`, `live`, `ended`, `failed`) to `~/.hark/meeting.json`, so a launcher can watch it rather than scrape terminal output.

None of this is required to use hark: the flags are ordinary ones (`--phone`, `--mirror`, `--launch`), documented in [docs/usage.md](docs/usage.md#mirroring-and-launchers).

## Documentation

- [Usage](docs/usage.md): modes, options, the Gradium ear, saved audio, logs, mirroring and launchers.
- [Format](docs/format.md): the text and JSONL contract an agent relies on.
- [How it works](docs/how-it-works.md): capture, both ears, speaker masks, turns, voice matching and evaluation.

## Development

```bash
HARK_DIR="$(mktemp -d /tmp/hk-test.XXXXXX)" uv run pytest
```

Tests isolate hark's state directory and refuse to touch the real `~/.hark`.
Gradium tests use a local websocket server and never contact the hosted API.
Use a short temporary `HARK_DIR` for every test or smoke run.

## License

MIT; see [LICENSE](LICENSE). Model weights and audiotee have their own licenses.
