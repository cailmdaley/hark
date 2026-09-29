# How it works

```
mic ──────────────┐                               ┌─ me     (call mode: not diarized)
                  ├─ 16 kHz tracks ─ diarizer ─ masks ─ one ASR decoder per speaker ─ turns ─ .txt + .jsonl
system audio ─────┘   (audiotee)                  └─ S1…S8
```

## Capture

The mic is read through PortAudio (`sounddevice`). System audio comes from [audiotee](https://github.com/makeusabrew/audiotee), a Swift program that opens a Core Audio process tap and writes raw PCM to stdout. hark supervises it: if the tap stalls for 5 s or exits, hark restarts it with a backoff that grows to at most 30 s, and logs each restart. A mic that delivers nothing for 5 s is reopened the same way. Both tracks are captured at 16 kHz mono and padded to hold the wall clock, so timestamps stay true even when a device falls behind.

In call mode the mic is one speaker (`me`) and only system audio is diarized. In room mode the mic is diarized.

## Recognition and diarization

Both models are NVIDIA's, run in MLX via [mlx-audio](https://github.com/Blaizzy/mlx-audio)'s `SpeakerStreamingSession`, and combined the way NVIDIA's integration guide describes:

- **Nemotron-3-Diarization** (`mlx-community/Nemotron-3-Diarization`) comes from NVIDIA's streaming Sortformer line. For every 80 ms frame it gives each of up to 8 speakers a probability of speaking. With the default buffer it sees about 1.04 s of audio before committing.
- **Nemotron 3.5 streaming ASR** (`mlx-community/nemotron-3.5-asr-streaming-0.6b`) runs as one decoder per speaker. Each decoder hears only the frames the diarizer assigns to its speaker, so words come out already attributed. Nothing aligns words to speakers after the fact.

### Speaker masks

The mask is temporal, not a voice separator: a frame is either passed to a speaker's decoder or not. When two speakers hold the same frame (a backchannel under someone's sentence, or the overlapping edges of a turn), the mixed audio goes to both decoders and both transcribe the louder voice. The same words then appear on two lines.

hark therefore gives each frame only to its most probable speaker (`--speaker-mask exclusive`, the default). On AMI meeting-corpus recordings and Zoom calls this cut words copied into the wrong speaker's line by 7–25×, while word recall moved by under a point. `--speaker-mask shared` restores mlx-audio's behaviour, where every speaker above probability 0.5 hears the frame.

## Turns

The decoders emit tokens; hark groups them into conversational turns, not pause-delimited fragments:

- brief silences keep a turn open
- a sustained reply from someone else (at least 0.8 s of speech) closes it
- when nobody takes over, `--gap` seconds of silence (3 by default) close it
- short backchannels ("mm-hm", "right") don't end another speaker's turn
- a monologue longer than 30 s is split at its longest late pause

Each speaker's turn closes independently, which is why lines from different speakers can land slightly out of order.

## Voice matching

Voice matching applies to diarized slots (`S1`…), never to the call-mode mic. Enrolled voices (`hark enroll`, at least 5 s of audio) are [WeSpeaker](https://github.com/wenet-e2e/wespeaker) ResNet34 embeddings, run through ONNX Runtime. For each diarized slot, hark collects the most recent 15 s of that slot's speech and embeds it once the slot has 5 s, then again after every further 20 s. A slot claims a voice when the cosine similarity is at least 0.55 and leads the next enrolled voice by 0.21. The 0.55 floor sits above the worst impostor clips seen in testing, which scored about 0.5.

A name belongs to one slot at a time. When several slots claim the same voice, the strongest gets it only if it leads every other claimant by the same 0.21 margin. If that winner isn't the current holder, hark hands the name over: it writes `# S1 = S1` for the old holder, then names the new one. Names a human sets are never touched, and never given to another slot.

## Evaluating diarization

Two tools measure changes to the pipeline against real audio:

```bash
uv run python -m hark.replay AUDIO OUTDIR [--mask exclusive]
uv run scripts/diar_eval.py OUTDIR
```

`hark.replay` runs a file through the live pipeline (with the live default mask; `--mask default` gives mlx-audio's shared one, and `--start`/`--duration` cut a span) and dumps the diarizer's per-frame probabilities, every per-speaker token, and the resulting turns. `diar_eval.py` scores a dump: diarization error rate (DER), words duplicated across slots, and which words landed in a slot whose speaker didn't say them. The last one needs a reference: AMI word alignments (an `<meeting>.rttm` beside the audio, with `words/` and `corpusResources/`) or a Zoom `.transcript.vtt`.

Saved session audio (`<stem>.system.wav`) replays through the same path, so a real meeting where diarization went wrong becomes a test case.
