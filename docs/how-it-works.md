# How it works

```
mic ─ 300 ms lookback + pause gate ─┐             ┌─ me     (call mode: not diarized)
                                    ├─ 16 kHz tracks ─ diarizer ─ masks ─ one ASR decoder per speaker ─ turns ─ .txt + .jsonl
system audio ───────────────────────┘ (audiotee)   └─ S1…S8
```

`--ear local|gradium` selects recognition and speaker attribution.
The default is local when MLX imports, otherwise Gradium.
Sources, recording, timestamps, naming, source health, mirroring and the transcript sink are shared.
The diagram shows the local ear; Gradium replaces the diarizer/ASR chain with cloud recognition and CPU phrase embeddings.

## Capture

The mic is read through PortAudio (`sounddevice`). System audio comes from [audiotee](https://github.com/makeusabrew/audiotee), a Swift program that opens a Core Audio process tap and writes raw PCM to stdout. hark supervises it: if the tap stalls for 5 s or exits, hark restarts it with a backoff that grows to at most 30 s, and logs each restart. A mic that delivers nothing for 5 s is reopened the same way. Both tracks are captured at 16 kHz mono and padded to hold the wall clock, so timestamps stay true even when a device falls behind.

A background watcher pauses the mic while a dictation app records.
Every 75 ms it reads CoreAudio's process-object list with `ctypes` and checks each process's input-capture flag and bundle ID.
It matches `aquavoice` and `aqua-voice` by default (Aqua Voice records through `aquavoice.macOSBridge`; the second covers its app processes) and ignores hark's own process.
`--pause-for` accepts comma-separated bundle-ID substrings; `--pause-for none` disables app detection.
`hark pause` and `hark resume` pause the mic by hand through a flag file the watcher also polls.
Each pause becomes a wall-clock mute interval: from 300 ms before it was detected to when it ends, plus 300 ms after a dictation app stops.
The mic gate holds the newest 300 ms of mic audio before passing it on, so an interval can still reach the first words captured before detection.
Samples are timed by when the device delivered them, not by their position in the padded track, because a mic can run up to a second behind that timeline after a stall.
Muted samples become zeros for both the ASR and the saved WAV, so the track keeps its length and timeline.
System and phone audio are neither delayed nor muted, and the lost-source watch reads the device's own signal, so a pause never looks like a lost mic.

In call mode the mic is one speaker (`me`) and only system audio is diarized. In room mode the mic is diarized.

## Local recognition and diarization

Both models are NVIDIA's, run in MLX via [mlx-audio](https://github.com/Blaizzy/mlx-audio)'s `SpeakerStreamingSession`, and combined the way NVIDIA's integration guide describes:

- **Nemotron-3-Diarization** (`mlx-community/Nemotron-3-Diarization`) comes from NVIDIA's streaming Sortformer line. For every 80 ms frame it gives each of up to 8 speakers a probability of speaking. With the default buffer it sees about 1.04 s of audio before committing.
- **Nemotron 3.5 streaming ASR** (`mlx-community/nemotron-3.5-asr-streaming-0.6b`) runs as one decoder per speaker. Each decoder hears only the frames the diarizer assigns to its speaker, so words come out already attributed. Nothing aligns words to speakers after the fact.

### Speaker masks

The mask is temporal, not a voice separator: a frame is either passed to a speaker's decoder or not. When two speakers hold the same frame (a backchannel under someone's sentence, or the overlapping edges of a turn), the mixed audio goes to both decoders and both transcribe the louder voice. The same words then appear on two lines.

hark therefore gives each frame only to its most probable speaker (`--speaker-mask exclusive`, the default). On AMI meeting-corpus recordings and Zoom calls this cut words copied into the wrong speaker's line by 7–25×, while word recall moved by under a point. `--speaker-mask shared` restores mlx-audio's behaviour, where every speaker above probability 0.5 hears the frame.

## Gradium recognition and speakers

Gradium receives base64-encoded s16le PCM in 1,280-sample frames: 80 ms at the source's 16 kHz rate.
The websocket uses `pcm_16000`, the `default` model, an `x-api-key` header and `en`, `fr` or `any` as its language.
The `ready` message reports the model's sample rate and frame size, not necessarily the submitted audio's rate; the input stays 16 kHz.
Audio leaves the host in this mode.

### Gating and the source clock

A frame whose RMS is at least `0.001` opens a speech request, including the preceding 320 ms of audio.
The speech burst includes an 800 ms silence tail, then flushes without ending its request.
Long silence is discarded rather than submitted.
A subsequent burst can share the same connection, with its own pre-roll.
This is an energy gate, not a semantic speech detector; sufficiently loud background noise passes it.
A request ends after 58 seconds of submitted audio, near 1,200 recognised characters, or 60 seconds of source-clock quiet.
The quiet deadline sends EOS and drains the decoder before the observed 120-second provider no-output timeout; no socket is held until the next speech.
The upload bound limits replay work and leaves decoder-tail headroom below an inferred 60-second billing unit.

Each request carries piecewise integer-sample mappings from its compressed cloud clock to the original track.
Projecting Gradium's `start_s` and `stop_s` through those mappings yields source-relative timestamps, including any silence PhoneSource padded or the gate discarded.
A word crossing a cloud join can have several disjoint source intervals; the omitted gap is not included in its embedding.
Overlapping intervals are unioned so each source sample counts once.
The source anchor turns those offsets into wall times.
The saved WAV stays on the full padded clock; it is not the shortened cloud stream.

### Live phrases

Gradium returns mostly word-sized `text` / `end_text` pairs, identified by `stream_id`.
Four seconds of unique recognised audio form a phrase; an eight-second source-wall span, an 800 ms gap, a speech-flush acknowledgement or the request's end closes it too.
The phrase's samples are concatenated from the raw track archive and embedded with the same CPU WeSpeaker model used for enrolled voices.
A confident enrolled identity, using the naming floor `0.55` and bank-candidate margin `0.21`, reuses its anchored slot before centroid comparison.
Confidently different identities cannot share a slot; weak or ambiguous claims do not anchor it.
Without a confident bank match, cosine similarity below `0.55` against every eligible running centroid creates a new `S<n>` slot; otherwise the closest receives the embedding.
Identity anchors affect slot reuse, not display names or manual-name precedence.
Centroids average unit-normalized embeddings, one vote per embedded phrase.
Less than four seconds of audio inherits the preceding speaker without changing its centroid.

Completed phrases are emitted while their request remains open, so a monologue does not wait for request rotation.
There is one speaker per phrase: a brief second speaker or overlapping speech can be absorbed into its dominant voice.
Speaker slots can fragment, and this does not provide the local ear's frame-level diarization.
Two-second windows fragment heavily in the AMI calibration; longer windows trade latency and short-turn mistakes for more stable voices.

A final `text` can arrive without `end_text` before EOS.
The next word's start supplies its end when available; a speech-flush acknowledgement or EOS instead infers the end from the real submitted audio duration, excluding transport padding.
The log marks inference, and a late end for an inferred flush tail is ignored.
A final word therefore need not wait for the next speech burst or request rotation.

### Recovery and resources

Startup validates authentication and releases its socket without submitting audio.
Each bounded request has a fresh connection and can contain multiple gated speech bursts.
A dropped connection retries with cancellable exponential backoff from 0.5 seconds to a 30-second ceiling for the meeting's duration, uploading only its uncommitted audio suffix.
Each retry has a zero-based provider clock mapped both to original request samples and to the source; the original mapping stays intact for queued accepted results.
Replay starts at the earliest known stream's finalized horizon or the all-stream flush commitment, whichever is later.
A shortened frame is padded for transport, but those padding samples never enter source intervals or embeddings.
A drop after all input is committed retires the request without replay, a retry socket or outage notices.
Startup failures permit capture after the first failure but remain quiet until speech actually waits.
Lost/back markers cover that waiting interval: the earliest unacknowledged source sample starts it, and the current capture horizon ends it when recognition resumes.
Finalized words are retained independently of successful EOS.
Unended reset text stays a bounded fallback hypothesis rather than advancing a committed horizon over undecoded audio.
A retry replaces its stream's hypothesis; recognition shutdown can finalize the retained text once.
Replay suppresses segments starting before the preceding attempt's accepted horizon, per text stream.
If the service changes its segmentation across that boundary, an overlapping continuation can be skipped; hark logs the suppression.

The backlog holds at most 120 seconds of audio, including the in-flight request.
When full during realtime capture, it omits further recognition with source-interval notices rather than stopping capture or WAV recording.
Non-realtime file input instead waits for capacity through recoverable outages; its final drain has a fixed 30-second budget.
Phone capture has a separate ten-minute sample-count bound, independent of relay packet size; admitted audio is drained to the WAV before a capture-overflow error surfaces.
The worker watches advancing ASR progress and cancellation while input, a flush or EOS is outstanding.
A flushed, idle connection can wait for the next speech burst; heartbeats without progress do not keep outstanding work alive.
Authentication failures are terminal and reach the shared lifecycle as `# gradium …`, a failed owned `meeting.json` and `# ended`.
Other provider/network failures remain degraded until recovery or shutdown.
Timestamp anomalies are clamped and logged, stray word ends are ignored, and unexpected EOS restarts the request rather than masquerading as authentication.
An embedding failure preserves the recognised text, keeps the previous speaker and disables clustering with a log warning.
The embedder is warmed before capture, with at most four ONNX CPU threads.

A private PCM16 archive in system temporary storage retains the full source clock for delayed recognition and voice matching, even with `--no-save-audio`.
It stays open through independent final track flushes and closes at the end of the meeting.
A rolling source-clock buffer cannot cover old queued recognition across extended outages; the archive also supports naming when WAV saving is disabled.
The saved WAV is a separate, optional artifact.

The session log records submitted audio seconds, including replay, and credit balances when the metering endpoint responds.
A launcher-owned `meeting.json` exposes the latest balance and submitted seconds under `ear`.
Only live capture with `--launch`, or explicit output physically beneath the HARK home's `meetings/` directory, owns that record; standalone and file capture leave it untouched.
Metering failure does not stop a meeting; there is no automatic spending limit.
Balances can lag settled charges.
The [measured billing table](usage.md#gradium) fits rounding the service's audio-progress clock to 15-second units, including an observed 1.04-second decoder tail; this is not a published guarantee.
Sharing a gated socket through short quiet avoids both submitted silence and repeated short-request charges; quiet lasting 60 source seconds closes it before provider expiry.
The 58-second upload cap leaves two seconds of headroom below a 60-second unit, while the character cap bounds transcript size.

## Local turns

The decoders emit tokens; hark groups them into conversational turns, not pause-delimited fragments:

- brief silences keep a turn open
- a sustained reply from someone else (at least 0.8 s of speech) closes it
- when nobody takes over, `--gap` seconds of silence (3 by default) close it
- short backchannels ("mm-hm", "right") don't end another speaker's turn
- a monologue longer than 30 s is split at its longest late pause

Each speaker's turn closes independently, which is why lines from different speakers can land slightly out of order.

## Voice matching

Voice matching applies to anonymous slots (`S1`…) from either ear, never to the call-mode mic. Enrolled voices (`hark enroll`, at least 5 s of audio) are [WeSpeaker](https://github.com/wenet-e2e/wespeaker) ResNet34 embeddings, run through ONNX Runtime. For each diarized slot, hark collects the most recent 15 s of that slot's speech and embeds it once the slot has 5 s, then again after every further 20 s. A slot claims a voice when the cosine similarity is at least 0.55 and leads the next enrolled voice by 0.21. The 0.55 floor sits above the worst impostor clips seen in testing, which scored about 0.5.

A name belongs to one slot at a time. When several slots claim the same voice, the strongest gets it only if it leads every other claimant by the same 0.21 margin. If that winner isn't the current holder, hark hands the name over: it writes `# S1 = S1` for the old holder, then names the new one. Names a human sets are never touched, and never given to another slot.

## Evaluating local diarization

Two tools measure changes to the pipeline against real audio:

```bash
uv run python -m hark.replay AUDIO OUTDIR [--mask exclusive]
uv run scripts/diar_eval.py OUTDIR
```

`hark.replay` runs a file through the live pipeline (with the live default mask; `--mask default` gives mlx-audio's shared one, and `--start`/`--duration` cut a span) and dumps the diarizer's per-frame probabilities, every per-speaker token, and the resulting turns. `diar_eval.py` scores a dump: diarization error rate (DER), words duplicated across slots, and which words landed in a slot whose speaker didn't say them. The last one needs a reference: AMI word alignments (an `<meeting>.rttm` beside the audio, with `words/` and `corpusResources/`) or a Zoom `.transcript.vtt`.

Saved session audio (`<stem>.system.wav`) replays through the same path, so a real meeting where diarization went wrong becomes a test case.
