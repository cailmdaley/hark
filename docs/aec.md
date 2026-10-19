# Acoustic echo cancellation

Default mic + system capture applies WebRTC AEC3 inside hark, using audiotee's system track as the far-end reference.
The mic is captured raw through PortAudio, including with `--mic DEV`; hark doesn't enable device-wide voice processing, change routing, or adjust output volume.
`--room`, `--phone`, file replay, enrollment and mic-only fallback have no AEC.
The optional saved mic WAV contains exactly the processed, pause-muted PCM the models hear.

## Library and timing

[LiveKit RTC 1.1.21](https://pypi.org/project/livekit/1.1.21/) provides `AudioProcessingModule`, with echo cancellation enabled and gain control, high-pass filtering and noise suppression disabled.
Its [Python API](https://docs.livekit.io/reference/python/livekit/rtc/apm.html) wraps [WebRTC's audio processing module](https://github.com/livekit/rust-sdks/blob/main/webrtc-sys/src/apm.cpp).
The release has macOS arm64 and Linux x86_64/aarch64 wheels.
The macOS wheel is installed and exercised by native-processing tests; the Linux x86_64 wheel resolves and installs with uv's platform selector, but isn't executed on this Mac.
The wrapper needs no LiveKit account, connection or server.

Each track's `drain()` pads its own sample timeline, which can lag sample arrival by up to a second.
Start anchors alone therefore don't align echo.
The pure `hark/aec.py` stage buffers at least two seconds of mic audio, waits for the reference window, and estimates coarse mic/reference lag once per second by waveform correlation over ±1.5 seconds.
It retains the last confident estimate during silence or unrelated near-end speech.
The reference is fed 50 ms ahead of the estimated echo, then the mic is processed in 10 ms frames; AEC3 handles finer delay and drift.
Shutdown preserves the exact mic length, including a short last frame.
Paused and padded zero samples stay zero after processing.

`hark/dedupe.py` drops residual mic copies only when their wall spans overlap system speech and their normalized word sequences closely match.
It tolerates small spelling differences in long words, not added/omitted words or general semantic substitutions; short acknowledgments survive.
Mic utterances wait three seconds after finalization to allow the system copy to finish first.
System lines are immediate; drops and alignment changes go to `<stem>.log`.
Differently segmented copies, delayed completion beyond that hold, or changed recognition can escape deduplication.

## Built-in speaker measurement

On macOS 26.5.2, a short trial uses **MacBook Pro Speakers and MacBook Pro Microphone**, system output volume **44**, input volume **99**, and playback gain **0.35**.
The reference is Samantha's synthesized sentence: “The purple telescope measures seven distant galaxies. Please write down the number forty two.”
Raw mic, processed mic and system reference are captured simultaneously, with RMS measured over the same reference-length window selected from the raw mic.

| Metric | Raw mic | Software AEC mic |
|---|---:|---:|
| RMS | 0.031319 | 0.0040762 |
| Absolute normalized correlation with original speech | 0.49265 | 0.007325 |
| Absolute reference projection gain | 0.097371 | 0.0001884 |
| Local-ear replay | 14/14 reference words, with “measure” for “measures” | No transcript lines |

The measured RMS reduction is **17.71 dB**; reference projection is **54.27 dB** lower.
These include ambient noise and acoustic filtering, not a laboratory echo-return-loss measurement.
The estimated coarse mic lag is **+104 ms** after accounting for source start anchors.
An isolated default-mode live check produces **one system utterance, zero mic utterances**, and exits cleanly after SIGINT.
The original default devices and the built-in speaker's muted/zero-volume settings are restored after testing.
Muted-speaker baselines have no detectable leakage and aren't evidence of cancellation.

## Double-talk and real-call limits

The offline double-talk test repeats the measured acoustic mic/system pair and adds an independent Daniel speech clip to the mic during far-end playback: “My blue notebook contains five careful measurements. We should check the calibration tomorrow.”
The near-only control recognizes **13/13 words**, both raw and after AEC with a silent reference.

| Near speech RMS relative to recorded echo + noise | Near-end words recognized after AEC |
|---|---:|
| +7.95 dB | 13/13 |
| +1.93 dB | 6/13 |
| −4.09 dB | 6/13 |

At the lower two levels only “We should check the calibration tomorrow” survives recognition.
The raw +1.93 dB mixture recognizes 7/13 near words, so the failure isn't attributable entirely to AEC; software processing still doesn't meet full preservation at that level.
A synthetic pulsed-tone regression checks that double-talk isn't universally silenced, but isn't a speech-quality test.
**Simultaneous near-end speech preservation is not established across practical volume levels.**

A 25-second excerpt of a saved raw two-track call yields **6.88 dB RMS reduction** over its final 20 seconds.
Raw mic and system replays contain the same remote passage; processed mic replay yields no lines.
Its estimated mic lag is around **−265 ms**, with some lower-confidence alternative peaks around −200 ms.
A separate raw-call pair has weak correlations outside its opening, so it doesn't support a reliable drift estimate.
Offline AEC processing takes about **0.2 s per 25–28 s of audio** on this Mac (roughly 0.7–0.9% of one core by elapsed time).
These short tests don't establish long-call drift handling, device-switch recovery, Zoom mic coexistence or live human double-talk; those need a real call.

## Guarded reproduction

**Never play probes into an existing recording or into AirPods/headphones.**
Before each playback, check `pgrep -fl hark`, the default devices (`system_profiler SPAudioDataType`), and the managed meeting API.
The probe requires an explicit inactive API state and rejects local live hark CLI commands, including standalone recordings.
API failure or an unknown state prevents playback.
These checks aren't a shared recording lock: keep other launchers idle and require a quiet window.
The script targets the built-in mic and speakers explicitly, so connected AirPods aren't used.
It doesn't unmute speakers or change volume; verify that the built-in speakers are audible before interpreting a baseline, and restore any manually changed settings afterwards.

```bash
say -v Samantha -o /tmp/hark-reference.aiff \
  'The purple telescope measures seven distant galaxies. Please write down the number forty two.'
HARK_DIR="$(mktemp -d /tmp/hk-aec.XXXXXX)" uv run python scripts/aec-probe.py \
  /tmp/hark-reference.aiff /tmp/hark-aec-result --play --volume 0.35
```

Use a new output directory for each run.
The script limits playback to eight seconds and saves `raw.wav`, `aec.wav`, `system.wav` and `metrics.json`, including device names and volume settings.
It rejects all-zero PCM as inconclusive and requires a detectable raw baseline before interpreting attenuation.
Replay each WAV with `hark --file`, a separate temporary `HARK_DIR`, and separate transcript paths; this doesn't play sound.
Keep mic WAVs private because they include whatever else the room microphone hears.
