# Acoustic echo cancellation

Default two-track live capture uses the Mac's VoiceProcessingIO microphone path when `bin/voice-mic` is built and the default microphone is selected.
The helper selects the default input and output devices, negotiates matching mono 16 kHz client formats, and renders silence rather than replaying the microphone or system audio.
VoiceProcessingIO uses the output device as its acoustic echo reference.
This is Apple's voice-processing Audio Unit, the underlying alternative to `AVAudioEngine.inputNode.setVoiceProcessingEnabled`.
Automatic gain control is disabled; other-app ducking is set to its minimum, which is not a guarantee of zero ducking.

Build it with `scripts/build-voice-mic.sh`.
Room-only capture, enrollment and an explicitly selected `--mic` use the raw microphone.
A missing helper leaves dual capture running with a warning and no echo cancellation.
A helper failure is logged and retried; padded silence is not evidence of successful cancellation.
AEC is device-dependent and is not a guarantee that every remote word disappears.

## MacBook device measurement

A short hardware trial on macOS 26.5.2 uses the built-in MacBook Pro Speakers and MacBook Pro Microphone.
The speaker volume is 44, unmuted, with `afplay` gain 0.35; the trial does not change routing or system volume.
The reference is Samantha's synthesized sentence: “The purple telescope measures seven distant galaxies. Please write down the number forty two.”
Raw PortAudio capture and processed VoiceProcessingIO capture are sequential trials at 16 kHz mono.
The meeting API reports a failed, inactive meeting before playback.

| Metric | Raw mic | VoiceProcessingIO mic |
|---|---:|---:|
| RMS over a reference-length window | 0.007355 | 0.0007750 |
| Absolute peak normalized correlation with reference | 0.1688 | 0.1346 |
| Absolute least-squares reference projection gain | 0.007836 | 0.0006584 |
| Local-ear replay | All 14 reference words | No transcript lines |

The measured RMS reduction is **19.5 dB** (processed RMS is 10.5% of raw RMS).
The reference projection is **21.5 dB** lower.
A second trial at system volume 38 and `afplay` gain 1 measures 16.6 dB RMS reduction, with recognizable reference fragments in the raw transcript and no processed transcript lines.
These metrics include room noise and acoustic filtering; neither is a laboratory echo-return-loss measurement.
Each trial's window is selected by maximum absolute waveform correlation, so its offset can differ between raw and processed capture.
No transcript lines means no recognized leakage in this short trial, not zero acoustic leakage.
These short trials do not establish near-end speech preservation, double-talk behavior, performance with external speakers, or reliable cancellation across volume levels.

## Reproducing the probe

Do not play test audio while a meeting is recording.
The probe checks `http://localhost:4000/api/v1/meeting` and ordinary local hark CLI processes before capture and immediately before each playback.
It refuses playback unless the API explicitly contains `meeting: null` or a meeting whose state is `ended` or `failed`, with no local live hark command detected.
API failure or an unknown state also prevents playback.
The API covers managed meetings; process detection also covers ordinary standalone CLI capture, but is not a shared recording lock.
Keep other launchers idle during the probe: these checks cannot prevent a recording from starting after the check.
It never stops another recording, changes routing, or changes the system volume.
Keep clips short and quiet.

```bash
scripts/build-voice-mic.sh
say -v Samantha -o /tmp/hark-reference.aiff \
  'The purple telescope measures seven distant galaxies. Please write down the number forty two.'
HARK_DIR="$(mktemp -d /tmp/hk-aec.XXXXXX)" uv run python scripts/aec-probe.py \
  /tmp/hark-reference.aiff /tmp/hark-aec-result --play --volume 0.35
```

Use a new output directory for each run.
The script records raw and processed WAVs, helper logs and `metrics.json`, including device and volume settings.
It limits the reference to eight seconds, rejects all-zero captured PCM as inconclusive, and requires a detectable raw baseline before interpreting attenuation.
A quiet or muted playback that never reaches the mic cannot demonstrate AEC.
Nonzero processed samples still do not establish near-end speech preservation; that needs a separate near-end control.
Replay each WAV with `hark --file` and a separate temporary `HARK_DIR` to check recognized words too.
Keep the WAVs private: they also contain whatever else the microphone hears.

## API evidence and implementation limits

[Chromium's macOS input implementation](https://github.com/chromium/chromium/blob/main/media/audio/apple/audio_low_latency_input.cc) selects an input and an output device for VoiceProcessingIO and provides a silent output callback.
This is evidence that macOS supports a device-level reference without routing the meeting through hark's own playback.
[Apple's voice-processing overview](https://developer.apple.com/videos/play/wwdc2019/510/) describes the AVAudioEngine path; [its ducking overview](https://developer.apple.com/videos/play/wwdc2023/10235/) distinguishes minimum ducking from disabled ducking.

An AVAudioEngine probe on this Mac reports a nine-channel input format.
Default conversion produces zero-valued PCM; explicitly selecting channel zero produces audio but does not establish that the selected channel is the processed uplink.
Connecting an output branch in that graph fails with OSStatus `-10875` during initialization.
Those samples are not counted as successful AEC.
The mono VoiceProcessingIO helper avoids the undocumented channel mapping and initializes successfully on the real devices.

If a particular device cannot use this path, the next option is WebRTC AEC3 with the captured system track as an explicit render reference, including delay and clock-drift estimation.
That path requires its own device measurements and double-talk checks; it is not implemented here.
