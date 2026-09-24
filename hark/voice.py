"""Local WeSpeaker embeddings and transcript-backed speaker naming."""

from __future__ import annotations

from collections import defaultdict, deque
import numpy as np

MODEL_REPO = "soniqo/WeSpeaker-ResNet34-LM-ONNX"
MODEL_REVISION = "9df39b49edc6f896ebc928e9d14832936d60d9f6"
MODEL_FILE = "wespeaker-resnet34.onnx"


def _model_path():
    from huggingface_hub import hf_hub_download

    return hf_hub_download(MODEL_REPO, MODEL_FILE, revision=MODEL_REVISION)


def _fbank(samples):
    x = np.asarray(samples, dtype=np.float32).reshape(-1) * 32768.0
    frame_len, frame_step, n_fft = 400, 160, 512
    if len(x) < frame_len:
        x = np.pad(x, (0, frame_len - len(x)))
    frames = np.lib.stride_tricks.sliding_window_view(x, frame_len)[::frame_step]
    frames = frames - frames.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(frames * np.hamming(frame_len), n=n_fft, axis=1)).astype(np.float32) ** 2
    hz_to_mel = lambda hz: 1127.0 * np.log1p(hz / 700.0)
    edges = np.linspace(hz_to_mel(20), hz_to_mel(8000), 82)
    mel = hz_to_mel(np.arange(n_fft // 2 + 1, dtype=np.float32) * 16000 / n_fft)
    filters = np.maximum(0, np.minimum((mel[None, :] - edges[:-2, None]) / (edges[1:-1, None] - edges[:-2, None]),
                                       (edges[2:, None] - mel[None, :]) / (edges[2:, None] - edges[1:-1, None])))
    return np.log(np.maximum(power @ filters.T, 1e-10))[None].astype(np.float32)


class Embedder:
    def __init__(self):
        import onnxruntime as ort

        self.session = ort.InferenceSession(_model_path(), providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    def __call__(self, samples):
        vector = self.session.run(None, {self.input: _fbank(samples)})[0][0]
        return (vector / max(float(np.linalg.norm(vector)), 1e-12)).astype(np.float32)


class AudioBuffer:
    """A rolling mono sample buffer addressed by seconds from track start."""

    def __init__(self, seconds=45):
        self.limit = seconds * 16000
        self.start = 0
        self.samples = np.zeros(0, dtype=np.float32)

    def append(self, start, samples):
        if not self.samples.size:
            self.start = start
        expected = self.start + len(self.samples) / 16000
        if abs(start - expected) > 1 / 16000:
            self.start, self.samples = start, np.asarray(samples, dtype=np.float32).copy()
        else:
            self.samples = np.concatenate((self.samples, samples))
        if len(self.samples) > self.limit:
            drop = len(self.samples) - self.limit
            self.samples = self.samples[drop:]
            self.start += drop / 16000

    def slice(self, start, end):
        if end <= start or end <= self.start or start >= self.start + len(self.samples) / 16000:
            return self.samples[:0]
        lo = max(0, round((start - self.start) * 16000))
        hi = min(len(self.samples), round((end - self.start) * 16000))
        return self.samples[lo:hi]


class VoiceMatcher:
    """Name diarizer slots from finished-token speech, abstaining on ambiguity."""

    def __init__(self, voices, sink, embedder=None, threshold=0.40, margin=0.10):
        self.voices, self.sink = voices, sink
        self.embedder = embedder or Embedder()
        self.threshold, self.margin = threshold, margin
        self.seconds = defaultdict(float)
        self.checked = defaultdict(int)
        self.clips = defaultdict(lambda: deque())
        self.segments = self.clips

    def finished(self, track, utterance):
        slot = utterance.speaker
        if not slot.startswith("S") or slot in self.sink.names:
            return
        spans = getattr(utterance, "speech", [(utterance.start, utterance.end)])
        spans = sorted(spans)
        merged = []
        for start, end in spans:
            if merged and start - merged[-1][1] < 0.5:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        key = (id(track), slot)
        clip = self.clips[key]
        for start, end in merged:
            self.seconds[key] += end - start
            audio = track.audio.slice(start - 0.1, end + 0.1)
            if audio.size:
                clip.append(audio.copy())
        while sum(part.size for part in clip) > 15 * 16000:
            excess = sum(part.size for part in clip) - 15 * 16000
            if excess >= clip[0].size:
                clip.popleft()
            else:
                clip[0] = clip[0][excess:]
        total = self.seconds[key]
        threshold = 5 if self.checked[key] == 0 else self.checked[key] + 20
        if total < threshold:
            return
        self.checked[key] = total
        audio = np.concatenate(clip) if clip else np.zeros(0, dtype=np.float32)
        if audio.size < 5 * 16000:
            return
        embedding = self.embedder(audio)
        scores = sorted(((float(np.dot(embedding, vector)), name)
                         for name, vector in self.voices.items()), reverse=True)
        best, name = scores[0]
        second = scores[1][0] if len(scores) > 1 else 0.0
        if best >= self.threshold and best - second >= self.margin:
            self.sink.name(slot, name)
            from .capture import log

            log(f"voice: {slot} = {name} ({best:.2f}, next {second:.2f})")
