"""Local WeSpeaker embeddings and transcript-backed speaker naming."""

from __future__ import annotations

from collections import defaultdict, deque
import os
import tempfile
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

        options = ort.SessionOptions()
        options.intra_op_num_threads = min(4, os.cpu_count() or 1)
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(_model_path(), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    def __call__(self, samples):
        vector = self.session.run(None, {self.input: _fbank(samples)})[0][0]
        return (vector / max(float(np.linalg.norm(vector)), 1e-12)).astype(np.float32)


class OnlineCluster:
    """Assign mono 16 kHz speech to S1… by cosine running centroids.

    Clips shorter than `minimum_duration` inherit the last slot without updating it.
    A cosine below `threshold` creates a new slot; embeddings are unit-normalized.
    """

    def __init__(self, embedder=None, threshold=0.55, minimum_duration=1.8):
        self.embedder = embedder
        self.threshold = threshold
        self.minimum_duration = minimum_duration
        self.centroids = []
        self.counts = []
        self.previous = "S1"

    def assign(self, samples):
        if len(samples) < self.minimum_duration * 16000:
            return self.previous
        if self.embedder is None:
            self.embedder = Embedder()
        vector = np.asarray(self.embedder(samples), dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(vector).all() or norm < 1e-12:
            raise ValueError("invalid speaker embedding")
        vector = vector / norm
        scores = [float(np.dot(vector, c / np.linalg.norm(c))) for c in self.centroids]
        index = int(np.argmax(scores)) if scores else 0
        if not scores or scores[index] < self.threshold:
            index = len(self.centroids)
            self.centroids.append(vector.copy())
            self.counts.append(1)
        else:
            n = self.counts[index]
            self.centroids[index] = (n * self.centroids[index] + vector) / (n + 1)
            self.counts[index] += 1
        self.previous = f"S{index + 1}"
        return self.previous

    __call__ = assign


class AudioArchive:
    """Disk-backed raw track audio; delayed segments never lose their samples."""

    def __init__(self):
        self.file = tempfile.TemporaryFile()
        self.size = 0

    def append(self, start, samples):
        if round(start * 16000) != self.size:
            raise ValueError("non-contiguous track audio")
        self.file.seek(0, 2)
        self.file.write(np.asarray(samples, dtype="<f4").tobytes())
        self.size += len(samples)

    def slice(self, start, end):
        lo, hi = max(0, round(start * 16000)), min(self.size, round(end * 16000))
        self.file.seek(lo * 4)
        return np.frombuffer(self.file.read(max(0, hi - lo) * 4), dtype="<f4").copy()

    def close(self):
        self.file.close()


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
    """Name diarizer slots from finished-token speech, abstaining on ambiguity.

    Each (track, slot) keeps its latest cosine score against every enrolled voice.
    A slot claims a voice when that score clears `threshold` and leads its
    runner-up voice by `margin`; a voice goes to its strongest claimant only if
    it leads every other claimant by `margin`, so a name is held by at most one
    slot. A clear new winner takes the name from its holder, which is written
    back to its own label (`# S1 = S1`). Slots whose name a human set or
    changed are left alone, and their names are never handed to another slot.
    """

    def __init__(self, voices, sink, embedder=None, threshold=0.55, margin=0.21):
        self.voices, self.sink = voices, sink
        self.embedder = embedder or Embedder()
        self.threshold, self.margin = threshold, margin
        self.seconds = defaultdict(float)
        self.checked = defaultdict(int)
        self.clips = defaultdict(lambda: deque())
        self.segments = self.clips
        self.scores = {}  # (track, slot) -> {voice: latest cosine}
        self.given = {}  # slot -> the name this matcher wrote for it

    def _automatic(self, slot):
        return self.sink.names.get(slot) == self.given.get(slot)

    def _claim(self, scores):
        ranked = sorted(((score, name) for name, score in scores.items()), reverse=True)
        best, name = ranked[0]
        second = ranked[1][0] if len(ranked) > 1 else 0.0
        return name if best >= self.threshold and best - second >= self.margin else None

    def finished(self, track, utterance):
        slot = utterance.speaker
        if not slot.startswith("S") or not self._automatic(slot):
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
        self.scores[key] = {name: float(np.dot(embedding, vector)) for name, vector in self.voices.items()}
        self._assign(self.scores[key])

    def _assign(self, latest):
        from .capture import log

        held = set(self.sink.names.values())
        for voice in sorted(latest, key=latest.get, reverse=True):
            claimants = sorted(((scores[voice], slot) for (_, slot), scores in self.scores.items()
                                if self._automatic(slot) and self._claim(scores) == voice), reverse=True)
            if not claimants:
                continue
            (best, winner), rest = claimants[0], claimants[1:]
            holder = next((slot for slot, name in self.given.items()
                           if name == voice and self.sink.names.get(slot) == voice), None)
            if winner == holder or (rest and best - rest[0][0] < self.margin):
                continue
            if holder is None and voice in held:
                continue  # a human gave this name to a slot
            if holder is not None:
                self.sink.name(holder, holder)
                self.given[holder] = None
                log(f"voice: {holder} = {holder} ({voice} moves to {winner})")
            self.sink.name(winner, voice)
            self.given[winner] = voice
            held = set(self.sink.names.values())
            log(f"voice: {winner} = {voice} ({best:.2f}, next {rest[0][0] if rest else 0.0:.2f})")
