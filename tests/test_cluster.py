import numpy as np
import pytest

from hark.capture import WavRecorder, load_audio
from hark.voice import AudioArchive, OnlineCluster


def test_cluster_returns_to_centroid_and_short_inherits():
    vectors = iter([[1, 0], [0, 1], [1, 0], [0.8, 0.6]])
    cluster = OnlineCluster(embedder=lambda _: next(vectors))
    speech = np.ones(2 * 16000, np.float32)
    assert cluster.assign(speech) == "S1"
    assert cluster(speech) == "S2"
    assert cluster.assign(np.ones(16000)) == "S2"
    assert cluster(speech) == "S1"
    assert cluster(speech) == "S1"
    assert cluster.counts == [3, 1]
    np.testing.assert_allclose(cluster.centroids[0], [2.8 / 3, 0.6 / 3])


def test_cluster_similarity_floor_and_injected_minimum():
    c = OnlineCluster(lambda x: x[:2], minimum_duration=0)
    assert c.assign(np.array([1., 0])) == "S1"
    assert c.assign(np.array([0.55, np.sqrt(1 - 0.55**2)])) == "S1"
    assert c.assign(np.array([-1., 0])) == "S2"
    with pytest.raises(ValueError, match="invalid speaker"):
        c.assign(np.zeros(2))


def test_archive_keeps_old_audio_across_model_delay():
    archive = AudioArchive()
    archive.append(0, np.full(16000, 0.125, np.float32))
    archive.append(1, np.zeros(100 * 16000, np.float32))
    assert np.array_equal(archive.slice(0, 1), np.full(16000, 0.125, np.float32))
    assert not hasattr(archive, "samples")
    archive.close()


def test_embedder_caps_cpu_threads(monkeypatch):
    from types import SimpleNamespace
    import sys
    from hark import voice

    options = []
    def session(path, *, sess_options, providers):
        options.append(sess_options)
        assert providers == ["CPUExecutionProvider"]
        return SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name="features")])
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(SessionOptions=SimpleNamespace,
                                                                  InferenceSession=session))
    monkeypatch.setattr(voice, "_model_path", lambda: "local-model")
    monkeypatch.setattr(voice.os, "cpu_count", lambda: 24)
    voice.Embedder()
    assert options[0].intra_op_num_threads == 4
    assert options[0].inter_op_num_threads == 1


def test_cluster_warms_model_without_changing_centroids(monkeypatch):
    calls = []
    monkeypatch.setattr("hark.voice.Embedder", lambda: lambda x: calls.append(len(x)) or np.array([1., 0.]))
    c = OnlineCluster()
    c.warm()
    assert calls == [16000]
    assert c.centroids == []
    assert c.assign(np.ones(32000)) == "S1"


def test_dependencies_have_portable_platform_markers():
    import tomllib
    from pathlib import Path

    metadata = tomllib.loads(Path("pyproject.toml").read_text())
    deps = metadata["project"]["dependencies"]
    assert all("sys_platform == 'darwin'" in dep for dep in deps if dep.startswith(("mlx-audio", "sounddevice")))
    assert any(dep.startswith("websockets") for dep in deps)
    assert any(dep.startswith("huggingface-hub") for dep in deps)


def test_portable_loader_replays_pcm_exactly(tmp_path):
    samples = np.arange(-16000, 16000, dtype=np.float32) / 32768
    recorder = WavRecorder(tmp_path / "track.wav")
    recorder.write(samples)
    recorder.close()
    assert np.array_equal(load_audio(recorder.path), samples)
