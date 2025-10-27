"""Unit tests for embedding creation."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock
from typing import cast
import pytest
import numpy as np

# --- TensorFlow mock ---
mock_tf = ModuleType("tensorflow")
mock_tf.random = SimpleNamespace(set_seed=lambda *a, **kw: None)
sys.modules["tensorflow"] = mock_tf
sys.modules["tensorflow.compat"] = ModuleType("tensorflow.compat")
sys.modules["tensorflow.compat.v2"] = ModuleType("tensorflow.compat.v2")

# --- Keras mock ---
mock_keras = ModuleType("tensorflow.keras")
mock_keras_applications = ModuleType("tensorflow.keras.applications")
mock_keras_xception = ModuleType("tensorflow.keras.applications.xception")
mock_keras_layers = ModuleType("tensorflow.keras.layers")
mock_keras_models = ModuleType("tensorflow.keras.models")

setattr(mock_keras_applications, "Xception", MagicMock())
setattr(mock_keras_xception, "preprocess_input", MagicMock())

for layer in [
    "Dense", "Dropout", "GlobalAveragePooling2D", "Input", "Conv2D", "BatchNormalization"
    ]:
    setattr(mock_keras_layers, layer, MagicMock())

setattr(mock_keras_models, "Model", MagicMock())

# Register mocked keras modules
sys.modules.update({
    "tensorflow.keras": mock_keras,
    "tensorflow.keras.applications": mock_keras_applications,
    "tensorflow.keras.applications.xception": mock_keras_xception,
    "tensorflow.keras.layers": mock_keras_layers,
    "tensorflow.keras.models": mock_keras_models,
})

# --- MTCNN mock ---
mock_mtcnn = ModuleType("mtcnn")
setattr(mock_mtcnn, "MTCNN", MagicMock())
sys.modules["mtcnn"] = cast(ModuleType, mock_mtcnn)

# Import module under test
from deepfake_recognition.data_processing import embedding_creation as emb


def test_extract_video_frames_uniform_reads_k_frames(monkeypatch):
    """Test that extract_video_frames_uniform reads exactly k frames and releases capture."""
    fake_frame = np.ones((10, 10, 3), dtype=np.uint8)
    cap_mock = MagicMock()
    cap_mock.get.return_value = 20
    cap_mock.read.return_value = (True, fake_frame)
    cap_mock.release = MagicMock()

    # Patch cv2 inside the target module
    monkeypatch.setattr(
        "deepfake_recognition.data_processing.embedding_creation.cv2.VideoCapture", lambda _: cap_mock
    )
    monkeypatch.setattr(
        "deepfake_recognition.data_processing.embedding_creation.cv2.cvtColor", lambda f, _: f
    )

    frames = emb.extract_video_frames_uniform("video.mp4", (299, 299), k=5)
    assert len(frames) == 5
    cap_mock.release.assert_called_once()


def test_extract_video_frames_uniform_handles_read_error(monkeypatch):
    """Test that extract_video_frames_uniform handles failed reads gracefully."""
    cap_mock = MagicMock()
    cap_mock.get.return_value = 1
    cap_mock.read.return_value = (False, None)
    cap_mock.release = MagicMock()

    monkeypatch.setattr(
        "deepfake_recognition.data_processing.embedding_creation.cv2.VideoCapture", 
        lambda _: cap_mock
    )
    monkeypatch.setattr(
        "deepfake_recognition.data_processing.embedding_creation.cv2.cvtColor",
        lambda *_: (_ for _ in ()).throw(Exception("bad frame")),
    )

    frames = emb.extract_video_frames_uniform("x.mp4", (5, 5), k=1)
    assert frames[0].shape == (5, 5, 3)
    assert np.all(frames[0] == 0)
    cap_mock.release.assert_called_once()


def test_preprocess_for_Xception(monkeypatch):
    """Test that preprocess_for_Xception resizes and scales frames correctly."""
    fake_frame = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    monkeypatch.setattr(
        "cv2.resize", lambda f, s, interpolation: np.zeros(s + (3,), dtype=np.uint8)
    )
    monkeypatch.setattr(
        "deepfake_recognition.data_processing.embedding_creation.preprocess_input", 
        lambda x: x / 255.0
    )

    arr = emb.preprocess_for_Xception([fake_frame], (10, 10))
    assert arr.shape == (1, 10, 10, 3)
    assert np.all(arr <= 1.0)


def test_build_frame_embeddings(monkeypatch):
    """Test that build_frame_embeddings calls model.predict once per frame."""
    fake_model = MagicMock()
    fake_model.predict.return_value = np.ones((1, 2048))
    frames_np = np.random.rand(2, 299, 299, 3).astype(np.float32)

    result = emb.build_frame_embeddings(fake_model, frames_np)
    assert result.shape == (2, 2048)
    assert fake_model.predict.call_count == 2


def test_aggregate_video_embeddings_mean_and_sum():
    """Test that aggregate_video_embeddings computes mean and sum correctly."""
    embs = [np.array([[1, 2, 3]]), np.array([[4, 5, 6]])]
    mean_emb = emb.aggregate_video_embeddings(embs, "mean")
    sum_emb = emb.aggregate_video_embeddings(embs, "sum")

    np.testing.assert_array_equal(mean_emb, np.mean(np.vstack(embs), axis=0))
    np.testing.assert_array_equal(sum_emb, np.sum(np.vstack(embs), axis=0))


def test_main_computes_embeddings(monkeypatch, tmp_path):
    """Test that main() runs end-to-end using mocks and does not crash."""
    cfg_mock = MagicMock()
    cfg_mock.EMBEDDING_DIR = tmp_path
    cfg_mock.SAMPLED_OUTPUT_DIR = tmp_path
    cfg_mock.SIZE_FOR_XCEPTION = (299, 299)
    cfg_mock.USE_CACHED_EMBEDDINGS = False
    cfg_mock.EMBEDDING_AGGREGATION = "mean"
    cfg_mock.FRAMES_PER_VIDEO = 1
    cfg_mock.EMISSIONS_OUTPUT_DIR = tmp_path

    # Fake directory structure
    for split in ["train", "val", "test"]:
        for lbl in ["real", "fake"]:
            d = tmp_path / split / lbl
            d.mkdir(parents=True)
            (d / "vid.mp4").write_text("fake video")

    # Core mocks
    monkeypatch.setattr("deepfake_recognition.data_processing.embedding_creation.cfg", cfg_mock)
    monkeypatch.setattr("os.listdir", lambda _: ["vid.mp4"])
    monkeypatch.setattr(
        "deepfake_recognition.data_processing.embedding_creation.extract_video_frames_uniform", 
        lambda *a, **k: [np.zeros((10, 10, 3))]
    )
    monkeypatch.setattr(
        "deepfake_recognition.data_processing.embedding_creation.preprocess_for_Xception", 
        lambda *a, **k: np.zeros((1, 10, 10, 3))
    )
    monkeypatch.setattr(
        "deepfake_recognition.data_processing.embedding_creation.build_frame_embeddings", 
        lambda m, x: np.ones((1, 2048))
    )
    monkeypatch.setattr(
        "deepfake_recognition.data_processing.embedding_creation.aggregate_video_embeddings", 
        lambda e, a: np.ones(2048)
    )

    # Mock keras + model
    monkeypatch.setattr(
        "deepfake_recognition.data_processing.embedding_creation.Xception", 
        MagicMock()
    )
    monkeypatch.setattr(
        "deepfake_recognition.data_processing.embedding_creation.GlobalAveragePooling2D", 
        MagicMock()
    )
    fake_model = MagicMock()
    fake_model.output_shape = (None, 2048)
    monkeypatch.setattr(
        "deepfake_recognition.data_processing.embedding_creation.Model", 
        MagicMock(return_value=fake_model)
    )

    # Misc mocks
    monkeypatch.setattr("builtins.print", lambda *a, **kw: None)
    monkeypatch.setattr(
        "deepfake_recognition.data_processing.embedding_creation.EmissionsTracker",
        MagicMock(return_value=MagicMock(start=lambda: None, stop=lambda: 0.0)),
    )
    monkeypatch.setattr(
        "deepfake_recognition.data_processing.embedding_creation.tf",
        SimpleNamespace(random=SimpleNamespace(set_seed=lambda *a, **kw: None)),
    )

    # Mock kerass
    mock_keras_file_utils = ModuleType("keras.src.utils.file_utils")
    setattr(mock_keras_file_utils, "get_file", lambda *a, **kw: "/tmp/fake_xception_weights.h5")
    sys.modules["keras.src.utils.file_utils"] = mock_keras_file_utils

    import deepfake_recognition.data_processing.embedding_creation as emb
    emb.main()
