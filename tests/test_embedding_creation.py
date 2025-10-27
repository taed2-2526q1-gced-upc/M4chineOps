import sys
import pytest
import numpy as np
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock
from typing import cast

# --- MOCK de TensorFlow (mínimo y funcional) ---
import numpy as np
from types import ModuleType, SimpleNamespace

mock_tf = ModuleType("tensorflow")
mock_tf.__path__ = []  # permite importar submódulos
mock_tf.random = SimpleNamespace(set_seed=lambda *a, **kw: None)
sys.modules["tensorflow"] = mock_tf
sys.modules["tensorflow.compat"] = ModuleType("tensorflow.compat")
sys.modules["tensorflow.compat.v2"] = ModuleType("tensorflow.compat.v2")

# --- MOCK de Keras y submódulos ---
mock_keras = ModuleType("tensorflow.keras")
mock_keras_applications = ModuleType("tensorflow.keras.applications")
mock_keras_xception = ModuleType("tensorflow.keras.applications.xception")
mock_keras_layers = ModuleType("tensorflow.keras.layers")
mock_keras_models = ModuleType("tensorflow.keras.models")

setattr(mock_keras_applications, "Xception", MagicMock())
setattr(mock_keras_xception, "preprocess_input", MagicMock())

for layer_name in [
    "Dense",
    "Dropout",
    "GlobalAveragePooling2D",
    "Input",
    "Conv2D",
    "BatchNormalization",
]:
    setattr(mock_keras_layers, layer_name, MagicMock())

setattr(mock_keras_models, "Model", MagicMock())

# ✅ Registrar módulos en sys.modules
sys.modules["tensorflow"] = mock_tf
sys.modules["tensorflow.keras"] = mock_keras
sys.modules["tensorflow.keras.applications"] = mock_keras_applications
sys.modules["tensorflow.keras.applications.xception"] = mock_keras_xception
sys.modules["tensorflow.keras.layers"] = mock_keras_layers
sys.modules["tensorflow.keras.models"] = mock_keras_models

# --- MOCK de MTCNN ---
mock_mtcnn = ModuleType("mtcnn")
setattr(mock_mtcnn, "MTCNN", MagicMock())
sys.modules["mtcnn"] = cast(ModuleType, mock_mtcnn)

from deepfake_recognition.data_processing import embedding_creation as emb




# ---------------------------------------------------------------------------
# TESTS: extract_video_frames_uniform
# ---------------------------------------------------------------------------

def test_extract_video_frames_uniform_reads_k_frames(monkeypatch):
    """
    Test: extract_video_frames_uniform returns exactly k frames.
    """
    fake_frame = np.ones((10, 10, 3), dtype=np.uint8)
    cap_mock = MagicMock()
    cap_mock.get.return_value = 20  # total frames
    cap_mock.read.return_value = (True, fake_frame)

    monkeypatch.setattr("cv2.VideoCapture", lambda _: cap_mock)
    monkeypatch.setattr("cv2.cvtColor", lambda f, _: f)

    frames = emb.extract_video_frames_uniform("video.mp4", (299, 299), k=5)
    assert len(frames) == 5
    cap_mock.release.assert_called_once()


def test_extract_video_frames_uniform_handles_read_error(monkeypatch):
    """
    Test: extract_video_frames_uniform handles read errors gracefully.
    """
    cap_mock = MagicMock()
    cap_mock.get.return_value = 1
    cap_mock.read.return_value = (False, None)

    monkeypatch.setattr("cv2.VideoCapture", lambda _: cap_mock)
    monkeypatch.setattr("cv2.cvtColor", lambda f, _: (_ for _ in ()).throw(Exception("bad frame")))

    frames = emb.extract_video_frames_uniform("x.mp4", (5, 5), k=1)
    assert frames[0].shape == (5, 5, 3)
    assert np.all(frames[0] == 0)


# ---------------------------------------------------------------------------
# TESTS: preprocess_for_Xception
# ---------------------------------------------------------------------------

def test_preprocess_for_Xception(monkeypatch):
    """
    Test: preprocess_for_Xception resizes and preprocesses correctly.
    """
    fake_frame = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    monkeypatch.setattr("cv2.resize", lambda f, s, interpolation: np.zeros(s + (3,), dtype=np.uint8))
    monkeypatch.setattr("deepfake_recognition.data_processing.embedding_creation.preprocess_input", lambda x: x / 255.0)

    arr = emb.preprocess_for_Xception([fake_frame], (10, 10))
    assert arr.shape == (1, 10, 10, 3)
    assert np.all(arr <= 1.0)


# ---------------------------------------------------------------------------
# TESTS: build_frame_embeddings
# ---------------------------------------------------------------------------

def test_build_frame_embeddings(monkeypatch):
    """
    Test: build_frame_embeddings calls model.predict once per frame.
    """
    fake_model = MagicMock()
    fake_model.predict.return_value = np.ones((1, 2048))
    frames_np = np.random.rand(2, 299, 299, 3).astype(np.float32)

    result = emb.build_frame_embeddings(fake_model, frames_np)
    assert result.shape == (2, 2048)
    assert fake_model.predict.call_count == 2


# ---------------------------------------------------------------------------
# TESTS: aggregate_video_embeddings
# ---------------------------------------------------------------------------

def test_aggregate_video_embeddings_mean_and_sum():
    """
    Test: aggregate_video_embeddings aggregates correctly.
    """
    embs = [np.array([[1, 2, 3]]), np.array([[4, 5, 6]])]

    mean_emb = emb.aggregate_video_embeddings(embs, "mean")
    sum_emb = emb.aggregate_video_embeddings(embs, "sum")

    np.testing.assert_array_equal(mean_emb, np.mean(np.vstack(embs), axis=0))
    np.testing.assert_array_equal(sum_emb, np.sum(np.vstack(embs), axis=0))


# ---------------------------------------------------------------------------
# TESTS: main (mocked execution)
# ---------------------------------------------------------------------------

def test_main_computes_embeddings(monkeypatch, tmp_path):
    cfg_mock = MagicMock()
    cfg_mock.EMBEDDING_DIR = tmp_path
    cfg_mock.SAMPLED_OUTPUT_DIR = tmp_path
    cfg_mock.SIZE_FOR_XCEPTION = (299, 299)
    cfg_mock.USE_CACHED_EMBEDDINGS = False
    cfg_mock.EMBEDDING_AGGREGATION = "mean"
    cfg_mock.FRAMES_PER_VIDEO = 1

    # Fake dirs
    for split in ["train", "val", "test"]:
        for lbl in ["real", "fake"]:
            d = tmp_path / split / lbl
            d.mkdir(parents=True)
            (d / "vid.mp4").write_text("fake video")

    # Parches
    monkeypatch.setattr("deepfake_recognition.data_processing.embedding_creation.cfg", cfg_mock)
    monkeypatch.setattr("os.listdir", lambda p: ["vid.mp4"])
    monkeypatch.setattr("deepfake_recognition.data_processing.embedding_creation.extract_video_frames_uniform", lambda *a, **kw: [np.zeros((10, 10, 3))])
    monkeypatch.setattr("deepfake_recognition.data_processing.embedding_creation.preprocess_for_Xception", lambda *a, **kw: np.zeros((1, 10, 10, 3)))
    monkeypatch.setattr("deepfake_recognition.data_processing.embedding_creation.build_frame_embeddings", lambda m, x: np.ones((1, 5)))

    monkeypatch.setattr("deepfake_recognition.data_processing.embedding_creation.aggregate_video_embeddings", lambda e, a: np.ones(5))

    # ✅ mock directly in sys.modules before import
    mock_keras_file_utils = ModuleType("keras.src.utils.file_utils")
    setattr(mock_keras_file_utils, "get_file", lambda *a, **kw: "/tmp/fake_xception_weights.h5")
    sys.modules["keras.src.utils.file_utils"] = mock_keras_file_utils


    # 🚫 Mock de Xception dentro del propio módulo
    monkeypatch.setattr(
        "deepfake_recognition.data_processing.embedding_creation.Xception",
        MagicMock(return_value=MagicMock())
    )

    # Otros mocks
    monkeypatch.setattr("deepfake_recognition.data_processing.embedding_creation.GlobalAveragePooling2D", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr("deepfake_recognition.data_processing.embedding_creation.Model", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr("builtins.print", lambda *a, **kw: None)
    monkeypatch.setattr(
        "deepfake_recognition.data_processing.embedding_creation.EmissionsTracker",
        MagicMock(return_value=MagicMock(start=lambda: None, stop=lambda: 0.0))
    )

    import deepfake_recognition.data_processing.embedding_creation as emb
    emb.main()