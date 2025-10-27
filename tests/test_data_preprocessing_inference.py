import sys
import pytest
import numpy as np 

from types import ModuleType
from unittest.mock import MagicMock, patch
from typing import cast

# --- MOCK de TensorFlow y submódulos usados por el proyecto ---
mock_tf = ModuleType("tensorflow")
mock_keras = ModuleType("tensorflow.keras")
mock_keras_applications = ModuleType("tensorflow.keras.applications")
mock_keras_xception = ModuleType("tensorflow.keras.applications.xception")
mock_keras_layers = ModuleType("tensorflow.keras.layers")
mock_keras_models = ModuleType("tensorflow.keras.models")

# Añadir mocks de clases/funciones que el código podría importar
setattr(mock_keras_applications, "Xception", MagicMock())
setattr(mock_keras_xception, "preprocess_input", MagicMock())

# Añadir mocks de capas comunes
for layer_name in [
    "Dense",
    "Dropout",
    "GlobalAveragePooling2D",
    "Input",
    "Conv2D",
    "BatchNormalization",
]:
    setattr(mock_keras_layers, layer_name, MagicMock())

# Mock para modelos Keras
setattr(mock_keras_models, "Model", MagicMock())

# Registrar los módulos en sys.modules
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

# --- Ahora ya puedes importar tu módulo sin que falle ---
import deepfake_recognition.data_processing.data_preprocessing_inference as dpi


# ---------- FIXTURES ----------

@pytest.fixture
def mock_detector():
    """Mock del detector MTCNN."""
    detector = MagicMock()
    # Por defecto devuelve una cara detectada
    detector.detect_faces.return_value = [{'box': [10, 10, 20, 20]}]
    return detector


@pytest.fixture
def mock_video_capture():
    """Mock del cv2.VideoCapture."""
    cap = MagicMock()
    cap.get.return_value = 10  # total de frames
    cap.read.return_value = (True, np.ones((100, 100, 3), dtype=np.uint8))
    return cap


@pytest.fixture
def fake_img_size():
    """Tamaño de imagen simulado."""
    return (64, 64)


@pytest.fixture(autouse=True)
def patch_cv2_and_cfg(tmp_path, mock_video_capture):
    """Parchea cv2 y cfg globalmente."""
    with patch("cv2.VideoCapture", return_value=mock_video_capture), \
         patch("cv2.cvtColor", side_effect=lambda f, c: f), \
         patch("cv2.imwrite", return_value=True), \
         patch("cv2.rectangle", side_effect=lambda *args, **kwargs: None), \
         patch("os.path.exists", return_value=True), \
         patch("deepfake_recognition.config") as mock_cfg:

        mock_cfg.RAW_DATA_DIR = str(tmp_path / "raw")
        mock_cfg.PROCESSED_DATA_DIR = str(tmp_path / "processed")
        mock_cfg.FRAMES_PER_VIDEO = 5

        yield mock_cfg


# ---------- TESTS extract_face_frames ----------

@pytest.mark.parametrize("faces_detected", [True, False])
def test_extract_face_frames(mock_detector, faces_detected, fake_img_size):
    if not faces_detected:
        mock_detector.detect_faces.return_value = []

    frames = dpi.extract_face_frames(
        filepath="video.mp4",
        detector=mock_detector,
        IMG_SIZE=fake_img_size,
        k=3
    )

    assert isinstance(frames, list)
    assert len(frames) == 3

    for frame in frames:
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3

    if faces_detected:
        # Si detecta caras, los frames no deben ser completamente negros
        assert not np.all(frames[0] == 0)
    else:
        # Si no detecta caras, los frames son vacíos (negros)
        assert np.all(frames[0] == 0)


# ---------- TESTS extract_and_save_face_paths ----------

def test_extract_and_save_face_paths_success(mock_detector, tmp_path):
    boxed_paths, face_paths = dpi.extract_and_save_face_paths(
        filepath="video.mp4",
        detector=mock_detector,
        k=3
    )

    # Validación general
    assert len(boxed_paths) == 3
    assert len(face_paths) == 3
    for p in boxed_paths + face_paths:
        assert isinstance(p, str)
        assert p.endswith(".jpg")


def test_extract_and_save_face_paths_with_exception(mock_detector):
    """Simula error en detector.detect_faces para cubrir except."""
    mock_detector.detect_faces.side_effect = Exception("error")

    boxed_paths, face_paths = dpi.extract_and_save_face_paths(
        filepath="video.mp4",
        detector=mock_detector,
        k=2
    )

    assert boxed_paths == ["", ""]
    assert face_paths == ["", ""]


# ---------- TEST main (mocked) ----------

def test_main_with_missing_csv(monkeypatch):
    """El main debe continuar cuando el CSV no existe."""
    mock_detector_class = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(dpi, "MTCNN", mock_detector_class)
    monkeypatch.setattr("os.path.exists", lambda path: False)

    dpi.main()  
