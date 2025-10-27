import sys
import pytest
import numpy as np

from types import ModuleType
from unittest.mock import MagicMock, patch
from typing import cast

# --- MOCK TensorFlow and its submodules used by the project ---
mock_tf = ModuleType("tensorflow")
mock_keras = ModuleType("tensorflow.keras")
mock_keras_applications = ModuleType("tensorflow.keras.applications")
mock_keras_xception = ModuleType("tensorflow.keras.applications.xception")
mock_keras_layers = ModuleType("tensorflow.keras.layers")
mock_keras_models = ModuleType("tensorflow.keras.models")

# Mock necessary functions/classes to avoid importing real TensorFlow
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

# Register modules in sys.modules so imports work
sys.modules["tensorflow"] = mock_tf
sys.modules["tensorflow.keras"] = mock_keras
sys.modules["tensorflow.keras.applications"] = mock_keras_applications
sys.modules["tensorflow.keras.applications.xception"] = mock_keras_xception
sys.modules["tensorflow.keras.layers"] = mock_keras_layers
sys.modules["tensorflow.keras.models"] = mock_keras_models

# --- MOCK MTCNN ---
mock_mtcnn = ModuleType("mtcnn")
setattr(mock_mtcnn, "MTCNN", MagicMock())
sys.modules["mtcnn"] = cast(ModuleType, mock_mtcnn)

# --- Import the module to test ---
import deepfake_recognition.data_processing.data_preprocessing_inference as dpi


# ---------- FIXTURES ----------

@pytest.fixture
def mock_detector():
    """
    Mock for the MTCNN detector.
    Returns one detected face by default.
    """
    detector = MagicMock()
    detector.detect_faces.return_value = [{'box': [10, 10, 20, 20]}]
    return detector


@pytest.fixture
def mock_video_capture():
    """
    Mock for cv2.VideoCapture.
    Simulates a video with 10 frames of ones (dummy image data).
    """
    cap = MagicMock()
    cap.get.return_value = 10  # total frames
    cap.read.side_effect = [(True, np.ones((100, 100, 3), dtype=np.uint8)) for _ in range(10)]
    return cap


@pytest.fixture
def fake_img_size():
    """Simulated image size for empty frames."""
    return (64, 64)


@pytest.fixture(autouse=True)
def patch_cv2_and_cfg(tmp_path, mock_video_capture):
    """
    Globally patch OpenCV functions and the config module for tests.
    Provides temporary directories for RAW, PROCESSED, and API_UPLOADS.
    """
    with patch("deepfake_recognition.data_processing.data_preprocessing_inference.cv2.VideoCapture", return_value=mock_video_capture), \
         patch("deepfake_recognition.data_processing.data_preprocessing_inference.cv2.cvtColor", side_effect=lambda f, c: f), \
         patch("deepfake_recognition.data_processing.data_preprocessing_inference.cv2.imwrite", return_value=True), \
         patch("deepfake_recognition.data_processing.data_preprocessing_inference.cv2.rectangle", side_effect=lambda *args, **kwargs: None), \
         patch("os.path.exists", return_value=True), \
         patch("deepfake_recognition.config") as mock_cfg:

        # Patch config variables
        mock_cfg.RAW_DATA_DIR = str(tmp_path / "raw")
        mock_cfg.FRAMES_PER_VIDEO = 12
        mock_cfg.API_UPLOADS_DATA_DIR = str(tmp_path / "api_uploads")

        yield mock_cfg


# ---------- TESTS extract_face_frames ----------

@pytest.mark.parametrize("faces_detected", [True, False])
def test_extract_face_frames(mock_detector, faces_detected, fake_img_size):
    """
    Test extraction of face frames from a video.

    - If faces are detected, frames should contain nonzero pixels.
    - If no faces are detected, frames should be empty (all zeros).
    """
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
        # Ensure at least some pixels are nonzero when a face is detected
        assert np.any(frames[0] != 0)
    else:
        # Empty frame (all zeros) when no faces detected
        assert np.all(frames[0] == 0)


# ---------- TESTS extract_and_save_face_paths ----------

def test_extract_and_save_face_paths_success(mock_detector, tmp_path):
    """
    Test extracting and saving face frames successfully.
    Checks that paths are returned and are strings.
    """
    boxed_paths, face_paths = dpi.extract_and_save_face_paths(
        detector=mock_detector,
        filepath="video.mp4",
        filename="sample_video",
        k=3
    )

    assert len(boxed_paths) == 3
    assert len(face_paths) == 3
    for p in boxed_paths + face_paths:
        assert isinstance(p, str)


def test_extract_and_save_face_paths_with_exception(mock_detector):
    """
    Simulate an exception during face detection.
    Function should return empty paths for all frames.
    """
    mock_detector.detect_faces.side_effect = Exception("error")

    boxed_paths, face_paths = dpi.extract_and_save_face_paths(
        detector=mock_detector,
        filepath="video.mp4",
        filename="broken_video",
        k=2
    )

    assert boxed_paths == ["", ""]
    assert face_paths == ["", ""]


# ---------- TEST MAIN PLACEHOLDER ----------

def test_main_placeholder_exists():
    """
    Ensure the module has the necessary functions and does not fail
    due to the absence of a main() function.
    """
    assert hasattr(dpi, "extract_face_frames")
    assert hasattr(dpi, "extract_and_save_face_paths")
