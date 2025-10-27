import io
import sys
import pytest
import numpy as np

from types import ModuleType
from unittest.mock import MagicMock, patch
from typing import cast
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------
#  MOCK TensorFlow and MTCNN dependencies to safely import the API
# ---------------------------------------------------------------------
mock_tf = ModuleType("tensorflow")
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

sys.modules["tensorflow"] = mock_tf
sys.modules["tensorflow.keras"] = mock_keras
sys.modules["tensorflow.keras.applications"] = mock_keras_applications
sys.modules["tensorflow.keras.applications.xception"] = mock_keras_xception
sys.modules["tensorflow.keras.layers"] = mock_keras_layers
sys.modules["tensorflow.keras.models"] = mock_keras_models

mock_mtcnn = ModuleType("mtcnn")
setattr(mock_mtcnn, "MTCNN", MagicMock())
sys.modules["mtcnn"] = cast(ModuleType, mock_mtcnn)

# ---------------------------------------------------------------------
#  Import API after mocks
# ---------------------------------------------------------------------
import deepfake_recognition.api.api as api


# ---------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------
@pytest.fixture
def client():
    """Creates a reusable FastAPI test client."""
    return TestClient(api.app)


@pytest.fixture
def mock_model():
    """Creates a fake model that returns deterministic probabilities."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.2, 0.8], [0.3, 0.7]])
    return model


# ---------------------------------------------------------------------
#  Basic endpoint tests
# ---------------------------------------------------------------------
def test_read_root(client):
    """Root endpoint should return a welcome message."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "Deepfake Recognition API" in data["message"]


def test_health_check_ok(client):
    """Health endpoint should return model information."""
    response = client.get("/health")
    data = response.json()
    assert response.status_code == 200
    assert "status" in data
    assert "model_path" in data
    assert "feature_dim" in data
    assert isinstance(data["model_loaded"], bool)


def test_health_check_error_state(monkeypatch, client):
    """Covers /health response when model_error is not None."""
    monkeypatch.setattr(api, "model_error", "load failure")
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "error"
    assert data["model_loaded"] is False


# ---------------------------------------------------------------------
#  Deepfake detection tests
# ---------------------------------------------------------------------
@patch("deepfake_recognition.api.api.model", new_callable=lambda: MagicMock())
@patch("deepfake_recognition.api.api.extract_face_frames")
@patch("deepfake_recognition.api.api.preprocess_for_Xception")
@patch("deepfake_recognition.api.api.build_frame_embeddings")
def test_detect_deepfake_success(
    mock_build_emb, mock_preprocess, mock_extract, mock_model, client
):
    """Simulates successful deepfake detection with fully mocked pipeline."""
    mock_extract.return_value = [np.ones((64, 64, 3), dtype=np.uint8)]
    mock_preprocess.return_value = np.random.rand(1, 64, 64, 3)
    mock_build_emb.return_value = np.random.rand(2, 2048)
    mock_model.predict_proba.return_value = np.array([[0.1, 0.9], [0.2, 0.8]])

    dummy_video = io.BytesIO(b"fake video data")
    files = {"video": ("test.mp4", dummy_video, "video/mp4")}

    with patch("deepfake_recognition.api.api.cv2.VideoCapture") as mock_cap:
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda x: {5: 10, 7: 30}.get(x, 10)
        mock_cap.return_value = cap
        response = client.post("/detect_deepfake", files=files)

    assert response.status_code == 200, response.text
    data = response.json()
    assert "deepfake_probability" in data
    assert "is_deepfake" in data
    assert isinstance(data["deepfake_probability"], float)


@patch("deepfake_recognition.api.api.model", new_callable=lambda: MagicMock())
def test_detect_deepfake_invalid_format(mock_model, client):
    """Rejects unsupported file formats (not .mp4)."""
    bad_file = io.BytesIO(b"not a video")
    files = {"video": ("test.txt", bad_file, "text/plain")}
    response = client.post("/detect_deepfake", files=files)
    assert response.status_code == 400
    assert "Invalid file format" in response.text


@patch("deepfake_recognition.api.api.model", None)
def test_detect_deepfake_model_not_loaded(client):
    """Returns 503 if the model is not loaded."""
    fake_video = io.BytesIO(b"fake data")
    files = {"video": ("video.mp4", fake_video, "video/mp4")}
    response = client.post("/detect_deepfake", files=files)
    assert response.status_code == 503
    assert "Model not loaded" in response.text


# ---------------------------------------------------------------------
#  Face extraction tests
# ---------------------------------------------------------------------
@patch("deepfake_recognition.api.api.extract_and_save_face_paths")
@patch("deepfake_recognition.api.api.cv2.VideoCapture")
def test_extract_faces_from_video_api(mock_cap, mock_extract, client, tmp_path):
    """Validates /download_faces_from_video endpoint behavior."""
    cap = MagicMock()
    cap.isOpened.return_value = True
    mock_cap.return_value = cap

    # Create temporary fake image files
    temp_face = tmp_path / "face1.jpg"
    temp_box = tmp_path / "box1.jpg"
    temp_face.write_text("data")
    temp_box.write_text("data")

    mock_extract.return_value = ([str(temp_box)], [str(temp_face)])

    video_file = io.BytesIO(b"fake video data")
    files = {"video": ("video.mp4", video_file, "video/mp4")}
    response = client.post("/download_faces_from_video", files=files)

    assert response.status_code == 200, response.text
    assert "Content-Disposition" in response.headers
    assert response.headers["Content-Disposition"].endswith(".zip")


@patch("deepfake_recognition.api.api.cv2.VideoCapture")
def test_extract_faces_from_video_api_failure(mock_cap, client):
    """Checks proper 500 response when video cannot be opened."""
    cap = MagicMock()
    cap.isOpened.return_value = False
    mock_cap.return_value = cap

    video_file = io.BytesIO(b"fake video data")
    files = {"video": ("video.mp4", video_file, "video/mp4")}
    response = client.post("/download_faces_from_video", files=files)
    assert response.status_code == 500
    assert "Error extracting faces" in response.text


def test_display_faces_from_video_api_failure(client):
    """Ensures HTML endpoint handles failure gracefully."""
    # Patch cv2.VideoCapture to simulate failure
    with patch("deepfake_recognition.api.api.cv2.VideoCapture") as mock_cap:
        cap = MagicMock()
        cap.isOpened.return_value = False
        mock_cap.return_value = cap

        video_file = io.BytesIO(b"fake video data")
        files = {"video": ("bad_video.mp4", video_file, "video/mp4")}
        response = client.post("/display_faces_from_video", files=files)

    assert response.status_code == 500
    assert "Error extracting faces" in response.text
