import io
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

import deepfake_recognition.api as api


@pytest.fixture
def client():
    """Creates a FastAPI test client."""
    return TestClient(api.app)


# ------------------------------
#  BASIC ENDPOINT TESTS
# ------------------------------

def test_read_root(client):
    """Tests that the root endpoint returns a welcome message."""
    response = client.get('/')
    assert response.status_code == 200
    assert 'message' in response.json()
    assert 'Deepfake Recognition API' in response.json()['message']


def test_health_check_ok(client):
    """Tests that /health returns model info and health status."""
    response = client.get('/health')
    data = response.json()
    assert response.status_code == 200
    assert 'status' in data
    assert 'model_path' in data
    assert 'feature_dim' in data


# ------------------------------
#  MOCKED MODEL TESTS
# ------------------------------

@pytest.fixture
def mock_model():
    """Creates a fake ML model that returns deterministic probabilities."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.2, 0.8], [0.3, 0.7]])
    return model


@patch("deepfake_recognition.api.model", new_callable=lambda: MagicMock())
@patch("deepfake_recognition.api.extract_face_frames")
@patch("deepfake_recognition.api.preprocess_for_Xception")
@patch("deepfake_recognition.api.build_frame_embeddings")
def test_detect_deepfake_success(
    mock_build_emb, mock_preprocess, mock_extract, mock_model, client
):
    """Tests successful deepfake detection with mocked data."""
    # Mock intermediate steps
    mock_extract.return_value = [np.ones((64, 64, 3), dtype=np.uint8)]
    mock_preprocess.return_value = np.random.rand(1, 64, 64, 3)
    mock_build_emb.return_value = np.random.rand(2, 2048)
    mock_model.predict_proba.return_value = np.array([[0.1, 0.9], [0.2, 0.8]])

    # Create dummy video file
    dummy_video = io.BytesIO(b"fake video data")
    files = {"video": ("test.mp4", dummy_video, "video/mp4")}

    with patch("cv2.VideoCapture") as mock_cap:
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda x: {5: 10, 7: 30}.get(x, 10)  # FPS & frame count
        mock_cap.return_value = cap

        response = client.post("/detect_deepfake", files=files)

    assert response.status_code == 200
    data = response.json()
    assert "deepfake_probability" in data
    assert "is_deepfake" in data
    assert isinstance(data["deepfake_probability"], float)


def test_detect_deepfake_invalid_format(client):
    """Tests rejection of unsupported video formats."""
    bad_file = io.BytesIO(b"not a video")
    files = {"video": ("test.txt", bad_file, "text/plain")}
    response = client.post("/detect_deepfake", files=files)
    assert response.status_code == 400
    assert "Invalid file format" in response.text


@patch("deepfake_recognition.api.model", None)
def test_detect_deepfake_model_not_loaded(client):
    """Tests that API returns 503 when model is missing."""
    fake_video = io.BytesIO(b"fake data")
    files = {"video": ("video.mp4", fake_video, "video/mp4")}
    response = client.post("/detect_deepfake", files=files)
    assert response.status_code == 503
    assert "Model not loaded" in response.text


# ------------------------------
#  EXTRACT FACES TEST
# ------------------------------

@patch("deepfake_recognition.api.extract_and_save_face_paths")
@patch("cv2.VideoCapture")
def test_extract_faces_from_video_api(mock_cap, mock_extract, client, tmp_path):
    """Tests /extract_faces_from_video endpoint with mocked data and zip generation."""
    # Mock video capture to simulate valid video
    cap = MagicMock()
    cap.isOpened.return_value = True
    mock_cap.return_value = cap

    # Create fake extracted image paths
    temp_file1 = tmp_path / "face1.jpg"
    temp_file2 = tmp_path / "box1.jpg"
    temp_file1.write_text("data")
    temp_file2.write_text("data")

    mock_extract.return_value = ([str(temp_file1)], [str(temp_file2)])

    video_file = io.BytesIO(b"fake video data")
    files = {"video": ("video.mp4", video_file, "video/mp4")}

    response = client.post("/extract_faces_from_video", files=files)
    assert response.status_code == 200
    assert response.headers["Content-Disposition"].endswith(".zip")


@patch("cv2.VideoCapture")
def test_extract_faces_from_video_api_failure(mock_cap, client):
    """Tests error handling when video cannot be opened."""
    cap = MagicMock()
    cap.isOpened.return_value = False
    mock_cap.return_value = cap

    video_file = io.BytesIO(b"fake video data")
    files = {"video": ("video.mp4", video_file, "video/mp4")}

    response = client.post("/extract_faces_from_video", files=files)
    assert response.status_code == 500
    assert "Error extracting faces" in response.text
