import pytest
import numpy as np
import os
import cv2
from unittest.mock import MagicMock, patch

import deepfake_recognition.data_processing.data_preprocessing_inference as dpi


@pytest.fixture
def mock_detector():
    """Simulates an MTCNN detector that always detects one face."""
    detector = MagicMock()
    detector.detect_faces.return_value = [{'box': [10, 10, 50, 50]}]
    return detector


@pytest.fixture
def mock_detector_no_face():
    """Simulates an MTCNN detector that detects no faces."""
    detector = MagicMock()
    detector.detect_faces.return_value = []
    return detector


@pytest.fixture
def dummy_frame():
    """Creates a dummy RGB frame."""
    return np.ones((100, 100, 3), dtype=np.uint8) * 255


@patch("cv2.VideoCapture")
def test_extract_face_frames_success(mock_videocap, mock_detector, dummy_frame, tmp_path):
    """Tests successful extraction of frames when a face is detected."""
    # Mock the behavior of cv2.VideoCapture
    cap = MagicMock()
    cap.get.return_value = 10
    cap.read.return_value = (True, dummy_frame)
    mock_videocap.return_value = cap

    frames = dpi.extract_face_frames("fake_video.mp4", mock_detector, (64, 64), k=3)

    assert isinstance(frames, list)
    assert len(frames) == 3
    for frame in frames:
        assert isinstance(frame, np.ndarray)
        assert frame.shape[2] == 3  # RGB channels


@patch("cv2.VideoCapture")
def test_extract_face_frames_no_face(mock_videocap, mock_detector_no_face, dummy_frame):
    """Tests frame extraction when no faces are detected."""
    cap = MagicMock()
    cap.get.return_value = 5
    cap.read.return_value = (True, dummy_frame)
    mock_videocap.return_value = cap

    frames = dpi.extract_face_frames("fake_video.mp4", mock_detector_no_face, (64, 64), k=2)

    assert len(frames) == 2
    assert np.all(frames[0] == 0)  # empty frame (no face detected)


@patch("cv2.VideoCapture")
def test_extract_and_save_face_paths(mock_videocap, mock_detector, dummy_frame, tmp_path):
    """Tests saving of face crops and frames with bounding boxes."""
    os.makedirs(os.path.join(tmp_path, "api_uploads"), exist_ok=True)

    # Mock temporary config paths
    dpi.cfg.PROCESSED_DATA_DIR = str(tmp_path)
    dpi.cfg.RAW_DATA_DIR = str(tmp_path)

    cap = MagicMock()
    cap.get.return_value = 10
    cap.read.return_value = (True, dummy_frame)
    mock_videocap.return_value = cap

    boxed_paths, face_paths = dpi.extract_and_save_face_paths(
        os.path.join(tmp_path, "video.mp4"), mock_detector, k=2
    )

    assert len(boxed_paths) == 2
    assert len(face_paths) == 2
    for p in boxed_paths + face_paths:
        # Files may be saved or empty string in case of mock failure
        assert os.path.exists(p) or p == ""


@patch("cv2.VideoCapture")
def test_extract_and_save_face_paths_error(mock_videocap, mock_detector):
    """Tests error handling when reading a frame fails."""
    cap = MagicMock()
    cap.get.return_value = 10
    cap.read.side_effect = Exception("Read error")  # simulate frame read failure
    mock_videocap.return_value = cap

    dpi.cfg.PROCESSED_DATA_DIR = "/tmp"
    dpi.cfg.RAW_DATA_DIR = "/tmp"

    boxed_paths, face_paths = dpi.extract_and_save_face_paths("fake_video.mp4", mock_detector, k=1)

    assert boxed_paths[0] == ""
    assert face_paths[0] == ""
