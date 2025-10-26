import io
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import cv2

import deepfake_recognition.app_streamlit as app


# ------------------------------
#  check_api_health
# ------------------------------

@patch("deepfake_recognition.app_streamlit.requests.get")
@patch("deepfake_recognition.app_streamlit.st")
def test_check_api_health_success(mock_st, mock_get):
    """Tests successful API health check returning JSON."""
    mock_get.return_value.json.return_value = {"status": "ok"}
    mock_get.return_value.raise_for_status.return_value = None

    result = app.check_api_health("http://fake-url/health")
    assert result["status"] == "ok"
    mock_st.error.assert_not_called()
    mock_st.stop.assert_not_called()


@patch("deepfake_recognition.app_streamlit.requests.get")
@patch("deepfake_recognition.app_streamlit.st")
def test_check_api_health_failure(mock_st, mock_get):
    """Tests failed API health check raising a RequestException."""
    from requests.exceptions import RequestException
    mock_get.side_effect = RequestException("Connection error")

    with pytest.raises(SystemExit):
        app.check_api_health("http://bad-url")

    mock_st.error.assert_called_once()


# ------------------------------
#  call_api_endpoint
# ------------------------------

@patch("deepfake_recognition.app_streamlit.requests.post")
@patch("deepfake_recognition.app_streamlit.st.spinner")
def test_call_api_endpoint_success(mock_spinner, mock_post):
    """Tests successful POST call to API endpoint."""
    fake_response = MagicMock(status_code=200)
    mock_post.return_value = fake_response

    buffer = io.BytesIO(b"video data")
    result = app.call_api_endpoint("http://fake-url/detect", buffer, "video.mp4")

    assert result.status_code == 200
    mock_post.assert_called_once()


@patch("deepfake_recognition.app_streamlit.requests.post")
@patch("deepfake_recognition.app_streamlit.st")
def test_call_api_endpoint_request_exception(mock_st, mock_post):
    """Tests failed API request returning None."""
    from requests.exceptions import RequestException
    mock_post.side_effect = RequestException("Timeout")

    buffer = io.BytesIO(b"video data")
    result = app.call_api_endpoint("http://fake-url", buffer, "video.mp4")

    assert result is None
    mock_st.error.assert_called_once()


# ------------------------------
#  check_video_duration
# ------------------------------

@patch("deepfake_recognition.app_streamlit.cv2.VideoCapture")
@patch("deepfake_recognition.app_streamlit.st")
def test_check_video_duration_under_limit(mock_st, mock_cv):
    """Tests that short videos pass duration check."""
    cap = MagicMock()
    cap.get.side_effect = lambda x: {cv2.CAP_PROP_FPS: 30, cv2.CAP_PROP_FRAME_COUNT: 300}.get(x, 0)
    mock_cv.return_value = cap

    fake_file = MagicMock()
    fake_file.getbuffer.return_value = b"data"

    assert app.check_video_duration(fake_file, max_duration=20) is True
    mock_st.error.assert_not_called()


@patch("deepfake_recognition.app_streamlit.cv2.VideoCapture")
@patch("deepfake_recognition.app_streamlit.st")
def test_check_video_duration_too_long(mock_st, mock_cv):
    """Tests that long videos trigger an error and return False."""
    cap = MagicMock()
    cap.get.side_effect = lambda x: {cv2.CAP_PROP_FPS: 30, cv2.CAP_PROP_FRAME_COUNT: 90000}.get(x, 0)
    mock_cv.return_value = cap

    fake_file = MagicMock()
    fake_file.getbuffer.return_value = b"data"

    result = app.check_video_duration(fake_file, max_duration=10)
    assert result is False
    mock_st.error.assert_called_once()


# ------------------------------
#  display_detection_results
# ------------------------------

@patch("deepfake_recognition.app_streamlit.st")
def test_display_detection_results_deepfake(mock_st):
    """Tests rendering of detection results for a deepfake video."""
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "summary": "Deepfake detected",
        "deepfake_probability": 0.95,
        "is_deepfake": True,
        "frame_probabilities": [0.9, 0.95],
        "frames_used": 2
    }

    app.display_detection_results(fake_response, threshold=0.5)
    mock_st.error.assert_called()
    mock_st.metric.assert_called_once()


@patch("deepfake_recognition.app_streamlit.st")
def test_display_detection_results_authentic(mock_st):
    """Tests rendering of detection results for an authentic video."""
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "summary": "Authentic video",
        "deepfake_probability": 0.1,
        "is_deepfake": False,
        "frame_probabilities": [0.1, 0.05],
        "frames_used": 2
    }

    app.display_detection_results(fake_response, threshold=0.5)
    mock_st.success.assert_called()
    mock_st.metric.assert_called_once()


@patch("deepfake_recognition.app_streamlit.st")
def test_display_detection_results_bad_json(mock_st):
    """Tests that invalid JSON responses are handled gracefully."""
    fake_response = MagicMock()
    fake_response.json.side_effect = ValueError("Invalid JSON")

    app.display_detection_results(fake_response, threshold=0.5)
    mock_st.error.assert_called_with("Could not parse JSON response from detection endpoint.")
