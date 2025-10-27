"""Unit tests for the Deepfake Recognition Streamlit application."""

import io
from unittest.mock import MagicMock, patch
import cv2

import deepfake_recognition.api.app_streamlit as app


@patch("deepfake_recognition.api.app_streamlit.requests.get")
@patch("deepfake_recognition.api.app_streamlit.st")
def test_check_api_health_success(mock_st, mock_get):
    """Test successful API health check returning valid JSON."""
    mock_get.return_value.json.return_value = {"status": "ok"}
    mock_get.return_value.raise_for_status.return_value = None

    result = app.check_api_health("http://fake-url/health")
    assert result["status"] == "ok"
    mock_st.error.assert_not_called()
    mock_st.stop.assert_not_called()


@patch("deepfake_recognition.api.app_streamlit.requests.get")
@patch("deepfake_recognition.api.app_streamlit.st")
def test_check_api_health_failure(mock_st, mock_get):
    """
    Test that a failed API health check calls st.error() and st.stop().
    Since st.stop() does not raise SystemExit, we simply assert it was called.
    """
    from requests.exceptions import RequestException
    mock_get.side_effect = RequestException("Connection error")

    app.check_api_health("http://bad-url")

    mock_st.error.assert_called_once()
    mock_st.stop.assert_called_once()


@patch("deepfake_recognition.api.app_streamlit.requests.post")
@patch("deepfake_recognition.api.app_streamlit.st.spinner")
def test_call_api_endpoint_success(mock_spinner, mock_post):
    """Test successful POST call to an API endpoint."""
    fake_response = MagicMock(status_code=200)
    mock_post.return_value = fake_response

    buffer = io.BytesIO(b"video data")
    result = app.call_api_endpoint("http://fake-url/detect", buffer, "video.mp4")

    assert result.status_code == 200
    mock_post.assert_called_once()


@patch("deepfake_recognition.api.app_streamlit.requests.post")
@patch("deepfake_recognition.api.app_streamlit.st")
def test_call_api_endpoint_request_exception(mock_st, mock_post):
    """Test failed API request returning None and displaying an error."""
    from requests.exceptions import RequestException
    mock_post.side_effect = RequestException("Timeout")

    buffer = io.BytesIO(b"video data")
    result = app.call_api_endpoint("http://fake-url", buffer, "video.mp4")

    assert result is None
    mock_st.error.assert_called_once()



@patch("deepfake_recognition.api.app_streamlit.cv2.VideoCapture")
@patch("deepfake_recognition.api.app_streamlit.st")
def test_check_video_duration_under_limit(mock_st, mock_cv):
    """Test that short videos pass the duration check."""
    cap = MagicMock()
    cap.get.side_effect = lambda x: {cv2.CAP_PROP_FPS: 30, cv2.CAP_PROP_FRAME_COUNT: 300}.get(x, 0)
    mock_cv.return_value = cap

    fake_file = MagicMock()
    fake_file.getbuffer.return_value = b"data"

    assert app.check_video_duration(fake_file, max_duration=20) is True
    mock_st.error.assert_not_called()


@patch("deepfake_recognition.api.app_streamlit.cv2.VideoCapture")
@patch("deepfake_recognition.api.app_streamlit.st")
def test_check_video_duration_too_long(mock_st, mock_cv):
    """Test that overly long videos trigger an error and return False."""
    cap = MagicMock()
    cap.get.side_effect = lambda x:{cv2.CAP_PROP_FPS:30, cv2.CAP_PROP_FRAME_COUNT: 90000}.get(x, 0)
    mock_cv.return_value = cap

    fake_file = MagicMock()
    fake_file.getbuffer.return_value = b"data"

    result = app.check_video_duration(fake_file, max_duration=10)
    assert result is False
    mock_st.error.assert_called_once()


@patch("deepfake_recognition.api.app_streamlit.st")
def test_display_detection_results_deepfake(mock_st):
    """Test rendering of detection results for a deepfake video."""
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "summary": "Deepfake detected",
        "deepfake_probability": 0.95,
        "is_deepfake": True,
        "frame_probabilities": [0.9, 0.95],
        "frames_used": 2,
    }

    app.display_detection_results(fake_response, threshold=0.5)
    mock_st.error.assert_called()
    mock_st.metric.assert_called_once()


@patch("deepfake_recognition.api.app_streamlit.st")
def test_display_detection_results_authentic(mock_st):
    """Test rendering of detection results for an authentic video."""
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "summary": "Authentic video",
        "deepfake_probability": 0.1,
        "is_deepfake": False,
        "frame_probabilities": [0.1, 0.05],
        "frames_used": 2,
    }

    app.display_detection_results(fake_response, threshold=0.5)
    mock_st.success.assert_called()
    mock_st.metric.assert_called_once()


@patch("deepfake_recognition.api.app_streamlit.st")
def test_display_detection_results_bad_json(mock_st):
    """Test that invalid JSON responses are handled gracefully."""
    fake_response = MagicMock()
    fake_response.json.side_effect = ValueError("Invalid JSON")

    app.display_detection_results(fake_response, threshold=0.5)
    mock_st.error.assert_called_with("Could not parse JSON response from detection endpoint.")


@patch("deepfake_recognition.api.app_streamlit.st")
@patch("deepfake_recognition.api.app_streamlit.call_api_endpoint")
def test_display_video_and_faces_success(mock_call, mock_st):
    """Test that successful /display_faces_from_video response renders HTML."""
    fake_file = MagicMock()
    fake_file.name = "video.mp4"
    fake_file.type = "video/mp4"

    mock_resp = MagicMock(status_code=200, text="<html>Faces</html>")
    mock_call.return_value = mock_resp

    app.display_video_and_faces(fake_file)
    mock_st.components.v1.html.assert_called_once()


@patch("deepfake_recognition.api.app_streamlit.st")
@patch("deepfake_recognition.api.app_streamlit.call_api_endpoint")
def test_display_video_and_faces_error(mock_call, mock_st):
    """Test error handling when /display_faces_from_video fails."""
    fake_file = MagicMock()
    fake_file.name = "video.mp4"
    fake_file.type = "video/mp4"

    mock_resp = MagicMock(status_code=500)
    mock_resp.json.return_value = {"detail": "Failed"}
    mock_call.return_value = mock_resp

    app.display_video_and_faces(fake_file)
    mock_st.error.assert_called_once()


@patch("deepfake_recognition.api.app_streamlit.st")
@patch("deepfake_recognition.api.app_streamlit.call_api_endpoint")
def test_download_boxes_and_faces_success(mock_call, mock_st):
    """Test that ZIP download is offered when API returns 200."""
    fake_file = MagicMock()
    fake_file.name = "video.mp4"
    buffer = io.BytesIO(b"data")

    mock_resp = MagicMock(status_code=200, content=b"zipcontent")
    mock_call.return_value = mock_resp

    app.download_boxes_and_faces(fake_file, buffer)
    mock_st.download_button.assert_called_once()


@patch("deepfake_recognition.api.app_streamlit.st")
@patch("deepfake_recognition.api.app_streamlit.call_api_endpoint")
def test_download_boxes_and_faces_error(mock_call, mock_st):
    """Test error message when download endpoint fails."""
    fake_file = MagicMock()
    fake_file.name = "video.mp4"
    buffer = io.BytesIO(b"data")

    mock_resp = MagicMock(status_code=500)
    mock_resp.json.return_value = {"detail": "error"}
    mock_call.return_value = mock_resp

    app.download_boxes_and_faces(fake_file, buffer)
    mock_st.error.assert_called_once()


@patch("deepfake_recognition.api.app_streamlit.st")
@patch("deepfake_recognition.api.app_streamlit.check_api_health")
def test_main_health_ok(mock_health, mock_st):
    """Minimal test for main() with healthy API response."""
    mock_health.return_value = {"status": "ok", "threshold": 0.5, "model_loaded": True}

    # Mock UI interactions
    mock_st.file_uploader.return_value = None  # Skip rest of logic

    app.main()

    mock_st.sidebar.success.assert_called_once()
    mock_st.title.assert_called_once()


@patch("deepfake_recognition.api.app_streamlit.st")
@patch("deepfake_recognition.api.app_streamlit.check_api_health")
def test_main_health_fail(mock_health, mock_st):
    """Minimal test for main() with failed API health response."""
    mock_health.return_value = {"status": "fail"}

    # Ensure uploader does nothing (prevents bytes-like error)
    mock_st.file_uploader.return_value = None

    app.main()

    mock_st.sidebar.error.assert_called_once()
    mock_st.stop.assert_called_once()