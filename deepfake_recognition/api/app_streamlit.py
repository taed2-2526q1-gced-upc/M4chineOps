import streamlit as st
import requests
import io
import os
import tempfile
import cv2


# 1. Configuration
API_BASE_URL = os.getenv('API_BASE_URL', 'http://nattech.fib.upc.edu:40410')
DETECT_ENDPOINT = f'{API_BASE_URL}/detect_deepfake'
DISPLAY_FACES_ENDPOINT  = f'{API_BASE_URL}/display_faces_from_video'
DOWNLOAD_FACES_ENDPOINT  = f'{API_BASE_URL}/download_faces_from_video'
HEALTH_ENDPOINT = f'{API_BASE_URL}/health'


# 2. Helper functions
def check_api_health(url: str):
    """
    Checks the health of the FastAPI backend.

    Args:
        url (str): Health endpoint URL.
    """

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Could not connect to the API: {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error checking API health: {e}")
        st.stop()


def call_api_endpoint(url: str, file_buffer: io.BytesIO, filename: str) -> requests.Response | None:
    """
    Generic function to call a POST endpoint with a file.

    Args:
        url (str): API endpoint URL.
        file_buffer (io.BytesIO): In-memory file buffer.
        filename (str): Name of the file being sent.

    Returns:
        requests.Response: Response from the API. None if there was an error.
    """

    files = {'video': (filename, file_buffer, 'video/mp4')}
    try:
        with st.spinner(f"Sending video to API for {url.split('/')[-1]}..."):
            response = requests.post(url, files=files, timeout=300)
        return response
    except requests.exceptions.RequestException as e:
        st.error(f'⚠️ API request failed: {e}')
        return None


def check_video_duration(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile, max_duration: int = 60) -> bool:
    """
    Checks the duration of the uploaded video in seconds.

    Args:
        uploaded_file: Uploaded video file.
        max_duration (int): Maximum allowed duration in seconds.

    Returns:
        bool: True if duration is within limit, False otherwise.
    """

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps and fps > 0 else 0
    cap.release()
    os.remove(tmp_path)

    if duration > max_duration:
        st.error(f"❌ The video is too long ({duration:.1f} seconds). Please upload a video under {max_duration} seconds.")
        return False

    return True


def display_video_and_faces(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile):
    """Displays the uploaded video and extracted faces (title below video)."""

    st.markdown("---")
    st.header("Uploaded Video and Extracted Faces")

    # show the uploaded video first
    st.subheader("Uploaded Video")
    st.video(uploaded_file, format=uploaded_file.type)

    # show extracted faces below 
    st.subheader("Extracted Faces")

    faces_response = call_api_endpoint(DISPLAY_FACES_ENDPOINT, uploaded_file, uploaded_file.name)

    if faces_response and faces_response.status_code == 200:
        st.components.v1.html(
            faces_response.text,
            height=600,
            scrolling=True
        )
    elif faces_response is not None:
        st.error(f"❌ Error extracting faces (Status: {faces_response.status_code}): {faces_response.json().get('detail', 'Unknown error')}")


def download_boxes_and_faces(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile, file_buffer: io.BytesIO):
    """Creates a download link for a ZIP with faces and boxed frames."""

    st.subheader("Download the Extracted Faces and Boxed Frames")

    vid_name = uploaded_file.name
    vid_name_no_ext = os.path.splitext(vid_name)[0]
    download_response = call_api_endpoint(DOWNLOAD_FACES_ENDPOINT, file_buffer, vid_name)

    if download_response and download_response.status_code == 200:
        st.download_button(
            label='⬇️ Download ZIP file',
            data=download_response.content,
            file_name=f'{vid_name_no_ext}_boxes_and_faces.zip',
            mime='application/zip'
        )
    elif download_response is not None:
        st.error(f"❌ Error generating the downloadable file (Status: {download_response.status_code}): {download_response.json().get('detail', 'Unknown error')}")


def display_detection_results(detection_response: requests.Response, threshold: float):
    """Displays the deepfake detection results."""

    st.markdown("---")
    st.header("Deepfake Detection Results")

    try:
        data = detection_response.json()
    except Exception:
        st.error("Could not parse JSON response from detection endpoint.")
        return

    summary = data.get("summary", "N/A")
    prob = data.get("deepfake_probability", 0.0)
    is_deepfake = data.get("is_deepfake", False)

    if is_deepfake:
        st.error(f"### {summary}")
        st.markdown("**This video is likely a Deepfake!**")
    else:
        st.success(f"### {summary}")
        st.markdown("**This video appears to be Authentic.**")

    st.metric(
        label="Deepfake Probability (P)",
        value=f"{prob * 100:.2f}%",
        delta_color="off"
    )
    st.markdown(f"*Classification Threshold: P ≥ **{threshold}***")

    with st.expander("Show Detailed Frame Probabilities"):
        frame_probs = data.get("frame_probabilities", [])
        if frame_probs:
            st.write(f"Frames used for analysis: **{data.get('frames_used', 0)}**")
            st.line_chart({"Deepfake Probability": frame_probs})
            st.caption("Probability of deepfake per extracted frame.")
        else:
            st.info("No frame probabilities available.")


# 3. Main Streamlit App
def main():
    st.set_page_config(
        page_title="Deepfake Recognition",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    st.title("🎬 Deepfake Recognition System")
    st.markdown("Upload a video to analyze if it is a deepfake using a CNN + Logistic Regression model.")

    # 1. Check API health
    health_data = check_api_health(HEALTH_ENDPOINT)
    deepfake_threshold = health_data.get("threshold", 0.5)

    if health_data.get("status") == "ok":
        st.sidebar.success("✅ API is connected and healthy.")
        st.sidebar.markdown(f"Model loaded: **{health_data.get('model_loaded')}**")
        st.sidebar.markdown(f"Deepfake Threshold: **{deepfake_threshold}**")
    else:
        st.sidebar.error("❌ API health check failed.")
        st.sidebar.json(health_data)
        st.stop()

    # 2. File uploader
    uploaded_file = st.file_uploader(
        "Upload a video file (MP4 only, under 60 seconds)",
        type=["mp4"],
        key="video_uploader"
    )

    if uploaded_file is not None:
        if not check_video_duration(uploaded_file, max_duration=60):
            st.stop()

        file_buffer = io.BytesIO(uploaded_file.getvalue())

        if st.button("🚀 Start Deepfake Analysis", type="primary"):
            st.session_state['analysis_started'] = True

            # 3. Display video + extracted faces
            display_video_and_faces(uploaded_file)

            # 4. Download boxes and faces
            file_buffer.seek(0)
            download_boxes_and_faces(uploaded_file, file_buffer)

            # 5. Deepfake detection
            file_buffer.seek(0)
            detection_response = call_api_endpoint(DETECT_ENDPOINT, file_buffer, uploaded_file.name)

            if detection_response and detection_response.status_code == 200:
                display_detection_results(detection_response, deepfake_threshold)
            elif detection_response is not None:
                error_detail = detection_response.json().get('detail', 'Unknown error during detection.')
                st.error(f"❌ Deepfake detection failed (Status: {detection_response.status_code}): {error_detail}")


if __name__ == "__main__":
    main()
