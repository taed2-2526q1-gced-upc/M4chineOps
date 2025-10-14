from fastapi import FastAPI, UploadFile, File
import tempfile
import os
import cv2
from mtcnn import MTCNN
from data_processing.data_frame_extraction import extract_face_frames_from_video
import data_processing.config as cfg
import random




app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Deepfake Recognition API"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/detect_deepfake")
async def detect_deepfake(video: UploadFile = File(...)):
    """
    Detects whether the uploaded video is a deepfake.
    """
    # TODO: Replace with your deepfake detection model
    fake_probability = random.uniform(0, 1)
    is_deepfake = fake_probability > 0.5

    return {
        "filename": video.filename,
        "deepfake_probability": round(fake_probability, 3),
        "is_deepfake": is_deepfake
    }
@app.post("/extract_faces_from_video")
async def extract_faces_from_video_api(video: UploadFile = File(...)):
    """
    Endpoint que usa la función extract_face_frames_from_video
    para extraer rostros de un video subido (o dataset).
    """
    import tempfile
    import os
    import cv2
    from mtcnn import MTCNN
    from data_processing.data_frame_extraction import extract_face_frames_from_video
    import data_processing.config as cfg

    # Guardar temporalmente el video subido
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await video.read())
        video_path = tmp.name

    # Información simulada (como si viniera del dataset)
    video_info = {
        "label": 0,
        "filepath": video_path,
        "frames": int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
    }

    detector = MTCNN()
    num_frames_to_extract = 5
    split = "api_uploads"

    os.makedirs(os.path.join(cfg.PROCESSED_DATA_DIR, f"{split}_data"), exist_ok=True)

    frame_paths = extract_face_frames_from_video(detector, video_info, split, num_frames_to_extract)

    os.remove(video_path)

    return {
        "filename": video.filename,
        "num_extracted_frames": len(frame_paths),
        "frames": frame_paths
    }
