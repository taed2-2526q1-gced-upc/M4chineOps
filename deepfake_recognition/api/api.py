from fastapi import FastAPI, UploadFile, File, HTTPException
import os, tempfile, cv2, numpy as np, traceback, joblib
from typing import List
from mtcnn import MTCNN
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
from codecarbon import EmissionsTracker

from deepfake_recognition.data_processing.data_frame_extraction import extract_face_frames_from_video
import deepfake_recognition.config as cfg

app = FastAPI(
    title="🎬 Deepfake Recognition API",
    description="Deepfake detection API using pretrained CNN embeddings + Logistic Regression (.pkl model).",
    version="2.0",
    contact={"name": "M4chineOps Team"},
)

# -------- Model loading --------
MODEL_PATH = str(cfg.MODEL_PATH)
PROCESSED_DATA_DIR = str(cfg.PROCESSED_DATA_DIR)

print(f"🔍 Looking for model at: {MODEL_PATH}")

try:
    obj = joblib.load(MODEL_PATH)
    if isinstance(obj, dict) and "model" in obj:
        model = obj["model"]
    else:
        model = obj
    model_error = None
    print(f"✅ Model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    model = None
    model_error = f"Could not load model from '{MODEL_PATH}': {e}"
    print(f"❌ {model_error}")

# -------- Feature extractor (CNN) --------
print("📦 Loading CNN feature extractor (Xception)...")
feature_extractor = Xception(weights="imagenet", include_top=False, pooling="avg")
print("✅ CNN feature extractor ready (2048-dim output)")

# -------- Face detector --------
detector = MTCNN()

# -------- Utilities --------
def _read_images(paths: List[str]) -> List[np.ndarray]:
    imgs = []
    for p in paths:
        im = cv2.imread(p)
        if im is None:
            continue
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        imgs.append(im)
    return imgs


def _extract_cnn_features(imgs: List[np.ndarray]) -> np.ndarray:
    """Extract 2048-D CNN embeddings from frames using Xception."""
    if len(imgs) == 0:
        return np.empty((0, 2048), dtype=np.float32)

    processed = []
    for im in imgs:
        im = cv2.resize(im, (224, 224))
        processed.append(im)
    X = np.stack(processed).astype(np.float32)
    X = preprocess_input(X)
    feats = feature_extractor.predict(X, verbose=0)
    return feats


# -------- API Routes --------
@app.get("/")
def read_root():
    return {"message": "Welcome to the Deepfake Recognition API (CNN + .pkl model)"}


@app.get("/health")
def health_check():
    return {
        "status": "ok" if model_error is None else "error",
        "model_loaded": model_error is None,
        "model_path": MODEL_PATH,
        "threshold": cfg.DEEPFAKE_THRESHOLD,
        "feature_dim": 2048,
        "error": model_error,
    }


@app.post("/detect_deepfake")
async def detect_deepfake(video: UploadFile = File(...)):
    """Upload a video and detect whether it’s a deepfake using CNN features + trained model."""
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {model_error}")

    try:
        EMISSIONS_OUTPUT_DIR = str(cfg.EMISSIONS_OUTPUT_DIR)

        # CodeCarbon tracker
        tracker = EmissionsTracker(output_dir = EMISSIONS_OUTPUT_DIR, project_name='deepfake_recognition_model_inference')
        tracker.start()

        # Save video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(await video.read())
            video_path = tmp.name

        # Video info
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video.")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        video_info = {"label": 0, "filepath": video_path, "frames": total_frames}
        split = "api_uploads"
        os.makedirs(os.path.join(PROCESSED_DATA_DIR, f"{split}_data"), exist_ok=True)

        # Extract faces
        frame_paths = extract_face_frames_from_video(
            detector=detector,
            video_info=video_info,
            split=split,
            num_frames_to_extract=cfg.FRAMES_PER_VIDEO,
        )
        imgs_rgb = _read_images(frame_paths)

        # Fallback: if no faces found
        if len(imgs_rgb) == 0:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video (fallback).")
            N = min(cfg.FRAMES_PER_VIDEO, max(total_frames, 1))
            idxs = np.linspace(0, total_frames - 1, num=N, dtype=int)
            for i in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                ok, frame = cap.read()
                if ok and frame is not None:
                    imgs_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()

        if len(imgs_rgb) == 0:
            raise ValueError("No frames could be extracted from the video.")

        # Extract CNN features
        X = _extract_cnn_features(imgs_rgb)

        # Predict per frame
        if hasattr(model, "predict_proba"):
            frame_probs = model.predict_proba(X)[:, 1].tolist()
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X)
            frame_probs = (1 / (1 + np.exp(-scores))).tolist()
        else:
            preds = model.predict(X).astype(float).tolist()
            frame_probs = preds

        # Aggregate
        deepfake_prob = float(np.mean(frame_probs))
        is_deepfake = deepfake_prob >= cfg.DEEPFAKE_THRESHOLD

        tracker.stop()

        return {
            "filename": video.filename,
            "frames_used": len(frame_probs),
            "deepfake_probability": round(deepfake_prob, 4),
            "is_deepfake": is_deepfake,
            "threshold": cfg.DEEPFAKE_THRESHOLD,
            "frame_probabilities": [round(p, 4) for p in frame_probs],
            "summary": "😱 Deepfake detected" if is_deepfake else "✅ Authentic video",
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    finally:
        if "video_path" in locals() and os.path.exists(video_path):
            os.remove(video_path)


@app.post("/extract_faces_from_video")
async def extract_faces_from_video_api(video: UploadFile = File(...)):
    """Debug endpoint to view extracted faces."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await video.read())
        video_path = tmp.name

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video.")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        video_info = {"label": 0, "filepath": video_path, "frames": total_frames}
        split = "api_uploads"
        os.makedirs(os.path.join(PROCESSED_DATA_DIR, f"{split}_data"), exist_ok=True)

        frame_paths = extract_face_frames_from_video(
            detector=detector,
            video_info=video_info,
            split=split,
            num_frames_to_extract=cfg.FRAMES_PER_VIDEO,
        )

        return {
            "filename": video.filename,
            "num_extracted_frames": len(frame_paths),
            "frames": frame_paths,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting faces: {e}")

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)
