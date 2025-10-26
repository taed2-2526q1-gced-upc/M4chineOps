from fastapi import FastAPI, UploadFile, File, HTTPException
import io, os, cv2, traceback, joblib
import zipfile, tempfile
import numpy as np
import base64

from mtcnn import MTCNN

from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model

from fastapi.responses import HTMLResponse, StreamingResponse

from deepfake_recognition.data_processing.data_preprocessing_inference import extract_and_save_face_paths, extract_face_frames
from deepfake_recognition.data_processing.embedding_creation import preprocess_for_Xception, build_frame_embeddings
import deepfake_recognition.config as cfg

# 1. Configuration 
FRAMES_PER_VIDEO = cfg.FRAMES_PER_VIDEO
DEEPFAKE_THRESHOLD = cfg.DEEPFAKE_THRESHOLD
IMG_SIZE = cfg.SIZE_FOR_XCEPTION

app = FastAPI(
    title='🎬 Deepfake Recognition API',
    description='Deepfake detection API using pretrained CNN embeddings + Logistic Regression (.pkl model).',
    version='1.0',
    contact={'name': 'M4chineOps Team'},
)

# 2. Model loading 
MODEL_PATH = str(cfg.MODEL_PATH)
print(f'Looking for model at: {MODEL_PATH}...')

try:
    obj = joblib.load(MODEL_PATH)
    if isinstance(obj, dict) and 'model' in obj:
        model = obj['model']
    else:
        model = obj
    model_error = None
    print(f'Model loaded successfully from: {MODEL_PATH}!')
except Exception as e:
    model = None
    model_error = f"Could not load model from '{MODEL_PATH}': {e}"
    print(f'{model_error}')


# 3. Model Initialization
# Kera's Xception model for embeddings
base = Xception(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
emb_model = Model(inputs=base.input, outputs=GlobalAveragePooling2D()(base.output))  
emb_dim = emb_model.output_shape[-1] # output dim: 2048
print(f'Xception embedding model loaded. Embedding size: {emb_dim}')

# face detector
detector = MTCNN()


# 4. Api endpoints
@app.get('/')
def read_root():
    return {'message': 'Welcome to the Deepfake Recognition API (CNN + Logistic Regression [.pkl] model)'}


@app.get('/health')
def health_check():
    return {
        'status': 'ok' if model_error is None else 'error',
        'model_loaded': model_error is None,
        'model_path': MODEL_PATH,
        'threshold': DEEPFAKE_THRESHOLD,
        'feature_dim': 2048,
        'error': model_error,
    }


@app.post('/detect_deepfake')
async def detect_deepfake(video: UploadFile = File(...)):
    """Upload a video and detect whether it's a deepfake using CNN features + trained model."""

    if model is None:
        raise HTTPException(status_code=503, detail=f'Model not loaded: {model_error}')

    # validate file format
    allowed_types = ['video/mp4', 'video/mpeg', 'video/quicktime']
    if video.content_type not in allowed_types or not video.filename.lower().endswith('.mp4'):
        raise HTTPException(
            status_code=400,
            detail='Invalid file format. Only .mp4 videos are supported.',
        )
    
    try:
        # save video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(await video.read())
            tmp.flush()
            video_path = tmp.name
        
        # check if video can be opened
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail='Could not open video file.')

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps and fps > 0 else 0

        if duration_sec > 60: # max video duration = 60 seconds
            cap.release()
            raise HTTPException(
                status_code=400,
                detail=f'The uploaded video too long ({duration_sec:.1f}s). Maximum allowed duration is 60 seconds.',
            )

        cap.release()
        video_name = os.path.splitext(os.path.basename(video_path))[0] # without the .mp4 extension

        print(f'\n Processing video: {video_name}.mp4...')
        frames = extract_face_frames(video_path, detector, IMG_SIZE, k = cfg.FRAMES_PER_VIDEO)
        x = preprocess_for_Xception(frames, IMG_SIZE)

        X = build_frame_embeddings(emb_model, x) 

        # predict per frame
        if hasattr(model, 'predict_proba'):
            frame_probs = model.predict_proba(X)[:, 1].tolist()
        elif hasattr(model, 'decision_function'):
            scores = model.decision_function(X)
            frame_probs = (1 / (1 + np.exp(-scores))).tolist()
        else:
            preds = model.predict(X).astype(float).tolist()
            frame_probs = preds

        deepfake_prob = float(np.mean(frame_probs))
        is_deepfake = deepfake_prob >= DEEPFAKE_THRESHOLD

        return {
            'filename': video.filename,
            'duration_sec': round(duration_sec, 2),
            'frames_used': len(frame_probs),
            'deepfake_probability': round(deepfake_prob, 4),
            'is_deepfake': is_deepfake,
            'threshold': DEEPFAKE_THRESHOLD,
            'frame_probabilities': [round(p, 4) for p in frame_probs],
            'summary': '😱 Deepfake detected' if is_deepfake else '✅ Authentic video',
        }

    except HTTPException:
        raise  # re-raise cleanly formatted FastAPI HTTP errors
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f'Inference error: {e}')


@app.post('/display_faces_from_video', response_class=HTMLResponse)
async def display_faces_from_video_api(video: UploadFile = File(...)):
    """Shows faces and bouding boxes extracted directly as an HTML object."""

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(await video.read())
        tmp.flush()
        video_path = tmp.name
        video_name = os.path.splitext(os.path.basename(video.filename))[0]

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError('The video could not be opened.')
        cap.release()

        API_UPLOADS_DIR = str(cfg.API_UPLOADS_DATA_DIR)
        if not os.path.exists(API_UPLOADS_DIR):
            os.makedirs(API_UPLOADS_DIR, exist_ok=True)

        boxed_paths, face_paths = extract_and_save_face_paths(
            detector=detector,
            filepath=video_path,
            filename=video_name,
            k=FRAMES_PER_VIDEO,
        )

        # converting images to Base64 to display them in HTML
        html_box_imgs = ''
        for p in boxed_paths:
            if os.path.exists(p):
                with open(p, 'rb') as f:
                    encoded = base64.b64encode(f.read()).decode('utf-8')
                html_box_imgs += f'<img src="data:image/jpeg;base64,{encoded}" style="width:200px; margin:5px; border-radius:8px;">'
        
        html_face_imgs = ''
        for p in face_paths:
            if os.path.exists(p):
                with open(p, 'rb') as f:
                    encoded = base64.b64encode(f.read()).decode('utf-8')
                html_face_imgs += f'<img src="data:image/jpeg;base64,{encoded}" style="width:200px; margin:5px; border-radius:8px;">'

        html_content = f"""
        <html>
        <head>
            <title>Faces extracted from {video.filename}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    text-align: center;
                    background-color: #f6f6f6;
                }}
                h1 {{
                    color: #333;
                }}
                .container {{
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: center;
                    background-color: white;
                    padding: 10px;
                    border-radius: 12px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                    margin-bottom: 20px;
                }}
            </style>
        </head>
        <body>
            <h1>🧑‍🦰 Extracted Faces From Video: {video.filename}</h1>
            <p>Total: {len(face_paths)} imatges</p>
            <div class="container">{html_face_imgs}</div>

            <h1>🟩 Extracted Bounding Boxes Video: {video.filename}</h1>
            <p>Total: {len(boxed_paths)} imatges</p>
            <div class="container">{html_box_imgs}</div>
        </body>
        </html>
        """

        return HTMLResponse(content=html_content, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error extracting faces: {e}')


@app.post('/download_faces_from_video', response_class=HTMLResponse)
async def download_faces_from_video_api(video: UploadFile = File(...)):
    """Retrieves faces and bounding boxes into a downloadable .zip file."""

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(await video.read())
        tmp.flush()
        video_path = tmp.name
        video_name = os.path.splitext(os.path.basename(video.filename))[0]

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError('The video could not be opened.')
        cap.release()

        API_UPLOADS_DIR = str(cfg.API_UPLOADS_DATA_DIR)
        if not os.path.exists(API_UPLOADS_DIR):
            os.makedirs(API_UPLOADS_DIR, exist_ok=True)

        boxed_paths, face_paths = extract_and_save_face_paths(
            detector=detector,
            filepath=video_path,
            filename=video_name,
            k=FRAMES_PER_VIDEO,
        )

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zipf:
            for p in boxed_paths + face_paths: 
                with open(p, 'rb') as f:
                    zipf.writestr(os.path.basename(p), f.read())
                os.remove(p)
        zip_buffer.seek(0)

        # return as downloadable file
        return StreamingResponse(
            zip_buffer,
            media_type='application/x-zip-compressed',
            headers={'Content-Disposition': f'attachment; filename={video_name}_boxes_and_faces.zip'}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error extracting faces: {e}')