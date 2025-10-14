from fastapi import FastAPI, UploadFile, File
from typing import List
import random
import face_recognition
import tempfile
import os

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

@app.post("/detect_faces")
async def detect_faces(image: UploadFile = File(...)):
    """
    Detect faces in an uploaded image using the face_recognition model.
    Returns bounding boxes (top, right, bottom, left) for each detected face.
    """
    # Guardar temporalmente la imagen subida
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(await image.read())
        tmp_path = tmp.name

    # Cargar imagen y detectar rostros
    image_np = face_recognition.load_image_file(tmp_path)
    face_locations = face_recognition.face_locations(image_np)

    # Eliminar archivo temporal
    os.remove(tmp_path)

    # Formatear los resultados
    faces = [
        {"top": t, "right": r, "bottom": b, "left": l}
        for (t, r, b, l) in face_locations
    ]

    return {
        "filename": image.filename,
        "num_faces": len(faces),
        "faces": faces
    }