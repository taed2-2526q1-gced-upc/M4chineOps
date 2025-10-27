import io
import sys
import pytest
import numpy as np 

from types import ModuleType
from unittest.mock import MagicMock, patch
from typing import cast

from fastapi.testclient import TestClient

# --- MOCK de TensorFlow y submódulos usados por el proyecto ---
mock_tf = ModuleType("tensorflow")
mock_keras = ModuleType("tensorflow.keras")
mock_keras_applications = ModuleType("tensorflow.keras.applications")
mock_keras_xception = ModuleType("tensorflow.keras.applications.xception")
mock_keras_layers = ModuleType("tensorflow.keras.layers")
mock_keras_models = ModuleType("tensorflow.keras.models")

# Añadir mocks de clases/funciones que el código podría importar
setattr(mock_keras_applications, "Xception", MagicMock())
setattr(mock_keras_xception, "preprocess_input", MagicMock())

# Añadir mocks de capas comunes
for layer_name in [
    "Dense",
    "Dropout",
    "GlobalAveragePooling2D",
    "Input",
    "Conv2D",
    "BatchNormalization",
]:
    setattr(mock_keras_layers, layer_name, MagicMock())

# Mock para modelos Keras
setattr(mock_keras_models, "Model", MagicMock())

# Registrar los módulos en sys.modules
sys.modules["tensorflow"] = mock_tf
sys.modules["tensorflow.keras"] = mock_keras
sys.modules["tensorflow.keras.applications"] = mock_keras_applications
sys.modules["tensorflow.keras.applications.xception"] = mock_keras_xception
sys.modules["tensorflow.keras.layers"] = mock_keras_layers
sys.modules["tensorflow.keras.models"] = mock_keras_models

# --- MOCK de MTCNN ---
mock_mtcnn = ModuleType("mtcnn")
setattr(mock_mtcnn, "MTCNN", MagicMock())
sys.modules["mtcnn"] = cast(ModuleType, mock_mtcnn)

# --- Now safe to import your API ---
import deepfake_recognition.api.api as api


# ------------------------------
#  FIXTURES
# ------------------------------

@pytest.fixture
def client():
    """Crea un cliente de pruebas de FastAPI."""
    return TestClient(api.app)


@pytest.fixture
def mock_model():
    """Crea un modelo simulado que devuelve probabilidades predecibles."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.2, 0.8], [0.3, 0.7]])
    return model


# ------------------------------
#  BASIC ENDPOINT TESTS
# ------------------------------

def test_read_root(client):
    """Tests que el endpoint raíz devuelve el mensaje esperado."""
    response = client.get('/')
    assert response.status_code == 200
    data = response.json()
    assert 'message' in data
    assert 'Deepfake Recognition API' in data['message']


def test_health_check_ok(client):
    """Tests que /health devuelve el estado del modelo."""
    response = client.get('/health')
    data = response.json()
    assert response.status_code == 200
    assert 'status' in data
    assert 'model_path' in data
    assert 'feature_dim' in data


# ------------------------------
#  MOCKED MODEL TESTS
# ------------------------------

@patch("deepfake_recognition.api.api.model", new_callable=lambda: MagicMock())
@patch("deepfake_recognition.api.api.extract_face_frames")
@patch("deepfake_recognition.api.api.preprocess_for_Xception")
@patch("deepfake_recognition.api.api.build_frame_embeddings")
def test_detect_deepfake_success(
    mock_build_emb, mock_preprocess, mock_extract, mock_model, client
):
    """Tests de detección exitosa de deepfake con datos simulados."""
    # Mocks de pasos intermedios
    mock_extract.return_value = [np.ones((64, 64, 3), dtype=np.uint8)]
    mock_preprocess.return_value = np.random.rand(1, 64, 64, 3)
    mock_build_emb.return_value = np.random.rand(2, 2048)
    mock_model.predict_proba.return_value = np.array([[0.1, 0.9], [0.2, 0.8]])

    # Crear archivo de video simulado
    dummy_video = io.BytesIO(b"fake video data")
    files = {"video": ("test.mp4", dummy_video, "video/mp4")}

    with patch("cv2.VideoCapture") as mock_cap:
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda x: {5: 10, 7: 30}.get(x, 10)
        mock_cap.return_value = cap

        response = client.post(
            "/detect_deepfake",
            files=files,
            headers={"content-type": "multipart/form-data"}  # 👈 fix MIME handling
        )

    assert response.status_code == 200
    data = response.json()
    assert "deepfake_probability" in data
    assert "is_deepfake" in data
    assert isinstance(data["deepfake_probability"], float)


def test_detect_deepfake_invalid_format(client):
    """Tests rechazo de formatos de archivo no soportados."""
    bad_file = io.BytesIO(b"not a video")
    files = {"video": ("test.txt", bad_file, "text/plain")}
    response = client.post("/detect_deepfake", files=files)
    assert response.status_code == 400
    assert "Invalid file format" in response.text


@patch("deepfake_recognition.api.api.model", None)
def test_detect_deepfake_model_not_loaded(client):
    """Tests que el API devuelve 503 cuando el modelo no está cargado."""
    fake_video = io.BytesIO(b"fake data")
    files = {"video": ("video.mp4", fake_video, "video/mp4")}
    response = client.post("/detect_deepfake", files=files)
    assert response.status_code == 503
    assert "Model not loaded" in response.text


# ------------------------------
#  EXTRACT FACES TESTS
# ------------------------------

@patch("deepfake_recognition.api.api.extract_and_save_face_paths")
@patch("cv2.VideoCapture")
def test_extract_faces_from_video_api(mock_cap, mock_extract, client, tmp_path):
    """Tests del endpoint /extract_faces_from_video con mocks."""
    cap = MagicMock()
    cap.isOpened.return_value = True
    mock_cap.return_value = cap

    # Crear imágenes simuladas
    temp_file1 = tmp_path / "face1.jpg"
    temp_file2 = tmp_path / "box1.jpg"
    temp_file1.write_text("data")
    temp_file2.write_text("data")

    # 👇 corregimos el orden de retorno (boxes primero, faces después)
    mock_extract.return_value = ([str(temp_file2)], [str(temp_file1)])

    video_file = io.BytesIO(b"fake video data")
    files = {"video": ("video.mp4", video_file, "video/mp4")}

    response = client.post(
        "/extract_faces_from_video",
        files=files,
        headers={"content-type": "multipart/form-data"}  # opcional, consistente
    )
    assert response.status_code == 200
    assert response.headers["Content-Disposition"].endswith(".zip")



@patch("cv2.VideoCapture")
def test_extract_faces_from_video_api_failure(mock_cap, client):
    """Tests manejo de errores cuando no se puede abrir el video."""
    cap = MagicMock()
    cap.isOpened.return_value = False
    mock_cap.return_value = cap

    video_file = io.BytesIO(b"fake video data")
    files = {"video": ("video.mp4", video_file, "video/mp4")}

    response = client.post("/extract_faces_from_video", files=files)
    assert response.status_code == 500
    assert "Error extracting faces" in response.text
