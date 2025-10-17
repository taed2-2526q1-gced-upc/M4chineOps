# -*- coding: utf-8 -*-
"""Entrenamiento del modelo de detección de deepfakes con CodeCarbon y MLflow"""

import os
import glob
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix

from codecarbon import EmissionsTracker
import mlflow

# Config reproducible
np.random.seed(42)
tf.random.set_seed(42)

# === Paths y parámetros ===
VIDEO_ROOT = "data/sampled_videos"
SPLITS = ["train", "test"]
CLASSES = ["real", "fake"]
label_map = {"real": 1, "fake": 0}

FRAMES_PER_VIDEO = 10
AGGREGATION = "mean"
IMG_SIZE = (299, 299)
BATCH_SIZE = 32

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_OUT = os.path.join(OUTPUT_DIR, "video_embeddings.csv")
MODEL_OUT = os.path.join(OUTPUT_DIR, "model.joblib")

# === Funciones auxiliares ===
def sample_frame_indices(num_frames, k):
    if num_frames <= 0:
        return []
    return [int(round(i * (num_frames - 1) / max(1, k - 1))) for i in range(k)] if k > 1 else [0]

def read_video_frames_uniform(path, k=10):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = sample_frame_indices(total, k)
    frames = []
    for target in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames

def preprocess_frames(frames, img_size=(299, 299)):
    arr = []
    for f in frames:
        f = cv2.resize(f, img_size, interpolation=cv2.INTER_AREA)
        arr.append(f.astype(np.float32))
    x = np.stack(arr, axis=0)
    return preprocess_input(x)

def embed_frames(frames_np, model, batch_size=32):
    embs = []
    for i in range(0, frames_np.shape[0], batch_size):
        batch = frames_np[i:i + batch_size]
        e = model.predict(batch, verbose=0)
        embs.append(e)
    return np.vstack(embs)

def aggregate_video_embeddings(frame_embs, how="mean"):
    if frame_embs.size == 0:
        return None
    return frame_embs.sum(axis=0) if how == "sum" else frame_embs.mean(axis=0)

def split_Xy(df_in, emb_dim):
    feats = [f"e{j}" for j in range(emb_dim)]
    X = df_in[feats].values
    y = df_in["label"].values.astype(int)
    return X, y

# === Entrenamiento ===
def main():
    # CodeCarbon tracker
    tracker = EmissionsTracker(output_dir="emissions")
    tracker.start()

    mlflow.start_run()

    print("Cargando modelo base Xception...")
    base = Xception(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    emb_model = Model(inputs=base.input, outputs=GlobalAveragePooling2D()(base.output))
    emb_dim = emb_model.output_shape[-1]

    rows = []
    for split in SPLITS:
        for cls in CLASSES:
            folder = os.path.join(VIDEO_ROOT, split, cls)
            if not os.path.isdir(folder):
                print(f"[WARN] No existe: {folder}")
                continue
            videos = sorted(glob.glob(os.path.join(folder, "*.mp4")))
            print(f"{split}/{cls}: {len(videos)} vídeos")
            for vp in tqdm(videos, desc=f"{split}/{cls}"):
                frames = read_video_frames_uniform(vp, k=FRAMES_PER_VIDEO)
                if len(frames) == 0:
                    continue
                x = preprocess_frames(frames, IMG_SIZE)
                frame_embs = embed_frames(x, emb_model, BATCH_SIZE)
                vid_emb = aggregate_video_embeddings(frame_embs, AGGREGATION)
                if vid_emb is None:
                    continue
                row = {"split": split, "video_path": vp, "label": label_map[cls]}
                for j in range(emb_dim):
                    row[f"e{j}"] = vid_emb[j]
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(CSV_OUT, index=False)
    print(f"Embeddings guardados en {CSV_OUT}")

    df_train = df[df["split"] == "train"].copy()
    df_test = df[df["split"] == "test"].copy()

    X_train, y_train = split_Xy(df_train, emb_dim)
    X_test, y_test = split_Xy(df_test, emb_dim)

    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(max_iter=5000, solver="lbfgs")
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_scores = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_scores)

    print("\n=== RESULTADOS ===")
    print("Accuracy:", round(acc, 4))
    print("F1:", round(f1, 4))
    print("ROC-AUC:", round(roc, 4))

    # Log metrics in MLflow
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc)

    # Stop CodeCarbon
    emissions = tracker.stop()
    mlflow.log_metric("carbon_emissions_kg", emissions)

    mlflow.end_run()

    # Save model
    import joblib
    joblib.dump(clf, MODEL_OUT)
    print(f"Modelo guardado en {MODEL_OUT}")

if __name__ == "__main__":
    main()
