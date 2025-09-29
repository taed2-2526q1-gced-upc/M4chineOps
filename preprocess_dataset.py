import os
import cv2
import numpy as np

def video_to_clips(video_path, out_dir, clip_len=16, resize=(224,224)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resize)
        frames.append(frame)
        # Cuando juntamos clip_len frames → guardamos
        if len(frames) == clip_len:
            clip_name = os.path.join(
                out_dir,
                f"{os.path.basename(video_path).split('.')[0]}_{idx}.npy"
            )
            np.save(clip_name, np.array(frames))
            frames = []
            idx += 1
    cap.release()

def process_folder(in_dir, out_dir, clip_len=16):
    os.makedirs(out_dir, exist_ok=True)
    for fname in os.listdir(in_dir):
        if fname.endswith(".mp4"):
            video_to_clips(os.path.join(in_dir, fname), out_dir, clip_len)

# Procesar dataset completo
splits = ["train", "val", "test"]  # ✅ añadimos test
labels = ["real", "fake"]

for split in splits:
    for label in labels:
        in_dir = f"data/ffpp/{split}/{label}"
        out_dir = f"data/ffpp_clips/{split}/{label}"
        process_folder(in_dir, out_dir)
