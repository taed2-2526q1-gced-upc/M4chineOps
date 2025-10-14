import os
import cv2
import numpy as np

def npy_to_mp4(npy_path, mp4_path, fps=30):
    frames = np.load(npy_path)
    h, w = frames.shape[1:3]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(mp4_path, fourcc, fps, (w, h))
    for f in frames:
        # Convertir RGB → BGR porque OpenCV usa BGR
        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    out.release()

base_dir = "data"  # carpeta donde tienes train/val/test
output_base = "videomae_data"  # carpeta destino para los .mp4

for split in ["train", "val", "test"]:
    for label in ["real", "fake"]:
        npy_dir = os.path.join(base_dir, split, label)
        mp4_dir = os.path.join(output_base, split, label)
        if not os.path.exists(npy_dir):
            print(f"⚠️ No existe: {npy_dir}, saltando...")
            continue
        os.makedirs(mp4_dir, exist_ok=True)

        npy_files = [f for f in os.listdir(npy_dir) if f.endswith(".npy")]
        for f in npy_files:
            npy_path = os.path.join(npy_dir, f)
            mp4_path = os.path.join(mp4_dir, f.replace(".npy", ".mp4"))
            npy_to_mp4(npy_path, mp4_path)
        print(f"✅ Convertidos {len(npy_files)} clips en {mp4_dir}")
