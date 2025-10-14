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
        # Save when joining clip_len frames
        if len(frames) == clip_len:
            clip_name = os.path.join(
                out_dir,
                f"{os.path.splitext(os.path.basename(video_path))[0]}_{idx}.npy"
            )
            np.save(clip_name, np.array(frames))
            frames = []
            idx += 1
    cap.release()


def process_folder(in_dir, out_dir, clip_len=16):
    """Convert all videos in a folder into .npy clips"""
    if not os.path.exists(in_dir):
        print(f"⚠️ Skipping missing folder: {in_dir}")
        return
    os.makedirs(out_dir, exist_ok=True)
    videos = [f for f in os.listdir(in_dir) if f.endswith(".mp4")]
    if not videos:
        print(f"⚠️ No .mp4 files found in {in_dir}")
        return
    for fname in videos:
        video_to_clips(os.path.join(in_dir, fname), out_dir, clip_len)


# Process the complete dataset using your current structure
splits = ["train", "val", "test"]
labels = ["real", "fake"]

for split in splits:
    for label in labels:
        in_dir = f"data/ffpp/{split}/{label}"              # ✅ your actual dataset folders
        out_dir = f"data/ffpp_clips/{split}/{label}"    # ✅ output location for clips
        process_folder(in_dir, out_dir)
