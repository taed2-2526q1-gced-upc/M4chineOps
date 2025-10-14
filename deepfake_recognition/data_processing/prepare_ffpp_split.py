import os
import glob
import random
import shutil

# Input/output paths
RAW_DIR = "data/raw"
OUT_DIR = "data/ffpp"

TRAIN_SPLIT = 0.8  # 80% train, 20% val
SEED = 42
random.seed(SEED)

# 1. Create destination folders
for split in ["train", "val"]:
    for label in ["real", "fake"]:
        os.makedirs(os.path.join(OUT_DIR, split, label), exist_ok=True)

# 2. Obtain real videos
real_videos = glob.glob(os.path.join(RAW_DIR, "original_sequences/youtube/c40/videos/*.mp4"))

# 3. Obtain fake videos (from all methods)
fake_videos = []
methods = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
for m in methods:
    fake_videos.extend(
        glob.glob(os.path.join(RAW_DIR, f"manipulated_sequences/{m}/c40/videos/*.mp4"))
    )

print(f"Reales: {len(real_videos)}, Falsos: {len(fake_videos)}")

# 4. Split train/val
def split_and_copy(videos, label):
    random.shuffle(videos)
    n_train = int(len(videos) * TRAIN_SPLIT)
    train_v = videos[:n_train]
    val_v = videos[n_train:]

    for v in train_v:
        shutil.copy(v, os.path.join(OUT_DIR, "train", label))
    for v in val_v:
        shutil.copy(v, os.path.join(OUT_DIR, "val", label))

split_and_copy(real_videos, "real")
split_and_copy(fake_videos, "fake")

print("Prepared dataset in", OUT_DIR)
