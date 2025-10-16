import os
import shutil
import random
import runpy
import pytest

@pytest.fixture
def fake_video_structure(tmp_path):
    """
    Create a temporary directory structure that mimics the DeepFake raw dataset.
    Includes both real and fake videos in their respective subfolders.
    """
    raw_dir = tmp_path / "data" / "raw"
    out_dir = tmp_path / "data" / "ffpp"

    # Create directories for real and fake videos
    real_dir = raw_dir / "original_sequences" / "youtube" / "c40" / "videos"
    fake_base = raw_dir / "manipulated_sequences"
    methods = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]

    real_dir.mkdir(parents=True)
    for m in methods:
        (fake_base / m / "c40" / "videos").mkdir(parents=True)

    # Create dummy real and fake .mp4 files
    for i in range(10):
        (real_dir / f"real_{i}.mp4").write_text("real")
    for m in methods:
        for j in range(4):
            (fake_base / m / "c40" / "videos" / f"{m.lower()}_{j}.mp4").write_text("fake")

    return raw_dir, out_dir


def test_dataset_preparation_script_with_ratio(fake_video_structure, tmp_path):
    """
    Test that:
    1. The output folder structure is correctly created.
    2. The script copies files according to the 80/20 split ratio.
    3. All videos are included and properly copied.
    """

    raw_dir, out_dir = fake_video_structure

    # Generate a temporary version of the dataset preparation script using the fake paths
    script_path = tmp_path / "prepare_dataset.py"
    script_code = f"""
import os, glob, random, shutil
RAW_DIR = r"{raw_dir}"
OUT_DIR = r"{out_dir}"
TRAIN_SPLIT = 0.8
SEED = 42
random.seed(SEED)
# Create output directories
for split in ["train", "val"]:
    for label in ["real", "fake"]:
        os.makedirs(os.path.join(OUT_DIR, split, label), exist_ok=True)
# Gather real and fake videos
real_videos = glob.glob(os.path.join(RAW_DIR, "original_sequences/youtube/c40/videos/*.mp4"))
fake_videos = []
methods = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
for m in methods:
    fake_videos.extend(glob.glob(os.path.join(RAW_DIR, f"manipulated_sequences/{{m}}/c40/videos/*.mp4")))
# Split and copy helper
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
"""

    script_path.write_text(script_code)

    # Execute the generated script
    runpy.run_path(str(script_path))

    # 1️. Check that all output folders exist
    for split in ["train", "val"]:
        for label in ["real", "fake"]:
            path = out_dir / split / label
            assert path.exists(), f"Missing directory: {path}"

    # 2️. Check total number of copied files
    real_train = list((out_dir / "train" / "real").glob("*.mp4"))
    real_val = list((out_dir / "val" / "real").glob("*.mp4"))
    fake_train = list((out_dir / "train" / "fake").glob("*.mp4"))
    fake_val = list((out_dir / "val" / "fake").glob("*.mp4"))

    # Validate all files were copied
    assert len(real_train) + len(real_val) == 10
    assert len(fake_train) + len(fake_val) == 16  # 4 methods × 4 files each

    # 3️. Verify 80/20 split (rounded)
    def approx_ratio(train_count, val_count):
        total = train_count + val_count
        if total == 0:
            return 0
        return round(train_count / total, 1)

    real_ratio = approx_ratio(len(real_train), len(real_val))
    fake_ratio = approx_ratio(len(fake_train), len(fake_val))

    assert real_ratio == 0.8, f"Expected 0.8, got {real_ratio}"
    assert fake_ratio == 0.8, f"Expected 0.8, got {fake_ratio}"

    # 4️. Ensure files are correctly copied (content check)
    for f in real_train + real_val:
        assert f.read_text() == "real"
    for f in fake_train + fake_val:
        assert f.read_text() == "fake"
