import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys
from unittest.mock import MagicMock

# Preemptively mock cv2 before importing the module to avoid OpenCV dependency
sys.modules["cv2"] = MagicMock()

from deepfake_recognition.data_processing import data_sampling_and_metadata as ds


@pytest.fixture
def tmp_dirs(tmp_path):
    """
    Fixture: creates temporary directories simulating the real/fake datasets.
    """
    src_real = tmp_path / "real_videos"
    src_fake = tmp_path / "fake_videos"
    dst_dir = tmp_path / "output"
    metadata_dir = tmp_path / "metadata"

    for d in [src_real, src_fake, dst_dir, metadata_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Create dummy MP4 files to simulate real/fake videos
    for i in range(5):
        (src_real / f"real_{i}.mp4").write_text("dummy")
        (src_fake / f"fake_{i}.mp4").write_text("dummy")

    return src_real, src_fake, dst_dir, metadata_dir


def test_sample_split_creates_expected_structure(tmp_dirs, monkeypatch):
    """
    Test: sample_split() should generate the correct folder structure, CSV files, and metadata.
    """
    src_real, src_fake, dst_dir, metadata_dir = tmp_dirs

    # Mock cv2.VideoCapture to avoid opening real videos
    mock_cap = MagicMock()
    mock_cap.get.side_effect = lambda prop: {
        7: 10,     # CAP_PROP_FRAME_COUNT
        3: 1920,   # CAP_PROP_FRAME_WIDTH
        4: 1080,   # CAP_PROP_FRAME_HEIGHT
    }.get(prop, 0)
    mock_cap.isOpened.return_value = True

    # Patch both cv2.VideoCapture and shutil.copy
    with patch("cv2.VideoCapture", return_value=mock_cap), \
         patch("shutil.copy", side_effect=lambda src, dst: Path(dst).write_text("copied")):
        ds.sample_split(str(src_real), str(src_fake), str(dst_dir), str(metadata_dir), n_per_class=3)

    # Validate the output folder structure
    for split in ["train", "val", "test"]:
        for label in ["real", "fake"]:
            path = dst_dir / split / label
            assert path.exists(), f"Missing directory {path}"

    # Check that all metadata CSV files exist and contain expected columns
    for csv_file in ["train_fake.csv", "train_real.csv", 
                     "val_fake.csv", "val_real.csv",
                     "test_fake.csv", "test_real.csv"]:
        csv_path = metadata_dir / csv_file
        assert csv_path.exists(), f"Missing {csv_file}"
        df = pd.read_csv(csv_path)
        assert all(col in df.columns for col in ["filename", "label", "frames", "width", "height"])


def test_main_creates_metadata_dir_and_calls_sample_split(tmp_path):
    """
    Test: main() should create METADATA_DIR if it doesn’t exist and call sample_split().
    """
    meta_dir = tmp_path / "metadata"
    meta_dir_exists_before = meta_dir.exists()

    cfg_mock = MagicMock()
    cfg_mock.REAL_DATA_SUBDIR = tmp_path / "real"
    cfg_mock.FAKE_DATA_SUBDIR = tmp_path / "fake"
    cfg_mock.SAMPLED_OUTPUT_DIR = tmp_path / "output"
    cfg_mock.METADATA_DIR = meta_dir
    cfg_mock.N_SAMPLES_PER_CLASS = 2

    with patch("deepfake_recognition.data_processing.data_sampling_and_metadata.cfg", cfg_mock), \
         patch("deepfake_recognition.data_processing.data_sampling_and_metadata.sample_split") as mock_split:
        ds.main()
        mock_split.assert_called_once()

    # Ensure the metadata directory was created
    assert meta_dir.exists() or meta_dir_exists_before is True
