"""Unit tests for the Data Download."""

import json
import sys
import time
from unittest.mock import patch, MagicMock
import pytest

from deepfake_recognition.data_processing import data_download


@pytest.fixture(scope="module")
def tmp_output_dir(tmp_path_factory):
    """
    Fixture: creates a temporary directory to simulate the download destination.
    """
    return tmp_path_factory.mktemp("data_download_test")


@pytest.fixture(scope="function")
def fake_args(tmp_output_dir):
    """
    Fixture: simulates command-line arguments for the downloader.
    """
    args = MagicMock()
    args.dataset = "original"
    args.type = "videos"
    args.compression = "raw"
    args.num_videos = 2
    args.output_path = str(tmp_output_dir)
    args.server = "EU"
    args.base_url = "http://example.com/v3/"
    args.tos_url = "http://example.com/tos.pdf"
    args.deepfakes_model_url = (
        "http://example.com/v3/manipulated_sequences/Deepfakes/models/"
    )
    return args


# ---------------------------------------------------------------------------
# BASIC UNIT TESTS
# ---------------------------------------------------------------------------

def test_safe_urlretrieve_success(tmp_output_dir):
    """
    Test: safe_urlretrieve() completes successfully on the first attempt.
    """
    fake_file = tmp_output_dir / "file.txt"

    with patch("urllib.request.urlretrieve") as mock_retrieve:
        mock_retrieve.return_value = None
        data_download.safe_urlretrieve("http://example.com/file", str(fake_file))
        mock_retrieve.assert_called_once()


def test_safe_urlretrieve_fails_after_retries(tmp_output_dir):
    """
    Test: safe_urlretrieve() raises an exception after all retries fail.
    """
    with patch("urllib.request.urlretrieve", side_effect=Exception("Fail")):
        with pytest.raises(Exception, match="Failed to download"):
            data_download.safe_urlretrieve(
                "http://example.com/file",
                str(tmp_output_dir / "file.txt"),
                retries=2,
                delay=0
            )


def test_reporthook_progress_output(capsys):
    """
    Test: reporthook() prints progress messages correctly.
    """
    data_download.reporthook(0, 1024, 2048)  # initialize start time
    time.sleep(0.001)
    data_download.reporthook(1, 1024, 2048)  # simulate progress
    out, _ = capsys.readouterr()
    assert "Progress:" in out


def test_parse_args_defaults(monkeypatch):
    """
    Test: parse_args() sets correct defaults for EU server.
    """
    monkeypatch.setattr(sys, "argv", ["data_download.py", "/tmp/output"])
    args = data_download.parse_args()
    assert args.output_path == "/tmp/output"
    assert args.server == "EU"
    assert args.base_url.startswith("http://canis.vc.in.tum.de")
    assert args.tos_url.endswith("FaceForensics_TOS.pdf")


def test_parse_args_different_servers(monkeypatch):
    """
    Test: parse_args() constructs correct base URLs for different servers.
    """
    for server, expected_url in [
        ("EU2", "http://kaldir.vc.in.tum.de/faceforensics/"),
        ("CA", "http://falas.cmpt.sfu.ca:8100/"),
    ]:
        argv = ["data_download.py", "/tmp/output", "--server", server]
        monkeypatch.setattr(sys, "argv", argv)
        args = data_download.parse_args()
        assert args.base_url.startswith(expected_url)

    # Invalid server → argparse should trigger SystemExit
    argv = ["data_download.py", "/tmp/output", "--server", "XX"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        data_download.parse_args()


def test_safe_urlretrieve_retries_on_failure(tmp_output_dir):
    """
    Test: safe_urlretrieve() retries multiple times before failing.
    """
    with patch("urllib.request.urlretrieve", side_effect=Exception("Fail")):
        with pytest.raises(Exception, match="Failed to download"):
            data_download.safe_urlretrieve(
                "http://example.com/file",
                str(tmp_output_dir / "f.txt"),
                retries=3,
                delay=0,
            )


# ---------------------------------------------------------------------------
# MOCKED DOWNLOAD TESTS
# ---------------------------------------------------------------------------

def test_download_file_creates_file(tmp_output_dir):
    """
    Test: download_file() creates a temporary file and renames it successfully.
    """
    fake_url = "http://example.com/testfile"
    out_file = tmp_output_dir / "downloaded.txt"

    with patch("urllib.request.urlretrieve") as mock_retrieve:
        data_download.download_file(fake_url, str(out_file))
        assert out_file.exists()
        mock_retrieve.assert_called_once()


def test_download_file_skips_existing_file(tmp_output_dir):
    """
    Test: download_file() skips downloading if the file already exists.
    """
    existing_file = tmp_output_dir / "exists.txt"
    existing_file.write_text("already here")

    with patch("urllib.request.urlretrieve") as mock_retrieve, \
         patch("deepfake_recognition.data_processing.data_download.tqdm.write") as mock_tqdm_write:
        data_download.download_file("http://example.com/whatever", str(existing_file))
        mock_retrieve.assert_not_called()
        mock_tqdm_write.assert_called_once()


def test_download_files_calls_download_file(tmp_output_dir):
    """
    Test: download_files() internally calls download_file() for each listed file.
    """
    files = ["file1.txt", "file2.txt"]
    with patch(
        "deepfake_recognition.data_processing.data_download.download_file"
    ) as mock_dl:
        data_download.download_files(files, "http://example.com/", \
        str(tmp_output_dir), report_progress=False)
        assert mock_dl.call_count == len(files)


# ---------------------------------------------------------------------------
# MAIN FUNCTION BEHAVIOR
# ---------------------------------------------------------------------------

def test_main_exits_if_data_exists(fake_args, tmp_output_dir):
    """
    Test: main() should exit early if the dataset already exists.
    """
    with patch("deepfake_recognition.config.GLOBAL_DATA_DIR", tmp_output_dir), \
         patch("os.path.exists", return_value=True), \
         patch("sys.exit") as mock_exit:
        data_download.main(fake_args)
        mock_exit.assert_called_once_with(0)


def test_main_downloads_when_no_data(monkeypatch, fake_args, tmp_output_dir):
    """
    Test: main() continues when data does not exist and triggers download_files().
    """
    monkeypatch.setattr("builtins.input", lambda _: "y")

    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps([["a", "b"], ["c", "d"]]).encode("utf-8")

    with patch("deepfake_recognition.config.GLOBAL_DATA_DIR", tmp_output_dir / "no_data"), \
         patch("os.path.exists", return_value=False), \
         patch("urllib.request.urlopen", return_value=mock_response), \
         patch("deepfake_recognition.data_processing.data_download.download_files") as mock_dl:
        data_download.main(fake_args)
        mock_dl.assert_called()


def test_main_masks_and_models_paths(monkeypatch, fake_args, tmp_output_dir):
    """
    Test: main() covers the branches for 'masks' and 'models' download types.
    """
    monkeypatch.setattr("builtins.input", lambda _: "y")
    fake_args.dataset = "Deepfakes"
    fake_args.type = "masks"

    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps([["a", "b"], ["c", "d"]]).encode("utf-8")

    # Case 1: Masks branch (skip FaceShifter warning)
    with patch("deepfake_recognition.config.GLOBAL_DATA_DIR", tmp_output_dir / "no_masks"), \
         patch("os.path.exists", return_value=False), \
         patch("urllib.request.urlopen", return_value=mock_response), \
         patch("deepfake_recognition.data_processing.data_download.download_files") as mock_dl:
        data_download.main(fake_args)
        mock_dl.assert_called()

    # Case 2: Models branch (Deepfakes only)
    fake_args.type = "models"
    with patch("deepfake_recognition.config.GLOBAL_DATA_DIR", tmp_output_dir / "no_models"), \
         patch("os.path.exists", return_value=False), \
         patch("urllib.request.urlopen", return_value=mock_response), \
         patch("deepfake_recognition.data_processing.data_download.download_files") as mock_dl:
        data_download.main(fake_args)
        mock_dl.assert_called()


def test_main_models_invalid_dataset(monkeypatch, fake_args, tmp_output_dir):
    """
    Test: main() aborts when 'models' are requested for a non-Deepfakes dataset.
    """
    fake_args.dataset = "Face2Face"
    fake_args.type = "models"
    monkeypatch.setattr("builtins.input", lambda _: "y")

    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps([["a", "b"]]).encode("utf-8")

    with patch("deepfake_recognition.config.GLOBAL_DATA_DIR", tmp_output_dir / "no_models2"), \
         patch("os.path.exists", return_value=False), \
         patch("urllib.request.urlopen", return_value=mock_response), \
         patch("deepfake_recognition.data_processing.data_download.download_files") as mock_dl:
        data_download.main(fake_args)
        mock_dl.assert_not_called()  # abort expected


def test_main_masks_faceshifter(monkeypatch, fake_args, tmp_output_dir):
    """
    Test: main() aborts when attempting to download masks for FaceShifter.
    """
    fake_args.dataset = "FaceShifter"
    fake_args.type = "masks"
    monkeypatch.setattr("builtins.input", lambda _: "y")

    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps([["a", "b"]]).encode("utf-8")

    with patch("deepfake_recognition.config.GLOBAL_DATA_DIR", tmp_output_dir / "no_masks2"), \
         patch("os.path.exists", return_value=False), \
         patch("urllib.request.urlopen", return_value=mock_response), \
         patch("deepfake_recognition.data_processing.data_download.download_files") as mock_dl:
        data_download.main(fake_args)
        mock_dl.assert_not_called()  # abort expected


def test_main_original_youtube_videos(monkeypatch, fake_args, tmp_output_dir):
    """
    Test: main() downloads the special 'original_youtube_videos' dataset.
    """
    fake_args.dataset = "original_youtube_videos"
    monkeypatch.setattr("builtins.input", lambda _: "y")

    with patch("deepfake_recognition.config.GLOBAL_DATA_DIR", tmp_output_dir / "no_youtube"), \
         patch("os.path.exists", return_value=False), \
         patch("deepfake_recognition.data_processing.data_download.download_file") as mock_dl:
        data_download.main(fake_args)
        mock_dl.assert_called_once()
