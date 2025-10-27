import pandas as pd
import pytest

import deepfake_recognition.ge_validate as ge_validate


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create a fake data directory structure with sample files."""
    data_root = tmp_path / "data"
    raw_real = data_root / "raw" / "real"
    raw_fake = data_root / "raw" / "fake"
    meta_dir = data_root / "metadata"
    raw_real.mkdir(parents=True)
    raw_fake.mkdir(parents=True)
    meta_dir.mkdir(parents=True)

    # Create dummy mp4 files
    for i in range(2):
        (raw_real / f"video_real_{i}.mp4").write_bytes(b"abc")
        (raw_fake / f"video_fake_{i}.mp4").write_bytes(b"abc")

    # Create a metadata CSV
    df = pd.DataFrame({
        "filename": ["video_real_0.mp4", "video_fake_0.mp4"],
        "split": ["train", "val"],
        "label": ["real", "fake"]
    })
    df.to_csv(meta_dir / "metadata.csv", index=False)

    return data_root


def test_df_raw_reads_files_correctly(tmp_data_dir):
    """Checks that df_raw builds a correct DataFrame from video files."""
    df = ge_validate.df_raw(tmp_data_dir / "raw")
    assert not df.empty
    assert set(df["label"].unique()) == {"real", "fake"}
    assert all(df["ext"].isin(["mp4"]))
    assert all(col in df.columns for col in ["filepath", "filename", "label", "size_bytes"])


def test_df_meta_reads_and_merges_csvs(tmp_data_dir):
    """Checks that df_meta merges CSV metadata files correctly."""
    df = ge_validate.df_meta(tmp_data_dir / "metadata")
    assert not df.empty
    assert set(df.columns) == {"filename", "split", "label"}
    assert {"train", "val"}.issubset(set(df["split"]))


def test_exps_raw_and_meta_are_valid():
    """Ensures the expectation definitions return lists of dicts with proper structure."""
    for func in [ge_validate.exps_raw, ge_validate.exps_meta]:
        exps = func()
        assert isinstance(exps, list)
        assert all("t" in e for e in exps)
        assert all(isinstance(e["t"], str) for e in exps)


def test_run_ge_calls_expectations(monkeypatch, tmp_path):
    """Mocks PandasDataset to verify expectations are executed."""
    called_expectations = []

    class MockGE:
        def __init__(self, df):
            self.df = df

        def validate(self):
            return {"success": True}

        def __getattr__(self, item):
            def _mock_fn(**kwargs):
                called_expectations.append((item, kwargs))
                return True
            return _mock_fn

    monkeypatch.setattr(ge_validate, "PandasDataset", MockGE)
    df = pd.DataFrame({"filepath": ["a"], "filename": ["a.mp4"], "label": ["real"], "ext": ["mp4"], "size_bytes": [100]})
    exps = [{"t": "expect_column_to_exist", "k": {"column": "filepath"}}]

    result = ge_validate.run_ge(df, exps, "test_result", tmp_path)
    assert result["success"]
    assert called_expectations[0][0] == "expect_column_to_exist"
    assert (tmp_path / "test_result.json").exists()


def test_validate_raw_passes_with_mock(monkeypatch, tmp_data_dir):
    """Tests validate_raw succeeds when mock Great Expectations always passes."""
    def mock_run_ge(df, exps, name, outdir):
        return {"success": True}

    monkeypatch.setattr(ge_validate, "run_ge", mock_run_ge)
    ge_validate.validate_raw(tmp_data_dir, min_per_class=1)


def test_validate_raw_fails_with_low_samples(monkeypatch, tmp_data_dir):
    """Tests that validate_raw exits if there are too few samples per class."""
    def mock_die(msg):
        raise SystemExit(msg)

    monkeypatch.setattr(ge_validate, "die", mock_die)

    # Delete fake class to simulate insufficient data
    fake_dir = tmp_data_dir / "raw" / "fake"
    for f in fake_dir.glob("*"):
        f.unlink()

    with pytest.raises(SystemExit):
        ge_validate.validate_raw(tmp_data_dir, min_per_class=2)


def test_validate_metadata_passes_with_mock(monkeypatch, tmp_data_dir):
    """Tests validate_metadata passes successfully when GE passes."""
    def mock_run_ge(df, exps, name, outdir):
        return {"success": True}

    monkeypatch.setattr(ge_validate, "run_ge", mock_run_ge)
    ge_validate.validate_metadata(tmp_data_dir)


def test_validate_metadata_detects_leakage(monkeypatch, tmp_data_dir):
    """Ensures validate_metadata dies if a filename appears in multiple splits."""
    def mock_run_ge(df, exps, name, outdir):
        return {"success": True}

    def mock_die(msg):
        raise SystemExit(msg)

    monkeypatch.setattr(ge_validate, "run_ge", mock_run_ge)
    monkeypatch.setattr(ge_validate, "die", mock_die)

    # Duplicate filename with different split
    df = pd.DataFrame({
        "filename": ["dup.mp4", "dup.mp4"],
        "split": ["train", "val"],
        "label": ["real", "real"]
    })
    meta_dir = tmp_data_dir / "metadata"
    df.to_csv(meta_dir / "leak.csv", index=False)

    with pytest.raises(SystemExit, match="Leakage"):
        ge_validate.validate_metadata(tmp_data_dir)
