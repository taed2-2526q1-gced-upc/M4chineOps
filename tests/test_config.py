from pathlib import Path
import sys
import pytest


@pytest.fixture
def reload_config_module():
    """Reloads config.py fresh for each test to ensure no cached values."""
    if "deepfake_recognition.config" in sys.modules:
        del sys.modules["deepfake_recognition.config"]
    import deepfake_recognition.config as config
    return config


def test_proj_root_is_valid(reload_config_module):
    """Ensures that PROJ_ROOT points to a valid directory."""
    config = reload_config_module
    assert isinstance(config.PROJ_ROOT, Path)
    assert config.PROJ_ROOT.exists()  # should exist or be creatable


def test_data_directories_defined(reload_config_module):
    """Checks that all key data directories are defined as Path objects."""
    config = reload_config_module
    expected_dirs = [
        config.GLOBAL_DATA_DIR,
        config.RAW_DATA_DIR,
        config.PROCESSED_DATA_DIR,
        config.SAMPLED_OUTPUT_DIR,
        config.METADATA_DIR,
        config.EMBEDDING_DIR,
        config.MODELS_DIR,
        config.EMISSIONS_OUTPUT_DIR,
        config.API_UPLOADS_DIR
    ]
    for d in expected_dirs:
        assert isinstance(d, Path), f"{d} is not a Path object"


def test_constants_have_expected_types(reload_config_module):
    """Verifies that main constants have the correct data types and valid ranges."""
    cfg = reload_config_module
    assert isinstance(cfg.N_SAMPLES_PER_CLASS, int)
    assert isinstance(cfg.SIZE_FOR_XCEPTION, tuple)
    assert isinstance(cfg.FRAMES_PER_VIDEO, int)
    assert isinstance(cfg.EMBEDDING_AGGREGATION, str)
    assert isinstance(cfg.DEEPFAKE_THRESHOLD, float)
    assert 0 <= cfg.DEEPFAKE_THRESHOLD <= 1, "Threshold must be between 0 and 1"


def test_logger_initialization_does_not_fail(reload_config_module):
    """Ensures that the Loguru logger is properly initialized and functional."""
    cfg = reload_config_module
    cfg.logger.info("Test message from config")
    assert hasattr(cfg.logger, "info")
    assert callable(cfg.logger.info)


def test_tqdm_integration_graceful(monkeypatch):
    """Simulates a missing tqdm module and verifies config still imports cleanly."""
    sys_modules_backup = sys.modules.copy()
    sys.modules["tqdm"] = None

    try:
        if "deepfake_recognition.config" in sys.modules:
            del sys.modules["deepfake_recognition.config"]
        import deepfake_recognition.config as config
        assert True  # should import successfully
    finally:
        sys.modules = sys_modules_backup
