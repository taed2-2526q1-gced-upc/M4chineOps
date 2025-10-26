from pathlib import Path
import sys
import pytest


@pytest.fixture
def reload_config_module():
    """Reloads config.py cleanly for each test."""
    if "deepfake_recognition.config" in sys.modules:
        del sys.modules["deepfake_recognition.config"]
    import deepfake_recognition.config as config
    return config


def test_proj_root_is_valid(reload_config_module):
    """Checks that PROJ_ROOT is a valid directory path."""
    config = reload_config_module
    assert isinstance(config.PROJ_ROOT, Path)
    # The project root should exist or be creatable
    assert config.PROJ_ROOT.exists()


def test_data_directories_defined(reload_config_module):
    """Ensures all expected data directories are Path objects."""
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
    """Validates that key constants have correct data types."""
    cfg = reload_config_module
    assert isinstance(cfg.N_SAMPLES_PER_CLASS, int)
    assert isinstance(cfg.SIZE_FOR_XCEPTION, tuple)
    assert isinstance(cfg.FRAMES_PER_VIDEO, int)
    assert isinstance(cfg.EMBEDDING_AGGREGATION, str)
    assert isinstance(cfg.DEEPFAKE_THRESHOLD, float)
    assert 0 <= cfg.DEEPFAKE_THRESHOLD <= 1, "Threshold must be between 0 and 1"


def test_logger_initialization_does_not_fail(reload_config_module):
    """Ensures the Loguru logger is correctly set up."""
    cfg = reload_config_module
    cfg.logger.info("Test message from config")
    # No exceptions should occur — just verifying it logs successfully
    assert hasattr(cfg.logger, "info")
    assert callable(cfg.logger.info)


def test_tqdm_integration_graceful(monkeypatch):
    """Ensures config handles missing tqdm gracefully."""
    # Simulate tqdm not being installed
    sys_modules_backup = sys.modules.copy()
    sys.modules["tqdm"] = None

    try:
        if "deepfake_recognition.config" in sys.modules:
            del sys.modules["deepfake_recognition.config"]
        import deepfake_recognition.config as config
        assert True  # Should import successfully
    finally:
        sys.modules = sys_modules_backup
