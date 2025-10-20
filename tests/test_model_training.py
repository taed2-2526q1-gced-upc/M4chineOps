import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.linear_model import LogisticRegression

from deepfake_recognition.data_processing import model_training as mt


# --------------------------------------------------------------------
# Test: split_Xy
# --------------------------------------------------------------------
def test_split_Xy_returns_correct_shapes():
    """split_Xy should return X, y with correct types and shapes."""
    df = pd.DataFrame({
        'e0': [0.1, 0.2, 0.3],
        'e1': [1.0, 1.1, 1.2],
        'label': ['real', 'fake', 'real']
    })

    X, y = mt.split_Xy(df)

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape == (3, 2)
    assert set(y) == {0, 1}


# --------------------------------------------------------------------
# Test: tune_hyperparameters
# --------------------------------------------------------------------
def test_tune_hyperparameters_returns_best_model(monkeypatch):
    """tune_hyperparameters should return estimator, params, and score."""

    # Fake training data
    X_train = np.random.rand(10, 3)
    y_train = np.random.randint(0, 2, size=10)

    # Mock GridSearchCV to avoid actual training
    mock_gs = MagicMock()
    mock_gs.best_estimator_ = LogisticRegression()
    mock_gs.best_params_ = {'C': 1, 'penalty': 'l2'}
    mock_gs.best_score_ = 0.95
    mock_gs.fit.return_value = None

    with patch("deepfake_recognition.data_processing.model_training.GridSearchCV", return_value=mock_gs):
        est, params, score = mt.tune_hyperparameters(X_train, y_train)

    assert isinstance(est, LogisticRegression)
    assert params == {'C': 1, 'penalty': 'l2'}
    assert score == 0.95
    mock_gs.fit.assert_called_once()


# --------------------------------------------------------------------
# Optional: Mock main() high-level behavior
# --------------------------------------------------------------------
def test_main_reads_data_and_calls_tune(monkeypatch, tmp_path):
    """Main function should read CSVs and call tune_hyperparameters once."""

    fake_df = pd.DataFrame({
        'e0': np.random.rand(5),
        'e1': np.random.rand(5),
        'label': np.random.choice(['real', 'fake'], 5)
    })

    # Mock read_csv to return fake data
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: fake_df)

    # Mock configuration values
    monkeypatch.setattr(mt.cfg, "EMBEDDING_DIR", tmp_path)
    monkeypatch.setattr(mt.cfg, "EMBEDDING_AGGREGATION", "mean")

    # Mock tune_hyperparameters
    mock_tune = MagicMock(return_value=(LogisticRegression(), {'C': 1}, 0.9))
    monkeypatch.setattr(mt, "tune_hyperparameters", mock_tune)

    # Mock pickle.dump to avoid writing files
    with patch("pickle.dump") as mock_dump:
        mt.main()

    mock_tune.assert_called_once()
    mock_dump.assert_called_once()
