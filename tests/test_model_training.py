"""Unit tests for Model Training."""

from unittest.mock import patch, MagicMock
import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

import deepfake_recognition.modeling.model_training as mt


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


def test_tune_hyperparameters_returns_best_model(monkeypatch):
    """tune_hyperparameters should return estimator, params, and score."""

    X_train = np.random.rand(10, 3)
    y_train = np.random.randint(0, 2, size=10)

    mock_gs = MagicMock()
    mock_gs.best_estimator_ = LogisticRegression()
    mock_gs.best_params_ = {'C': 1, 'penalty': 'l2'}
    mock_gs.best_score_ = 0.95
    mock_gs.fit.return_value = None

    with patch("deepfake_recognition.modeling.model_training.GridSearchCV", return_value=mock_gs):
        est, params, score = mt.tune_hyperparameters(X_train, y_train, scoring='roc_auc')

    assert isinstance(est, LogisticRegression)
    assert params == {'C': 1, 'penalty': 'l2'}
    assert score == 0.95
    mock_gs.fit.assert_called_once()


def test_main_reads_data_and_runs_pipeline(monkeypatch, tmp_path):
    """Main should read CSVs, call tune_hyperparameters once, and save model."""

    fake_df = pd.DataFrame({
        'e0': [0.1, 0.2, 0.3],
        'e1': [1.0, 1.1, 1.2],
        'label': ['real', 'fake', 'real']
    })

    # Mock pd.read_csv to return fake_df for all embedding files
    def fake_read_csv(path, *args, **kwargs):
        return fake_df
    monkeypatch.setattr(pd, "read_csv", fake_read_csv)

    # Mock config paths
    monkeypatch.setattr(mt.cfg, "EMBEDDING_DIR", tmp_path)
    monkeypatch.setattr(mt.cfg, "EMBEDDING_AGGREGATION", "mean")
    monkeypatch.setattr(mt.cfg, "EMISSIONS_OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(mt.cfg, "MODEL_PATH", tmp_path / "model.pkl")

    # Mock EmissionsTracker
    mock_tracker = MagicMock()
    mock_tracker.start.return_value = None
    mock_tracker.stop.return_value = 0.0
    monkeypatch.setattr(mt, "EmissionsTracker", MagicMock(return_value=mock_tracker))

    # Mock MLflow functions
    monkeypatch.setattr(mt.mlflow, "set_experiment", MagicMock())
    monkeypatch.setattr(mt.mlflow, "start_run", MagicMock())
    monkeypatch.setattr(mt.mlflow.sklearn, "log_model", MagicMock())
    monkeypatch.setattr(mt.mlflow, "log_params", MagicMock())
    monkeypatch.setattr(mt.mlflow, "log_metric", MagicMock())
    monkeypatch.setattr(mt.mlflow, "log_artifact", MagicMock())

    # Mock tune_hyperparameters
    mock_tune = MagicMock(return_value=(LogisticRegression(), {'C': 1}, 0.9))
    monkeypatch.setattr(mt, "tune_hyperparameters", mock_tune)

    # Mock pickle.dump
    with patch("pickle.dump") as mock_dump:
        mt.main()

    # Verify calls
    mock_tune.assert_called_once()
    mock_dump.assert_called_once()
    mock_tracker.start.assert_called_once()
    mock_tracker.stop.assert_called_once()
