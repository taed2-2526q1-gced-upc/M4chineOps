import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix

import deepfake_recognition.config as cfg


def split_Xy(df_in: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Split DataFrame into features and labels.

    Args:
        df_in (pd.DataFrame): DataFrame with embeddings and labels.
    """

    feature_cols = [c for c in df_in.columns if c != 'label']
    X = df_in[feature_cols].values.astype(np.float32)
    y = df_in['label'].map({'real': 0, 'fake': 1}).values.astype(int)
    
    return X, y


def tune_hyperparameters(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Tune hyperparameters for Logistic Regression using GridSearchCV.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
    """

    parameters = {
        'penalty': ['l2', None],
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['lbfgs', 'newton-cholesky', 'saga'],
        'max_iter': [1000, 2000, 5000],
    }

    lr = LogisticRegression(random_state = 42)

    grid_search = GridSearchCV(estimator = lr, param_grid = parameters, cv = 5, scoring = 'roc_auc')
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_


def main():
    EMBEDDING_DIR = str(cfg.EMBEDDING_DIR)  

    # shuffle data (right now it's sorted by label)
    df_train = pd.read_csv(os.path.join(EMBEDDING_DIR, f'train_{cfg.EMBEDDING_AGGREGATION}_video_embeddings.csv')).sample(frac=1, random_state=42).reset_index(drop=True)
    df_val = pd.read_csv(os.path.join(EMBEDDING_DIR, f'val_{cfg.EMBEDDING_AGGREGATION}_video_embeddings.csv')).sample(frac=1, random_state=42).reset_index(drop=True)
    df_test  = pd.read_csv(os.path.join(EMBEDDING_DIR, f'test_{cfg.EMBEDDING_AGGREGATION}_video_embeddings.csv')).sample(frac=1, random_state=42).reset_index(drop=True)

    X_train, y_train = split_Xy(df_train)
    X_val,   y_val   = split_Xy(df_val)
    X_test,  y_test  = split_Xy(df_test)

    # scale features 
    print('Scaling features...')
    scaler = StandardScaler(with_mean = True, with_std = True)
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train) # recommended for LR
    X_val_scaled = scaler.transform(X_val)
    
    # hyperparameter tuning with train set + prediction with best_estimator_
    print('Tuning hyperparameters with GridSearchCV...')
    # best_lr = tune_hyperparameters(X_train_scaled, y_train)
    best_lr = LogisticRegression(random_state = 42)

    best_lr.fit(X_train_scaled, y_train)
    y_pred = best_lr.predict(X_val_scaled)
    y_scores = best_lr.predict_proba(X_val_scaled)[:, 1]

    acc  = accuracy_score(y_val, y_pred)
    f1   = f1_score(y_val, y_pred)
    roc = roc_auc_score(y_val, y_scores)

    print('\n=== RESULTS FOR TRAIN-VAL ===')
    print('Accuracy:', round(acc, 4))
    print('F1-score:', round(f1, 4))
    print('ROC-AUC:', round(roc, 4))
    print('\nConfusion Matrix:\n', confusion_matrix(y_val, y_pred))
    print('\nClassification Report:\n', classification_report(y_val, y_pred, digits=3))


if __name__ == "__main__":
    main()