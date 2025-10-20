import os
import numpy as np
import pandas as pd
import pickle
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix

import deepfake_recognition.config as cfg


def split_Xy(df_in: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Split DataFrame into features and labels."""
    feature_cols = [c for c in df_in.columns if c != 'label']
    X = df_in[feature_cols].values.astype(np.float32)
    y = df_in['label'].map({'real': 0, 'fake': 1}).values.astype(int)
    return X, y


def tune_hyperparameters(X_train: pd.DataFrame, y_train: pd.Series):
    """Tune hyperparameters for Logistic Regression using GridSearchCV."""
    param_grid = [
        {
            'penalty': ['l2', None],
            'C': [0.01, 0.1, 1, 10],
            'solver': ['lbfgs', 'saga'],
            'max_iter': [2000]
        },
        {
            'penalty': ['elasticnet'],
            'C': [0.1, 1, 10],
            'solver': ['saga'],
            'l1_ratio': [0.3, 0.5, 0.7],
            'max_iter': [3000]
        }
    ]

    lr = LogisticRegression(random_state=42)
    grid_search = GridSearchCV(
        estimator=lr,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print("\n=== BEST PARAMETERS FOUND ===")
    print(grid_search.best_params_)
    print("Best ROC-AUC:", round(grid_search.best_score_, 4))

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def main():
    EMBEDDING_DIR = str(cfg.EMBEDDING_DIR)

    df_train = pd.read_csv(os.path.join(EMBEDDING_DIR, f'train_{cfg.EMBEDDING_AGGREGATION}_video_embeddings.csv')).sample(frac=1, random_state=42).reset_index(drop=True)
    df_val = pd.read_csv(os.path.join(EMBEDDING_DIR, f'val_{cfg.EMBEDDING_AGGREGATION}_video_embeddings.csv')).sample(frac=1, random_state=42).reset_index(drop=True)
    df_test  = pd.read_csv(os.path.join(EMBEDDING_DIR, f'test_{cfg.EMBEDDING_AGGREGATION}_video_embeddings.csv')).sample(frac=1, random_state=42).reset_index(drop=True)

    X_train, y_train = split_Xy(df_train)
    X_val,   y_val   = split_Xy(df_val)
    X_test,  y_test  = split_Xy(df_test)

    # scale features
    print('Scaling features...')
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # hyperparameter tuning
    print('Tuning hyperparameters with GridSearchCV...')
    best_lr, best_params, best_cv_score = tune_hyperparameters(X_train_scaled, y_train)

    # train final model
    best_lr.fit(X_train_scaled, y_train)
    y_pred = best_lr.predict(X_val_scaled)
    y_scores = best_lr.predict_proba(X_val_scaled)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    roc = roc_auc_score(y_val, y_scores)

    print('\n=== RESULTS FOR TRAIN-VAL ===')
    print('Accuracy:', round(acc, 4))
    print('F1-score:', round(f1, 4))
    print('ROC-AUC:', round(roc, 4))
    print('\nConfusion Matrix:\n', confusion_matrix(y_val, y_pred))
    print('\nClassification Report:\n', classification_report(y_val, y_pred, digits=3))

    # save model + scaler + metrics
    results = {
        'model': best_lr,
        'scaler': scaler,
        'best_params': best_params,
        'cv_roc_auc': round(best_cv_score, 4),
        'metrics_val': {
            'accuracy': round(acc, 4),
            'f1_score': round(f1, 4),
            'roc_auc': round(roc, 4),
        },
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    output_path = os.path.join(EMBEDDING_DIR, f'logreg_model_{datetime.now():%Y%m%d_%H%M%S}.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"\nModel and results stored in: {output_path}")


if __name__ == "__main__":
    main()
