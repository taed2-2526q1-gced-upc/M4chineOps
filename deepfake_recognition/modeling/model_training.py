import os
import numpy as np
import pandas as pd
import pickle
import mlflow

from datetime import datetime
from codecarbon import EmissionsTracker

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, make_scorer, \
                            confusion_matrix, classification_report

import deepfake_recognition.config as cfg


def split_Xy(df_in: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Split DataFrame into features and labels.

    Args:
        df_in (pd.DataFrame): Input DataFrame with features and 'label'

    Returns:
        X (np.ndarray): Feature array.
        y (np.ndarray): Label array.
    """

    feature_cols = [c for c in df_in.columns if c != 'label']
    X = df_in[feature_cols].values.astype(np.float32)
    y = df_in['label'].map({'real': 0, 'fake': 1}).values.astype(int)

    return X, y


def tune_hyperparameters(X_train: pd.DataFrame, y_train: pd.Series, scoring) -> tuple[LogisticRegression, dict, float]:
    """
    Tune hyperparameters for Logistic Regression using GridSearchCV.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        scoring: Scoring function for evaluation.

    Returns:
        best_model (LogisticRegression): Best Logistic Regression model.
        best_params (dict): Best hyperparameters found.
        best_score (float): Best cross-validation score achieved.
    """

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
        cv=3,
        scoring=scoring,
        n_jobs=-1,
        verbose=3
    )

    grid_search.fit(X_train, y_train)

    print("\n=== BEST PARAMETERS FOUND ===")
    print(grid_search.best_params_)

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def main():
    EMBEDDING_DIR = str(cfg.EMBEDDING_DIR)
    EMISSIONS_OUTPUT_DIR = str(cfg.EMISSIONS_OUTPUT_DIR)
    MODEL_PATH = str(cfg.MODEL_PATH)

    # CodeCarbon tracker
    tracker = EmissionsTracker(output_dir = EMISSIONS_OUTPUT_DIR, project_name='deepfake_recognition_model_training')
    tracker.start()

    df_train = pd.read_csv(os.path.join(EMBEDDING_DIR, f'train_{cfg.EMBEDDING_AGGREGATION}_video_embeddings.csv')).sample(frac=1, random_state=42).reset_index(drop=True)
    df_val = pd.read_csv(os.path.join(EMBEDDING_DIR, f'val_{cfg.EMBEDDING_AGGREGATION}_video_embeddings.csv')).sample(frac=1, random_state=42).reset_index(drop=True)
    df_test  = pd.read_csv(os.path.join(EMBEDDING_DIR, f'test_{cfg.EMBEDDING_AGGREGATION}_video_embeddings.csv')).sample(frac=1, random_state=42).reset_index(drop=True)

    # start Mlflow experiment
    experiment_name = 'Model Training and Evaluation'
    try:
        mlflow.set_experiment(experiment_name)
        print(f'MLflow experiment {experiment_name} created successfully!')
    except Exception as e:
        print(f'Unexpected Error when creating the experiment: {e}')

    X_train, y_train = split_Xy(df_train)
    X_val,   y_val   = split_Xy(df_val)

    # we'll do cross validation with GridSearchCV, we can use train + val
    X_train = np.vstack((X_train, X_val))
    y_train = np.hstack((y_train, y_val))

    X_test,  y_test  = split_Xy(df_test)

    # scale features
    print('Scaling features...')
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    date_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_name = 'LR_finetuned_' + date_now
    with mlflow.start_run(run_name = run_name):
        # hyperparameter tuning
        best_lr, best_params, best_cv_score = tune_hyperparameters(X_train_scaled, y_train, scoring='roc_auc')     

        mlflow.sklearn.log_model(best_lr.fit(X_train_scaled, y_train), name='model')
        print(f'{run_name} model logged successfully!')

        mlflow.log_params(best_params)
        print(f'{run_name} parameters logged successfully!')
        
        # evaluation for X_train + X_val
        y_pred = best_lr.predict(X_test_scaled)
        y_scores = best_lr.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_scores)

        print('\n=== RESULTS FOR TEST ===')
        print('Accuracy:', round(acc, 4))
        print('F1-score:', round(f1, 4))
        print('ROC-AUC:', round(roc, 4))
        print('\nConfusion Matrix:\n', confusion_matrix(y_test, y_pred))
        print('\nClassification Report:\n', classification_report(y_test, y_pred, digits=3))

        mlflow.log_metric('Accuracy', acc)
        mlflow.log_metric('F1_Score', f1)
        mlflow.log_metric('ROC_AUC', roc)
        mlflow.log_metric('CV_ROC_AUC', best_cv_score)
        print(f'Train+Val metrics logged successfully!')

    # save model + scaler + metrics
    results = {
        'model': best_lr,
        'scaler': scaler,
        'best_params': best_params,
        'cv_roc_auc': round(best_cv_score, 4),
        'metrics_test': {
            'accuracy': round(acc, 4),
            'f1_score': round(f1, 4),
            'roc_auc': round(roc, 4),
        },
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(MODEL_PATH, 'wb') as f:
       pickle.dump(results, f)
    mlflow.log_artifact(MODEL_PATH, artifact_path='model_files') # model_files is the folder name in mlflow artifacts
    print(f'Model saved at {MODEL_PATH} and logged to MLflow successfully!')

    emissions = tracker.stop()
    print(f'Emissions tracked: {emissions} kg CO2')
    

if __name__ == "__main__":
    main()
