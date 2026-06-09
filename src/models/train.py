from time import time

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import IsolationForest


def train_model(X_clean: pd.DataFrame, anomaly_fraction: float):
    """
    Trains an Isolation Forest on CLEAN data only.

    Why clean data only?
      Isolation Forest learns the distribution of normal behaviour.
      If anomalies are present during training, the model's boundary
      of "normal" is shifted, making anomalies harder to detect at
      inference time.

    No labels are used — this is fully unsupervised.
    Evaluation metrics live in run_pipeline.py, not here.

    Args:
        X_clean:          Encoded clean feature matrix (no anomalies, no target col)
        anomaly_fraction: Contamination hint passed to IsolationForest.
                          Since training data is clean this should be small (≤ 0.02).

    Returns:
        iso_forest: Trained IsolationForest model
    """
    print(f"Training on clean data | Shape: {X_clean.shape}")

    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=anomaly_fraction,
        random_state=42,
        n_jobs=-1
    )

    with mlflow.start_run(run_name="isolation_forest_clean", nested=True):

        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("contamination", anomaly_fraction)
        mlflow.log_param("training_data", "clean_only")

        print("Fitting Isolation Forest on clean data...")
        start = time()
        iso_forest.fit(X_clean)
        train_time = time() - start

        mlflow.log_metric("train_time", train_time)
        mlflow.sklearn.log_model(iso_forest, "isolation_forest_model")

        print(f"Training complete in {train_time:.2f}s")
        print("Model saved to MLflow.")

    return iso_forest