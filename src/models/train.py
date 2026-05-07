import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score

def train_model(df: pd.DataFrame, target_col: str, anomaly_fraction: float):
    """
    Trains an Isolation Forest model, scores anomalies, and logs with MLflow.

    Args:
        df (pd.DataFrame): The baseline DataFrame containing features and labels.
        target_col (str): Name of the ground-truth anomaly label column.
        anomaly_fraction (float): The expected proportion of outliers (contamination).
        
    Returns:
        iso_forest: The trained model.
        anomaly_scores: Array of the calculated anomaly scores.
    """
    # --- 1. Prepare Data ---
    X = df.drop(columns=[target_col])
    y_true = df[target_col]
    
    print(f"Baseline features and labels are prepared.")
    print(f"Feature matrix shape: {X.shape}")

    # --- 2. Instantiate Model ---
    iso_forest = IsolationForest(
        n_estimators=50,
        contamination=anomaly_fraction,
        random_state=42,
        n_jobs=-1
    )

    # Start MLflow tracking run
    with mlflow.start_run(run_name="isolation_forest_baseline"):
        print("\nTraining the Isolation Forest model on the polluted data...")
        
        # --- 3. Train Model ---
        iso_forest.fit(X)
        
        # --- 4. Predict and Transform Scores ---
        # Get raw anomaly scores and invert so HIGHER = more anomalous
        raw_anomaly_scores = iso_forest.decision_function(X)
        anomaly_scores = -1 * raw_anomaly_scores
        
        # Isolation Forest predict() outputs 1 (normal) and -1 (anomaly)
        # We map this to 0 (normal) and 1 (anomaly) to match ground truth y_true
        raw_preds = iso_forest.predict(X)
        mapped_preds = np.where(raw_preds == -1, 1, 0)
        
        # Calculate Metrics
        acc = accuracy_score(y_true, mapped_preds)
        rec = recall_score(y_true, mapped_preds)
        auc = roc_auc_score(y_true, anomaly_scores)
        
        # --- 5. Log to MLflow ---
        # Log Params
        mlflow.log_param("n_estimators", 50)
        mlflow.log_param("contamination", anomaly_fraction)
        
        # Log Metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("roc_auc", auc)
        
        # Log Model
        mlflow.sklearn.log_model(iso_forest, "isolation_forest_model")
        
        # Log Dataset for UI visibility
        train_ds = mlflow.data.from_pandas(df, source="polluted_baseline_data")
        mlflow.log_input(train_ds, context="training_and_evaluation")
        
        print("Training complete.")
        print(f"Metrics logged - Accuracy: {acc:.4f}, Recall: {rec:.4f}, ROC AUC: {auc:.4f}")
        
        return iso_forest, anomaly_scores

# Example usage inside your orchestration script:
# model, scores = train_model(df=df_baseline, target_col='anomaly_label', anomaly_fraction=0.02)