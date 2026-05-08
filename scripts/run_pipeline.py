#!/usr/bin/env python3
"""
Runs sequentially: load → preprocess → validate → inject anomalies → feature engineering → model training
"""

import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, precision_score, recall_score,
    f1_score, roc_auc_score
)
from sklearn.ensemble import IsolationForest

# === Fix import path for local modules ===
# ESSENTIAL: Allows imports from the src/ directory structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local modules - Core pipeline components
from src.data.preprocess import preprocess_data, inject_anomalies
from src.features.build_features import build_feature_pipeline
from src.utils.validate_data import validate_claims_data

def main(args):
    """
    Main training pipeline function that orchestrates the complete ML workflow 
    for Health Insurance Fraud Detection using Isolation Forest.
    """
    
    # === MLflow Setup - ESSENTIAL for experiment tracking ===
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mlruns_path = args.mlflow_uri or f"file://{project_root}/mlruns"
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment(args.experiment)

    # Start MLflow run
    with mlflow.start_run(run_name="Isolation_Forest_Baseline"):
        # === Log hyperparameters and configuration ===
        mlflow.log_param("model", "IsolationForest")
        mlflow.log_param("anomaly_fraction", args.anomaly_fraction)
        mlflow.log_param("test_size", args.test_size)

        # === STAGE 1: Data Loading ===
        print("🔄 Loading raw Medicare claims data...")
        df_raw = pd.read_csv(args.input)
        print(f"✅ Data loaded: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

        # === STAGE 2: Data Preprocessing ===
        print("🔧 Preprocessing data (cleaning missing values & fixing formats)...")
        df_clean = preprocess_data(df_raw)
        print(f"✅ Data preprocessed: {df_clean.shape[0]} rows remaining.")

        # === STAGE 3: CRITICAL Data Quality Validation ===
        print("🔍 Validating data quality with Great Expectations...")
        is_valid, failed = validate_claims_data(df_clean)
        mlflow.log_metric("data_quality_pass", int(is_valid))

        if not is_valid:
            import json
            mlflow.log_text(json.dumps(failed, indent=2), artifact_file="failed_expectations.json")
            raise ValueError(f"❌ Data quality check failed. Issues: {failed}")
        else:
            print("✅ Data validation passed. Logged to MLflow.")

        # === STAGE 4: Anomaly Injection (Ground Truth Generation) ===
        print(f"💉 Injecting synthetic fraud rings (fraction: {args.anomaly_fraction})...")
        df_polluted, anomaly_label = inject_anomalies(df_clean, anomaly_fraction=args.anomaly_fraction)
        print(f"✅ Anomalies injected. Generated {anomaly_label.sum()} fraudulent claims.")

        # === STAGE 5: Feature Engineering ===
        print("🛠️  Building features (Label Encoding & Feature Selection)...")
        target = args.target
        df_final = build_feature_pipeline(df_polluted, anomaly_label)
        
        # Save processed dataset for reproducibility
        processed_path = os.path.join(project_root, "data", "processed", "claims_processed.csv")
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df_final.to_csv(processed_path, index=False)
        print(f"✅ Processed dataset saved to {processed_path} | Shape: {df_final.shape}")

        # === STAGE 6: Train/Test Split ===
        print("📊 Splitting data...")
        X = df_final.drop(columns=[target])
        y = df_final[target]
        
        # Stratified split to maintain exactly 2% fraud in both training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=args.test_size, 
            stratify=y, 
            random_state=42
        )
        print(f"✅ Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

        # === STAGE 7: Model Training (Isolation Forest) ===
        print("🤖 Training Isolation Forest Baseline...")
        
        model = IsolationForest(
            n_estimators=300,                  # Optimized higher tree count
            contamination=args.anomaly_fraction, # Tell the model exactly how much fraud to expect
            max_samples=0.8,                   # Prevent overfitting
            random_state=42,
            n_jobs=-1
        )

        t0 = time.time()
        # IMPORTANT: Fit on the polluted training data
        model.fit(X_train) 
        train_time = time.time() - t0
        mlflow.log_metric("train_time", train_time)
        print(f"✅ Model trained in {train_time:.2f} seconds")

        # === STAGE 8: Model Evaluation ===
        print("📊 Evaluating model performance...")
        
        t1 = time.time()
        # Get raw decision scores (lower is more anomalous, invert so higher = anomalous)
        anomaly_scores = -1 * model.decision_function(X_test)
        
        # Isolation forest outputs -1 for anomalies, 1 for normal. Map this to 1 and 0.
        raw_preds = model.predict(X_test)
        y_pred = np.where(raw_preds == -1, 1, 0)
        
        pred_time = time.time() - t1
        mlflow.log_metric("pred_time", pred_time)

        # Calculate metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, anomaly_scores) # Golden metric for fraud
        
        # Log all metrics for experiment tracking
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall) 
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        
        print(f"🎯 Model Performance:")
        print(f"   Precision: {precision:.3f} | Recall: {recall:.3f}")
        print(f"   F1 Score: {f1:.3f} | ROC AUC: {roc_auc:.3f}")

        # === STAGE 9: Model Serialization and Logging ===
        print("💾 Saving model to MLflow...")
        mlflow.sklearn.log_model(
            model, 
            artifact_path="model" 
        )
        print("✅ Baseline model saved to MLflow.")

        # === Final Performance Summary ===
        print(f"\n⏱️  Performance Summary:")
        print(f"   Training time: {train_time:.2f}s")
        print(f"   Inference time: {pred_time:.4f}s")
        
        print(f"\n📈 Detailed Classification Report:")
        print(classification_report(y_test, y_pred, digits=3))

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run complete ML Pipeline for Medicare Fraud Detection")
    p.add_argument("--input", type=str, required=True,
                   help="path to raw CSV (e.g., data/raw/DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv)")
    p.add_argument("--target", type=str, default="anomaly_label")
    p.add_argument("--anomaly_fraction", type=float, default=0.02)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--experiment", type=str, default="Medicare Fraud Baseline")
    p.add_argument("--mlflow_uri", type=str, default=None,
                   help="override MLflow tracking URI, else uses project_root/mlruns")

    args = p.parse_args()
    main(args)

"""
# Run the pipeline from your terminal using:

python scripts/run_pipeline.py \
    --input data/raw/DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv \
    --target anomaly_label
"""