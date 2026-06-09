#!/usr/bin/env python3
"""
Pipeline order:
  load → preprocess → validate → build clean features → train IF →
  inject anomalies → transform polluted → score → evaluate
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.metrics import (
    classification_report, precision_score,
    recall_score, f1_score, roc_auc_score
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.preprocess import preprocess_data, inject_anomalies
from src.features.build_features import (
    build_clean_feature_pipeline,
    build_polluted_feature_pipeline
)
from src.utils.validate_data import validate_claims_data
from src.models.train import train_model


def main(args):

    # === MLflow Setup ===
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mlruns_path = args.mlflow_uri or os.path.join(project_root, "mlruns")
    mlflow.set_tracking_uri(Path(mlruns_path).as_uri())
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name="IF_Clean_Training_Pipeline"):

        mlflow.log_param("model", "IsolationForest")
        mlflow.log_param("anomaly_fraction", args.anomaly_fraction)
        mlflow.log_param("training_approach", "clean_data_only")

        # === STAGE 1: Data Loading ===
        print("🔄 Loading raw Medicare claims data...")
        df_raw = pd.read_csv(args.input)
        print(f"✅ Loaded: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

        # === STAGE 2: Preprocessing ===
        print("\n🔧 Preprocessing data...")
        df_clean = preprocess_data(df_raw)
        print(f"✅ Preprocessed: {df_clean.shape[0]} rows remaining")

        # === STAGE 3: Data Validation ===
        print("\n🔍 Validating data quality...")
        is_valid, failed = validate_claims_data(df_clean)
        mlflow.log_metric("data_quality_pass", int(is_valid))

        if not is_valid:
            import json
            mlflow.log_text(
                json.dumps(failed, indent=2),
                artifact_file="failed_expectations.json"
            )
            raise ValueError(f"❌ Data quality check failed: {failed}")
        print("✅ Validation passed.")

        # === STAGE 4: Build Features on CLEAN Data ===
        # Encoders are fitted here on normal data only and saved to disk.
        # These same encoders will be reused in Stage 6 to encode polluted data,
        # ensuring FRAUD_NPI_1 and other unseen values are handled consistently.
        print("\n🛠️  Building features on CLEAN data (fitting + saving encoders)...")

        encoder_path = os.path.join(
            project_root, "models", "encoders", "label_encoders.pkl"
        )
        X_clean, encoders = build_clean_feature_pipeline(df_clean, encoder_path)

        clean_path = os.path.join(
            project_root, "data", "processed", "claims_clean_features.csv"
        )
        os.makedirs(os.path.dirname(clean_path), exist_ok=True)
        X_clean.to_csv(clean_path, index=False)
        print(f"✅ Clean features saved → {clean_path} | Shape: {X_clean.shape}")

        # === STAGE 5: Train Isolation Forest on CLEAN Data ===
        # The model learns the distribution of NORMAL claims.
        # No anomalies have been injected yet — this is intentional.
        print("\n🤖 Training Isolation Forest on CLEAN data...")

        t0 = time.time()
        model = train_model(
            X_clean=X_clean,
            anomaly_fraction=args.anomaly_fraction
        )
        train_time = time.time() - t0
        mlflow.log_metric("train_time", train_time)
        print(f"✅ Training complete in {train_time:.2f}s")

        # === STAGE 6: Inject Anomalies ===
        # Anomaly injection happens AFTER training.
        # The model has never seen these fraudulent patterns.
        print(f"\n💉 Injecting synthetic fraud (fraction: {args.anomaly_fraction})...")
        df_polluted, anomaly_label = inject_anomalies(
            df_clean,
            anomaly_fraction=args.anomaly_fraction
        )
        print(f"✅ Injected {anomaly_label.sum()} anomalies into {len(anomaly_label)} claims")

        # === STAGE 7: Transform Polluted Data ===
        # Uses the encoders fitted in Stage 4 — same integer mappings.
        # Unseen values like FRAUD_NPI_1 are encoded as -1 (out-of-distribution signal).
        print("\n🛠️  Transforming polluted data using pre-fitted encoders...")
        df_final = build_polluted_feature_pipeline(df_polluted, anomaly_label, encoders)

        polluted_path = os.path.join(
            project_root, "data", "processed", "claims_polluted_features.csv"
        )
        df_final.to_csv(polluted_path, index=False)
        print(f"✅ Polluted features saved → {polluted_path} | Shape: {df_final.shape}")

        # === STAGE 8: Score Polluted Data ===
        # No train/test split — we trained on ALL clean data.
        # We now score ALL polluted data and evaluate against injected labels.
        print("\n📊 Scoring all polluted data...")

        X_polluted = df_final.drop(columns=[args.target])
        y_true = df_final[args.target]

        t1 = time.time()
        # Invert scores: raw decision_function gives lower = more anomalous.
        # After *-1, higher score = more anomalous (correct direction for ROC-AUC).
        anomaly_scores = -1 * model.decision_function(X_polluted)
        raw_preds = model.predict(X_polluted)
        # Map IF output: -1 (anomaly) → 1, 1 (normal) → 0
        y_pred = np.where(raw_preds == -1, 1, 0)
        score_time = time.time() - t1

        # === STAGE 9: Evaluate ===
        print("\n📈 Evaluating model performance...")

        precision = precision_score(y_true, y_pred)
        recall    = recall_score(y_true, y_pred)
        f1        = f1_score(y_true, y_pred)
        roc_auc   = roc_auc_score(y_true, anomaly_scores)

        mlflow.log_metric("precision",  precision)
        mlflow.log_metric("recall",     recall)
        mlflow.log_metric("f1",         f1)
        mlflow.log_metric("roc_auc",    roc_auc)
        mlflow.log_metric("score_time", score_time)

        print(f"\n🎯 Model Performance:")
        print(f"   Precision : {precision:.3f}")
        print(f"   Recall    : {recall:.3f}")
        print(f"   F1 Score  : {f1:.3f}")
        print(f"   ROC AUC   : {roc_auc:.3f}  ← primary metric")

        print(f"\n⏱️  Performance Summary:")
        print(f"   Training time : {train_time:.2f}s  (on {len(X_clean)} clean claims)")
        print(f"   Scoring time  : {score_time:.4f}s (on {len(X_polluted)} polluted claims)")

        print(f"\n📋 Detailed Classification Report:")
        print(classification_report(y_true, y_pred, digits=3))


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Medicare Fraud Detection — Clean Training Pipeline"
    )
    p.add_argument("--input",            type=str,   required=True)
    p.add_argument("--target",           type=str,   default="anomaly_label")
    p.add_argument("--anomaly_fraction", type=float, default=0.02)
    p.add_argument("--experiment",       type=str,   default="Medicare Fraud Clean Training")
    p.add_argument("--mlflow_uri",       type=str,   default=None)
    args = p.parse_args()
    main(args)

"""
Run with:
python scripts/run_pipeline.py --input data/raw/DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv
"""