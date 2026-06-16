#!/usr/bin/env python3
"""
Model Comparison Pipeline - Train and compare Isolation Forest, One-Class SVM, CBLOF, and ECOD.
Saves metrics comparison to data/processed/model_comparison.csv.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

# Fix import path for local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.preprocess import preprocess_data, inject_anomalies
from src.features.build_features import build_feature_pipeline, load_encoders
from src.utils.validate_data import validate_claims_data

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# PyOD imports (with clean error fallback)
try:
    from pyod.models.cblof import CBLOF
    from pyod.models.ecod import ECOD
except ImportError:
    print("\n❌ Error: The 'pyod' library is required to run CBLOF and ECOD models.")
    print("Please install it using your terminal: pip install pyod\n")
    sys.exit(1)

def main(args):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # === STAGE 1: Data Loading ===
    print("🔄 Loading raw Medicare claims data...")
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file not found: {args.input}")
        sys.exit(1)
        
    df_raw = pd.read_csv(args.input)
    print(f"✅ Data loaded: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

    # === STAGE 2: Data Preprocessing ===
    print("🔧 Preprocessing data (cleaning missing values & fixing formats)...")
    df_clean = preprocess_data(df_raw)
    print(f"✅ Data preprocessed: {df_clean.shape[0]} rows remaining.")

    # === STAGE 3: Data Quality Validation ===
    print("🔍 Validating data quality with Great Expectations...")
    is_valid, failed = validate_claims_data(df_clean)
    if not is_valid:
        print(f"❌ Data validation failed! Issues: {failed}")
        raise ValueError(f"Data validation failed. Issues: {failed}")
    print("✅ Data validation passed.")

    # === STAGE 4: Anomaly Injection ===
    print(f"💉 Injecting synthetic fraud rings (fraction: {args.anomaly_fraction})...")
    df_polluted, anomaly_label = inject_anomalies(df_clean, anomaly_fraction=args.anomaly_fraction)
    print(f"✅ Anomalies injected. Generated {anomaly_label.sum()} fraudulent claims.")

    # === STAGE 5: Feature Engineering ===
    print("🛠️  Building features (Label Encoding & Feature Selection)...")
    try:
        encoders = load_encoders()
    except FileNotFoundError:
        encoders = None
    df_final = build_feature_pipeline(df_polluted, anomaly_label, encoders)

    # === STAGE 6: Train/Test Split ===
    print("📊 Splitting data...")
    X = df_final.drop(columns=['anomaly_label'])
    y = df_final['anomaly_label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=args.test_size, 
        stratify=y, 
        random_state=42
    )
    print(f"✅ Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

    # Scale data for distance-based models (One-Class SVM)
    print("⚖️ Scaling features for distance-based models...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Models dict to store models and their parameters
    models = {
        "Isolation Forest": {
            "model": IsolationForest(n_estimators=50, contamination=args.anomaly_fraction, random_state=42, n_jobs=-1),
            "use_scaled": False,
            "pyod_format": False
        },
        "One-Class SVM": {
            "model": OneClassSVM(kernel='rbf', gamma='scale', nu=args.anomaly_fraction, verbose=0),
            "use_scaled": True,
            "pyod_format": False
        },
        "CBLOF": {
            "model": CBLOF(n_clusters=5, contamination=args.anomaly_fraction, use_weights=True, random_state=42, n_jobs=-1),
            "use_scaled": False,
            "pyod_format": True
        },
        "ECOD": {
            "model": ECOD(contamination=args.anomaly_fraction),
            "use_scaled": False,
            "pyod_format": True
        }
    }

    comparison_results = []

    print("\n🤖 Training and Evaluating Models...")
    for name, config in models.items():
        print(f"\n--- {name} ---")
        clf = config["model"]
        
        # Select train/test matrices
        X_tr = X_train_scaled if config["use_scaled"] else X_train
        X_te = X_test_scaled if config["use_scaled"] else X_test
        
        # Fit model
        clf.fit(X_tr)
        print("Training complete.")

        # Predict and score
        if config["pyod_format"]:
            # PyOD outputs 1 for anomaly, 0 for normal
            y_pred = clf.predict(X_te)
            scores = clf.decision_function(X_te)  # PyOD: higher is more anomalous
        else:
            # sklearn models output -1 for anomaly, 1 for normal
            raw_preds = clf.predict(X_te)
            y_pred = np.where(raw_preds == -1, 1, 0)
            scores = -1 * clf.decision_function(X_te)  # Invert so higher is more anomalous
            
        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, scores)

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        print(f"Metrics: Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")
        print(f"Confusion Matrix:\n  TN: {tn:6d} | FP: {fp:6d}\n  FN: {fn:6d} | TP: {tp:6d}")

        comparison_results.append({
            "Model": name,
            "Accuracy": round(accuracy, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(roc_auc, 4),
            "True Negatives": tn,
            "False Positives": fp,
            "False Negatives": fn,
            "True Positives": tp
        })

    # Save metrics to CSV
    df_compare = pd.DataFrame(comparison_results)
    out_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "model_comparison.csv")
    df_compare.to_csv(out_path, index=False)
    print(f"\n✅ Model comparison summary saved to: {out_path}")
    print("\nSummary Table:")
    print(df_compare[["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]].to_string(index=False))

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train and compare 4 unsupervised anomaly detection models")
    p.add_argument("--input", type=str, default="data/raw/DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv",
                   help="path to raw claims CSV file")
    p.add_argument("--anomaly_fraction", type=float, default=0.02,
                   help="proportion of anomalies to inject (expected contamination rate)")
    p.add_argument("--test_size", type=float, default=0.2,
                   help="proportion of data to use for testing")
    
    args = p.parse_args()
    main(args)
