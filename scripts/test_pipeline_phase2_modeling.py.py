import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
import optuna

print("=== Phase 2: Baseline Modeling with Isolation Forest ===")

# 1. Load Processed Data
df = pd.read_csv("data/processed/claims_processed.csv")
target_col = "anomaly_label"

# Ensure target is formatted correctly
assert df[target_col].isna().sum() == 0, f"{target_col} has NaNs"
assert set(df[target_col].unique()) <= {0, 1}, f"{target_col} not 0/1"

X = df.drop(columns=[target_col])
y = df[target_col]

# Split data (stratify ensures anomalies are distributed evenly in train and test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

def objective(trial):
    # Define hyperparameter search space for Isolation Forest
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_samples": trial.suggest_float("max_samples", 0.1, 1.0),
        "max_features": trial.suggest_float("max_features", 0.5, 1.0),
        # Assuming your injected anomaly fraction is around 2% (0.02)
        # We let Optuna search tightly around that range
        "contamination": trial.suggest_float("contamination", 0.01, 0.05),
        "random_state": 42,
        "n_jobs": -1
    }
    
    # Instantiate model
    model = IsolationForest(**params)
    
    # Train on the full polluted training set
    model.fit(X_train)
    
    # Evaluate on the test set
    # decision_function gives lower scores to anomalies. 
    # We multiply by -1 so HIGHER scores = MORE anomalous (Standard for ROC-AUC)
    anomaly_scores = -1 * model.decision_function(X_test)
    
    # Calculate ROC-AUC as the optimization target
    auc = roc_auc_score(y_test, anomaly_scores)
    
    return auc

# Run the hyperparameter optimization
# We want to MAXIMIZE the ROC-AUC score
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

print("\n=== Tuning Results ===")
print("Best Params:", study.best_params)
print(f"Best ROC-AUC: {study.best_value:.4f}")

# --- Optional: Train Final Model with Best Params ---
print("\nTraining final baseline model with best parameters...")
best_model = IsolationForest(**study.best_params, random_state=42, n_jobs=-1)
best_model.fit(X_train)

final_scores = -1 * best_model.decision_function(X_test)
final_auc = roc_auc_score(y_test, final_scores)
print(f"Final Test ROC-AUC: {final_auc:.4f}")