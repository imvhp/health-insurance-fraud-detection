"""
INFERENCE PIPELINE - Production Serving for PTL Anomaly Detection
==================================================================

Each of the 4 models has its own native scoring convention:

  IsolationForest  decision_function → pos=normal, neg=anomaly, boundary at 0
                   typical range: [-0.10, +0.14], offset_ ≈ -0.54
                   HIGH  : score <  0
                   MEDIUM: 0 ≤ score < +0.03  (near boundary)
                   LOW   : score ≥ +0.03

  OCSVM            decision_function → pos=normal, neg=anomaly, boundary at 0
                   typical range: [-3.5, +3.0], offset_ ≈ +3.45 (OCSVM stores its own)
                   HIGH  : score <  0
                   MEDIUM: 0 ≤ score < +0.5
                   LOW   : score ≥ +0.5

  CBLOF  (PyOD)    decision_function → higher = more anomalous, threshold_ = boundary
                   typical range: [0.4, 64], threshold_ ≈ 8.28
                   HIGH  : score ≥ threshold_ * 1.5
                   MEDIUM: threshold_ ≤ score < threshold_ * 1.5
                   LOW   : score <  threshold_

  ECOD   (PyOD)    decision_function → higher = more anomalous, threshold_ = boundary
                   typical range: [32, 110], threshold_ ≈ 67.12
                   HIGH  : score ≥ threshold_ * 1.5
                   MEDIUM: threshold_ ≤ score < threshold_ * 1.5
                   LOW   : score <  threshold_

risk_percentage is derived from each model's own boundary, NOT a shared sigmoid.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib

# Suppress noisy but harmless sklearn/pyod warnings during single-claim serving
warnings.filterwarnings("ignore", message="Skipping features without any observed values", category=UserWarning)
warnings.filterwarnings("ignore", message="Precision loss occurred in moment calculation", category=RuntimeWarning)

# Setup path imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Import PTL modules
from features import add_domain_features
from model_selection import build_data_profile, score_models_for_profile


# ============ MODEL DIRECTORY DETECTION ============

def find_model_dir() -> str:
    """Locates the directory containing preprocessor.joblib and model files.

    Priority order:
      1. Latest retrain folder (models/retrain/<timestamp>/) — sorted by
         directory name descending so the newest timestamp wins.
      2. Baseline model (models/baseline/)
      3. Bare models/ root (last resort)
    """
    # 1. Retrain folders — newest first (folder names are YYYYMMDD_HHMMSS)
    retrain_root = os.path.join(PROJECT_ROOT, "models", "retrain")
    if os.path.exists(retrain_root):
        subdirs = sorted(
            [d for d in os.listdir(retrain_root)
             if os.path.isdir(os.path.join(retrain_root, d))],
            reverse=True,
        )
        for d in subdirs:
            path = os.path.join(retrain_root, d)
            if os.path.exists(os.path.join(path, "preprocessor.joblib")):
                return path

    # 2. Baseline model
    baseline = os.path.join(PROJECT_ROOT, "models", "baseline")
    if os.path.exists(os.path.join(baseline, "preprocessor.joblib")):
        return baseline

    # 3. Last resort
    return os.path.join(PROJECT_ROOT, "models")


def get_model_version(model_dir: str) -> str:
    """Derive a human-readable version string from the model directory path.

    * Retrain folders  → 1.0.X (where X is chronological order)
    * Baseline folder   → "1.0.0"
    * Anything else     → "1.0.0"
    """
    dirname = os.path.basename(model_dir)
    if dirname == "baseline":
        return "1.0.0"
    
    retrain_root = os.path.join(PROJECT_ROOT, "models", "retrain")
    if os.path.exists(retrain_root):
        subdirs = sorted(
            [d for d in os.listdir(retrain_root)
             if os.path.isdir(os.path.join(retrain_root, d))]
        )
        try:
            index = subdirs.index(dirname)
            return f"1.0.{index + 1}"
        except ValueError:
            pass
            
    return "1.0.0"


# ============ LAZY GLOBAL LOADING ============

_preprocessor = None
_models = {}
_numeric_cols = []
_categorical_cols = []
_model_dir_used = None


def _load_artifacts_from(model_dir: str):
    """Internal: load preprocessor + models from *model_dir* into globals."""
    global _preprocessor, _models, _numeric_cols, _categorical_cols, _model_dir_used

    print(f"[INFO] Loading serving artifacts from: {model_dir}")

    preprocessor_path = os.path.join(model_dir, "preprocessor.joblib")
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor not found at: {preprocessor_path}")

    _preprocessor = joblib.load(preprocessor_path)
    _models = {}

    # Extract training-time feature names from preprocessor ColumnTransformer
    try:
        _numeric_cols = list(_preprocessor.transformers[0][2])
        _categorical_cols = list(_preprocessor.transformers[1][2])
    except Exception as e:
        print(f"[WARNING] Failed to extract column lists from preprocessor: {e}")
        _numeric_cols = []
        _categorical_cols = []

    # Load available models
    for name in ["IsolationForest", "CBLOF", "OCSVM", "ECOD"]:
        model_path = os.path.join(model_dir, f"{name}.joblib")
        if os.path.exists(model_path):
            try:
                _models[name] = joblib.load(model_path)
                print(f"[SUCCESS] Loaded model: {name}")
            except Exception as e:
                print(f"[WARNING] Failed to load model {name} from {model_path}: {e}")

    if not _models:
        raise ValueError(f"No valid models found in {model_dir}")

    _model_dir_used = model_dir


def reload_serving_artifacts():
    """Force a full reload from the best available model directory.

    Call this after a retrain cycle completes so the server immediately
    picks up the new model without requiring a restart.
    """
    global _preprocessor
    _preprocessor = None          # clear cache flag so _load_artifacts_from runs
    best_dir = find_model_dir()
    _load_artifacts_from(best_dir)
    print(f"[INFO] Serving artifacts reloaded — now using: {best_dir}")


def get_serving_artifacts():
    """Return cached preprocessor and models, reloading when a newer model
    directory has appeared (e.g. after a retrain cycle)."""
    global _preprocessor, _models, _numeric_cols, _categorical_cols, _model_dir_used

    # Always check whether a newer model directory is now available
    best_dir = find_model_dir()

    if _preprocessor is not None and _model_dir_used == best_dir:
        # Cache is valid — same directory, no change
        return _preprocessor, _models, _numeric_cols, _categorical_cols, _model_dir_used

    if _preprocessor is not None and _model_dir_used != best_dir:
        print(f"[INFO] New model directory detected: {best_dir} (was: {_model_dir_used}). Reloading...")

    _load_artifacts_from(best_dir)
    return _preprocessor, _models, _numeric_cols, _categorical_cols, _model_dir_used


# ============ MODEL-NATIVE RISK CLASSIFICATION ============

def classify_risk(model_name: str, raw_score: float, clf) -> tuple:
    """
    Classify risk using each model's own native scoring system.
    Returns (risk_level: str, risk_percentage: float, prediction: str)

    - risk_level: "LOW" | "MEDIUM" | "HIGH"
    - risk_percentage: 0-100 (derived from model-native boundary, not a forced sigmoid)
    - prediction: "NORMAL" (LOW) or "ANOMALY" (MEDIUM or HIGH)
    """

    if model_name in ("ECOD", "CBLOF"):
        # --- PyOD models ---
        # decision_function: higher score = more anomalous; threshold_ = decision boundary
        threshold = float(getattr(clf, "threshold_", 1.0)) or 1.0

        # risk_percentage:
        #   0%   → score = 0             (perfectly normal)
        #   50%  → score = threshold_    (exactly at boundary)
        #   100% → score = 2×threshold_  (twice as anomalous as boundary)
        risk_pct = float(np.clip((raw_score / threshold) * 50.0, 0.0, 100.0))

        if raw_score >= threshold * 1.5:
            level = "HIGH"
        elif raw_score >= threshold:
            level = "MEDIUM"
        else:
            level = "LOW"

    elif model_name == "IsolationForest":
        # --- IsolationForest ---
        # decision_function: positive = normal, negative = anomaly, boundary at 0
        # offset_ is negative; typical scale ≈ |offset_| / 5 gives a reasonable MEDIUM band
        abs_offset = abs(float(getattr(clf, "offset_", -0.1))) or 0.1
        medium_band = abs_offset * 0.05   # thin band right around the 0 boundary

        # risk_percentage:
        #   0%   → score = +abs_offset   (far normal side)
        #   50%  → score = 0             (decision boundary)
        #   100% → score = -abs_offset   (far anomaly side)
        risk_pct = float(np.clip((0.5 - raw_score / (2.0 * abs_offset)) * 100.0, 0.0, 100.0))

        if raw_score >= medium_band:
            level = "LOW"
        elif raw_score >= 0:
            level = "MEDIUM"
        else:
            level = "HIGH"

    else:
        # --- OCSVM ---
        # decision_function: positive = normal, negative = anomaly, boundary at 0
        # offset_ is stored as the raw margin; use it as scale
        offset_raw = getattr(clf, "offset_", [0.5])
        scale = abs(float(np.ravel(offset_raw)[0])) or 0.5
        medium_band = scale * 0.1   # 10% of scale as MEDIUM band

        # risk_percentage same shape as IsolationForest
        risk_pct = float(np.clip((0.5 - raw_score / (2.0 * scale)) * 100.0, 0.0, 100.0))

        if raw_score >= medium_band:
            level = "LOW"
        elif raw_score >= 0:
            level = "MEDIUM"
        else:
            level = "HIGH"

    prediction = "NORMAL" if level == "LOW" else "ANOMALY"
    return level, round(risk_pct, 2), prediction


# ============ INFERENCE LOGIC ============

def predict_claims(input_data: list[dict], contamination: float = 0.05) -> dict:
    """
    Main inference logic.
    1. Featurizes the claim(s)
    2. Builds a data profile
    3. Scores the models for suitability and selects the best one
    4. Transforms the features using the preprocessor
    5. Feeds them to the selected model
    6. Classifies each claim using the model's native scoring system
    """
    preprocessor, models, numeric_cols, categorical_cols, model_dir = get_serving_artifacts()

    # 1. Featurize
    df = pd.DataFrame(input_data)
    df_feats = add_domain_features(df)

    # Ensure ID columns exist
    if "CLM_ID" not in df_feats.columns:
        df_feats["CLM_ID"] = [f"CLAIM_{i}" for i in range(len(df_feats))]
    if "PRVDR_NUM" not in df_feats.columns:
        df_feats["PRVDR_NUM"] = "UNKNOWN"

    # Align features with preprocessor columns
    expected_cols = list(numeric_cols) + list(categorical_cols)
    for col in expected_cols:
        if col not in df_feats.columns:
            df_feats[col] = np.nan

    X_df = df_feats[expected_cols]

    # 2. Build data profile & select model
    profile = build_data_profile(
        X_df=X_df,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        contamination=contamination,
        has_labels=False
    )

    suitability_df = score_models_for_profile(profile)
    suitability_report = suitability_df.to_dict(orient="records")

    # Find the top-ranked model that is actually loaded
    selected_model_name = None
    for row in suitability_df.sort_values("rank")["model"]:
        if row in models:
            selected_model_name = row
            break

    if selected_model_name is None:
        selected_model_name = list(models.keys())[0]

    clf = models[selected_model_name]

    # 3. Transform features
    X_transformed = preprocessor.transform(X_df)

    # 4. Score with the selected model (raw native scores)
    raw_scores = np.asarray(clf.decision_function(X_transformed), dtype=float)

    # 5. Classify each claim using model-native thresholds
    results = []
    for i in range(len(df)):
        raw = float(raw_scores[i])
        risk_level, risk_pct, pred_label = classify_risk(selected_model_name, raw, clf)

        claim_id = str(df_feats.iloc[i].get("CLM_ID", f"CLAIM_{i}"))
        provider_id = str(df_feats.iloc[i].get("PRVDR_NUM", "UNKNOWN"))

        results.append({
            "claim_id": claim_id,
            "provider_id": provider_id,
            "anomaly_score": round(raw, 6),
            "risk_level": risk_level,
            "risk_percentage": risk_pct,
            "prediction": pred_label,
            "model_selected": selected_model_name,
        })

    return {
        "model_selected": selected_model_name,
        "model_version": get_model_version(model_dir),
        "model_directory": model_dir,
        "suitability_report": suitability_report,
        "predictions": results,
    }
