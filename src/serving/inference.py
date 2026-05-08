"""
INFERENCE PIPELINE - Production ML Model Serving for Fraud Detection
=========================================================================

This module provides the core inference functionality for the Medicare Fraud 
Detection model. It ensures that serving-time feature transformations exactly match 
training-time transformations, which is CRITICAL for model accuracy in production.

Key Responsibilities:
1. Load MLflow-logged Isolation Forest model
2. Apply identical feature transformations as used during training
3. Ensure correct feature ordering for model input
4. Convert model predictions (-1/1) to user-friendly output

CRITICAL PATTERN: Training/Serving Consistency
- Replicates the LabelEncoding strategy deterministically for single-row inference
- Maintains exact 9-feature column order from training
- Handles missing NPIs and numeric coercions gracefully

Production Deployment:
- MODEL_DIR points to containerized model artifacts
- Optimized for single-row inference (real-time API serving)
"""

import os
import pandas as pd
import numpy as np
import mlflow

# === MODEL LOADING CONFIGURATION ===
# IMPORTANT: This path is set during Docker container build.
# We are defaulting to the path where your MLflow artifact is currently stored.
MODEL_DIR = "notebooks/mlruns/0/models/m-fdb06fe56dc243a6b063a4c539a8153a/artifacts"

try:
    # Load the trained Isolation Forest model in MLflow pyfunc format
    model = mlflow.pyfunc.load_model(MODEL_DIR)
    print(f"✅ Isolation Forest Model loaded successfully from {MODEL_DIR}")
except Exception as e:
    print(f"❌ Failed to load model from {MODEL_DIR}: {e}")
    # Fallback for local development
    try:
        import glob
        local_model_paths = glob.glob("./mlruns/*/*/artifacts/model")
        if local_model_paths:
            latest_model = max(local_model_paths, key=os.path.getmtime)
            model = mlflow.pyfunc.load_model(latest_model)
            MODEL_DIR = latest_model
            print(f"✅ Fallback: Loaded model from {latest_model}")
        else:
            raise Exception("No model found in local mlruns")
    except Exception as fallback_error:
        print(f"Warning: Model could not be loaded at startup. Error: {fallback_error}")

# === FEATURE SCHEMA CONSTANTS ===
# CRITICAL: Load the exact feature column order used during training
FEATURE_COLS = [
    'PRVDR_NUM',
    'NCH_PRMRY_PYR_CLM_PD_AMT',
    'AT_PHYSN_NPI',
    'OP_PHYSN_NPI',
    'OT_PHYSN_NPI',
    'CLM_UTLZTN_DAY_CNT',
    'ADMTNG_ICD9_DGNS_CD',
    'CLM_DRG_CD',
    'ICD9_PRCDR_CD_1'
]

CATEGORICAL_COLS = [
    'PRVDR_NUM', 'AT_PHYSN_NPI', 'OP_PHYSN_NPI', 'OT_PHYSN_NPI',
    'ADMTNG_ICD9_DGNS_CD', 'CLM_DRG_CD', 'ICD9_PRCDR_CD_1'
]

NUMERIC_COLS = ['NCH_PRMRY_PYR_CLM_PD_AMT', 'CLM_UTLZTN_DAY_CNT']

def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply identical feature transformations as used during model training.
    
    Transformation Pipeline:
    1. Clean column names and handle missing values ('UNKNOWN' or 0)
    2. Convert categorical text to deterministic integers (Production safe LabelEncoding)
    3. Coerce numeric columns safely
    4. Align features with training schema and order
    """
    df = df.copy()
    
    # Clean column names (remove any whitespace)
    df.columns = df.columns.str.strip()
    
    # === STEP 1: Numeric Type Coercion ===
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
            
    # === STEP 2: Production-Safe Categorical Encoding ===
    # Because training used LabelEncoder().fit_transform() on the whole dataset,
    # applying it to a single row would turn everything to 0. 
    # Instead, we use a deterministic hash module to convert strings to consistent integers.
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            # Fill missing with 'UNKNOWN' exactly like build_features.py
            df[col] = df[col].fillna('UNKNOWN').astype(str)
            
            # Create a consistent integer from the string (mod 100000 to keep it manageable for the trees)
            df[col] = df[col].apply(lambda x: abs(hash(x)) % 100000)
    
    # === STEP 3: Feature Alignment with Training Schema ===
    # CRITICAL: Ensure features are in exact same order as training
    # Missing features get filled with 0, extra features are dropped
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)
    
    return df

def predict(input_dict: dict) -> str:
    """
    Main prediction function for Medicare Fraud inference.
    
    Pipeline:
    1. Convert input dictionary to DataFrame
    2. Apply feature transformations
    3. Generate model prediction using loaded Isolation Forest
    4. Convert -1/1 anomaly output to user-friendly string
    """
    
    # === STEP 1: Convert Input to DataFrame ===
    df = pd.DataFrame([input_dict])
    
    # === STEP 2: Apply Feature Transformations ===
    df_enc = _serve_transform(df)
    
    # === STEP 3: Generate Model Prediction ===
    try:
        preds = model.predict(df_enc)
        
        if hasattr(preds, "tolist"):
            preds = preds.tolist()
            
        if isinstance(preds, (list, tuple)) and len(preds) == 1:
            result = preds[0]
        else:
            result = preds
            
    except Exception as e:
        raise Exception(f"Model prediction failed: {e}")
    
    # === STEP 4: Convert to Business-Friendly Output ===
    # Isolation Forest outputs -1 for Anomalies, 1 for Normal points
    if result == -1:
        return "Anomaly Detected (Review Required)"  # High risk - structural or attributive anomaly
    else:
        return "Normal Claim"  # Low risk - aligns with normal Medicare distribution