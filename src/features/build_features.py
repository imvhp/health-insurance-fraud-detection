import os
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# === CONSTANTS ===
BASELINE_FEATURES = [
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


# ============================================================
# SHARED HELPER
# ============================================================

def select_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """Selects the 9 features required for Isolation Forest."""
    df_baseline = df[BASELINE_FEATURES].copy()
    print(f"Selected {df_baseline.shape[1]} baseline features.")
    return df_baseline


# ============================================================
# CLEAN DATA PATH  (fit + transform + save encoders)
# ============================================================

def build_clean_feature_pipeline(
    df_clean: pd.DataFrame,
    encoder_save_path: str = "models/encoders/label_encoders.pkl"
) -> tuple[pd.DataFrame, dict]:
    """
    Used on CLEAN data only (before anomaly injection).

    Steps:
      1. Select 9 baseline features
      2. Fit a LabelEncoder per categorical column
      3. Save encoders to disk for later use on polluted data
      4. Return encoded DataFrame (no target column) + encoder dict

    Args:
        df_clean:          Clean preprocessed DataFrame (no injected anomalies)
        encoder_save_path: Where to persist the fitted encoders

    Returns:
        X_clean:   Encoded feature matrix ready for IF training
        encoders:  Dict of {col_name: fitted LabelEncoder}
    """
    df_selected = select_baseline_features(df_clean)
    df_encoded = df_selected.copy()
    encoders = {}

    for col in CATEGORICAL_COLS:
        df_encoded[col] = df_encoded[col].fillna('UNKNOWN').astype(str)

        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        encoders[col] = le

    # Persist encoders so polluted transform uses identical mappings
    os.makedirs(os.path.dirname(encoder_save_path), exist_ok=True)
    joblib.dump(encoders, encoder_save_path)
    print(f"Encoders fitted and saved → {encoder_save_path}")
    print(f"Clean feature matrix shape: {df_encoded.shape}")

    return df_encoded, encoders


# ============================================================
# POLLUTED DATA PATH  (transform-only using saved encoders)
# ============================================================

def build_polluted_feature_pipeline(
    df_polluted: pd.DataFrame,
    anomaly_label: pd.Series,
    encoders: dict
) -> pd.DataFrame:
    """
    Used on POLLUTED data (after anomaly injection).

    Steps:
      1. Select 9 baseline features
      2. Transform categoricals using the SAME encoders fitted on clean data
         → Unseen values (e.g. FRAUD_NPI_1) get mapped to -1
      3. Attach ground-truth anomaly_label for evaluation

    Args:
        df_polluted:   DataFrame after inject_anomalies()
        anomaly_label: Ground-truth Series (0 = normal, 1 = anomaly)
        encoders:      Dict returned by build_clean_feature_pipeline()

    Returns:
        df_final: Encoded features + anomaly_label column
    """
    df_selected = select_baseline_features(df_polluted)
    df_encoded = df_selected.copy()

    for col in CATEGORICAL_COLS:
        df_encoded[col] = df_encoded[col].fillna('UNKNOWN').astype(str)
        le = encoders[col]

        # 1. Build an O(1) dictionary mapping from the encoder classes
        mapping_dict = {category: idx for idx, category in enumerate(le.classes_)}
        
        # 2. Map values using optimized pandas vectorization. 
        # Unseen entries (like FRAUD_NPI_1) will automatically become NaN, then filled with -1.
        df_encoded[col] = df_encoded[col].map(mapping_dict).fillna(-1).astype(int)

    # Attach ground-truth label — used only for evaluation, never for training
    df_encoded['anomaly_label'] = anomaly_label.values

    print(f"Polluted feature matrix shape: {df_encoded.shape}")
    print(f"Anomalies in polluted set:     {anomaly_label.sum()} / {len(anomaly_label)}")

    return df_encoded


# ============================================================
# CONVENIENCE LOADER  (if encoders need to be reloaded from disk)
# ============================================================

def load_encoders(
    encoder_path: str = "models/encoders/label_encoders.pkl"
) -> dict:
    """Loads pre-fitted encoders from disk."""
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(
            f"Encoders not found at {encoder_path}. "
            "Run build_clean_feature_pipeline() first."
        )
    encoders = joblib.load(encoder_path)
    print(f"Encoders loaded from {encoder_path}")
    return encoders