import os

from fastapi import encoders
import joblib
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def select_baseline_features(df_polluted: pd.DataFrame) -> pd.DataFrame:
    """
    Step 1 & 2: Selects the specific subset of features required for the 
    Isolation Forest model based on the reference paper.
    """
    baseline_features = [
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
    
    df_baseline = df_polluted[baseline_features].copy()
    print(f"Created baseline DataFrame with {df_baseline.shape[1]} selected features.")
    return df_baseline

def encode_categorical_features(
    df: pd.DataFrame,
    encoders=None
) -> pd.DataFrame:

    categorical_cols = [
        'PRVDR_NUM', 'AT_PHYSN_NPI', 'OP_PHYSN_NPI', 'OT_PHYSN_NPI',
        'ADMTNG_ICD9_DGNS_CD', 'CLM_DRG_CD', 'ICD9_PRCDR_CD_1'
    ]

    df_encoded = df.copy()

    # Training mode: create encoders
    if encoders is None:
        encoders = {}

        for col in categorical_cols:
            df_encoded[col] = df_encoded[col].fillna('UNKNOWN').astype(str)

            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])

            encoders[col] = le

        os.makedirs("models/encoders", exist_ok=True)
        joblib.dump(encoders, "models/encoders/label_encoders.pkl")

    # Inference/training with saved encoders
    else:
        for col in categorical_cols:
            df_encoded[col] = (
                df_encoded[col]
                .fillna('UNKNOWN')
                .astype(str)
            )

            df_encoded[col] = encoders[col].transform(
                df_encoded[col]
            )

    return df_encoded

def finalize_baseline_dataset(df_encoded: pd.DataFrame, anomaly_label: pd.Series) -> pd.DataFrame:
    """
    Step 4: Merges features with ground-truth labels and shuffles the data 
    to ensure randomness for model evaluation.
    """
    # Add the ground truth label
    df_encoded['anomaly_label'] = anomaly_label
    
    # Shuffle the dataset (frac=1 shuffles all rows)
    df_final = df_encoded.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("\n--- Baseline DataFrame for Isolation Forest is ready ---")
    print(f"Final Shape: {df_final.shape}")
    return df_final

def build_feature_pipeline(df_polluted: pd.DataFrame, anomaly_label: pd.Series, encoders=None) -> pd.DataFrame:
    """
    Orchestration function to run the full feature engineering flow.
    """
    df_selected = select_baseline_features(df_polluted)
    df_encoded = encode_categorical_features(
        df_selected,
        encoders
    )
    df_final = finalize_baseline_dataset(df_encoded, anomaly_label)
    
    return df_final

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