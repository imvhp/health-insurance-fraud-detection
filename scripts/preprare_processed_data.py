import os, sys
import pandas as pd

# Make src importable from the root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import your custom modules
from src.data.preprocess import preprocess_data, inject_anomalies
from src.features.build_features import build_feature_pipeline 
# (Adjust the import paths above based on exactly what you named your files)

# Define file paths
RAW = "data/raw/DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv"
OUT = "data/processed/claims_processed.csv"

def main():
    print("🚀 Starting Data Processing Pipeline...")

    # 1) Load raw data
    print(f"\n[1/5] Loading raw data from {RAW}...")
    df_raw = pd.read_csv(RAW)

    # 2) Preprocess data (drops empty columns, fixes dates, handles NaNs)
    print(f"\n[2/5] Cleaning and preprocessing data...")
    df_clean = preprocess_data(df_raw)

    # 3) Inject anomalies (creates our "ground truth" labels)
    print(f"\n[3/5] Injecting anomalies (Fraction: 0.02)...")
    # Make sure your inject_anomalies function ends with: return df_polluted, anomaly_label
    df_polluted, anomaly_label = inject_anomalies(df_clean, anomaly_fraction=0.02)

    # Sanity checks for the labels
    assert anomaly_label.isna().sum() == 0, "anomaly_label contains NaNs!"
    assert set(anomaly_label.unique()) <= {0, 1}, "anomaly_label contains values other than 0 and 1!"
    print(f"      Anomaly injection successful. {anomaly_label.sum()} anomalies created.")

    # 4) Build features (selects columns, LabelEncodes, and merges labels)
    print(f"\n[4/5] Building features for Isolation Forest...")
    df_final = build_feature_pipeline(df_polluted, anomaly_label)

    # Sanity checks for the final dataset
    assert 'anomaly_label' in df_final.columns, "Target column 'anomaly_label' is missing!"
    assert df_final.isnull().sum().sum() == 0, "Missing values found in the final dataset!"

    # 5) Save processed dataset
    print(f"\n[5/5] Saving processed dataset...")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df_final.to_csv(OUT, index=False)
    
    print(f"\n✅ Pipeline Complete! Processed dataset saved to {OUT} | Shape: {df_final.shape}")

if __name__ == "__main__":
    main()