import pandas as pd
import numpy as np


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:

    # 1. Drop Empty Columns
    # Find all columns that start with 'HCPCS_CD_'
    hcpcs_cols_to_drop = [col for col in df.columns if col.startswith('HCPCS_CD_')]
    df.drop(columns=hcpcs_cols_to_drop, inplace=True)

    # 2. Handle Missing Values
    # Drop rows with missing essential identifiers/dates
    essential_cols = ['CLM_ID', 'DESYNPUF_ID', 'CLM_FROM_DT', 'CLM_THRU_DT', 'PRVDR_NUM', 'CLM_UTLZTN_DAY_CNT']
    df.dropna(subset=essential_cols, inplace=True)

    # Fill missing optional NPIs with a placeholder string 'missing'
    npi_cols = ['AT_PHYSN_NPI', 'OP_PHYSN_NPI', 'OT_PHYSN_NPI']
    df[npi_cols] = df[npi_cols].fillna('missing')

    # 3. Correct Data Types
    # Convert date columns from float/int to datetime objects
    date_cols = ['CLM_FROM_DT', 'CLM_THRU_DT', 'CLM_ADMSN_DT', 'NCH_BENE_DSCHRG_DT']
    for col in date_cols:
        # The conversion handles both float and int types by first casting to int, then to string.
        df[col] = pd.to_datetime(df[col].astype(int).astype(str), format='%Y%m%d')

    # Convert NPI columns to string type, handling the mixed types (float and our 'missing' string)
    for col in npi_cols:
        df[col] = df[col].apply(lambda x: str(int(x)) if x != 'missing' else x)

    # 4. Address Invalid Values
    # Correct negative payment amounts by taking the absolute value
    df['CLM_PMT_AMT'] = df['CLM_PMT_AMT'].abs()

    # 5. Reset Index
    # This is crucial for ensuring a clean, contiguous index for mapping later on.
    df.reset_index(drop=True, inplace=True)

    return df

def inject_anomalies(df: pd.DataFrame, anomaly_fraction: float = 0.02) -> pd.DataFrame:
    # The df DataFrame is our source of normal data.
    # We will create a copy to inject anomalies into.
    df_polluted = df.copy()

    # --- Anomaly Injection Parameters ---
    n_anomalies = int(len(df_polluted) * anomaly_fraction)

    # Get a set of random, unique indices to inject anomalies into. These indices are our ground truth.
    anomaly_indices = np.random.choice(df_polluted.index, n_anomalies, replace=False)

    # Split the anomaly indices for the two different types of anomalies
    n_attributive = n_anomalies // 2
    n_structural = n_anomalies - n_attributive
    attributive_indices = anomaly_indices[:n_attributive]
    structural_indices = anomaly_indices[n_attributive:]


    # --- 1. Inject Attributive Anomalies into df_polluted ---
    # Anomaly: Unusually high payment amounts.
    # We calculate a high outlier value based on the clean data's distribution.
    high_payment_threshold = df['NCH_PRMRY_PYR_CLM_PD_AMT'].quantile(0.99)
    # Generate random outlier payments that are clearly above the normal threshold.
    random_outlier_payments = high_payment_threshold * (1.5 + np.random.rand(n_attributive) * 2) # between 1.5x and 3.5x the threshold
    df_polluted.loc[attributive_indices, 'NCH_PRMRY_PYR_CLM_PD_AMT'] = random_outlier_payments

    # --- 2. Inject Structural Anomalies into df_polluted ---
    # Anomaly: A rare provider colluding with a new, fraudulent physician.
    # Find one of the rarest providers from the clean data.
    rarest_provider_str = df['PRVDR_NUM'].value_counts().idxmin()
    # Create a new, unique NPI to represent a fraudulent physician not seen in the clean data.
    fraudulent_physician_npi = 'FRAUD_NPI_1'
    # Assign the structural anomaly claims to this rare provider and fraudulent physician.
    df_polluted.loc[structural_indices, 'PRVDR_NUM'] = rarest_provider_str
    df_polluted.loc[structural_indices, 'AT_PHYSN_NPI'] = fraudulent_physician_npi
 

    # --- 3. Create the Ground-Truth Anomaly Label ---
    # Create a Series of zeros with the same index as our DataFrames.
    anomaly_label = pd.Series(0, index=df.index)
    # Set the value to 1 at the indices where we injected anomalies.
    anomaly_label.loc[anomaly_indices] = 1

    return df_polluted