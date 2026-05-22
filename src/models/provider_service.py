"""
PROVIDER SERVICE - Provider Statistics Aggregation
====================================================

Aggregates individual claims into provider-level statistics.

Core Function:
    compile_provider_stats(provider_id, claims_df) → provider_stats_dict

This module takes a dataframe of claims and compiles them into meaningful statistics:
- Total claims count
- Total cost (sum of claim amounts)
- Unique patient count
- Average claim amount
- Anomaly count and rate
- Claim distribution (by diagnosis, by DRG, etc.)

Your partner's web app will call these functions to populate provider dashboards
and provide context to investigators when reviewing high-risk providers.

EXAMPLE USAGE:
    from src.models.provider_service import compile_provider_stats
    import pandas as pd
    
    # Your partner has a dataframe with columns:
    # ['PRVDR_NUM', 'DESYNPUF_ID', 'CLM_PMT_AMT', 'is_anomaly', ...]
    claims_df = pd.read_csv('claims.csv')
    
    # Get stats for a specific provider
    stats = compile_provider_stats(provider_id="1234", claims_df=claims_df)
    
    # stats dict contains:
    {
        "provider_id": "1234",
        "total_claims": 500,
        "total_cost": 2500000.50,
        "unique_patients": 150,
        "average_claim_amount": 5000.00,
        "anomaly_count": 25,
        "anomaly_rate": 5.0,
        "min_claim": 1000.00,
        "max_claim": 50000.00,
        "std_dev_claim": 8500.00
    }
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np


def compile_provider_stats(
    provider_id: str,
    claims_df: pd.DataFrame,
    anomaly_column: str = "is_anomaly"
) -> Dict:
    """
    Compile comprehensive statistics for a single provider.
    
    This function aggregates all claims for a provider and returns statistics
    useful for provider dashboards and investigator context.
    
    Args:
        provider_id (str): Provider ID to filter claims for
                          Example: "1234", "1002CR"
        
        claims_df (pd.DataFrame): Dataframe with all claims. Must have columns:
                                 - "PRVDR_NUM": Provider ID
                                 - "DESYNPUF_ID": Patient ID
                                 - "CLM_PMT_AMT" or "NCH_PRMRY_PYR_CLM_PD_AMT": Claim amount
                                 - "is_anomaly": Boolean or "NORMAL"/"ANOMALY"
        
        anomaly_column (str): Name of column indicating anomalies.
                             Can be boolean, string, or numeric (0/1, -1/1)
    
    Returns:
        dict: Provider statistics with keys:
            - "provider_id": The provider ID
            - "total_claims": Number of claims
            - "total_cost": Sum of all claim amounts
            - "unique_patients": Count of distinct patients
            - "average_claim_amount": Mean claim value
            - "median_claim_amount": Median claim value
            - "min_claim": Lowest claim amount
            - "max_claim": Highest claim amount
            - "std_dev_claim": Standard deviation of claim amounts
            - "anomaly_count": Number of flagged claims
            - "anomaly_rate": Percentage of flagged claims
            - "claim_count_by_status": Breakdown by claim status
    
    EXAMPLE:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'PRVDR_NUM': ['1234', '1234', '1234', '5678'],
        ...     'DESYNPUF_ID': ['P1', 'P2', 'P1', 'P3'],
        ...     'CLM_PMT_AMT': [5000, 8000, 3000, 6000],
        ...     'is_anomaly': [False, True, False, False]
        ... })
        >>> stats = compile_provider_stats('1234', df)
        >>> print(stats['total_claims'])
        3
        >>> print(stats['anomaly_rate'])
        33.33
    """
    
    # Filter claims for this provider
    provider_claims = claims_df[claims_df['PRVDR_NUM'] == provider_id].copy()
    
    # Handle case where provider not found
    if provider_claims.empty:
        return {
            "provider_id": provider_id,
            "total_claims": 0,
            "total_cost": 0.0,
            "unique_patients": 0,
            "average_claim_amount": 0.0,
            "median_claim_amount": 0.0,
            "min_claim": 0.0,
            "max_claim": 0.0,
            "std_dev_claim": 0.0,
            "anomaly_count": 0,
            "anomaly_rate": 0.0,
            "claim_count_by_status": {}
        }
    
    # Get claim amount column (try multiple possible names)
    claim_amount_col = None
    for col_name in ['CLM_PMT_AMT', 'NCH_PRMRY_PYR_CLM_PD_AMT', 'claim_amount', 'amount']:
        if col_name in provider_claims.columns:
            claim_amount_col = col_name
            break
    
    if claim_amount_col is None:
        raise ValueError(f"No claim amount column found. Available: {provider_claims.columns.tolist()}")
    
    # ========== CALCULATE STATISTICS ==========
    
    # 1. BASIC COUNTS
    total_claims = len(provider_claims)
    unique_patients = provider_claims['DESYNPUF_ID'].nunique()
    
    # 2. COST METRICS
    claim_amounts = pd.to_numeric(provider_claims[claim_amount_col], errors='coerce').dropna()
    total_cost = float(claim_amounts.sum())
    average_claim_amount = float(claim_amounts.mean())
    median_claim_amount = float(claim_amounts.median())
    min_claim = float(claim_amounts.min())
    max_claim = float(claim_amounts.max())
    std_dev_claim = float(claim_amounts.std())
    
    # 3. ANOMALY METRICS
    anomaly_count, anomaly_rate = _calculate_anomaly_stats(
        provider_claims, 
        anomaly_column
    )
    
    return {
        "provider_id": provider_id,
        "total_claims": int(total_claims),
        "total_cost": round(total_cost, 2),
        "unique_patients": int(unique_patients),
        "average_claim_amount": round(average_claim_amount, 2),
        "median_claim_amount": round(median_claim_amount, 2),
        "min_claim": round(min_claim, 2),
        "max_claim": round(max_claim, 2),
        "std_dev_claim": round(std_dev_claim, 2),
        "anomaly_count": int(anomaly_count),
        "anomaly_rate": round(anomaly_rate, 2)  # percentage
    }


def _calculate_anomaly_stats(claims_df: pd.DataFrame, anomaly_column: str) -> tuple:
    """
    Helper function to count anomalies regardless of format.
    
    Handles different anomaly representations:
    - Boolean: True/False
    - String: "NORMAL"/"ANOMALY", "normal"/"anomaly"
    - Numeric: 1/0, -1/1
    
    Args:
        claims_df: DataFrame with claim data
        anomaly_column: Name of anomaly column
    
    Returns:
        Tuple of (anomaly_count: int, anomaly_rate: float)
    """
    
    if anomaly_column not in claims_df.columns:
        return 0, 0.0
    
    anomalies = claims_df[anomaly_column]
    total = len(anomalies)
    
    if total == 0:
        return 0, 0.0
    
    # Count anomalies based on type
    if anomalies.dtype == bool or anomalies.dtype == 'object':
        # String or boolean
        if anomalies.dtype == 'object':
            anomaly_count = sum(str(x).upper() in ["ANOMALY", "TRUE", "-1", "1"] for x in anomalies)
        else:
            anomaly_count = sum(anomalies)
    else:
        # Numeric: count non-zero or positive values
        anomaly_count = sum(anomalies > 0)
    
    anomaly_rate = (anomaly_count / total) * 100
    
    return int(anomaly_count), round(anomaly_rate, 2)


def get_provider_risk_context(
    provider_id: str,
    claims_df: pd.DataFrame,
    diagnosis_column: str = "ADMTNG_ICD9_DGNS_CD"
) -> Dict:
    """
    Get rich context about a provider for investigator review.
    
    Returns more detailed breakdown useful when an investigator is reviewing
    a flagged provider. Includes diagnosis codes and patterns.
    
    Args:
        provider_id: Provider ID
        claims_df: DataFrame with all claims
        diagnosis_column: Name of diagnosis code column
    
    Returns:
        dict: Rich context including:
            - provider_stats: Basic stats from compile_provider_stats()
            - top_diagnoses: Most common diagnosis codes
            - diagnosis_count: Number of unique diagnoses
            - recent_claims: Latest claims (if has date column)
    
    EXAMPLE:
        >>> context = get_provider_risk_context('1234', df)
        >>> print("Provider has", context['provider_stats']['total_claims'], "claims")
        >>> print("Top diagnosis:", context['top_diagnoses'][0])
    """
    
    # Get basic stats
    provider_stats = compile_provider_stats(provider_id, claims_df)
    
    if provider_stats['total_claims'] == 0:
        return {"provider_stats": provider_stats, "warning": "Provider not found"}
    
    # Filter for this provider
    provider_claims = claims_df[claims_df['PRVDR_NUM'] == provider_id]
    
    # Get diagnosis breakdown
    top_diagnoses = []
    diagnosis_count = 0
    
    if diagnosis_column in provider_claims.columns:
        diagnosis_counts = provider_claims[diagnosis_column].value_counts()
        diagnosis_count = len(diagnosis_counts)
        top_diagnoses = diagnosis_counts.head(5).to_dict()
    
    return {
        "provider_stats": provider_stats,
        "top_diagnoses": top_diagnoses,
        "diagnosis_count": diagnosis_count
    }


def compare_providers(
    provider_ids: list,
    claims_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compare multiple providers' statistics side-by-side.
    
    Useful for the web app to show provider comparison dashboard.
    
    Args:
        provider_ids: List of provider IDs to compare
        claims_df: DataFrame with all claims
    
    Returns:
        pd.DataFrame: One row per provider with comparison metrics
    
    EXAMPLE:
        >>> providers = ['1234', '5678', '9012']
        >>> comparison = compare_providers(providers, df)
        >>> print(comparison[['provider_id', 'total_claims', 'anomaly_rate']])
    """
    
    results = []
    
    for provider_id in provider_ids:
        stats = compile_provider_stats(provider_id, claims_df)
        results.append(stats)
    
    return pd.DataFrame(results)
