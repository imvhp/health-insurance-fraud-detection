"""
EXPLAINABILITY SERVICE - SHAP Explanations for Investigator Review
===================================================================

Generates human-readable explanations of WHY claims/providers are flagged.

Core Functions:
    generate_claim_explanation(claim_data, model) → explanation_dict
    summarize_provider_explanations(provider_id, claims_df, model) → summary_dict

This module uses SHAP (SHapley Additive exPlanations) to break down model predictions
into feature contributions. Investigators need to understand WHY a provider was flagged,
not just that they were flagged.

EXAMPLE USAGE:
    from src.models.explainability import generate_claim_explanation
    import shap
    
    # After model prediction, explain it
    explanation = generate_claim_explanation(claim_features, model)
    
    # explanation dict contains:
    {
        "top_factors": [
            {"feature": "claim_amount", "impact": 0.35, "direction": "high"},
            {"feature": "drg_code", "impact": 0.25, "direction": "unusual"}
        ],
        "summary": "Flagged because claim amount is 3x average (0.35 impact) + 
                   rare DRG code (0.25 impact)",
        "confidence": 0.92
    }
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Note: shap library is imported but optional
# If not available, we provide fallback explanations
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP library not installed. Fallback explanations only.")


def generate_claim_explanation(
    claim_data: Dict,
    model,
    feature_names: List[str] = None,
    use_shap: bool = True
) -> Dict:
    """
    Generate explanation for why a single claim was flagged.
    
    This is what investigators see when reviewing a high-risk claim.
    It explains WHICH FEATURES contributed to the anomaly score.
    
    Args:
        claim_data (dict): Claim features as dictionary
                          Keys should match model training features
                          Example: {
                              "claim_amount": 25000,
                              "drg_code": 470,
                              "diagnosis_code": "78650",
                              "hospital_days": 8
                          }
        
        model: Trained ML model (Isolation Forest)
        
        feature_names (list): Names of features (for readability)
                             If None, uses dict keys
        
        use_shap (bool): Use SHAP for explanations (True)
                        or fallback heuristic explanations (False)
    
    Returns:
        dict: Explanation with keys:
            - "top_factors": List of most impactful features
            - "summary": Human-readable explanation text
            - "confidence": How confident we are in the explanation (0-1)
            - "raw_score": Raw model anomaly score
            - "percentile": What percentile is this claim in?
    
    EXAMPLE:
        >>> claim = {
        ...     "claim_amount": 25000,
        ...     "days_in_hospital": 10,
        ...     "drg_code": 999  # rare code
        ... }
        >>> explain = generate_claim_explanation(claim, model)
        >>> print(explain['summary'])
        "Claim is anomalous due to: extremely high claim amount (70% impact), 
         unusual DRG code (20% impact)"
    """
    
    # Convert dict to proper format for model
    if isinstance(claim_data, dict):
        claim_array = np.array([list(claim_data.values())])
        if feature_names is None:
            feature_names = list(claim_data.keys())
    else:
        claim_array = np.array([claim_data])
    
    # Get model prediction
    try:
        raw_score = model.predict(claim_array)[0]
        anomaly_scores = model.score_samples(claim_array)[0]
    except Exception as e:
        return {
            "error": str(e),
            "summary": "Unable to generate explanation",
            "confidence": 0.0
        }
    
    # Generate explanation
    if use_shap and SHAP_AVAILABLE:
        explanation = _generate_shap_explanation(
            claim_array, 
            model, 
            feature_names
        )
    else:
        # Fallback: use heuristic explanation
        explanation = _generate_heuristic_explanation(
            claim_data, 
            feature_names,
            raw_score
        )
    
    return explanation


def _generate_shap_explanation(
    claim_array: np.ndarray,
    model,
    feature_names: List[str]
) -> Dict:
    """
    Use SHAP library to generate feature importance.
    
    SHAP provides principled way to explain model predictions.
    Each feature gets a "SHAP value" showing its contribution.
    
    Args:
        claim_array: Numpy array of claim features
        model: Trained Isolation Forest model
        feature_names: Names of features
    
    Returns:
        dict: SHAP-based explanation
    """
    
    try:
        # Create SHAP explainer for tree-based model
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(claim_array)
        
        # Get base value (average prediction)
        base_value = explainer.expected_value
        
        # Calculate feature importances (absolute SHAP values)
        feature_importance = np.abs(shap_values[0])
        
        # Sort by importance
        sorted_indices = np.argsort(feature_importance)[::-1]
        
        # Build factor list
        top_factors = []
        for idx in sorted_indices[:3]:  # Top 3 factors
            factor_name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
            impact = float(feature_importance[idx])
            direction = "high" if shap_values[0][idx] > 0 else "low"
            value = float(claim_array[0][idx])
            
            top_factors.append({
                "feature": factor_name,
                "impact": round(impact, 3),
                "direction": direction,
                "value": round(value, 2)
            })
        
        # Generate summary text
        summary_parts = []
        for factor in top_factors:
            summary_parts.append(
                f"{factor['feature']} is {factor['direction']} "
                f"(value: {factor['value']}, impact: {factor['impact']})"
            )
        
        summary = "Flagged due to: " + " + ".join(summary_parts)
        confidence = min(sum(f['impact'] for f in top_factors), 1.0)
        
        return {
            "top_factors": top_factors,
            "summary": summary,
            "confidence": round(confidence, 2),
            "method": "SHAP"
        }
        
    except Exception as e:
        # If SHAP fails, fall back to heuristic
        print(f"SHAP explanation failed: {e}. Using heuristic fallback.")
        return _generate_heuristic_explanation(
            dict(zip(feature_names, claim_array[0])),
            feature_names,
            None
        )


def _generate_heuristic_explanation(
    claim_data: Dict,
    feature_names: List[str],
    score: Optional[float] = None
) -> Dict:
    """
    Generate simple heuristic explanation (when SHAP unavailable).
    
    Uses rule-based logic to identify anomalous features without
    formal SHAP analysis.
    
    Args:
        claim_data: Dict of claim features
        feature_names: Feature names
        score: Model anomaly score
    
    Returns:
        dict: Simple explanation
    """
    
    # Check for common anomalies
    anomaly_flags = []
    
    for feature, value in claim_data.items():
        if "amount" in feature.lower() and value > 20000:
            anomaly_flags.append(f"{feature} is unusually high ({value})")
        
        if "drg" in feature.lower() or "diagnosis" in feature.lower():
            # These would be rare codes
            if isinstance(value, (int, float)) and value > 500:
                anomaly_flags.append(f"{feature} is uncommon ({value})")
    
    summary = "Flagged due to: " + " + ".join(anomaly_flags) if anomaly_flags else "Multiple unusual characteristics"
    
    return {
        "top_factors": [
            {"feature": flag.split()[0], "impact": 0.5, "direction": "unusual"}
            for flag in anomaly_flags
        ],
        "summary": summary,
        "confidence": 0.6,  # Lower confidence for heuristic
        "method": "Heuristic"
    }


def summarize_provider_explanations(
    provider_id: str,
    claims_df: pd.DataFrame,
    anomaly_scores: np.ndarray,
    model = None,
    top_n: int = 3
) -> Dict:
    """
    Summarize top risk factors across ALL claims for a provider.
    
    When investigator reviews a HIGH-risk provider, they need to know:
    "Why is this provider risky? What are the patterns?"
    
    This function aggregates individual claim explanations to show
    the MOST COMMON reasons for flagging.
    
    Args:
        provider_id: Provider to explain
        claims_df: All claims dataframe
        anomaly_scores: Array of anomaly scores for each claim
        model: ML model (optional, for SHAP analysis)
        top_n: Number of top factors to return
    
    Returns:
        dict: Summary of provider risk factors
            - "top_risk_factors": Most common features in anomalies
            - "anomaly_patterns": Patterns in the flagged claims
            - "summary": Overall explanation text
    
    EXAMPLE:
        >>> summary = summarize_provider_explanations('1234', claims_df, scores)
        >>> print(summary['summary'])
        "Provider is risky because:"
        "- 70% of anomalies have unusually high claim amounts"
        "- 40% have rare DRG codes"
        "- Pattern suggests systematic overbilling"
    """
    
    # Filter claims for provider
    provider_claims = claims_df[claims_df['PRVDR_NUM'] == provider_id].copy()
    
    if provider_claims.empty:
        return {"error": f"Provider {provider_id} not found"}
    
    # Identify which claims are anomalies
    high_anomaly_threshold = 50  # Top 50% of scores
    anomaly_mask = anomaly_scores >= high_anomaly_threshold
    
    flagged_claims = provider_claims[anomaly_mask]
    
    if flagged_claims.empty:
        return {"summary": "No anomalous claims to explain"}
    
    # Analyze patterns
    risk_factors = _extract_risk_patterns(flagged_claims)
    
    # Build summary
    summary_lines = [
        f"Provider {provider_id} is flagged for:",
    ]
    
    for factor, prevalence in risk_factors[:top_n]:
        summary_lines.append(f"- {factor} (present in {prevalence:.0f}% of flagged claims)")
    
    summary = "\n".join(summary_lines)
    
    return {
        "provider_id": provider_id,
        "top_risk_factors": risk_factors[:top_n],
        "flagged_claim_count": len(flagged_claims),
        "total_claims": len(provider_claims),
        "flagged_percentage": round((len(flagged_claims) / len(provider_claims)) * 100, 1),
        "summary": summary
    }


def _extract_risk_patterns(claims_df: pd.DataFrame) -> List[Tuple[str, float]]:
    """
    Extract common patterns from flagged claims.
    
    Returns list of (pattern_description, prevalence_percentage) tuples
    sorted by prevalence.
    """
    
    patterns = []
    total = len(claims_df)
    
    if total == 0:
        return patterns
    
    # Check for high claim amounts
    if 'CLM_PMT_AMT' in claims_df.columns or 'NCH_PRMRY_PYR_CLM_PD_AMT' in claims_df.columns:
        amount_col = 'CLM_PMT_AMT' if 'CLM_PMT_AMT' in claims_df.columns else 'NCH_PRMRY_PYR_CLM_PD_AMT'
        mean_amount = pd.to_numeric(claims_df[amount_col], errors='coerce').mean()
        high_amount = claims_df[pd.to_numeric(claims_df[amount_col], errors='coerce') > mean_amount * 1.5]
        if len(high_amount) > 0:
            patterns.append(("Unusually high claim amounts", (len(high_amount) / total) * 100))
    
    # Check for unusual utilization
    if 'CLM_UTLZTN_DAY_CNT' in claims_df.columns:
        mean_days = pd.to_numeric(claims_df['CLM_UTLZTN_DAY_CNT'], errors='coerce').mean()
        high_days = claims_df[pd.to_numeric(claims_df['CLM_UTLZTN_DAY_CNT'], errors='coerce') > mean_days * 1.5]
        if len(high_days) > 0:
            patterns.append(("Unusually long hospital stays", (len(high_days) / total) * 100))
    
    # Check for rare diagnosis codes
    if 'ADMTNG_ICD9_DGNS_CD' in claims_df.columns:
        diagnosis_counts = claims_df['ADMTNG_ICD9_DGNS_CD'].value_counts()
        rare_diagnoses = diagnosis_counts[diagnosis_counts < 5].index
        rare_claims = claims_df[claims_df['ADMTNG_ICD9_DGNS_CD'].isin(rare_diagnoses)]
        if len(rare_claims) > 0:
            patterns.append(("Rare or unusual diagnosis codes", (len(rare_claims) / total) * 100))
    
    # Sort by prevalence (highest first)
    patterns.sort(key=lambda x: x[1], reverse=True)
    
    return patterns
