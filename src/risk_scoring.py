"""Risk Scoring Service

Converts individual claim anomaly scores into provider-level risk scores (0-100%).
"""

from typing import List, Tuple, Optional
import numpy as np


def calculate_provider_risk_score(
    anomaly_scores: List[float],
    weighting_method: str = "weighted_average"
) -> Tuple[float, str]:
    """Main function: Calculate provider-level risk score from anomalies.
    
    Args:
        anomaly_scores: List of claim anomaly scores (0-100%)
        weighting_method: 'weighted_average' (default)
    
    Returns:
        Tuple of (risk_score: float 0-100, category: str)
    """
    if not anomaly_scores:
        return 0.0, "LOW"
    
    scores_array = np.array(anomaly_scores, dtype=float)
    
    if weighting_method == "weighted_average":
        weights = np.where(scores_array > 50, 1.5, 1.0)
        risk_score = np.average(scores_array, weights=weights)
    else:
        risk_score = np.mean(scores_array)
    
    risk_score = float(np.clip(risk_score, 0, 100))
    category = categorize_risk(risk_score)
    
    return risk_score, category


def categorize_risk(risk_score: float) -> str:
    """Categorize risk score into LOW/MEDIUM/HIGH.
    
    Args:
        risk_score: Risk score 0-100%
    
    Returns:
        Category: 'LOW', 'MEDIUM', or 'HIGH'
    """
    if risk_score < 30:
        return "LOW"
    elif risk_score < 70:
        return "MEDIUM"
    else:
        return "HIGH"


def get_routing_action(risk_score: float) -> str:
    """Get routing recommendation based on risk.
    
    Args:
        risk_score: Risk score 0-100%
    
    Returns:
        Action: 'AUTO_APPROVE', 'STANDARD_REVIEW', or 'CREATE_ALERT'
    """
    category = categorize_risk(risk_score)
    if category == "LOW":
        return "AUTO_APPROVE"
    elif category == "MEDIUM":
        return "STANDARD_REVIEW"
    else:
        return "CREATE_ALERT"


def get_risk_statistics(anomaly_scores: List[float]) -> dict:
    """Get detailed risk statistics.
    
    Args:
        anomaly_scores: List of claim anomaly scores
    
    Returns:
        Dict with statistics
    """
    if not anomaly_scores:
        return {
            "total_claims": 0,
            "mean_score": 0.0,
            "median_score": 0.0,
            "std_dev": 0.0,
            "min_score": 0.0,
            "max_score": 0.0,
            "anomaly_count": 0,
            "anomaly_rate": 0.0
        }
    
    scores_array = np.array(anomaly_scores, dtype=float)
    
    return {
        "total_claims": len(scores_array),
        "mean_score": float(np.mean(scores_array)),
        "median_score": float(np.median(scores_array)),
        "std_dev": float(np.std(scores_array)) if len(scores_array) > 1 else 0.0,
        "min_score": float(np.min(scores_array)),
        "max_score": float(np.max(scores_array)),
        "anomaly_count": int(np.sum(scores_array > 50)),
        "anomaly_rate": float(np.sum(scores_array > 50) / len(scores_array) * 100)
    }
