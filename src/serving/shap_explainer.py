"""
SHAP Explanation Module for PTL Anomaly Detection
==================================================

Generates per-claim feature-level explanations using SHAP.
Adapted from: Health Insurance Fraud Detection / src / models / explainability.py

Returned explanation dict — matches the existing API contract:
  {
      "top_factors": [
          {
              "feature":   str,    # feature name
              "impact":    float,  # |SHAP value| or heuristic magnitude
              "direction": str,    # "increases_risk" | "decreases_risk"
              "value":     float   # processed (scaled) feature value
          },
          ...                      # top-5 by impact
      ],
      "summary":     str,          # "CLAIM FLAGGED (X% Risk). <explanation>"
      "confidence":  float,        # 0-1; lower for heuristic fallback
      "method":      str,          # "SHAP" | "Heuristic"
      "provider_id": str           # passed through from inference pipeline
  }
"""

from __future__ import annotations

import warnings
import numpy as np

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

# Models that have a tree structure SHAP can exploit natively.
_TREE_MODELS = ("IsolationForest",)

# How many background samples for KernelExplainer (kept small to limit latency).
_KERNEL_BG_SAMPLES = 50

# Maximum features shown in top_factors.
_TOP_N = 5


def explain_claim(
    model_name: str,
    clf,
    x_row: np.ndarray,
    feature_names: list[str],
    risk_percentage: float = 0.0,
    provider_id: str = "UNKNOWN",
    background: np.ndarray | None = None,
) -> dict:
    """
    Generate a SHAP explanation for a single claim observation.

    Args:
        model_name:      Name of the active model.
        clf:             Fitted model object.
        x_row:           1-D numpy array — preprocessed feature vector for ONE claim.
        feature_names:   List of feature names matching preprocessor output columns.
        risk_percentage: risk_pct value from classify_risk (0-100), embedded in summary.
        provider_id:     Provider identifier, echoed in the returned dict.
        background:      Optional 2-D background samples for KernelExplainer.

    Returns:
        Explanation dict (see module docstring for schema).
    """
    if not _SHAP_AVAILABLE:
        return _heuristic_explanation(x_row, feature_names, risk_percentage, provider_id)

    x_2d = x_row.reshape(1, -1)

    try:
        if model_name in _TREE_MODELS:
            return _shap_tree(clf, x_2d, feature_names, risk_percentage, provider_id)
        else:
            return _shap_kernel(clf, x_2d, feature_names, risk_percentage, provider_id, background)
    except Exception as exc:
        warnings.warn(f"[shap_explainer] SHAP failed ({exc}); using heuristic fallback.")
        return _heuristic_explanation(x_row, feature_names, risk_percentage, provider_id)


# ─────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────

def _build_top_factors(
    shap_values_1d: np.ndarray,
    x_row_1d: np.ndarray,
    feature_names: list[str],
) -> list[dict]:
    """Convert raw SHAP values into a sorted top-factor list matching API schema."""
    abs_vals = np.abs(shap_values_1d)
    top_idx = np.argsort(abs_vals)[::-1][:_TOP_N]

    factors = []
    for idx in top_idx:
        sv = float(shap_values_1d[idx])
        factors.append({
            "feature":   feature_names[idx] if idx < len(feature_names) else f"feature_{idx}",
            "impact":    round(abs(sv), 3),
            "direction": "high anomaly risk" if sv > 0 else "low anomaly risk",
            "value":     round(float(x_row_1d[idx]), 2),
        })
    return factors


def _build_summary(risk_percentage: float, top_factors: list[dict]) -> str:
    """Build summary string matching the existing API format."""
    if risk_percentage <= 50.0:
        return "Claim falls within normal distribution parameters. Not flagged."
        
    risk_str = f"CLAIM FLAGGED ({risk_percentage:.1f}% Risk)."
    if not top_factors:
        return f"{risk_str} Multiple unusual characteristics"
    parts = [
        f"{f['feature']} {f['direction']} (value: {f['value']}, impact: {f['impact']})"
        for f in top_factors[:3]
    ]
    return f"{risk_str} Key driving factors: " + " | ".join(parts)


def _build_confidence(top_factors: list[dict]) -> float:
    """Confidence = min(sum of top impacts, 1.0). Same logic as reference project."""
    return round(min(sum(f["impact"] for f in top_factors), 1.0), 2)


def _wrap(
    top_factors: list[dict],
    risk_percentage: float,
    provider_id: str,
    method: str,
    confidence: float | None = None,
) -> dict:
    """Assemble the final explanation dict."""
    if confidence is None:
        confidence = _build_confidence(top_factors)
    return {
        "top_factors": top_factors,
        "summary":     _build_summary(risk_percentage, top_factors),
        "confidence":  confidence,
        "method":      method,
        "provider_id": provider_id,
    }


# ─────────────────────────────────────────────
# SHAP strategies
# ─────────────────────────────────────────────

def _shap_tree(
    clf,
    x_2d: np.ndarray,
    feature_names: list[str],
    risk_percentage: float,
    provider_id: str,
) -> dict:
    """SHAP TreeExplainer — fast and exact for IsolationForest."""
    explainer = shap.TreeExplainer(clf)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_values = explainer.shap_values(x_2d)

    if isinstance(shap_values, list):
        sv = np.array(shap_values[0]).ravel()
    else:
        sv = np.array(shap_values).ravel()

    # IsolationForest: inference.py negates decision_function so higher = more anomalous.
    # Mirror that sign flip so "increases_risk" aligns with the final risk score direction.
    sv = -sv

    top_factors = _build_top_factors(sv, x_2d[0], feature_names)
    return _wrap(top_factors, risk_percentage, provider_id, method="SHAP")


def _shap_kernel(
    clf,
    x_2d: np.ndarray,
    feature_names: list[str],
    risk_percentage: float,
    provider_id: str,
    background: np.ndarray | None,
) -> dict:
    """
    SHAP KernelExplainer — model-agnostic for CBLOF, OCSVM, ECOD.
    Wraps clf.decision_function so SHAP treats it as a black-box scorer.
    """
    is_sklearn_sign = hasattr(clf, "offset_")   # True for OCSVM (pos=normal)

    def _predict_fn(X: np.ndarray) -> np.ndarray:
        scores = clf.decision_function(X).ravel().astype(float)
        if is_sklearn_sign:
            scores = -scores    # unify: higher = more anomalous
        return scores

    if background is not None and len(background) > 0:
        bg = background[:_KERNEL_BG_SAMPLES]
    else:
        # Build a synthetic background centered at the training mean.
        # After StandardScaler, zeros = mean of training data (true "normal" baseline).
        # We add small Gaussian noise so KernelExplainer can integrate over a
        # realistic distribution instead of a single degenerate point.
        n_features = x_2d.shape[1]
        rng = np.random.default_rng(42)
        bg = rng.normal(loc=0.0, scale=0.1, size=(_KERNEL_BG_SAMPLES, n_features))

    explainer = shap.KernelExplainer(_predict_fn, bg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_values = explainer.shap_values(x_2d, nsamples=128, silent=True)

    sv = np.array(shap_values).ravel()
    top_factors = _build_top_factors(sv, x_2d[0], feature_names)
    return _wrap(top_factors, risk_percentage, provider_id, method="SHAP")



# ─────────────────────────────────────────────
# Heuristic fallback (no SHAP dependency)
# ─────────────────────────────────────────────

def _heuristic_explanation(
    x_row: np.ndarray,
    feature_names: list[str],
    risk_percentage: float,
    provider_id: str,
) -> dict:
    """
    Fallback when SHAP is unavailable or fails.

    Ranks features by absolute scaled value. Only features with magnitude > 2.0
    (i.e., more than 2 standard deviations from the mean) are included.
    If no feature stands out, returns an empty top_factors list and a default summary,
    matching the reference project's behavior.
    """
    abs_vals = np.abs(x_row)
    # Only consider features with significant deviation
    significant_idx = np.where(abs_vals > 2.0)[0]
    
    top_factors = []
    if len(significant_idx) > 0:
        # Sort significant indices by magnitude descending
        sorted_idx = significant_idx[np.argsort(abs_vals[significant_idx])[::-1]][:_TOP_N]
        for idx in sorted_idx:
            val = float(x_row[idx])
            top_factors.append({
                "feature":   feature_names[idx] if idx < len(feature_names) else f"feature_{idx}",
                "impact":    round(float(abs_vals[idx]), 3),
                "direction": "increases anomaly risk" if val > 0 else "decreases anomaly risk",
                "value":     round(val, 2),
            })

    return _wrap(
        top_factors,
        risk_percentage,
        provider_id,
        method="Heuristic",
        confidence=0.6,     # fixed lower confidence matching reference project
    )
