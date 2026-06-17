"""Heuristic and metric-based model selection for anomaly detection."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd


MODEL_NAMES = ["IsolationForest", "CBLOF", "OCSVM", "ECOD"]


@dataclass
class DataProfile:
    n_rows: int
    n_features: int
    n_numeric: int
    n_categorical: int
    categorical_ratio: float
    missing_ratio: float
    contamination: float
    has_labels: bool


def build_data_profile(
    X_df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    contamination: float,
    has_labels: bool,
) -> DataProfile:
    """Summarize the dataset characteristics used by model-selection rules."""
    n_rows = int(len(X_df))
    n_features = int(len(numeric_cols) + len(categorical_cols))
    n_numeric = int(len(numeric_cols))
    n_categorical = int(len(categorical_cols))
    categorical_ratio = n_categorical / max(n_features, 1)
    missing_ratio = float(X_df.isna().to_numpy().mean()) if n_rows and n_features else 0.0

    return DataProfile(
        n_rows=n_rows,
        n_features=n_features,
        n_numeric=n_numeric,
        n_categorical=n_categorical,
        categorical_ratio=float(categorical_ratio),
        missing_ratio=missing_ratio,
        contamination=float(contamination),
        has_labels=bool(has_labels),
    )


def _add_reason(reasons: list[str], condition: bool, text: str) -> None:
    if condition:
        reasons.append(text)


def score_models_for_profile(profile: DataProfile) -> pd.DataFrame:
    """
    Score how suitable each model is for a dataset before training.

    The scores are intentionally heuristic. They are used when no trusted labels
    are available, and are superseded by evaluation metrics when labels exist.
    """
    rows = []

    for model in MODEL_NAMES:
        score = 50.0
        reasons: list[str] = []

        if model == "IsolationForest":
            score += 18
            _add_reason(reasons, True, "good general-purpose baseline for mixed claim data")
            if profile.n_rows >= 10000:
                score += 8
                reasons.append("scales well to medium/large datasets")
            if profile.n_features >= 30:
                score += 5
                reasons.append("handles moderately high-dimensional feature sets")
            if profile.categorical_ratio > 0.30:
                score += 4
                reasons.append("works reasonably after frequency encoding categorical codes")

        elif model == "CBLOF":
            score += 12
            _add_reason(reasons, True, "useful when anomalies are small clusters far from large clusters")
            if 1000 <= profile.n_rows <= 100000:
                score += 10
                reasons.append("dataset size is suitable for cluster-based scoring")
            if profile.n_features <= 60:
                score += 8
                reasons.append("feature dimension is not too high for clustering")
            else:
                score -= 8
                reasons.append("very high-dimensional data can weaken cluster distance quality")
            if profile.categorical_ratio > 0.50:
                score -= 6
                reasons.append("many encoded categorical columns may make clusters less meaningful")

        elif model == "OCSVM":
            score += 5
            _add_reason(reasons, True, "captures nonlinear normal-data boundary")
            if profile.n_rows <= 10000:
                score += 12
                reasons.append("small/medium dataset keeps OCSVM practical")
            elif profile.n_rows > 30000:
                score -= 18
                reasons.append("large dataset makes OCSVM slow and less practical")
            if profile.n_features <= 40:
                score += 7
                reasons.append("lower-dimensional data is better for RBF boundary learning")
            else:
                score -= 6
                reasons.append("higher-dimensional data can reduce OCSVM stability")
            if profile.contamination > 0.10:
                score -= 5
                reasons.append("high expected anomaly rate is less ideal for one-class boundary learning")

        elif model == "ECOD":
            score += 14
            _add_reason(reasons, True, "fast distribution-tail detector")
            numeric_ratio = profile.n_numeric / max(profile.n_features, 1)
            if numeric_ratio >= 0.60:
                score += 12
                reasons.append("numeric-heavy data fits ECOD distribution-tail assumptions")
            if profile.n_rows >= 10000:
                score += 6
                reasons.append("efficient on large datasets")
            if profile.categorical_ratio > 0.50:
                score -= 8
                reasons.append("too many encoded categorical columns can weaken tail-based assumptions")
            if profile.contamination <= 0.05:
                score += 4
                reasons.append("low expected anomaly rate fits tail detection")

        rows.append(
            {
                "model": model,
                "suitability_score": round(float(np.clip(score, 0, 100)), 2),
                "reason": "; ".join(reasons),
            }
        )

    df = pd.DataFrame(rows).sort_values("suitability_score", ascending=False)
    df["rank"] = range(1, len(df) + 1)
    return df[["rank", "model", "suitability_score", "reason"]]


def choose_models_by_mode(suitability_df: pd.DataFrame, mode: str) -> list[str]:
    """Return model names to execute for a requested model mode."""
    ordered = suitability_df.sort_values("rank")["model"].tolist()
    if mode == "all":
        return MODEL_NAMES
    if mode == "auto":
        return ordered[:1]
    if mode == "recommended":
        return ordered[:2]
    raise ValueError(f"Unsupported model mode: {mode}")


def select_best_from_metrics(results_df: pd.DataFrame) -> dict:
    """
    Select best model from evaluation metrics.

    F1 is primary because fraud/anomaly tasks need balance between precision and
    recall. ROC-AUC and average precision break ties.
    """
    if results_df.empty:
        return {}

    sort_cols = [c for c in ["f1_score", "roc_auc", "average_precision"] if c in results_df.columns]
    if not sort_cols:
        return {}

    best = results_df.sort_values(sort_cols, ascending=False).iloc[0]
    return {
        "selected_model": best["model"],
        "selection_basis": "evaluation_metrics",
        "primary_metric": "f1_score",
        "f1_score": float(best.get("f1_score", np.nan)),
        "roc_auc": float(best.get("roc_auc", np.nan)),
        "average_precision": float(best.get("average_precision", np.nan)),
    }


def write_selection_report(
    output_dir: str,
    profile: DataProfile,
    suitability_df: pd.DataFrame,
    models_to_run: Iterable[str],
    metric_selection: dict | None = None,
) -> None:
    """Persist model-selection profile and recommendations."""
    import os

    os.makedirs(output_dir, exist_ok=True)
    suitability_df.to_csv(os.path.join(output_dir, "model_suitability.csv"), index=False)

    report = {
        "data_profile": asdict(profile),
        "models_to_run": list(models_to_run),
        "heuristic_selected_model": suitability_df.sort_values("rank").iloc[0]["model"],
        "metric_selection": metric_selection or {},
    }
    with open(os.path.join(output_dir, "model_selection_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
