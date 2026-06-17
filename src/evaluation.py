"""Hàm đánh giá và vẽ biểu đồ cho anomaly detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, precision_recall_curve
)


def safe_metric(fn, *args, default=np.nan, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return default


def evaluate_predictions(model_name: str, y_true, scores, y_pred, runtime: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    y_pred = np.asarray(y_pred).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "model": model_name,
        "roc_auc": safe_metric(roc_auc_score, y_true, scores),
        "average_precision": safe_metric(average_precision_score, y_true, scores),
        "precision": safe_metric(precision_score, y_true, y_pred, zero_division=0),
        "recall": safe_metric(recall_score, y_true, y_pred, zero_division=0),
        "f1_score": safe_metric(f1_score, y_true, y_pred, zero_division=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "runtime_seconds": runtime,
    }


def precision_at_k(y_true, scores, k_ratio: float = 0.01) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    k = max(1, int(len(scores) * k_ratio))
    idx = np.argsort(scores)[::-1][:k]
    return float(y_true[idx].mean())


def add_topk_metrics(row: dict, y_true, scores) -> dict:
    for k in [0.005, 0.01, 0.02, 0.05]:
        row[f"precision_at_{int(k*1000)/10:g}%"] = precision_at_k(y_true, scores, k)
    return row


def plot_roc_curves(score_dict: dict, y_true, output_path: str):
    plt.figure(figsize=(8, 6))
    for name, scores in score_dict.items():
        try:
            fpr, tpr, _ = roc_curve(y_true, scores)
            auc = roc_auc_score(y_true, scores)
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
        except Exception:
            continue
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_pr_curves(score_dict: dict, y_true, output_path: str):
    plt.figure(figsize=(8, 6))
    for name, scores in score_dict.items():
        try:
            precision, recall, _ = precision_recall_curve(y_true, scores)
            ap = average_precision_score(y_true, scores)
            plt.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
        except Exception:
            continue
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_confusion_matrices(model_outputs: dict, y_true, output_path: str):
    """Vẽ confusion matrix heatmap cho từng mô hình."""
    import seaborn as sns

    n_models = len(model_outputs)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]

    y_true = np.asarray(y_true).astype(int)

    for ax, (name, obj) in zip(axes, model_outputs.items()):
        pred = np.asarray(obj["pred"]).astype(int)
        cm = confusion_matrix(y_true, pred, labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Normal", "Anomaly"],
                    yticklabels=["Normal", "Anomaly"])
        ax.set_title(f"{name}")
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")

    fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_score_distributions(model_outputs: dict, y_true, output_path: str):
    """Vẽ phân phối anomaly score theo nhóm Normal/Anomaly cho từng mô hình."""
    n_models = len(model_outputs)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]

    y_true = np.asarray(y_true).astype(int)

    for ax, (name, obj) in zip(axes, model_outputs.items()):
        scores = np.asarray(obj["scores"], dtype=float)
        mask_normal = y_true == 0
        mask_anomaly = y_true == 1

        ax.hist(scores[mask_normal], bins=50, alpha=0.6, label="Normal", color="#2196F3", density=True)
        ax.hist(scores[mask_anomaly], bins=50, alpha=0.6, label="Anomaly", color="#F44336", density=True)
        ax.set_title(f"{name}")
        ax.set_xlabel("Anomaly Score")
        ax.set_ylabel("Density")
        ax.legend()

    fig.suptitle("Anomaly Score Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

