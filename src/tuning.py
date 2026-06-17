"""Tuning đơn giản cho 4 mô hình."""

from __future__ import annotations

from itertools import product
import pandas as pd
from evaluation import evaluate_predictions, add_topk_metrics
from models import run_isolation_forest, run_cblof, run_ocsvm, run_ecod


def tune_all_models(X_train, X_test, y_test, random_state=42) -> pd.DataFrame:
    results = []

    # Isolation Forest grid
    for n_estimators, max_samples, contamination in product(
        [100, 200, 300], ["auto", 0.5, 0.8], [0.01, 0.03, 0.05, 0.10]
    ):
        try:
            _, scores, pred, runtime = run_isolation_forest(
                X_train, X_test,
                contamination=contamination,
                random_state=random_state,
                n_estimators=n_estimators,
                max_samples=max_samples,
            )
            row = evaluate_predictions("IsolationForest", y_test, scores, pred, runtime)
            row.update({"n_estimators": n_estimators, "max_samples": max_samples, "contamination": contamination})
            results.append(add_topk_metrics(row, y_test, scores))
        except Exception as e:
            results.append({"model": "IsolationForest", "error": str(e)})

    # CBLOF grid
    for n_clusters, alpha, beta, contamination in product(
        [4, 8, 12, 16], [0.8, 0.9], [3, 5], [0.01, 0.03, 0.05, 0.10]
    ):
        try:
            _, scores, pred, runtime = run_cblof(
                X_train, X_test,
                contamination=contamination,
                random_state=random_state,
                n_clusters=n_clusters,
                alpha=alpha,
                beta=beta,
            )
            row = evaluate_predictions("CBLOF", y_test, scores, pred, runtime)
            row.update({"n_clusters": n_clusters, "alpha": alpha, "beta": beta, "contamination": contamination})
            results.append(add_topk_metrics(row, y_test, scores))
        except Exception as e:
            results.append({"model": "CBLOF", "error": str(e), "n_clusters": n_clusters})

    # OCSVM grid - giới hạn để tránh chạy quá lâu
    for nu, gamma in product([0.01, 0.03, 0.05, 0.10], ["scale", "auto", 0.001, 0.01]):
        try:
            _, scores, pred, runtime = run_ocsvm(
                X_train, X_test,
                nu=nu,
                gamma=gamma,
            )
            row = evaluate_predictions("OCSVM", y_test, scores, pred, runtime)
            row.update({"nu": nu, "gamma": gamma})
            results.append(add_topk_metrics(row, y_test, scores))
        except Exception as e:
            results.append({"model": "OCSVM", "error": str(e), "nu": nu, "gamma": gamma})

    # ECOD grid - chủ yếu thay threshold contamination
    for contamination in [0.01, 0.03, 0.05, 0.10]:
        try:
            _, scores, pred, runtime = run_ecod(X_train, X_test, contamination=contamination)
            row = evaluate_predictions("ECOD", y_test, scores, pred, runtime)
            row.update({"contamination": contamination})
            results.append(add_topk_metrics(row, y_test, scores))
        except Exception as e:
            results.append({"model": "ECOD", "error": str(e)})

    return pd.DataFrame(results)
