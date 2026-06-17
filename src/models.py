"""Các hàm train/predict cho 4 thuật toán anomaly detection."""

from __future__ import annotations

import time
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from pyod.models.cblof import CBLOF
from pyod.models.ecod import ECOD


def sklearn_pred_to_binary(pred):
    """sklearn anomaly: 1 normal, -1 anomaly -> binary: 0 normal, 1 anomaly."""
    return np.where(np.asarray(pred) == -1, 1, 0)


def run_isolation_forest(X_train, X_test, contamination=0.05, random_state=42,
                         n_estimators=200, max_samples="auto", max_features=1.0):
    start = time.time()
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train)
    scores = -model.decision_function(X_test)  # càng cao càng bất thường
    y_pred = sklearn_pred_to_binary(model.predict(X_test))
    return model, scores, y_pred, time.time() - start


def run_cblof(X_train, X_test, contamination=0.05, random_state=42,
              n_clusters=8, alpha=0.9, beta=5, use_weights=False):
    start = time.time()
    model = CBLOF(
        n_clusters=n_clusters,
        contamination=contamination,
        alpha=alpha,
        beta=beta,
        use_weights=use_weights,
        check_estimator=False,
        random_state=random_state,
    )
    model.fit(X_train)
    scores = model.decision_function(X_test)  # PyOD: càng cao càng bất thường
    y_pred = model.predict(X_test)            # PyOD: 0 normal, 1 anomaly
    return model, scores, y_pred, time.time() - start


def run_ocsvm(X_train, X_test, contamination=0.05, random_state=42,
              kernel="rbf", nu=0.05, gamma="scale"):
    # random_state không dùng cho OneClassSVM, giữ tham số để đồng nhất interface.
    start = time.time()
    model = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
    model.fit(X_train)
    scores = -model.decision_function(X_test)  # càng cao càng bất thường
    y_pred = sklearn_pred_to_binary(model.predict(X_test))
    return model, scores, y_pred, time.time() - start


def run_ecod(X_train, X_test, contamination=0.05, random_state=42):
    # ECOD gần như deterministic, random_state không dùng.
    start = time.time()
    model = ECOD(contamination=contamination)
    model.fit(X_train)
    scores = model.decision_function(X_test)
    y_pred = model.predict(X_test)
    return model, scores, y_pred, time.time() - start


MODEL_RUNNERS = {
    "IsolationForest": run_isolation_forest,
    "CBLOF": run_cblof,
    "OCSVM": run_ocsvm,
    "ECOD": run_ecod,
}
