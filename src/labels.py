"""Tạo nhãn bất thường synthetic cho dữ liệu CMS khi không có nhãn thật."""

from __future__ import annotations

import re
import numpy as np
import pandas as pd


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)


def create_synthetic_anomaly_label(df: pd.DataFrame, target_rate: float = 0.05) -> pd.Series:
    """
    Tạo label synthetic dựa trên các rule nghiệp vụ tương đối hợp lý.

    1 = bất thường giả lập
    0 = bình thường

    Lưu ý: đây KHÔNG phải nhãn gian lận thật.
    """
    n = len(df)
    score = pd.Series(0.0, index=df.index)

    # Claim amount cực lớn.
    if "CLM_PMT_AMT" in df.columns:
        amount = _num(df, "CLM_PMT_AMT")
        q95 = amount.quantile(0.95)
        q99 = amount.quantile(0.99)
        score += (amount > q95).astype(float) * 1.0
        score += (amount > q99).astype(float) * 2.0

    # Số ngày sử dụng/nằm viện cao.
    for col in ["CLM_UTLZTN_DAY_CNT", "claim_duration_days", "admission_duration_days"]:
        if col in df.columns:
            s = _num(df, col)
            score += (s > s.quantile(0.95)).astype(float) * 0.8
            score += (s > s.quantile(0.99)).astype(float) * 1.5

    # Amount/day quá cao.
    if "amount_per_utilization_day" in df.columns:
        s = _num(df, "amount_per_utilization_day")
        score += (s > s.quantile(0.97)).astype(float) * 1.5

    # Nhiều mã chẩn đoán/thủ thuật.
    for col in ["num_diagnosis_codes", "num_procedure_codes"]:
        if col in df.columns:
            s = _num(df, col)
            score += (s > s.quantile(0.95)).astype(float) * 0.7

    # Provider/physician xuất hiện rất hiếm nhưng claim amount cao.
    id_freq_cols = [c for c in df.columns if c.endswith("_freq")]
    if id_freq_cols and "CLM_PMT_AMT" in df.columns:
        amount = _num(df, "CLM_PMT_AMT")
        high_amount = amount > amount.quantile(0.90)
        for c in id_freq_cols[:10]:
            freq = _num(df, c)
            rare = freq <= freq.quantile(0.05)
            score += (rare & high_amount).astype(float) * 0.5

    # Nếu không có feature nào tạo score, chọn ngẫu nhiên rất ít để code vẫn chạy.
    if score.max() == 0 or score.isna().all():
        rng = np.random.default_rng(42)
        y = pd.Series(0, index=df.index)
        anomaly_count = max(1, int(n * target_rate))
        y.iloc[rng.choice(n, size=anomaly_count, replace=False)] = 1
        return y.astype(int)

    score = score.fillna(0)
    threshold = score.quantile(1 - target_rate)
    y = (score >= threshold).astype(int)

    # Đảm bảo tỷ lệ xấp xỉ target_rate nếu bị tie quá nhiều.
    anomaly_count = max(1, int(n * target_rate))
    if y.sum() > anomaly_count * 2 or y.sum() < max(1, anomaly_count // 2):
        top_idx = score.sort_values(ascending=False).head(anomaly_count).index
        y = pd.Series(0, index=df.index)
        y.loc[top_idx] = 1

    return y.astype(int)
