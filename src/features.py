"""
Feature engineering cho dữ liệu CMS Synthetic Claims.
Code được viết theo hướng robust: nếu thiếu cột thì tự bỏ qua, không làm lỗi chương trình.
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer


DATE_PATTERN = re.compile(r"(_DT$|DATE|FROM|THRU|ADMSN|DSCHRG)", re.IGNORECASE)
AMOUNT_PATTERN = re.compile(r"(AMT|PMT|PYR|DDCTBL|COINS|LBLTY|PER_DIEM|PAID)", re.IGNORECASE)
ID_PATTERN = re.compile(r"(ID$|_ID|NPI|PRVDR|PHYSN|CLM_ID|DESYNPUF)", re.IGNORECASE)
CODE_PATTERN = re.compile(r"(ICD|DGNS|PRCDR|DRG|HCPCS|CD$|CODE)", re.IGNORECASE)


def read_claims_csv(path: str, sample_size: int | None = None, random_state: int = 42) -> pd.DataFrame:
    """Đọc CSV và lấy mẫu nếu cần để chạy nhanh trên máy cá nhân."""
    df = pd.read_csv(path, low_memory=False)
    if sample_size is not None and sample_size > 0 and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
    return df


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo thêm feature nghiệp vụ từ dữ liệu claim.
    Hàm này không phụ thuộc tuyệt đối vào một schema cụ thể.
    """
    out = df.copy()

    # Chuẩn hóa các cột ngày dạng YYYYMMDD hoặc chuỗi ngày.
    date_cols = [c for c in out.columns if DATE_PATTERN.search(c)]
    for c in date_cols:
        out[c + "_parsed"] = pd.to_datetime(out[c], errors="coerce", format="%Y%m%d")
        # Nếu format YYYYMMDD không đúng, thử parse tự động.
        missing_ratio = out[c + "_parsed"].isna().mean()
        if missing_ratio > 0.95:
            out[c + "_parsed"] = pd.to_datetime(out[c], errors="coerce")

    # Tạo duration cho các cặp ngày phổ biến.
    possible_pairs = [
        ("CLM_FROM_DT", "CLM_THRU_DT", "claim_duration_days"),
        ("CLM_ADMSN_DT", "NCH_BENE_DSCHRG_DT", "admission_duration_days"),
        ("CLM_ADMSN_DT", "CLM_THRU_DT", "admission_to_claim_end_days"),
    ]
    for start, end, new_col in possible_pairs:
        s, e = start + "_parsed", end + "_parsed"
        if s in out.columns and e in out.columns:
            out[new_col] = (out[e] - out[s]).dt.days
            out[new_col] = out[new_col].clip(lower=0)

    # Feature từ claim amount / utilization nếu có.
    if "CLM_PMT_AMT" in out.columns:
        out["claim_amount_log1p"] = np.log1p(pd.to_numeric(out["CLM_PMT_AMT"], errors="coerce").clip(lower=0))

    if "CLM_UTLZTN_DAY_CNT" in out.columns and "CLM_PMT_AMT" in out.columns:
        amt = pd.to_numeric(out["CLM_PMT_AMT"], errors="coerce")
        days = pd.to_numeric(out["CLM_UTLZTN_DAY_CNT"], errors="coerce").replace(0, np.nan)
        out["amount_per_utilization_day"] = amt / days
        out["amount_per_utilization_day"] = out["amount_per_utilization_day"].replace([np.inf, -np.inf], np.nan)

    # Đếm số mã chẩn đoán/thủ thuật không rỗng trên mỗi claim.
    diag_cols = [c for c in out.columns if re.search(r"ICD.*DGNS|DGNS", c, re.IGNORECASE)]
    proc_cols = [c for c in out.columns if re.search(r"ICD.*PRCDR|PRCDR", c, re.IGNORECASE)]
    if diag_cols:
        out["num_diagnosis_codes"] = out[diag_cols].notna().sum(axis=1)
    if proc_cols:
        out["num_procedure_codes"] = out[proc_cols].notna().sum(axis=1)

    # Count/frequency feature theo provider/patient/physician nếu có.
    group_cols = [c for c in out.columns if ID_PATTERN.search(c)]
    for c in group_cols[:20]:
        freq = out[c].astype(str).value_counts(dropna=False)
        out[c + "_freq"] = out[c].astype(str).map(freq).astype(float)
        out[c + "_freq_log1p"] = np.log1p(out[c + "_freq"])

    # Xóa cột datetime parsed vì sklearn không xử lý trực tiếp datetime.
    parsed_cols = [c for c in out.columns if c.endswith("_parsed")]
    out = out.drop(columns=parsed_cols, errors="ignore")

    return out


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Frequency encoding cho biến phân loại.
    Phù hợp hơn LabelEncoder đối với anomaly detection vì không tạo thứ tự giả.
    """

    def __init__(self):
        self.maps_: dict[str, dict[str, float]] = {}
        self.columns_: list[str] = []

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X).copy()
        self.columns_ = list(X_df.columns)
        n = len(X_df)
        self.maps_ = {}
        for col in self.columns_:
            freq = X_df[col].astype(str).fillna("__MISSING__").value_counts(dropna=False) / max(n, 1)
            self.maps_[col] = freq.to_dict()
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        encoded = []
        for i, col in enumerate(self.columns_):
            s = X_df.iloc[:, i].astype(str).fillna("__MISSING__")
            encoded.append(s.map(self.maps_[col]).fillna(0.0).to_numpy())
        return np.vstack(encoded).T if encoded else np.empty((len(X_df), 0))


def select_feature_columns(df: pd.DataFrame, label_col: str | None = None) -> tuple[list[str], list[str]]:
    """Chọn cột số và cột phân loại cho mô hình."""
    exclude = set()
    if label_col and label_col in df.columns:
        exclude.add(label_col)

    # Loại cột định danh quá riêng biệt khỏi categorical thô, nhưng vẫn dùng feature frequency đã tạo.
    raw_id_cols = {c for c in df.columns if ID_PATTERN.search(c) and not c.endswith("_freq") and not c.endswith("_freq_log1p")}
    exclude |= raw_id_cols

    numeric_cols = []
    categorical_cols = []

    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
        elif pd.api.types.is_bool_dtype(df[c]):
            numeric_cols.append(c)
        else:
            # Chỉ giữ các cột code/categorical có ý nghĩa; tránh giữ ngày dạng raw.
            if CODE_PATTERN.search(c) or c.endswith("_freq"):
                categorical_cols.append(c)

    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """Tạo preprocessor gồm scale numeric và frequency encode categorical."""
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("freq", FrequencyEncoder()),
        ("scaler", StandardScaler())
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
