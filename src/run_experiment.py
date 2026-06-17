"""
Chạy 4 thuật toán phát hiện bất thường trên CMS Synthetic Claims.

Ví dụ:
python src/run_experiment.py --data data/raw/DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv --label-mode synthetic --sample-size 30000
python src/run_experiment.py --data data/raw/DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv --label-mode synthetic --sample-size 30000 --tune
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Fix Windows console encoding for Vietnamese characters
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump

from features import read_claims_csv, add_domain_features, select_feature_columns, build_preprocessor
from labels import create_synthetic_anomaly_label
from models import run_isolation_forest, run_cblof, run_ocsvm, run_ecod
from evaluation import evaluate_predictions, add_topk_metrics, plot_roc_curves, plot_pr_curves, plot_confusion_matrices, plot_score_distributions
from tuning import tune_all_models
from model_selection import (
    build_data_profile,
    choose_models_by_mode,
    score_models_for_profile,
    select_best_from_metrics,
    write_selection_report,
)


def parse_args():
    parser = argparse.ArgumentParser(description="CMS Fraud Anomaly Detection with 4 algorithms")
    parser.add_argument("--data", required=True, help="Đường dẫn file CSV dataset")
    parser.add_argument("--output-dir", default="outputs", help="Thư mục lưu kết quả")
    parser.add_argument("--model-dir", default="models", help="Thư mục lưu model")
    parser.add_argument("--label-mode", choices=["none", "synthetic", "column"], default="synthetic")
    parser.add_argument("--label-col", default=None, help="Tên cột nhãn nếu label-mode=column")
    parser.add_argument("--sample-size", type=int, default=30000, help="Số dòng lấy mẫu để chạy nhanh; 0 nghĩa là dùng toàn bộ")
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--contamination", type=float, default=0.05)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--model-mode",
        choices=["all", "recommended", "auto"],
        default="all",
        help="all=run 4 models; recommended=run top 2 suitable models; auto=run the most suitable model",
    )
    parser.add_argument("--tune", action="store_true", help="Bật tuning tham số")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    sample_size = None if args.sample_size == 0 else args.sample_size

    print("=" * 80)
    print("ĐỌC DỮ LIỆU")
    print("=" * 80)
    df_raw = read_claims_csv(args.data, sample_size=sample_size, random_state=args.random_state)
    print(f"Dataset shape ban đầu: {df_raw.shape}")

    print("\nTạo feature nghiệp vụ...")
    df_feat = add_domain_features(df_raw)
    print(f"Dataset shape sau feature engineering: {df_feat.shape}")

    y = None
    label_col = None
    if args.label_mode == "column":
        if not args.label_col or args.label_col not in df_feat.columns:
            raise ValueError("Bạn chọn label-mode=column nhưng label-col không tồn tại trong dataset.")
        label_col = args.label_col
        y = df_feat[label_col].astype(int)
    elif args.label_mode == "synthetic":
        label_col = "synthetic_anomaly_label"
        y = create_synthetic_anomaly_label(df_feat, target_rate=args.contamination)
        df_feat[label_col] = y
        print(f"Tỷ lệ synthetic anomaly: {y.mean():.4f}")
    else:
        print("Không dùng nhãn. Chương trình chỉ xuất anomaly score.")

    numeric_cols, categorical_cols = select_feature_columns(df_feat, label_col=label_col)
    print(f"\nSố cột numeric dùng cho mô hình: {len(numeric_cols)}")
    print(f"Số cột categorical frequency-encoded: {len(categorical_cols)}")

    X_df = df_feat[numeric_cols + categorical_cols].copy()

    profile = build_data_profile(
        X_df=X_df,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        contamination=args.contamination,
        has_labels=y is not None,
    )
    suitability_df = score_models_for_profile(profile)
    effective_model_mode = args.model_mode
    if args.model_mode == "auto" and y is not None:
        effective_model_mode = "all"
    models_to_run = choose_models_by_mode(suitability_df, effective_model_mode)

    print("\n" + "=" * 80)
    print("GOI Y CHON MODEL THEO DAC DIEM DU LIEU")
    print("=" * 80)
    print(suitability_df[["rank", "model", "suitability_score"]])
    if args.model_mode == "auto" and y is not None:
        print("\nmodel-mode=auto va co nhan danh gia: se chay ca 4 model roi chon theo metric.")
    print(f"Se chay: {', '.join(models_to_run)}")
    write_selection_report(args.output_dir, profile, suitability_df, models_to_run)

    if y is not None:
        X_train_df, X_test_df, y_train, y_test, raw_train, raw_test = train_test_split(
            X_df,
            y,
            df_raw,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=y,
        )
    else:
        X_train_df, X_test_df, raw_test = X_df, X_df, df_raw
        y_train, y_test = None, None

    print("\nTiền xử lý dữ liệu...")
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    X_train = preprocessor.fit_transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape : {X_test.shape}")
    dump(preprocessor, os.path.join(args.model_dir, "preprocessor.joblib"))

    print("\n" + "=" * 80)
    print("CHẠY THUẬT TOÁN")
    print("=" * 80)

    model_outputs = {}

    all_runners = {
        "IsolationForest": lambda: run_isolation_forest(X_train, X_test, contamination=args.contamination, random_state=args.random_state),
        "CBLOF": lambda: run_cblof(X_train, X_test, contamination=args.contamination, random_state=args.random_state),
        "OCSVM": lambda: run_ocsvm(X_train, X_test, contamination=args.contamination, random_state=args.random_state, nu=args.contamination),
        "ECOD": lambda: run_ecod(X_train, X_test, contamination=args.contamination, random_state=args.random_state),
    }
    runners = [(name, all_runners[name]) for name in models_to_run]

    for name, runner in runners:
        print(f"\nĐang chạy {name}...")
        try:
            model, scores, pred, runtime = runner()
            model_outputs[name] = {"model": model, "scores": scores, "pred": pred, "runtime": runtime}
            dump(model, os.path.join(args.model_dir, f"{name}.joblib"))
            print(f"Hoàn thành {name} trong {runtime:.2f} giây")
        except Exception as e:
            print(f"Lỗi khi chạy {name}: {e}")

    # Lưu scores cho từng claim trong test set.
    score_df = raw_test.reset_index(drop=True).copy()
    for name, obj in model_outputs.items():
        score_df[f"{name}_score"] = obj["scores"]
        score_df[f"{name}_prediction"] = obj["pred"]
    if y_test is not None:
        score_df["label_used_for_evaluation"] = np.asarray(y_test).astype(int)
    score_path = os.path.join(args.output_dir, "claim_anomaly_scores.csv")
    score_df.to_csv(score_path, index=False)
    print(f"\nĐã lưu score từng claim: {score_path}")

    # Top suspicious claims theo điểm trung bình chuẩn hóa rank.
    top_df = score_df.copy()
    score_cols = [c for c in top_df.columns if c.endswith("_score")]
    if score_cols:
        rank_scores = []
        for c in score_cols:
            rank_scores.append(top_df[c].rank(pct=True).to_numpy())
        top_df["ensemble_rank_score"] = np.mean(rank_scores, axis=0)
        top_df = top_df.sort_values("ensemble_rank_score", ascending=False)
        top_path = os.path.join(args.output_dir, "top_suspicious_claims.csv")
        top_df.head(200).to_csv(top_path, index=False)
        print(f"Đã lưu top hồ sơ đáng nghi: {top_path}")

    # Đánh giá nếu có nhãn.
    if y_test is not None:
        rows = []
        score_dict = {}
        for name, obj in model_outputs.items():
            row = evaluate_predictions(name, y_test, obj["scores"], obj["pred"], obj["runtime"])
            row = add_topk_metrics(row, y_test, obj["scores"])
            rows.append(row)
            score_dict[name] = obj["scores"]

        results_df = pd.DataFrame(rows).sort_values("f1_score", ascending=False)
        metric_selection = select_best_from_metrics(results_df)
        write_selection_report(args.output_dir, profile, suitability_df, models_to_run, metric_selection)
        result_path = os.path.join(args.output_dir, "model_comparison.csv")
        results_df.to_csv(result_path, index=False)
        print("\n" + "=" * 80)
        print("BẢNG SO SÁNH MÔ HÌNH")
        print("=" * 80)
        print(results_df)
        if metric_selection:
            print(
                "\nModel được chọn theo metric: "
                f"{metric_selection['selected_model']} "
                f"(F1={metric_selection['f1_score']:.4f}, "
                f"ROC-AUC={metric_selection['roc_auc']:.4f})"
            )
        print(f"\nĐã lưu bảng so sánh: {result_path}")

        plot_roc_curves(score_dict, y_test, os.path.join(args.output_dir, "roc_curves.png"))
        plot_pr_curves(score_dict, y_test, os.path.join(args.output_dir, "pr_curves.png"))
        plot_confusion_matrices(model_outputs, y_test, os.path.join(args.output_dir, "confusion_matrices.png"))
        plot_score_distributions(model_outputs, y_test, os.path.join(args.output_dir, "score_distributions.png"))
        print("Đã lưu biểu đồ ROC, PR, Confusion Matrix, Score Distribution vào outputs/")

        if args.tune:
            print("\n" + "=" * 80)
            print("TUNING THAM SỐ")
            print("=" * 80)
            tuning_df = tune_all_models(X_train, X_test, y_test, random_state=args.random_state)
            tuning_path = os.path.join(args.output_dir, "tuning_results.csv")
            tuning_df.to_csv(tuning_path, index=False)
            print(f"Đã lưu toàn bộ tuning: {tuning_path}")

            ok = tuning_df.dropna(subset=["f1_score"], how="any") if "f1_score" in tuning_df.columns else tuning_df
            if not ok.empty and "model" in ok.columns:
                best = ok.sort_values(["f1_score", "roc_auc"], ascending=False).groupby("model", as_index=False).head(1)
                best_path = os.path.join(args.output_dir, "best_tuning_results.csv")
                best.to_csv(best_path, index=False)
                print("\nBest tuning mỗi mô hình:")
                print(best)
                print(f"Đã lưu best tuning: {best_path}")
    else:
        print("\nKhông có nhãn nên không tính ROC-AUC/Precision/Recall/F1.")


if __name__ == "__main__":
    main()
