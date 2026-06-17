"""Retrain/update models with newly collected claim data.

This script prepares a combined dataset from the baseline CSV plus recent CSV
files, then calls run_experiment.py with model-mode=auto. It is designed to be
run manually or by a scheduler every few days.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Periodic retraining pipeline for CMS fraud anomaly models")
    parser.add_argument(
        "--base-data",
        default="data/DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv",
        help="Existing baseline CSV used as historical training data",
    )
    parser.add_argument(
        "--new-data-dir",
        default="data/new",
        help="Folder containing newly collected CSV files",
    )
    parser.add_argument("--days", type=int, default=3, help="Only include new CSV files modified in the last N days")
    parser.add_argument("--output-root", default="outputs/retrain", help="Root folder for retraining outputs")
    parser.add_argument("--model-root", default="models/retrain", help="Root folder for retrained models")
    parser.add_argument("--label-mode", choices=["none", "synthetic", "column"], default="synthetic")
    parser.add_argument("--label-col", default=None)
    parser.add_argument("--contamination", type=float, default=0.05)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--tune", action="store_true", help="Also run hyperparameter tuning after retraining")
    parser.add_argument(
        "--require-new-data",
        action="store_true",
        help="Stop if no recent new CSV files are found",
    )
    return parser.parse_args()


def recent_csv_files(new_data_dir: Path, days: int) -> list[Path]:
    if not new_data_dir.exists():
        return []

    cutoff = datetime.now() - timedelta(days=days)
    files = []
    for path in sorted(new_data_dir.glob("*.csv")):
        modified = datetime.fromtimestamp(path.stat().st_mtime)
        if modified >= cutoff:
            files.append(path)
    return files


def read_csvs(paths: list[Path]) -> list[pd.DataFrame]:
    frames = []
    for path in paths:
        frames.append(pd.read_csv(path, low_memory=False))
    return frames


def combine_data(base_data: Path, new_files: list[Path], combined_path: Path) -> dict:
    if not base_data.exists():
        raise FileNotFoundError(f"Base data not found: {base_data}")

    frames = [pd.read_csv(base_data, low_memory=False)]
    frames.extend(read_csvs(new_files))

    combined = pd.concat(frames, ignore_index=True, sort=False)
    before_dedup = len(combined)

    if "CLM_ID" in combined.columns:
        combined = combined.drop_duplicates(subset=["CLM_ID"], keep="last")
        dedup_key = "CLM_ID"
    else:
        combined = combined.drop_duplicates(keep="last")
        dedup_key = "all_columns"

    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(combined_path, index=False)

    return {
        "base_rows": int(len(frames[0])),
        "new_files": [str(p) for p in new_files],
        "new_file_count": len(new_files),
        "rows_before_dedup": int(before_dedup),
        "rows_after_dedup": int(len(combined)),
        "dedup_key": dedup_key,
        "combined_data": str(combined_path),
    }


def run_experiment(args, combined_path: Path, run_output_dir: Path, run_model_dir: Path) -> None:
    cmd = [
        sys.executable,
        "src/run_experiment.py",
        "--data",
        str(combined_path),
        "--label-mode",
        args.label_mode,
        "--sample-size",
        "0",
        "--model-mode",
        "auto",
        "--output-dir",
        str(run_output_dir),
        "--model-dir",
        str(run_model_dir),
        "--contamination",
        str(args.contamination),
        "--random-state",
        str(args.random_state),
    ]

    if args.label_col:
        cmd.extend(["--label-col", args.label_col])
    if args.tune:
        cmd.append("--tune")

    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root)
    model_root = Path(args.model_root)
    run_output_dir = output_root / timestamp
    run_model_dir = model_root / timestamp
    combined_path = run_output_dir / "combined_claims.csv"

    base_data = Path(args.base_data)
    new_data_dir = Path(args.new_data_dir)
    new_files = recent_csv_files(new_data_dir, args.days)

    if args.require_new_data and not new_files:
        raise RuntimeError(f"No new CSV files found in {new_data_dir} modified in the last {args.days} days.")

    print("=" * 80)
    print("RETRAIN CYCLE")
    print("=" * 80)
    print(f"Base data     : {base_data}")
    print(f"New data dir  : {new_data_dir}")
    print(f"Recent window : last {args.days} days")
    print(f"New CSV files : {len(new_files)}")

    manifest = combine_data(base_data, new_files, combined_path)
    print(f"Combined data : {combined_path}")
    print(f"Rows after dedup: {manifest['rows_after_dedup']}")

    run_experiment(args, combined_path, run_output_dir, run_model_dir)

    selection_report_path = run_output_dir / "model_selection_report.json"
    if selection_report_path.exists():
        with open(selection_report_path, "r", encoding="utf-8") as f:
            manifest["selection_report"] = json.load(f)

    manifest.update(
        {
            "timestamp": timestamp,
            "days": args.days,
            "output_dir": str(run_output_dir),
            "model_dir": str(run_model_dir),
            "label_mode": args.label_mode,
            "contamination": args.contamination,
        }
    )

    manifest_path = run_output_dir / "retrain_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Retrain manifest saved: {manifest_path}")


if __name__ == "__main__":
    main()
