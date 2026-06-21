"""Export claims from MySQL database to CSV matching CMS baseline format.

This script connects to the fraud_detection MySQL database, JOINs and PIVOTs
the normalized tables (raw_claims, raw_diagnoses, raw_procedures, raw_hcpcs)
into a flat CSV with columns identical to the baseline training data:

  DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv

The exported CSV is placed in data/new/ for retrain_cycle.py to pick up.

Usage:
    python src/export_claims_csv.py
    python src/export_claims_csv.py --since 2026-06-01
    python src/export_claims_csv.py --since 7d
    python src/export_claims_csv.py --all-statuses
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Constants: column structure matching baseline CSV
# ---------------------------------------------------------------------------
# Baseline CSV column order (81 columns)
BASELINE_COLUMNS = [
    "DESYNPUF_ID", "CLM_ID", "SEGMENT",
    "CLM_FROM_DT", "CLM_THRU_DT",
    "PRVDR_NUM", "CLM_PMT_AMT", "NCH_PRMRY_PYR_CLM_PD_AMT",
    "AT_PHYSN_NPI", "OP_PHYSN_NPI", "OT_PHYSN_NPI",
    "CLM_ADMSN_DT", "ADMTNG_ICD9_DGNS_CD",
    "CLM_PASS_THRU_PER_DIEM_AMT", "NCH_BENE_IP_DDCTBL_AMT",
    "NCH_BENE_PTA_COINSRNC_LBLTY_AM", "NCH_BENE_BLOOD_DDCTBL_LBLTY_AM",
    "CLM_UTLZTN_DAY_CNT", "NCH_BENE_DSCHRG_DT", "CLM_DRG_CD",
]
DIAG_COLS = [f"ICD9_DGNS_CD_{i}" for i in range(1, 11)]     # 10 columns
PROC_COLS = [f"ICD9_PRCDR_CD_{i}" for i in range(1, 7)]      # 6 columns
HCPCS_COLS = [f"HCPCS_CD_{i}" for i in range(1, 46)]         # 45 columns

ALL_COLUMNS = BASELINE_COLUMNS + DIAG_COLS + PROC_COLS + HCPCS_COLS

# Date columns that need YYYYMMDD integer format
DATE_COLUMNS = ["CLM_FROM_DT", "CLM_THRU_DT", "CLM_ADMSN_DT", "NCH_BENE_DSCHRG_DT"]

# DB columns to exclude (Spring Boot business fields, not in training data)
DB_EXCLUDE_COLS = [
    "RAW_CLAIM_ID", "CLAIM_HANDLER_ID", "INVESTIGATOR_ID",
    "CLAIM_STATUS", "CREATED_AT", "RESOLVED_AT", "VERSION",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export claims from MySQL to CSV matching CMS baseline format"
    )
    parser.add_argument(
        "--since",
        default=None,
        help=(
            "Only export claims created after this date. "
            "Accepts ISO date (2026-06-01) or relative (7d, 14d). "
            "Default: export all claims."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="data/new",
        help="Directory to save the exported CSV (default: data/new)",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Output filename. Default: claims_YYYYMMDD.csv",
    )
    parser.add_argument(
        "--all-statuses",
        action="store_true",
        help=(
            "Export claims of ALL statuses. By default, only exports "
            "APPROVED / REJECTED claims (verified by investigators)."
        ),
    )
    parser.add_argument("--db-host", default="localhost")
    parser.add_argument("--db-port", type=int, default=3306)
    parser.add_argument("--db-name", default="fraud_detection")
    parser.add_argument("--db-user", default="root")
    parser.add_argument("--db-password", default="PG123")
    return parser.parse_args()


def resolve_since(since_str: str | None) -> datetime | None:
    """Parse --since argument into a datetime."""
    if since_str is None:
        return None

    # Relative format: "7d", "14d"
    match = re.match(r"^(\d+)d$", since_str.strip(), re.IGNORECASE)
    if match:
        days = int(match.group(1))
        return datetime.now() - timedelta(days=days)

    # ISO date format
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y%m%d"):
        try:
            return datetime.strptime(since_str.strip(), fmt)
        except ValueError:
            continue

    raise ValueError(f"Cannot parse --since value: '{since_str}'. Use YYYY-MM-DD or Nd format.")


def get_connection(args):
    """Create a MySQL connection using mysql-connector-python."""
    try:
        import mysql.connector
        return mysql.connector.connect(
            host=args.db_host,
            port=args.db_port,
            database=args.db_name,
            user=args.db_user,
            password=args.db_password,
        )
    except ImportError:
        print("[ERROR] mysql-connector-python is required. Install with:")
        print("        pip install mysql-connector-python")
        sys.exit(1)


def pivot_child_table(
    conn, parent_id_col: str, child_table: str, value_col: str,
    output_prefix: str, max_slots: int, since: datetime | None,
    status_filter: bool,
) -> pd.DataFrame:
    """
    Read a child table and pivot rows into numbered columns.

    Example: raw_diagnoses → ICD9_DGNS_CD_1, ICD9_DGNS_CD_2, ...
    """
    query = f"""
        SELECT c.{parent_id_col}, ch.{value_col}
        FROM {child_table} ch
        JOIN raw_claims c ON c.RAW_CLAIM_ID = ch.RAW_CLAIM_ID
        WHERE 1=1
    """
    params = []
    if since:
        query += " AND c.CREATED_AT >= %s"
        params.append(since)
    if status_filter:
        query += " AND c.CLAIM_STATUS IN ('APPROVED', 'REJECTED')"

    query += f" ORDER BY c.{parent_id_col}, ch.{child_table.upper().rstrip('s')}_ID"
    # Fix: the ordering column name varies, just use the PK
    # Actually use a simpler ordering
    query = f"""
        SELECT c.CLM_ID, ch.{value_col}
        FROM {child_table} ch
        JOIN raw_claims c ON c.RAW_CLAIM_ID = ch.RAW_CLAIM_ID
        WHERE 1=1
    """
    if since:
        query += " AND c.CREATED_AT >= %s"
    if status_filter:
        query += " AND c.CLAIM_STATUS IN ('APPROVED', 'REJECTED')"

    df = pd.read_sql(query, conn, params=params if params else None)

    if df.empty:
        return pd.DataFrame(columns=["CLM_ID"])

    # Assign a row number per CLM_ID to pivot
    df["row_num"] = df.groupby("CLM_ID").cumcount() + 1
    # Keep only up to max_slots
    df = df[df["row_num"] <= max_slots]
    # Pivot
    df["col_name"] = df["row_num"].apply(lambda n: f"{output_prefix}_{n}")
    pivoted = df.pivot(index="CLM_ID", columns="col_name", values=value_col)
    pivoted = pivoted.reset_index()

    # Ensure all expected columns exist
    expected = [f"{output_prefix}_{i}" for i in range(1, max_slots + 1)]
    for col in expected:
        if col not in pivoted.columns:
            pivoted[col] = None

    return pivoted[["CLM_ID"] + expected]


def export_claims(args) -> Path:
    """Main export logic: query DB, pivot child tables, merge, output CSV."""
    since = resolve_since(args.since)
    status_filter = not args.all_statuses
    conn = get_connection(args)

    print("=" * 70)
    print("EXPORT CLAIMS FROM DATABASE")
    print("=" * 70)
    print(f"  Database     : {args.db_host}:{args.db_port}/{args.db_name}")
    print(f"  Since        : {since or 'ALL'}")
    print(f"  Status filter: {'APPROVED/REJECTED only' if status_filter else 'ALL statuses'}")

    # ---- 1. Query main claims table ----
    claim_cols = [
        "CLM_ID", "DESYNPUF_ID", "SEGMENT",
        "CLM_FROM_DT", "CLM_THRU_DT",
        "PRVDR_NUM", "CLM_PMT_AMT", "NCH_PRMRY_PYR_CLM_PD_AMT",
        "AT_PHYSN_NPI", "OP_PHYSN_NPI", "OT_PHYSN_NPI",
        "CLM_ADMSN_DT", "ADMTNG_ICD9_DGNS_CD",
        "CLM_PASS_THRU_PER_DIEM_AMT", "NCH_BENE_IP_DDCTBL_AMT",
        "NCH_BENE_PTA_COINSRNC_LBLTY_AM", "NCH_BENE_BLOOD_DDCTBL_LBLTY_AM",
        "CLM_UTLZTN_DAY_CNT", "NCH_BENE_DSCHRG_DT", "CLM_DRG_CD",
    ]
    query = f"SELECT {', '.join(claim_cols)} FROM raw_claims WHERE 1=1"
    params = []
    if since:
        query += " AND CREATED_AT >= %s"
        params.append(since)
    if status_filter:
        query += " AND CLAIM_STATUS IN ('APPROVED', 'REJECTED')"

    claims_df = pd.read_sql(query, conn, params=params if params else None)
    print(f"  Claims found : {len(claims_df)}")

    if claims_df.empty:
        print("\n[WARNING] No claims matched the criteria. No CSV exported.")
        conn.close()
        return None

    # ---- 2. Pivot child tables ----
    print("  Pivoting diagnoses...")
    diag_df = pivot_child_table(
        conn, "CLM_ID", "raw_diagnoses", "ICD9_DGNS_CD",
        "ICD9_DGNS_CD", 10, since, status_filter,
    )

    print("  Pivoting procedures...")
    proc_df = pivot_child_table(
        conn, "CLM_ID", "raw_procedures", "ICD9_PRCDR_CD",
        "ICD9_PRCDR_CD", 6, since, status_filter,
    )

    print("  Pivoting HCPCS codes...")
    hcpcs_df = pivot_child_table(
        conn, "CLM_ID", "raw_hcpcs", "HCPCS_CD",
        "HCPCS_CD", 45, since, status_filter,
    )

    conn.close()

    # ---- 3. Merge all into one flat DataFrame ----
    merged = claims_df
    if not diag_df.empty:
        merged = merged.merge(diag_df, on="CLM_ID", how="left")
    if not proc_df.empty:
        merged = merged.merge(proc_df, on="CLM_ID", how="left")
    if not hcpcs_df.empty:
        merged = merged.merge(hcpcs_df, on="CLM_ID", how="left")

    # ---- 4. Convert DATE columns to YYYYMMDD integer format ----
    for col in DATE_COLUMNS:
        if col in merged.columns:
            dt = pd.to_datetime(merged[col], errors="coerce")
            # Format as YYYYMMDD integer string (matching baseline CSV)
            merged[col] = dt.dt.strftime("%Y%m%d")
            # Replace NaT → NaN
            merged[col] = merged[col].where(merged[col].notna(), other=None)

    # ---- 5. Ensure all 81 columns exist, in correct order ----
    for col in ALL_COLUMNS:
        if col not in merged.columns:
            merged[col] = None

    merged = merged[ALL_COLUMNS]

    # ---- 6. Write CSV ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output_name:
        filename = args.output_name
    else:
        filename = f"claims_{datetime.now().strftime('%Y%m%d')}.csv"

    output_path = output_dir / filename
    merged.to_csv(output_path, index=False)

    print(f"\n  Output file  : {output_path}")
    print(f"  Rows exported: {len(merged)}")
    print(f"  Columns      : {len(merged.columns)}")
    print("=" * 70)

    return output_path


def main():
    args = parse_args()
    result = export_claims(args)
    if result:
        print(f"\n[SUCCESS] Exported to: {result}")
    else:
        print("\n[INFO] No data exported.")
        sys.exit(1)


if __name__ == "__main__":
    main()
