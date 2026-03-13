from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_d5 import build_issue_gap_dataset, get_issue_lag_config


VALIDATION_START = pd.Timestamp("2026-01-01")
VALIDATION_END = pd.Timestamp("2026-01-31")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build equivalent five-company total dataset for a configurable issue gap.")
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--issue-gap-days", type=int, default=6)
    parser.add_argument("--hourly-path", type=Path, default=Path("new/history_hourly.csv"))
    parser.add_argument("--output-prefix", default=None, help="Default is baseline_d{issue_gap_days}")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    new_dir = args.base_dir / "new"
    output_prefix = args.output_prefix or f"baseline_d{args.issue_gap_days}"
    hourly_path = args.base_dir / args.hourly_path

    hourly_dataset = pd.read_csv(hourly_path, parse_dates=["timestamp", "date"])
    ready = build_issue_gap_dataset(
        hourly_dataset[
            [
                "site_id",
                "site_name",
                "city_cn",
                "city_en",
                "province_cn",
                "timestamp",
                "date",
                "hour",
                "year",
                "month",
                "quarter",
                "weekofyear",
                "dayofmonth",
                "dayofweek",
                "dayofyear",
                "is_weekend",
                "is_month_start",
                "is_month_end",
                "season",
                "is_holiday_cn",
                "holiday_name_cn",
                "is_workday_cn",
                "is_makeup_workday",
                "date_type_cn",
                "load",
            ]
        ],
        issue_gap_days=args.issue_gap_days,
    )
    ready["split"] = np.where(
        ready["target_date"] < VALIDATION_START,
        "train",
        np.where(ready["target_date"] <= VALIDATION_END, "validation", "test"),
    )
    ready["is_fully_actual_5_company"] = ready["target_date"].ge(VALIDATION_START).astype(int)

    short_lags, weekly_lags, _ = get_issue_lag_config(args.issue_gap_days)
    dataset_path = new_dir / f"{output_prefix}_dataset.csv"
    summary_path = new_dir / f"{output_prefix}_dataset_summary.json"

    ready.to_csv(dataset_path, index=False, encoding="utf-8-sig")
    summary = {
        "dataset_path": str(dataset_path),
        "rows": int(len(ready)),
        "issue_gap_days": int(args.issue_gap_days),
        "short_lags": short_lags,
        "weekly_lags": weekly_lags,
        "split_counts": ready["split"].value_counts().to_dict(),
        "target_date_start": str(ready["target_date"].min().date()),
        "target_date_end": str(ready["target_date"].max().date()),
        "issue_date_start": str(ready["issue_date"].min().date()),
        "issue_date_end": str(ready["issue_date"].max().date()),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
