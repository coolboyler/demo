from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_baseline_d6 as shared  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Tianlang D-6 baseline using only 2025-2026 data.")
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--dataset-path", type=Path, default=Path("new/baseline_d6_dataset_2024fill.csv"))
    parser.add_argument("--issue-gap-days", type=int, default=6)
    parser.add_argument("--output-prefix", default="baseline_d6_tianlang_2024fill_shared")
    return parser.parse_args()


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    dataset = shared.load_dataset(dataset_path)
    available_years = sorted(set(dataset["target_date"].dt.year.unique()).intersection(dataset["issue_date"].dt.year.unique()))
    mask = dataset["target_date"].dt.year.isin(available_years) & dataset["issue_date"].dt.year.isin(available_years)
    return dataset.loc[mask].copy().reset_index(drop=True)


def main() -> None:
    args = parse_args()
    results_dir = args.base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = args.base_dir / args.dataset_path if not args.dataset_path.is_absolute() else args.dataset_path
    dataset = load_dataset(dataset_path)
    train, validation, test, split_info = shared.split_dataset(dataset)
    short_lags, weekly_lags, _ = shared.get_issue_lag_config(args.issue_gap_days)

    best_model_name, best_params, best_test_prediction, validation_rows, top_test_rows = shared.search_best_model(
        validation=validation,
        test=test,
        short_lags=short_lags,
        weekly_lags=weekly_lags,
    )

    prediction_frame = shared.make_daily_prediction_frame(test, best_model_name, best_test_prediction)
    prediction_frame["split"] = "test"
    prediction_frame["is_fully_actual_5_company"] = 1
    long_prediction_frame = shared.make_long_prediction_frame(prediction_frame)
    daily_metrics = shared.daily_total_metrics(prediction_frame)

    prediction_path = results_dir / f"{args.output_prefix}_test_daily.csv"
    long_path = results_dir / f"{args.output_prefix}_test_long.csv"
    summary_path = results_dir / f"{args.output_prefix}_summary.json"

    prediction_frame.to_csv(prediction_path, index=False)
    long_prediction_frame.to_csv(long_path, index=False)

    selected_test_metrics = next(row for row in top_test_rows if row["model"] == best_model_name)
    summary = {
        "task_definition": f"Tianlang D-{args.issue_gap_days} baseline using the retained 2024fill shared dataset.",
        "site_name": "天朗",
        "year_scope": sorted(dataset["target_date"].dt.year.unique().tolist()),
        "dataset_path": str(dataset_path),
        "rows": int(len(dataset)),
        "issue_gap_days": int(args.issue_gap_days),
        "short_lags": short_lags,
        "weekly_lags": weekly_lags,
        "split_info": split_info,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "selected_model": best_model_name,
        "selected_model_params": best_params,
        "selected_model_test_metrics": selected_test_metrics,
        "daily_total_metrics": daily_metrics,
        "validation_top_rows": validation_rows,
        "test_top_rows": top_test_rows,
        "prediction_outputs": {"daily": str(prediction_path), "long": str(long_path)},
    }
    shared.save_summary_json(summary, summary_path)
    print(summary)


if __name__ == "__main__":
    main()
