from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from china_holiday_official import build_official_holiday_frame
from forecast_core import make_long_prediction_frame, save_summary_json
from forecast_d6 import (
    get_max_actual_date,
    load_history,
    predict_target_date,
    prediction_output_to_history_row,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forecast a future date range with strict issue-gap forecasting first, then recursive planning."
    )
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--history-path", type=Path, default=Path("new/history_daily.csv"))
    parser.add_argument("--router-summary-path", type=Path, default=Path("results/best_d6_summary.json"))
    parser.add_argument("--issue-gap-days", type=int, default=6)
    parser.add_argument("--start-date", type=str, default=None, help="Default is max_actual_date + 1.")
    parser.add_argument("--end-date", type=str, required=True)
    parser.add_argument("--output-prefix", default="forecast_d6_range")
    return parser.parse_args()


def ensure_supported_holiday_year(end_date: pd.Timestamp) -> None:
    holiday_years = sorted(build_official_holiday_frame(pd.date_range("2024-01-01", "2026-12-31", freq="D"))["date"].dt.year.unique())
    max_supported_year = max(holiday_years)
    if end_date.year > max_supported_year:
        raise ValueError(
            f"Target end date {end_date.date()} exceeds official holiday support year {max_supported_year}. "
            "Please extend the official holiday table first."
        )


def main() -> None:
    args = parse_args()
    results_dir = args.base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    history = load_history(args.history_path)
    max_actual_date = get_max_actual_date(history)
    start_date = pd.Timestamp(args.start_date).normalize() if args.start_date else max_actual_date + pd.Timedelta(days=1)
    end_date = pd.Timestamp(args.end_date).normalize()
    if end_date < start_date:
        raise ValueError(f"End date {end_date.date()} is earlier than start date {start_date.date()}.")
    ensure_supported_holiday_year(end_date)

    working_history = history.copy()
    daily_frames: list[pd.DataFrame] = []

    for target_date in pd.date_range(start_date, end_date, freq="D"):
        prediction_bundle = predict_target_date(history=working_history, target_date=target_date, router_summary_path=args.router_summary_path, issue_gap_days=args.issue_gap_days)
        output_frame = prediction_bundle["output_frame"].copy()
        output_frame["source_max_actual_date"] = prediction_bundle["max_actual_date"]
        daily_frames.append(output_frame)

        history_row = prediction_output_to_history_row(
            target_row=prediction_bundle["target_row"],
            prediction_vector=prediction_bundle["prediction_vector"],
        )
        working_history = (
            pd.concat([working_history, history_row], ignore_index=True)
            .sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
            .reset_index(drop=True)
        )

    final_daily = pd.concat(daily_frames, ignore_index=True)
    final_long = make_long_prediction_frame(final_daily)

    daily_path = results_dir / f"{args.output_prefix}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
    long_path = results_dir / f"{args.output_prefix}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_long.csv"
    summary_path = results_dir / f"{args.output_prefix}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_summary.json"

    final_daily.to_csv(daily_path, index=False)
    final_long.to_csv(long_path, index=False)

    strict_mode = f"strict_d{args.issue_gap_days}"
    strict_days = int(final_daily["forecast_mode"].eq(strict_mode).sum())
    recursive_days = int(final_daily["forecast_mode"].eq("planning_recursive").sum())
    summary = {
        "task_definition": f"Range forecast using holiday-router with strict D-{args.issue_gap_days} first and recursive planning afterwards.",
        "history_path": str(args.history_path),
        "router_summary_path": str(args.router_summary_path),
        "issue_gap_days": int(args.issue_gap_days),
        "start_date": str(start_date.date()),
        "end_date": str(end_date.date()),
        "max_actual_date": str(max_actual_date.date()),
        "days": int(len(final_daily)),
        f"strict_d{args.issue_gap_days}_days": strict_days,
        "planning_recursive_days": recursive_days,
        "route_name_counts": final_daily["route_name"].value_counts(dropna=False).to_dict(),
        "prediction_outputs": {
            "daily": str(daily_path),
            "long": str(long_path),
        },
    }
    save_summary_json(summary, summary_path)
    print(summary)


if __name__ == "__main__":
    main()
