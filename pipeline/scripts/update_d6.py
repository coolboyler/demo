from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from china_holiday_official import build_official_holiday_frame
from forecast_core import make_long_prediction_frame, save_summary_json
from forecast_d6 import (
    compute_date_type_group,
    compute_refined_type,
    get_max_actual_date,
    load_history,
    predict_target_date,
)


LOAD_COLUMNS = [f"load_h{hour:02d}" for hour in range(24)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append one day of actual total load history and immediately issue the next formal D-6 forecast."
    )
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--actual-file", type=Path, required=True, help="CSV file with one-day actual 24-hour total load.")
    parser.add_argument("--actual-date", type=str, default=None, help="Required when the file does not include a date column.")
    parser.add_argument("--daily-history-path", type=Path, default=Path("new/history_daily.csv"))
    parser.add_argument("--hourly-history-path", type=Path, default=Path("new/history_hourly.csv"))
    parser.add_argument("--router-summary-path", type=Path, default=Path("results/best_d6_summary.json"))
    parser.add_argument("--issue-gap-days", type=int, default=6)
    parser.add_argument("--forecast-output-prefix", default="forecast_d6")
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args()


def normalize_hour_column(column: str) -> str | None:
    col = str(column).strip().lower()
    if col.startswith("load_h") and len(col) == 8:
        return col
    if col.startswith("h") and len(col) == 3 and col[1:].isdigit():
        return f"load_{col}"
    if col.endswith(":00") and len(col) == 5 and col[:2].isdigit():
        return f"load_h{col[:2]}"
    if col.isdigit() and 0 <= int(col) <= 23:
        return f"load_h{int(col):02d}"
    return None


def load_actual_profile(actual_file: Path, actual_date_arg: str | None) -> tuple[pd.Timestamp, np.ndarray]:
    frame = pd.read_csv(actual_file)
    frame.columns = [str(column).strip() for column in frame.columns]

    if {"hour", "load"}.issubset(frame.columns):
        target_date = pd.Timestamp(actual_date_arg or frame.get("date", pd.Series([None])).iloc[0]).normalize()
        if pd.isna(target_date):
            raise ValueError("Actual file in long format must provide date column or --actual-date.")
        hour_map: dict[int, float] = {}
        for row in frame.itertuples(index=False):
            hour_value = str(getattr(row, "hour")).strip()
            if hour_value.endswith(":00"):
                hour_int = int(hour_value[:2])
            else:
                hour_int = int(hour_value)
            hour_map[hour_int] = float(getattr(row, "load"))
        if sorted(hour_map) != list(range(24)):
            raise ValueError("Actual file must contain exactly 24 hourly rows from 0 to 23.")
        profile = np.array([hour_map[hour] for hour in range(24)], dtype=float)
        return target_date, profile

    renamed: dict[str, str] = {}
    for column in frame.columns:
        normalized = normalize_hour_column(column)
        if normalized is not None:
            renamed[column] = normalized
    wide = frame.rename(columns=renamed)
    if len(wide) != 1:
        raise ValueError("Wide actual file must contain exactly one row.")

    date_source = actual_date_arg
    if date_source is None and "date" in wide.columns:
        date_source = str(wide.loc[0, "date"])
    if date_source is None:
        raise ValueError("Actual file must include a date column or use --actual-date.")

    target_date = pd.Timestamp(date_source).normalize()
    missing = [column for column in LOAD_COLUMNS if column not in wide.columns]
    if missing:
        raise ValueError(f"Actual file is missing hourly columns: {missing}")
    profile = wide.loc[0, LOAD_COLUMNS].to_numpy(dtype=float)
    return target_date, profile


def season_name(month: int) -> str:
    if month in {12, 1, 2}:
        return "winter"
    if month in {3, 4, 5}:
        return "spring"
    if month in {6, 7, 8}:
        return "summer"
    return "autumn"


def holiday_family_name(refined_type: str) -> str:
    if refined_type.startswith("holiday:"):
        return refined_type.split(":", 1)[1]
    if refined_type.startswith("pre_"):
        return refined_type[4:].rsplit("_d", 1)[0]
    if refined_type.startswith("post_"):
        return refined_type[5:].rsplit("_d", 1)[0]
    if refined_type.startswith("makeup_workday"):
        return "makeup_workday"
    if refined_type.startswith("weekend_"):
        return "weekend"
    if refined_type.startswith("workday_"):
        return "workday"
    return "other"


def phase_group_name(refined_type: str) -> str:
    if refined_type.startswith("holiday:"):
        return "holiday"
    if refined_type.startswith("pre_"):
        return "pre"
    if refined_type.startswith("post_"):
        return "post"
    if refined_type.startswith("makeup_workday"):
        return "makeup"
    if refined_type.startswith("weekend_"):
        return "weekend"
    if refined_type.startswith("workday_"):
        return "workday"
    return "other"


def build_daily_history_row(target_date: pd.Timestamp, profile: np.ndarray, history_columns: list[str]) -> pd.DataFrame:
    holiday_info = build_official_holiday_frame([target_date]).iloc[0]
    refined_type = compute_refined_type(target_date)
    row = {
        "date": target_date,
        "year": int(target_date.year),
        "month": int(target_date.month),
        "quarter": int(((target_date.month - 1) // 3) + 1),
        "weekofyear": int(target_date.isocalendar().week),
        "dayofmonth": int(target_date.day),
        "dayofweek": int(target_date.dayofweek),
        "dayofyear": int(target_date.dayofyear),
        "is_weekend": int(target_date.dayofweek in [5, 6]),
        "is_month_start": int(target_date.is_month_start),
        "is_month_end": int(target_date.is_month_end),
        "season": season_name(target_date.month),
        "is_holiday_cn": int(holiday_info["is_holiday_cn"]),
        "holiday_name_cn": holiday_info["holiday_name_cn"],
        "is_workday_cn": int(holiday_info["is_workday_cn"]),
        "is_makeup_workday": int(holiday_info["is_makeup_workday"]),
        "date_type_cn": "holiday"
        if int(holiday_info["is_holiday_cn"]) == 1
        else "makeup_workday"
        if int(holiday_info["is_makeup_workday"]) == 1
        else "weekend"
        if int(target_date.dayofweek in [5, 6]) == 1
        else "workday",
        "refined_date_type": refined_type,
        "date_type_group": compute_date_type_group(refined_type),
        "holiday_family": holiday_family_name(refined_type),
        "phase_group": phase_group_name(refined_type),
        "split": "future_actual",
        "actual_company_count": 5,
        "imputed_company_count": 0,
        "is_fully_actual_5_company": 1,
        "actual_daily_total": float(profile.sum()),
        "imputed_daily_total": 0.0,
        "daily_total": float(profile.sum()),
        "is_actual_observation": 1,
    }
    for hour in range(24):
        row[f"load_h{hour:02d}"] = float(profile[hour])

    aligned = {column: row.get(column, np.nan) for column in history_columns}
    if "is_actual_observation" not in aligned:
        aligned["is_actual_observation"] = 1
    return pd.DataFrame([aligned])


def build_hourly_history_rows(
    target_date: pd.Timestamp,
    profile: np.ndarray,
    hourly_history: pd.DataFrame,
) -> pd.DataFrame:
    first = hourly_history.iloc[0]
    holiday_info = build_official_holiday_frame([target_date]).iloc[0]
    rows: list[dict[str, object]] = []
    for hour in range(24):
        timestamp = target_date + pd.Timedelta(hours=hour)
        rows.append(
            {
                "site_id": first["site_id"],
                "site_name": first["site_name"],
                "city_cn": first["city_cn"],
                "city_en": first["city_en"],
                "province_cn": first["province_cn"],
                "timestamp": timestamp,
                "date": target_date,
                "hour": hour,
                "year": int(target_date.year),
                "month": int(target_date.month),
                "quarter": int(((target_date.month - 1) // 3) + 1),
                "weekofyear": int(target_date.isocalendar().week),
                "dayofmonth": int(target_date.day),
                "dayofweek": int(target_date.dayofweek),
                "dayofyear": int(target_date.dayofyear),
                "is_weekend": int(target_date.dayofweek in [5, 6]),
                "is_month_start": int(target_date.is_month_start),
                "is_month_end": int(target_date.is_month_end),
                "season": season_name(target_date.month),
                "is_holiday_cn": int(holiday_info["is_holiday_cn"]),
                "holiday_name_cn": holiday_info["holiday_name_cn"],
                "is_workday_cn": int(holiday_info["is_workday_cn"]),
                "is_makeup_workday": int(holiday_info["is_makeup_workday"]),
                "date_type_cn": "holiday"
                if int(holiday_info["is_holiday_cn"]) == 1
                else "makeup_workday"
                if int(holiday_info["is_makeup_workday"]) == 1
                else "weekend"
                if int(target_date.dayofweek in [5, 6]) == 1
                else "workday",
                "split": "future_actual",
                "actual_company_count": 5,
                "imputed_company_count": 0,
                "is_fully_actual_5_company": 1,
                "actual_daily_total": float(profile.sum()),
                "imputed_daily_total": 0.0,
                "日电量": float(profile.sum()),
                "load": float(profile[hour]),
            }
        )
    frame = pd.DataFrame(rows)
    aligned = frame.reindex(columns=hourly_history.columns)
    return aligned


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir
    daily_history_path = base_dir / args.daily_history_path
    hourly_history_path = base_dir / args.hourly_history_path
    actual_file = base_dir / args.actual_file if not args.actual_file.is_absolute() else args.actual_file
    router_summary_path = base_dir / args.router_summary_path if not args.router_summary_path.is_absolute() else args.router_summary_path
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    target_date, profile = load_actual_profile(actual_file, args.actual_date)
    daily_history = load_history(daily_history_path)
    hourly_history = pd.read_csv(hourly_history_path, parse_dates=["timestamp", "date"])
    if "is_actual_observation" not in daily_history.columns:
        daily_history["is_actual_observation"] = 1

    max_actual_date = get_max_actual_date(daily_history)
    expected_next_date = max_actual_date + pd.Timedelta(days=1)
    if not args.allow_overwrite and target_date != expected_next_date:
        raise ValueError(
            f"Actual date must equal next day after current max actual date. "
            f"Expected {expected_next_date.date()}, got {target_date.date()}."
        )

    daily_row = build_daily_history_row(target_date, profile, daily_history.columns.tolist())
    daily_updated = (
        pd.concat([daily_history[daily_history["date"] != target_date], daily_row], ignore_index=True)
        .sort_values("date")
        .reset_index(drop=True)
    )
    hourly_rows = build_hourly_history_rows(target_date, profile, hourly_history)
    hourly_updated = (
        pd.concat([hourly_history[hourly_history["date"] != target_date], hourly_rows], ignore_index=True)
        .sort_values(["timestamp"])
        .reset_index(drop=True)
    )

    daily_updated.to_csv(daily_history_path, index=False, encoding="utf-8-sig")
    hourly_updated.to_csv(hourly_history_path, index=False, encoding="utf-8-sig")

    next_target_date = get_max_actual_date(daily_updated) + pd.Timedelta(days=args.issue_gap_days)
    prediction_bundle = predict_target_date(
        history=daily_updated,
        target_date=next_target_date,
        router_summary_path=router_summary_path,
        issue_gap_days=args.issue_gap_days,
    )
    output_frame = prediction_bundle["output_frame"]
    long_frame = make_long_prediction_frame(output_frame)

    daily_forecast_path = results_dir / f"{args.forecast_output_prefix}_{next_target_date.strftime('%Y%m%d')}.csv"
    long_forecast_path = results_dir / f"{args.forecast_output_prefix}_{next_target_date.strftime('%Y%m%d')}_long.csv"
    summary_path = results_dir / f"update_d6_{target_date.strftime('%Y%m%d')}_summary.json"

    output_frame.to_csv(daily_forecast_path, index=False)
    long_frame.to_csv(long_forecast_path, index=False)

    summary = {
        "task_definition": "Append one day of actual total load history and issue the next formal D-6 forecast.",
        "actual_date": str(target_date.date()),
        "actual_daily_total": float(profile.sum()),
        "updated_daily_history_path": str(daily_history_path),
        "updated_hourly_history_path": str(hourly_history_path),
        "issue_gap_days": int(args.issue_gap_days),
        "next_target_date": str(next_target_date.date()),
        "next_issue_date": str((next_target_date - pd.Timedelta(days=args.issue_gap_days)).date()),
        "forecast_mode": str(output_frame.loc[0, "forecast_mode"]),
        "route_name": str(output_frame.loc[0, "route_name"]),
        "pred_daily_total": float(output_frame.loc[0, "pred_daily_total"]),
        "prediction_outputs": {
            "daily": str(daily_forecast_path),
            "long": str(long_forecast_path),
        },
    }
    save_summary_json(summary, summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
