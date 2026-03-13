from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from china_holiday_official import OFFICIAL_HOLIDAY_MAP, build_official_holiday_frame
from forecast_core import make_long_prediction_frame, save_summary_json
from train_best_d6 import TARGET_COLUMNS, add_holiday_meta, apply_holiday_router
from train_equivalent_5_total_spring_special import load_rule_params
from train_d5 import get_issue_lag_config, same_type_scaled_profile, weighted_profile


LOAD_COLUMNS = [f"load_h{hour:02d}" for hour in range(24)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Operational future forecast using the holiday-router model with configurable issue gap.")
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--history-path", type=Path, default=Path("new/history_daily.csv"))
    parser.add_argument("--router-summary-path", type=Path, default=Path("results/best_d6_summary.json"))
    parser.add_argument("--issue-gap-days", type=int, default=6)
    parser.add_argument(
        "--target-date",
        type=str,
        default=None,
        help="Target date in YYYY-MM-DD format. Default is max_actual_date + issue_gap_days.",
    )
    parser.add_argument("--output-prefix", default="forecast_d6")
    return parser.parse_args()


def load_history(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"])
    if "is_actual_observation" not in frame.columns:
        frame["is_actual_observation"] = 1
    return frame


def to_model_history(history: pd.DataFrame, issue_gap_days: int) -> pd.DataFrame:
    frame = history.copy().sort_values("date").reset_index(drop=True)
    frame = frame.rename(
        columns={
            "date": "target_date",
            "year": "target_year",
            "month": "target_month",
            "quarter": "target_quarter",
            "weekofyear": "target_weekofyear",
            "dayofmonth": "target_dayofmonth",
            "dayofweek": "target_dayofweek",
            "dayofyear": "target_dayofyear",
            "is_weekend": "target_is_weekend",
            "is_month_start": "target_is_month_start",
            "is_month_end": "target_is_month_end",
            "season": "target_season",
            "is_holiday_cn": "target_is_holiday_cn",
            "holiday_name_cn": "target_holiday_name_cn",
            "is_workday_cn": "target_is_workday_cn",
            "is_makeup_workday": "target_is_makeup_workday",
            "date_type_cn": "target_date_type_cn",
            "refined_date_type": "target_refined_date_type",
            "date_type_group": "target_date_type_group",
        }
    )
    for hour in range(24):
        frame[f"target_load_h{hour:02d}"] = frame[f"load_h{hour:02d}"]
    frame["site_id"] = "all_equivalent5_ext"
    frame["site_name"] = "五家公司等价总量扩展版"
    frame["city_cn"] = "广东"
    frame["city_en"] = "Guangdong"
    frame["province_cn"] = "广东"
    frame["issue_date"] = frame["target_date"] - pd.Timedelta(days=issue_gap_days)
    frame["split"] = "history"
    frame["sample_id"] = "all_equivalent5_ext_" + frame["target_date"].dt.strftime("%Y%m%d")
    frame["same_type_reference_count"] = np.nan
    frame["same_type_last1_date"] = pd.NaT
    frame["same_type_last2_date"] = pd.NaT
    if "is_actual_observation" not in frame.columns:
        frame["is_actual_observation"] = 1
    return add_holiday_meta(frame)


def build_future_calendar_row(target_date: pd.Timestamp, issue_gap_days: int) -> pd.DataFrame:
    base = pd.DataFrame({"target_date": [target_date]})
    base["target_year"] = base["target_date"].dt.year
    base["target_month"] = base["target_date"].dt.month
    base["target_quarter"] = base["target_date"].dt.quarter
    base["target_weekofyear"] = base["target_date"].dt.isocalendar().week.astype(int)
    base["target_dayofmonth"] = base["target_date"].dt.day
    base["target_dayofweek"] = base["target_date"].dt.dayofweek
    base["target_dayofyear"] = base["target_date"].dt.dayofyear
    base["target_is_weekend"] = base["target_dayofweek"].isin([5, 6]).astype(int)
    base["target_is_month_start"] = base["target_date"].dt.is_month_start.astype(int)
    base["target_is_month_end"] = base["target_date"].dt.is_month_end.astype(int)
    base["target_season"] = base["target_month"].map(
        {
            12: "winter",
            1: "winter",
            2: "winter",
            3: "spring",
            4: "spring",
            5: "spring",
            6: "summer",
            7: "summer",
            8: "summer",
            9: "autumn",
            10: "autumn",
            11: "autumn",
        }
    )
    holiday_frame = build_official_holiday_frame([target_date])
    base = base.merge(holiday_frame.rename(columns={"date": "target_date"}), on="target_date", how="left")
    base["target_date_type_cn"] = np.select(
        [
            base["is_holiday_cn"].eq(1),
            base["is_makeup_workday"].eq(1),
            base["target_is_weekend"].eq(1),
        ],
        ["holiday", "makeup_workday", "weekend"],
        default="workday",
    )
    base = base.rename(
        columns={
            "is_holiday_cn": "target_is_holiday_cn",
            "holiday_name_cn": "target_holiday_name_cn",
            "is_workday_cn": "target_is_workday_cn",
            "is_makeup_workday": "target_is_makeup_workday",
        }
    )
    base["target_refined_date_type"] = base["target_date"].map(compute_refined_type)
    base["target_date_type_group"] = base["target_refined_date_type"].map(compute_date_type_group)
    base["site_id"] = "all_equivalent5_ext"
    base["site_name"] = "五家公司等价总量扩展版"
    base["city_cn"] = "广东"
    base["city_en"] = "Guangdong"
    base["province_cn"] = "广东"
    base["issue_date"] = base["target_date"] - pd.Timedelta(days=issue_gap_days)
    base["sample_id"] = "all_equivalent5_ext_" + base["target_date"].dt.strftime("%Y%m%d")
    base["split"] = "future"
    base["actual_h00"] = np.nan
    return add_holiday_meta(base)


def holiday_blocks() -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    holiday_dates = sorted(OFFICIAL_HOLIDAY_MAP.items(), key=lambda item: item[0])
    blocks: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    current_start: pd.Timestamp | None = None
    current_end: pd.Timestamp | None = None
    current_name: str | None = None
    for date_value, holiday_name in holiday_dates:
        if current_start is None:
            current_start = date_value
            current_end = date_value
            current_name = holiday_name
            continue
        if holiday_name == current_name and (date_value - current_end).days == 1:
            current_end = date_value
            continue
        blocks.append((current_start, current_end, str(current_name)))
        current_start = date_value
        current_end = date_value
        current_name = holiday_name
    if current_start is not None and current_end is not None and current_name is not None:
        blocks.append((current_start, current_end, str(current_name)))
    return blocks


HOLIDAY_BLOCKS = holiday_blocks()


def compute_refined_type(target_date: pd.Timestamp) -> str:
    holiday_name = OFFICIAL_HOLIDAY_MAP.get(pd.Timestamp(target_date).normalize())
    if holiday_name is not None:
        return f"holiday:{holiday_name}"
    for start_date, end_date, block_name in HOLIDAY_BLOCKS:
        days_before = (start_date - pd.Timestamp(target_date).normalize()).days
        if 1 <= days_before <= 3:
            return f"pre_{block_name}_d{days_before}"
        days_after = (pd.Timestamp(target_date).normalize() - end_date).days
        if 1 <= days_after <= 3:
            return f"post_{block_name}_d{days_after}"
    if pd.Timestamp(target_date).dayofweek >= 5:
        holiday_frame = build_official_holiday_frame([target_date])
        if int(holiday_frame.loc[0, "is_makeup_workday"]) == 1:
            return f"makeup_workday_w{pd.Timestamp(target_date).dayofweek}"
        return f"weekend_w{pd.Timestamp(target_date).dayofweek}"
    return f"workday_w{pd.Timestamp(target_date).dayofweek}"


def compute_date_type_group(refined_type: str) -> str:
    if refined_type.startswith("holiday:"):
        return "holiday"
    if refined_type.startswith("pre_"):
        return "pre_holiday"
    if refined_type.startswith("post_"):
        return "post_holiday"
    if refined_type.startswith("makeup_workday"):
        return "makeup_workday"
    if refined_type.startswith("weekend_"):
        return "weekend"
    return "workday"


def add_lag_features(target_row: pd.DataFrame, model_history: pd.DataFrame, issue_gap_days: int) -> pd.DataFrame:
    row = target_row.copy()
    target_date = row.loc[0, "target_date"]
    short_lags, weekly_lags, all_lags = get_issue_lag_config(issue_gap_days)
    lag_feature_values: dict[str, float] = {}
    lag_predicted_flags: list[int] = []
    for lag_day in all_lags:
        lag_date = target_date - pd.Timedelta(days=lag_day)
        lag_match = model_history[model_history["target_date"].eq(lag_date)]
        if lag_match.empty:
            raise ValueError(f"Missing historical profile for lag day D-{lag_day} ({lag_date.date()}).")
        lag_series = lag_match.iloc[0]
        lag_predicted_flags.append(int(1 - int(lag_series.get("is_actual_observation", 1))))
        for hour in range(24):
            lag_feature_values[f"lag{lag_day}_h{hour:02d}"] = float(lag_series[f"target_load_h{hour:02d}"])

    eligible = model_history[model_history["target_date"].le(row.loc[0, "issue_date"])]
    same_type = eligible[eligible["target_refined_date_type"].eq(row.loc[0, "target_refined_date_type"])].sort_values("target_date")
    last1 = same_type.iloc[-1] if len(same_type) >= 1 else None
    last2 = same_type.iloc[-2] if len(same_type) >= 2 else None
    same_type_feature_values: dict[str, float] = {}
    for hour in range(24):
        values = []
        if last1 is not None:
            values.append(float(last1[f"target_load_h{hour:02d}"]))
        if last2 is not None:
            values.append(float(last2[f"target_load_h{hour:02d}"]))
        same_type_feature_values[f"same_type_mean2_h{hour:02d}"] = float(np.mean(values)) if values else np.nan

    same_type_predicted_count = int(sum(1 - int(ref.get("is_actual_observation", 1)) for ref in [last1, last2] if ref is not None))
    meta_values = {
        "same_type_reference_count": int(min(len(same_type), 2)),
        "same_type_last1_date": last1["target_date"] if last1 is not None else pd.NaT,
        "same_type_last2_date": last2["target_date"] if last2 is not None else pd.NaT,
        "lag_predicted_count": int(sum(lag_predicted_flags)),
        "same_type_predicted_count": same_type_predicted_count,
        "used_predicted_history": int(sum(lag_predicted_flags) + same_type_predicted_count > 0),
    }
    feature_frame = pd.DataFrame([{**lag_feature_values, **same_type_feature_values, **meta_values}])
    return pd.concat([row.reset_index(drop=True), feature_frame], axis=1)


def base_rule_prediction(frame: pd.DataFrame, params: dict[str, float], issue_gap_days: int) -> np.ndarray:
    short_lags, weekly_lags, _ = get_issue_lag_config(issue_gap_days)
    short_weights = np.exp(-params["short_lambda"] * np.arange(len(short_lags)))
    weekly_weights = np.exp(-params["weekly_lambda"] * np.arange(len(weekly_lags)))
    short_profile = weighted_profile(frame, short_lags, short_weights)
    same_profile = same_type_scaled_profile(frame, short_weights, lag_days=short_lags)
    weekly_profile = weighted_profile(frame, weekly_lags, weekly_weights)
    short_alpha = 1.0 - params["same_alpha"] - params["weekly_alpha"]
    return short_alpha * short_profile + params["same_alpha"] * same_profile + params["weekly_alpha"] * weekly_profile


def load_router_config(router_summary_path: Path) -> tuple[dict[str, float], set[str], bool, dict[str, dict[str, float]]]:
    router_summary = json.loads(router_summary_path.read_text(encoding="utf-8"))
    rule_summary_path = Path(router_summary["rule_summary_path"])
    if not rule_summary_path.is_absolute():
        rule_summary_path = router_summary_path.parent.parent / rule_summary_path
    rule_params = load_rule_params(rule_summary_path)
    active_families = set(router_summary["active_holiday_families"])
    makeup_active = bool(router_summary["makeup_active"])
    ordinary_config = {
        str(key): {"top_k": int(value["top_k"]), "alpha": float(value["alpha"])}
        for key, value in router_summary.get("ordinary_similar_config", {}).items()
    }
    return rule_params, active_families, makeup_active, ordinary_config


def get_max_actual_date(history: pd.DataFrame) -> pd.Timestamp:
    actual_mask = history["is_actual_observation"].fillna(1).astype(int).eq(1)
    if actual_mask.any():
        return pd.Timestamp(history.loc[actual_mask, "date"].max()).normalize()
    return pd.Timestamp(history["date"].max()).normalize()


def build_prediction_output(
    target_row: pd.DataFrame,
    routed_prediction: np.ndarray,
    replacement_rows: list[dict[str, object]],
    best_model_name: str,
    forecast_mode: str,
) -> pd.DataFrame:
    output_frame = target_row[
        [
            "sample_id",
            "site_id",
            "site_name",
            "city_cn",
            "issue_date",
            "target_date",
            "target_refined_date_type",
            "target_holiday_name_cn",
            "target_date_type_group",
            "holiday_context_tag",
            "holiday_context_family",
            "holiday_context_side",
            "holiday_context_bucket",
            "days_since_last_holiday",
            "days_to_next_holiday",
            "last_holiday_family",
            "next_holiday_family",
            "same_type_reference_count",
            "same_type_last1_date",
            "same_type_last2_date",
            "lag_predicted_count",
            "same_type_predicted_count",
            "used_predicted_history",
        ]
    ].copy()
    output_frame["best_model_name"] = best_model_name
    output_frame["forecast_mode"] = forecast_mode
    for hour in range(24):
        output_frame[f"actual_h{hour:02d}"] = np.nan
        output_frame[f"pred_h{hour:02d}"] = routed_prediction[:, hour]
    output_frame["pred_daily_total"] = routed_prediction.sum(axis=1)
    output_frame["route_name"] = replacement_rows[0]["route_name"] if replacement_rows else "base_rule"
    output_frame["is_future_prediction"] = 1
    return output_frame


def predict_target_date(
    history: pd.DataFrame,
    target_date: pd.Timestamp,
    router_summary_path: Path,
    issue_gap_days: int,
) -> dict[str, Any]:
    model_history = to_model_history(history, issue_gap_days=issue_gap_days)
    raw_target_row = build_future_calendar_row(target_date, issue_gap_days=issue_gap_days)
    context_source_columns = [
        "target_date",
        "target_holiday_name_cn",
        "target_refined_date_type",
        "target_date_type_group",
    ]
    context_frame = pd.concat(
        [
            model_history[context_source_columns],
            raw_target_row[context_source_columns],
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["target_date"], keep="last")
    context_frame = add_holiday_meta(context_frame)
    context_row = context_frame[context_frame["target_date"].eq(target_date)].copy()
    target_row = raw_target_row.drop(
        columns=[
            column
            for column in [
                "holiday_family",
                "holiday_segment",
                "holiday_rel_key",
                "holiday_pos",
                "holiday_len",
                "holiday_context_tag",
                "holiday_context_family",
                "holiday_context_side",
                "holiday_context_bucket",
                "days_since_last_holiday",
                "days_to_next_holiday",
                "last_holiday_family",
                "next_holiday_family",
            ]
            if column in raw_target_row.columns
        ],
        errors="ignore",
    ).merge(context_row, on=context_source_columns, how="left")
    target_row = add_lag_features(
        target_row,
        model_history,
        issue_gap_days=issue_gap_days,
    )
    rule_params, active_families, makeup_active, ordinary_config = load_router_config(router_summary_path)
    base_prediction = base_rule_prediction(target_row, rule_params, issue_gap_days=issue_gap_days)
    routed_prediction, replacement_rows = apply_holiday_router(
        frame=target_row,
        full_history=model_history,
        base_prediction=base_prediction,
        active_families=active_families,
        makeup_active=makeup_active,
        ordinary_config_map=ordinary_config,
    )
    max_actual_date = get_max_actual_date(history)
    forecast_mode = f"strict_d{issue_gap_days}" if target_date <= max_actual_date + pd.Timedelta(days=issue_gap_days) else "planning_recursive"
    output_frame = build_prediction_output(
        target_row=target_row,
        routed_prediction=routed_prediction,
        replacement_rows=replacement_rows,
        best_model_name="holiday_router_future",
        forecast_mode=forecast_mode,
    )
    return {
        "output_frame": output_frame,
        "long_frame": make_long_prediction_frame(output_frame),
        "target_row": target_row,
        "prediction_vector": routed_prediction[0],
        "replacement_rows": replacement_rows,
        "max_actual_date": max_actual_date,
    }


def prediction_output_to_history_row(target_row: pd.DataFrame, prediction_vector: np.ndarray) -> pd.DataFrame:
    row = target_row.iloc[0]
    history_row = {
        "date": row["target_date"],
        "year": int(row["target_year"]),
        "month": int(row["target_month"]),
        "quarter": int(row["target_quarter"]),
        "weekofyear": int(row["target_weekofyear"]),
        "dayofmonth": int(row["target_dayofmonth"]),
        "dayofweek": int(row["target_dayofweek"]),
        "dayofyear": int(row["target_dayofyear"]),
        "is_weekend": int(row["target_is_weekend"]),
        "is_month_start": int(row["target_is_month_start"]),
        "is_month_end": int(row["target_is_month_end"]),
        "season": row["target_season"],
        "is_holiday_cn": int(row["target_is_holiday_cn"]),
        "holiday_name_cn": row["target_holiday_name_cn"],
        "is_workday_cn": int(row["target_is_workday_cn"]),
        "is_makeup_workday": int(row["target_is_makeup_workday"]),
        "date_type_cn": row["target_date_type_cn"],
        "refined_date_type": row["target_refined_date_type"],
        "date_type_group": row["target_date_type_group"],
        "holiday_family": row.get("holiday_family", ""),
        "phase_group": "future_prediction",
        "split": "future_prediction",
        "actual_company_count": np.nan,
        "imputed_company_count": 5,
        "is_fully_actual_5_company": 0,
        "actual_daily_total": np.nan,
        "imputed_daily_total": float(np.sum(prediction_vector)),
        "daily_total": float(np.sum(prediction_vector)),
        "is_actual_observation": 0,
    }
    for hour in range(24):
        history_row[f"load_h{hour:02d}"] = float(prediction_vector[hour])
    return pd.DataFrame([history_row])


def main() -> None:
    args = parse_args()
    results_dir = args.base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    history = load_history(args.history_path)
    max_actual_date = get_max_actual_date(history)
    target_date = pd.Timestamp(args.target_date).normalize() if args.target_date else max_actual_date + pd.Timedelta(days=args.issue_gap_days)
    if target_date > max_actual_date + pd.Timedelta(days=args.issue_gap_days):
        raise ValueError(
            f"Target date {target_date.date()} exceeds max_actual_date + {args.issue_gap_days} "
            f"({(max_actual_date + pd.Timedelta(days=args.issue_gap_days)).date()}). "
            f"Operational D-{args.issue_gap_days} forecasting should be run daily with updated actuals."
        )

    prediction_bundle = predict_target_date(
        history=history,
        target_date=target_date,
        router_summary_path=args.router_summary_path,
        issue_gap_days=args.issue_gap_days,
    )
    output_frame = prediction_bundle["output_frame"]
    long_frame = prediction_bundle["long_frame"]

    daily_path = results_dir / f"{args.output_prefix}_{target_date.strftime('%Y%m%d')}.csv"
    long_path = results_dir / f"{args.output_prefix}_{target_date.strftime('%Y%m%d')}_long.csv"
    summary_path = results_dir / f"{args.output_prefix}_{target_date.strftime('%Y%m%d')}_summary.json"
    match_path = results_dir / f"{args.output_prefix}_{target_date.strftime('%Y%m%d')}_matches.csv"

    output_frame.to_csv(daily_path, index=False)
    long_frame.to_csv(long_path, index=False)
    pd.DataFrame(prediction_bundle["replacement_rows"]).to_csv(match_path, index=False)
    summary = {
        "task_definition": f"Operational D-{args.issue_gap_days} future forecast using the holiday-router model.",
        "history_path": str(args.history_path),
        "router_summary_path": str(args.router_summary_path),
        "issue_gap_days": int(args.issue_gap_days),
        "target_date": str(target_date.date()),
        "issue_date": str((target_date - pd.Timedelta(days=args.issue_gap_days)).date()),
        "max_actual_date": str(max_actual_date.date()),
        "route_name": output_frame.loc[0, "route_name"],
        "forecast_mode": output_frame.loc[0, "forecast_mode"],
        "used_predicted_history": int(output_frame.loc[0, "used_predicted_history"]),
        "pred_daily_total": float(output_frame.loc[0, "pred_daily_total"]),
        "prediction_outputs": {
            "daily": str(daily_path),
            "long": str(long_path),
            "matches": str(match_path),
        },
    }
    save_summary_json(summary, summary_path)
    print(summary)


if __name__ == "__main__":
    main()
