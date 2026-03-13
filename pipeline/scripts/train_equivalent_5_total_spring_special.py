from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_core import (
    daily_total_metrics,
    make_daily_prediction_frame,
    make_long_prediction_frame,
    prediction_metrics_from_wide,
    save_summary_json,
)
from train_d5 import SHORT_LAGS, WEEKLY_LAGS, get_issue_lag_config, same_type_scaled_profile, weighted_profile


TARGET_COLUMNS = [f"target_load_h{hour:02d}" for hour in range(24)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a Spring Festival sequence model on top of the strict D-5 rule baseline.")
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--rule-summary-path", type=Path, required=True)
    parser.add_argument("--output-prefix", required=True)
    return parser.parse_args()


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        dataset_path,
        parse_dates=["issue_date", "target_date", "same_type_last1_date", "same_type_last2_date"],
    )


def add_spring_meta(dataset: pd.DataFrame) -> pd.DataFrame:
    frame = dataset.sort_values("target_date").copy().reset_index(drop=True)
    is_spring_festival = frame["target_holiday_name_cn"].eq("Spring Festival")

    spring_key = [""] * len(frame)
    holiday_pos = [np.nan] * len(frame)
    holiday_len = [np.nan] * len(frame)

    index = 0
    while index < len(frame):
        if not is_spring_festival.iloc[index]:
            index += 1
            continue

        start = index
        while index + 1 < len(frame) and is_spring_festival.iloc[index + 1]:
            index += 1
        end = index
        block_len = end - start + 1

        for offset in range(3):
            target_idx = start - 3 + offset
            if target_idx >= 0:
                spring_key[target_idx] = f"pre_d{3 - offset}"

        for offset in range(block_len):
            target_idx = start + offset
            spring_key[target_idx] = f"holiday_d{offset + 1}"
            holiday_pos[target_idx] = offset + 1
            holiday_len[target_idx] = block_len

        for offset in range(3):
            target_idx = end + 1 + offset
            if target_idx < len(frame):
                spring_key[target_idx] = f"post_d{offset + 1}"

        index = end + 1

    frame["spring_key"] = spring_key
    frame["holiday_pos"] = holiday_pos
    frame["holiday_len"] = holiday_len
    frame["target_year"] = frame["target_date"].dt.year
    frame["spring_segment"] = np.where(frame["spring_key"].eq(""), "non_spring", "spring_window")
    return frame


def load_rule_params(summary_path: Path) -> dict[str, float]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    params = summary["selected_model_params"]
    return {
        "short_lambda": float(params["short_lambda"]),
        "same_alpha": float(params["same_alpha"]),
        "weekly_lambda": float(params["weekly_lambda"]),
        "weekly_alpha": float(params["weekly_alpha"]),
    }


def rule_prediction(frame: pd.DataFrame, params: dict[str, float], issue_gap_days: int = 5) -> np.ndarray:
    short_lags, weekly_lags, _ = get_issue_lag_config(issue_gap_days)
    short_weights = np.exp(-params["short_lambda"] * np.arange(len(short_lags)))
    weekly_weights = np.exp(-params["weekly_lambda"] * np.arange(len(weekly_lags)))
    short_profile = weighted_profile(frame, short_lags, short_weights)
    same_profile = same_type_scaled_profile(frame, short_weights, lag_days=short_lags)
    weekly_profile = weighted_profile(frame, weekly_lags, weekly_weights)
    short_alpha = 1.0 - params["same_alpha"] - params["weekly_alpha"]
    return short_alpha * short_profile + params["same_alpha"] * same_profile + params["weekly_alpha"] * weekly_profile


def interpolate_holiday_profile(previous_holidays: pd.DataFrame, target_pos: int, target_len: int) -> np.ndarray:
    source = previous_holidays.sort_values("holiday_pos")
    source_matrix = source[TARGET_COLUMNS].to_numpy(dtype=float)
    source_x = np.linspace(0.0, 1.0, len(source_matrix))
    target_x = (target_pos - 1) / max(target_len - 1, 1) if target_len > 1 else 0.0
    return np.array([np.interp(target_x, source_x, source_matrix[:, hour]) for hour in range(24)], dtype=float)


def spring_sequence_profile(
    row: pd.Series | pd.core.frame.Pandas,
    spring_history: pd.DataFrame,
    holiday_history_mode: str = "mean_all",
    shoulder_history_mode: str = "last1",
) -> np.ndarray:
    history = spring_history[spring_history["target_year"] < int(row.target_year)].copy()
    if history.empty or not row.spring_key:
        raise ValueError("Spring sequence profile requested without spring history.")

    if str(row.spring_key).startswith("holiday_d"):
        holiday_years = sorted(history.loc[history["target_holiday_name_cn"].eq("Spring Festival"), "target_year"].unique())
        if holiday_history_mode == "last1":
            holiday_years = holiday_years[-1:]
        profiles = []
        for target_year in holiday_years:
            previous_holidays = history[
                history["target_year"].eq(target_year) & history["target_holiday_name_cn"].eq("Spring Festival")
            ]
            profiles.append(interpolate_holiday_profile(previous_holidays, int(row.holiday_pos), int(row.holiday_len)))
        return np.mean(profiles, axis=0)

    same_key_rows = history[history["spring_key"].eq(row.spring_key)].sort_values("target_date")
    if shoulder_history_mode == "last1":
        same_key_rows = same_key_rows.tail(1)
    else:
        same_key_rows = same_key_rows.tail(2)
    return same_key_rows[TARGET_COLUMNS].mean().to_numpy(dtype=float)


def spring_sequence_prediction(
    frame: pd.DataFrame,
    spring_history: pd.DataFrame,
    holiday_history_mode: str = "mean_all",
    shoulder_history_mode: str = "last1",
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for row in frame.itertuples(index=False):
        rows.append(
            spring_sequence_profile(
                row=row,
                spring_history=spring_history,
                holiday_history_mode=holiday_history_mode,
                shoulder_history_mode=shoulder_history_mode,
            )
        )
    return np.vstack(rows)


def build_segment_metrics(prediction_frame: pd.DataFrame, segment_name: str) -> dict[str, float | int | str]:
    actual = prediction_frame[[f"actual_h{hour:02d}" for hour in range(24)]].to_numpy(dtype=float)
    predicted = prediction_frame[[f"pred_h{hour:02d}" for hour in range(24)]].to_numpy(dtype=float)
    hourly = prediction_metrics_from_wide(actual, predicted)
    daily = daily_total_metrics(prediction_frame)
    return {
        "segment": segment_name,
        "days": int(len(prediction_frame)),
        "hourly_accuracy_percent": float(100 - hourly["wape_percent"]),
        "hourly_wape_percent": float(hourly["wape_percent"]),
        "daily_accuracy_percent": float(daily["daily_total_accuracy_percent"]),
        "daily_wape_percent": float(daily["daily_total_wape_percent"]),
        "daily_bias_percent": float(daily["daily_total_bias_percent"]),
    }


def build_report_markdown(
    dataset_path: Path,
    rule_params: dict[str, float],
    spring_backtest_rows: list[dict[str, object]],
    comparison_rows: list[dict[str, object]],
    prediction_path: Path,
    summary_path: Path,
) -> str:
    backtest_columns = [
        "model_variant",
        "days",
        "hourly_accuracy_percent",
        "daily_accuracy_percent",
        "daily_bias_percent",
    ]
    comparison_columns = [
        "model_variant",
        "segment",
        "days",
        "hourly_accuracy_percent",
        "daily_accuracy_percent",
        "daily_bias_percent",
    ]
    backtest_header = "| " + " | ".join(backtest_columns) + " |"
    comparison_header = "| " + " | ".join(comparison_columns) + " |"
    divider = "| " + " | ".join(["---"] * len(backtest_columns)) + " |"
    comparison_divider = "| " + " | ".join(["---"] * len(comparison_columns)) + " |"

    def row_md(row: dict[str, object], columns: list[str]) -> str:
        return "| " + " | ".join(str(row[column]) for column in columns) + " |"

    return "\n".join(
        [
            "# 春节专模评估",
            "",
            f"- 数据集：`{dataset_path}`",
            f"- 基础规则 D-5 参数：`{rule_params}`",
            "- 春节假期日：按历年春节序列插值后取均值。",
            "- 春节前后 3 天：取最近一年同序位日。",
            "",
            "## 2025 春节回测",
            backtest_header,
            divider,
            *[row_md(row, backtest_columns) for row in spring_backtest_rows],
            "",
            "## 2026 测试分段对比",
            comparison_header,
            comparison_divider,
            *[row_md(row, comparison_columns) for row in comparison_rows],
            "",
            f"- 预测输出：`{prediction_path}`",
            f"- 摘要输出：`{summary_path}`",
        ]
    )


def main() -> None:
    args = parse_args()
    new_dir = args.base_dir / "new"
    new_dir.mkdir(parents=True, exist_ok=True)

    dataset = add_spring_meta(load_dataset(args.dataset_path))
    spring_history = dataset[dataset["spring_key"].ne("")].copy()
    rule_params = load_rule_params(args.rule_summary_path)

    validation = dataset[dataset["split"].eq("validation")].copy().reset_index(drop=True)
    test = dataset[dataset["split"].eq("test")].copy().reset_index(drop=True)
    spring_2025 = dataset[
        dataset["target_year"].eq(2025) & dataset["spring_key"].ne("")
    ].copy().reset_index(drop=True)

    validation_rule_pred = rule_prediction(validation, rule_params)
    test_rule_pred = rule_prediction(test, rule_params)
    spring_2025_rule_pred = rule_prediction(spring_2025, rule_params)
    spring_2025_special_pred = spring_sequence_prediction(spring_2025, spring_history)

    test_special_pred = test_rule_pred.copy()
    spring_mask = test["spring_key"].ne("").to_numpy()
    if spring_mask.any():
        test_special_pred[spring_mask, :] = spring_sequence_prediction(test.loc[spring_mask].reset_index(drop=True), spring_history)

    validation_frame = make_daily_prediction_frame(validation, "rule_plus_spring_sequence", validation_rule_pred)
    test_frame = make_daily_prediction_frame(test, "rule_plus_spring_sequence", test_special_pred)
    test_frame["spring_segment"] = test["spring_segment"].to_numpy()
    long_prediction_frame = make_long_prediction_frame(test_frame)

    spring_backtest_rows = []
    for model_variant, prediction in [
        ("rule_d5", spring_2025_rule_pred),
        ("spring_sequence_only", spring_2025_special_pred),
    ]:
        frame = make_daily_prediction_frame(spring_2025, model_variant, prediction)
        row = {"model_variant": model_variant, **build_segment_metrics(frame, "spring_window")}
        spring_backtest_rows.append(row)

    comparison_rows = []
    for model_variant, prediction in [
        ("rule_d5", test_rule_pred),
        ("rule_plus_spring_sequence", test_special_pred),
    ]:
        frame = make_daily_prediction_frame(test, model_variant, prediction)
        frame["spring_segment"] = test["spring_segment"].to_numpy()
        comparison_rows.append({"model_variant": model_variant, **build_segment_metrics(frame, "all_test")})
        comparison_rows.append(
            {
                "model_variant": model_variant,
                **build_segment_metrics(frame[frame["spring_segment"].eq("non_spring")].copy(), "non_spring"),
            }
        )
        comparison_rows.append(
            {
                "model_variant": model_variant,
                **build_segment_metrics(frame[frame["spring_segment"].eq("spring_window")].copy(), "spring_window"),
            }
        )

    prediction_path = new_dir / f"{args.output_prefix}_test_predictions_daily.csv"
    long_path = new_dir / f"{args.output_prefix}_test_predictions_long.csv"
    summary_path = new_dir / f"{args.output_prefix}_forecast_summary.json"
    report_path = new_dir / f"{args.output_prefix}_report.md"
    segment_path = new_dir / f"{args.output_prefix}_segment_metrics.csv"
    backtest_path = new_dir / f"{args.output_prefix}_spring_backtest_metrics.csv"

    test_frame.to_csv(prediction_path, index=False)
    long_prediction_frame.to_csv(long_path, index=False)
    pd.DataFrame(comparison_rows).to_csv(segment_path, index=False)
    pd.DataFrame(spring_backtest_rows).to_csv(backtest_path, index=False)

    summary = {
        "task_definition": "Strict D-5 rule baseline with Spring Festival sequence replacement for spring-window dates.",
        "dataset_path": str(args.dataset_path),
        "rule_summary_path": str(args.rule_summary_path),
        "split_info": {
            "validation_target_start": str(validation["target_date"].min().date()),
            "validation_target_end": str(validation["target_date"].max().date()),
            "test_target_start": str(test["target_date"].min().date()),
            "test_target_end": str(test["target_date"].max().date()),
        },
        "selected_strategy": {
            "holiday_history_mode": "mean_all",
            "shoulder_history_mode": "last1",
            "note": "Holiday days use interpolated average across prior Spring Festival windows; pre/post days use the most recent same-sequence day.",
        },
        "spring_2025_backtest": spring_backtest_rows,
        "test_segment_metrics": comparison_rows,
        "validation_daily_total_metrics": daily_total_metrics(validation_frame),
        "daily_total_metrics": daily_total_metrics(test_frame),
        "selected_model_validation_metrics": prediction_metrics_from_wide(
            validation[TARGET_COLUMNS].to_numpy(dtype=float),
            validation_rule_pred,
        ),
        "selected_model_test_metrics": prediction_metrics_from_wide(
            test[TARGET_COLUMNS].to_numpy(dtype=float),
            test_special_pred,
        ),
        "prediction_outputs": {
            "daily": str(prediction_path),
            "long": str(long_path),
            "segment_metrics": str(segment_path),
            "spring_backtest": str(backtest_path),
            "report": str(report_path),
        },
    }
    save_summary_json(summary, summary_path)
    report_path.write_text(
        build_report_markdown(
            dataset_path=args.dataset_path,
            rule_params=rule_params,
            spring_backtest_rows=spring_backtest_rows,
            comparison_rows=comparison_rows,
            prediction_path=prediction_path,
            summary_path=summary_path,
        ),
        encoding="utf-8",
    )

    print(summary)


if __name__ == "__main__":
    main()
