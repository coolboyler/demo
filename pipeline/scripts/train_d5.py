from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data_lib import D5_DATASET_LEGACY_NAMES, D5_DATASET_NAME, D5_OUTPUT_PREFIX, resolve_existing_path
from forecast_core import (
    TARGET_COLUMNS,
    TARGET_HOURS,
    ensure_hourly_dataset,
    make_daily_prediction_frame,
    make_long_prediction_frame,
    per_site_metrics,
    prediction_metrics_from_wide,
    save_summary_json,
)
from holiday_lib import _build_refined_date_map


ISSUE_GAP_DAYS = 5
SHORT_LAGS = [5, 6, 7, 8, 9, 10, 11]
WEEKLY_LAGS = [7, 14, 21]
ALL_LAGS = sorted(set(SHORT_LAGS + WEEKLY_LAGS))


def get_issue_lag_config(issue_gap_days: int) -> tuple[list[int], list[int], list[int]]:
    short_lags = list(range(issue_gap_days, issue_gap_days + 7))
    weekly_lags = [7, 14, 21]
    all_lags = sorted(set(short_lags + weekly_lags))
    return short_lags, weekly_lags, all_lags


def build_issue_gap_dataset(hourly_dataset: pd.DataFrame, issue_gap_days: int = ISSUE_GAP_DAYS) -> pd.DataFrame:
    hourly = hourly_dataset.copy()
    hourly["date"] = pd.to_datetime(hourly["date"])
    short_lags, weekly_lags, all_lags = get_issue_lag_config(issue_gap_days)

    refined_date_map = _build_refined_date_map(hourly)

    daily_load_profile = hourly.pivot_table(index=["site_id", "date"], columns="hour", values="load")
    daily_load_profile.columns = [f"target_load_h{int(hour):02d}" for hour in daily_load_profile.columns]
    daily_load_profile = daily_load_profile.reset_index().sort_values(["site_id", "date"]).reset_index(drop=True)

    target_calendar = (
        hourly.groupby(["site_id", "date"], sort=True)
        .first()
        .reset_index()[
            [
                "site_id",
                "date",
                "site_name",
                "city_cn",
                "city_en",
                "province_cn",
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
            ]
        ]
        .merge(refined_date_map[["date", "refined_date_type", "date_type_group"]], on="date", how="left")
        .rename(
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
    )

    dataset = (
        target_calendar.merge(
            daily_load_profile.rename(columns={"date": "target_date"}),
            on=["site_id", "target_date"],
            how="left",
        )
        .sort_values(["site_id", "target_date"])
        .reset_index(drop=True)
    )
    dataset["issue_date"] = dataset["target_date"] - pd.Timedelta(days=issue_gap_days)

    lag_source = daily_load_profile.rename(columns={"date": "source_date"})
    lag_feature_names: list[str] = []
    for lag_day in all_lags:
        lag_frame = lag_source.copy()
        lag_frame["target_date"] = lag_frame["source_date"] + pd.Timedelta(days=lag_day)
        lag_frame = lag_frame.drop(columns=["source_date"])
        lag_frame = lag_frame.rename(
            columns={f"target_load_h{hour:02d}": f"lag{lag_day}_h{hour:02d}" for hour in TARGET_HOURS}
        )
        lag_feature_names.extend(
            [column for column in lag_frame.columns if column not in {"site_id", "target_date"}]
        )
        dataset = dataset.merge(lag_frame, on=["site_id", "target_date"], how="left")

    same_type_rows: list[dict[str, object]] = []
    for site_id, site_frame in dataset.groupby("site_id", sort=False):
        history: dict[str, list[tuple[pd.Timestamp, np.ndarray]]] = {}
        for row in site_frame.sort_values("target_date").itertuples(index=False):
            record = {"site_id": site_id, "target_date": row.target_date}
            all_matches = history.get(row.target_refined_date_type, [])
            eligible_matches = [match for match in all_matches if match[0] <= row.issue_date]
            last1 = eligible_matches[-1] if len(eligible_matches) >= 1 else None
            last2 = eligible_matches[-2] if len(eligible_matches) >= 2 else None
            record["same_type_reference_count"] = int(min(len(eligible_matches), 2))
            record["same_type_last1_date"] = last1[0] if last1 is not None else pd.NaT
            record["same_type_last2_date"] = last2[0] if last2 is not None else pd.NaT
            for hour in TARGET_HOURS:
                ref1_value = last1[1][hour] if last1 is not None else np.nan
                ref2_value = last2[1][hour] if last2 is not None else np.nan
                record[f"same_type_mean2_h{hour:02d}"] = (
                    np.nan
                    if np.isnan(ref1_value) and np.isnan(ref2_value)
                    else float(np.nanmean([ref1_value, ref2_value]))
                )
            profile = np.asarray([getattr(row, column) for column in TARGET_COLUMNS], dtype=float)
            history.setdefault(row.target_refined_date_type, []).append((row.target_date, profile))
            same_type_rows.append(record)

    dataset = dataset.merge(pd.DataFrame(same_type_rows), on=["site_id", "target_date"], how="left")

    required_columns = lag_feature_names + TARGET_COLUMNS
    dataset = dataset.dropna(subset=required_columns).sort_values(["target_date", "site_id"]).reset_index(drop=True)
    dataset["sample_id"] = dataset["site_id"] + "_" + dataset["target_date"].dt.strftime("%Y%m%d")
    dataset["issue_gap_days"] = int(issue_gap_days)
    return dataset


def build_d5_dataset(hourly_dataset: pd.DataFrame) -> pd.DataFrame:
    return build_issue_gap_dataset(hourly_dataset, issue_gap_days=ISSUE_GAP_DAYS)


def dataset_summary(dataset: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(len(dataset)),
        "columns": int(dataset.shape[1]),
        "sites": sorted(dataset["site_id"].unique().tolist()),
        "target_date_start": str(dataset["target_date"].min()),
        "target_date_end": str(dataset["target_date"].max()),
        "issue_date_start": str(dataset["issue_date"].min()),
        "issue_date_end": str(dataset["issue_date"].max()),
        "issue_gap_days": ISSUE_GAP_DAYS,
        "short_lags": SHORT_LAGS,
        "weekly_lags": WEEKLY_LAGS,
        "refined_type_count": int(dataset["target_refined_date_type"].nunique()),
    }


def weighted_profile(frame: pd.DataFrame, lag_days: list[int], weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    output = np.zeros((len(frame), 24))
    for hour in TARGET_HOURS:
        matrix = np.column_stack([frame[f"lag{lag_day}_h{hour:02d}"].to_numpy(dtype=float) for lag_day in lag_days])
        output[:, hour] = matrix @ weights
    return output


def same_type_scaled_profile(frame: pd.DataFrame, short_weights: np.ndarray, lag_days: list[int] | None = None) -> np.ndarray:
    short_weights = np.asarray(short_weights, dtype=float)
    short_weights = short_weights / short_weights.sum()
    lag_days = lag_days or SHORT_LAGS

    recent_profile = np.zeros((len(frame), 24))
    for lag_day, weight in zip(lag_days, short_weights):
        recent_profile += weight * np.column_stack(
            [frame[f"lag{lag_day}_h{hour:02d}"].to_numpy(dtype=float) for hour in TARGET_HOURS]
        )

    same_type_profile = np.column_stack(
        [frame[f"same_type_mean2_h{hour:02d}"].to_numpy(dtype=float) for hour in TARGET_HOURS]
    )
    same_type_profile = np.where(np.isnan(same_type_profile), recent_profile, same_type_profile)
    recent_total = recent_profile.sum(axis=1)
    same_type_total = np.maximum(same_type_profile.sum(axis=1), 1e-9)
    return same_type_profile * (recent_total / same_type_total)[:, None]


def build_report_markdown(
    dataset_path: Path,
    split_info: dict[str, str],
    validation_rows: list[dict[str, object]],
    test_rows: list[dict[str, object]],
    best_model_name: str,
    best_params: dict[str, object],
    prediction_path: Path,
    summary_path: Path,
) -> str:
    columns = ["model", "mae", "rmse", "wape_percent", "smape_percent", "mape_nonzero_percent", "r2"]
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"

    def row_to_markdown(row: dict[str, object]) -> str:
        return "| " + " | ".join(str(row[column]) for column in columns) + " |"

    return "\n".join(
        [
            "# D-5 发单口径数值外推报告",
            "",
            "## 口径",
            f"- 发单时点：`issue_date = target_date - {ISSUE_GAP_DAYS}` 天。",
            f"- 可用历史：仅允许使用 `D-{ISSUE_GAP_DAYS}` 及更早的负荷数据。",
            "- 同类型/同节日参考：只允许引用 `<= issue_date` 已发生的历史同类型日期。",
            "- 无天气，按站点逐日预测 24 点，再可按天汇总。",
            "",
            "## 切分",
            f"- 训练目标日区间：{split_info['train_target_start']} 到 {split_info['train_target_end']}",
            f"- 验证目标日区间：{split_info['validation_target_start']} 到 {split_info['validation_target_end']}",
            f"- 测试目标日区间：{split_info['test_target_start']} 到 {split_info['test_target_end']}",
            "",
            "## 验证结果",
            header,
            divider,
            *[row_to_markdown(row) for row in validation_rows],
            "",
            "## 测试结果",
            header,
            divider,
            *[row_to_markdown(row) for row in test_rows],
            "",
            f"最终选用模型：`{best_model_name}`",
            f"- 最优参数：`{best_params}`",
            "",
            "## 产物",
            f"- 数据集：`{dataset_path}`",
            f"- 测试集逐日预测：`{prediction_path}`",
            f"- 模型摘要 JSON：`{summary_path}`",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train D-5 issuance extrapolation load forecast.")
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset-name", default=D5_DATASET_NAME)
    parser.add_argument("--output-prefix", default=D5_OUTPUT_PREFIX)
    parser.add_argument("--validation-start", default="2025-11-01")
    parser.add_argument("--validation-end", default="2025-12-31")
    parser.add_argument("--test-start", default="2026-01-01")
    parser.add_argument("--test-end", default="2026-01-31")
    parser.add_argument("--rebuild-dataset", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_dir = args.base_dir / "processed"
    reports_dir = args.base_dir / "reports"
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    legacy_dataset_names = D5_DATASET_LEGACY_NAMES if args.dataset_name == D5_DATASET_NAME else ()
    dataset_path = resolve_existing_path(processed_dir, args.dataset_name, legacy_dataset_names)
    if args.rebuild_dataset or not dataset_path.exists():
        hourly_dataset, _ = ensure_hourly_dataset(base_dir=args.base_dir)
        dataset = build_d5_dataset(hourly_dataset)
        dataset_path = processed_dir / args.dataset_name
        dataset.to_csv(dataset_path, index=False)
    else:
        dataset = pd.read_csv(
            dataset_path,
            parse_dates=["issue_date", "target_date", "same_type_last1_date", "same_type_last2_date"],
        )

    validation_start = pd.Timestamp(args.validation_start)
    validation_end = pd.Timestamp(args.validation_end)
    test_start = pd.Timestamp(args.test_start)
    test_end = pd.Timestamp(args.test_end)

    train = dataset[dataset["target_date"] < validation_start].copy().reset_index(drop=True)
    validation = dataset[
        (dataset["target_date"] >= validation_start) & (dataset["target_date"] <= validation_end)
    ].copy().reset_index(drop=True)
    test = dataset[(dataset["target_date"] >= test_start) & (dataset["target_date"] <= test_end)].copy().reset_index(drop=True)

    split_info = {
        "train_target_start": str(train["target_date"].min()),
        "train_target_end": str(train["target_date"].max()),
        "validation_target_start": str(validation["target_date"].min()),
        "validation_target_end": str(validation["target_date"].max()),
        "test_target_start": str(test["target_date"].min()),
        "test_target_end": str(test["target_date"].max()),
    }

    validation_actual = validation[TARGET_COLUMNS].to_numpy()
    test_actual = test[TARGET_COLUMNS].to_numpy()

    validation_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    candidate_map: dict[str, tuple[np.ndarray, dict[str, object]]] = {}
    model_counter: dict[str, int] = {}

    for short_lambda in np.linspace(0.05, 0.60, 23):
        short_weights = np.exp(-short_lambda * np.arange(len(SHORT_LAGS)))
        validation_short = weighted_profile(validation, SHORT_LAGS, short_weights)
        test_short = weighted_profile(test, SHORT_LAGS, short_weights)

        same_validation = same_type_scaled_profile(validation, short_weights)
        same_test = same_type_scaled_profile(test, short_weights)
        for same_alpha in np.linspace(0.0, 0.50, 11):
            blend_validation = (1 - same_alpha) * validation_short + same_alpha * same_validation
            blend_test = (1 - same_alpha) * test_short + same_alpha * same_test
            model_counter["blend_short_same"] = model_counter.get("blend_short_same", 0) + 1
            model_id = f"blend_short_same_{model_counter['blend_short_same']:03d}"
            validation_rows.append({"model": model_id, **prediction_metrics_from_wide(validation_actual, blend_validation)})
            test_rows.append({"model": model_id, **prediction_metrics_from_wide(test_actual, blend_test)})
            candidate_map[model_id] = (
                blend_test,
                {"short_lambda": round(float(short_lambda), 4), "same_alpha": round(float(same_alpha), 4)},
            )

        for weekly_lambda in np.linspace(0.05, 0.60, 12):
            weekly_weights = np.exp(-weekly_lambda * np.arange(len(WEEKLY_LAGS)))
            validation_weekly = weighted_profile(validation, WEEKLY_LAGS, weekly_weights)
            test_weekly = weighted_profile(test, WEEKLY_LAGS, weekly_weights)
            for same_alpha in np.linspace(0.0, 0.35, 8):
                for weekly_alpha in np.linspace(0.0, 0.50, 11):
                    if same_alpha + weekly_alpha > 0.70:
                        continue
                    short_alpha = 1.0 - same_alpha - weekly_alpha
                    blend_validation = (
                        short_alpha * validation_short
                        + same_alpha * same_validation
                        + weekly_alpha * validation_weekly
                    )
                    blend_test = short_alpha * test_short + same_alpha * same_test + weekly_alpha * test_weekly
                    model_counter["blend_short_same_weekly"] = model_counter.get("blend_short_same_weekly", 0) + 1
                    model_id = f"blend_short_same_weekly_{model_counter['blend_short_same_weekly']:03d}"
                    validation_rows.append(
                        {"model": model_id, **prediction_metrics_from_wide(validation_actual, blend_validation)}
                    )
                    test_rows.append({"model": model_id, **prediction_metrics_from_wide(test_actual, blend_test)})
                    candidate_map[model_id] = (
                        blend_test,
                        {
                            "short_lambda": round(float(short_lambda), 4),
                            "same_alpha": round(float(same_alpha), 4),
                            "weekly_lambda": round(float(weekly_lambda), 4),
                            "weekly_alpha": round(float(weekly_alpha), 4),
                        },
                    )

    validation_rows = sorted(validation_rows, key=lambda row: row["wape_percent"])[:20]
    test_rows_by_model = {row["model"]: row for row in test_rows}
    best_validation_row = min(validation_rows, key=lambda row: row["wape_percent"])
    best_model_name = str(best_validation_row["model"])
    best_test_prediction, best_params = candidate_map[best_model_name]
    top_test_rows = sorted(
        [test_rows_by_model[row["model"]] for row in validation_rows],
        key=lambda row: row["wape_percent"],
    )

    prediction_frame = make_daily_prediction_frame(test, best_model_name, best_test_prediction)
    long_prediction_frame = make_long_prediction_frame(prediction_frame)
    site_rows = per_site_metrics(prediction_frame)

    prediction_path = processed_dir / f"{args.output_prefix}_test_predictions_daily.csv"
    long_path = processed_dir / f"{args.output_prefix}_test_predictions_long.csv"
    summary_path = processed_dir / f"{args.output_prefix}_forecast_summary.json"
    report_path = reports_dir / f"{args.output_prefix}_forecast_report.md"

    prediction_frame.to_csv(prediction_path, index=False)
    long_prediction_frame.to_csv(long_path, index=False)

    summary = {
        "task_definition": "Strict D-5 issuance forecast. Use only D-5 and earlier historical load plus same-type history available by issue time.",
        "anti_leakage_assumption": (
            "For target day D, issue_date is D-5. No D-4..D-1 realized load is used. Same-type references are restricted "
            "to dates that are <= issue_date."
        ),
        "dataset_summary": dataset_summary(dataset),
        "split_info": split_info,
        "selected_model": best_model_name,
        "selected_model_params": best_params,
        "selected_model_test_metrics": test_rows_by_model[best_model_name],
        "validation_top_rows": validation_rows,
        "test_top_rows": top_test_rows,
        "per_site_test_metrics": site_rows,
        "prediction_outputs": {"daily": str(prediction_path), "long": str(long_path)},
    }
    save_summary_json(summary, summary_path)
    report_path.write_text(
        build_report_markdown(
            dataset_path=dataset_path,
            split_info=split_info,
            validation_rows=validation_rows,
            test_rows=top_test_rows,
            best_model_name=best_model_name,
            best_params=best_params,
            prediction_path=prediction_path,
            summary_path=summary_path,
        ),
        encoding="utf-8",
    )

    print(summary)


if __name__ == "__main__":
    main()
