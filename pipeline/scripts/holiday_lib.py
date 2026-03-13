from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data_lib import (
    HOURLY_DATASET_LEGACY_NAMES,
    HOURLY_DATASET_NAME,
    build_hourly_dataset,
    markdown_table,
    regression_metrics,
    resolve_existing_path,
    write_json,
)


WEEKLY_LAG_DAYS = [7, 14, 21]
TARGET_HOURS = list(range(24))
TARGET_COLUMNS = [f"target_load_h{hour:02d}" for hour in TARGET_HOURS]


@dataclass
class SplitBundle:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    split_info: dict[str, str]


def ensure_hourly_dataset(
    base_dir: Path,
    dataset_name: str = HOURLY_DATASET_NAME,
    refresh_weather: bool = False,
) -> tuple[pd.DataFrame, Path]:
    legacy_names = HOURLY_DATASET_LEGACY_NAMES if dataset_name == HOURLY_DATASET_NAME else ()
    dataset_path = resolve_existing_path(base_dir / "processed", dataset_name, legacy_names)
    if dataset_path.exists():
        return pd.read_csv(dataset_path, parse_dates=["timestamp", "date"]), dataset_path

    hourly_dataset = build_hourly_dataset(
        raw_dir=base_dir / "raw",
        weather_dir=base_dir / "weather",
        refresh_weather=refresh_weather,
    )
    dataset_path = base_dir / "processed" / dataset_name
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    hourly_dataset.to_csv(dataset_path, index=False)
    return hourly_dataset, dataset_path


def _build_refined_date_map(hourly_dataset: pd.DataFrame) -> pd.DataFrame:
    calendar = (
        hourly_dataset.groupby("date", sort=True)
        .first()
        .reset_index()[
            [
                "date",
                "dayofweek",
                "is_weekend",
                "is_holiday_cn",
                "holiday_name_cn",
                "is_makeup_workday",
                "date_type_cn",
            ]
        ]
        .sort_values("date")
        .reset_index(drop=True)
    )
    calendar["holiday_name_cn"] = calendar["holiday_name_cn"].fillna("non_holiday")

    holiday_blocks: list[dict[str, object]] = []
    in_block = False
    previous_date: pd.Timestamp | None = None
    previous_holiday_name: str | None = None
    for row in calendar.itertuples(index=False):
        holiday_name = row.holiday_name_cn if row.is_holiday_cn == 1 else None
        if row.is_holiday_cn == 1:
            if (
                not in_block
                or holiday_name != previous_holiday_name
                or previous_date is None
                or (row.date - previous_date).days != 1
            ):
                if in_block:
                    holiday_blocks[-1]["end"] = previous_date
                holiday_blocks.append({"holiday_name": holiday_name, "start": row.date, "end": row.date})
                in_block = True
            previous_holiday_name = holiday_name
            previous_date = row.date
        else:
            if in_block:
                holiday_blocks[-1]["end"] = previous_date
                in_block = False
                previous_holiday_name = None
            previous_date = row.date

    if in_block and previous_date is not None:
        holiday_blocks[-1]["end"] = previous_date

    def refined_type(row: pd.Series) -> str:
        if int(row["is_holiday_cn"]) == 1:
            return f"holiday:{row['holiday_name_cn']}"

        for block in holiday_blocks:
            day_before = (pd.Timestamp(block["start"]) - row["date"]).days
            if 1 <= day_before <= 3:
                return f"pre_{block['holiday_name']}_d{day_before}"

            day_after = (row["date"] - pd.Timestamp(block["end"])).days
            if 1 <= day_after <= 3:
                return f"post_{block['holiday_name']}_d{day_after}"

        if int(row["is_makeup_workday"]) == 1:
            return f"makeup_workday_w{int(row['dayofweek'])}"
        if int(row["is_weekend"]) == 1:
            return f"weekend_w{int(row['dayofweek'])}"
        return f"workday_w{int(row['dayofweek'])}"

    calendar["refined_date_type"] = calendar.apply(refined_type, axis=1)
    calendar["date_type_group"] = np.select(
        [
            calendar["refined_date_type"].str.startswith("holiday:"),
            calendar["refined_date_type"].str.startswith("pre_"),
            calendar["refined_date_type"].str.startswith("post_"),
            calendar["refined_date_type"].str.startswith("makeup_workday"),
            calendar["refined_date_type"].str.startswith("weekend_"),
        ],
        ["holiday", "pre_holiday", "post_holiday", "makeup_workday", "weekend"],
        default="workday",
    )
    return calendar


def _make_same_type_reference_features(dataset: pd.DataFrame) -> pd.DataFrame:
    same_type_rows: list[dict[str, object]] = []
    load_columns = [f"target_load_h{hour:02d}" for hour in TARGET_HOURS]

    for site_id, site_frame in dataset.groupby("site_id", sort=False):
        history: dict[str, list[tuple[pd.Timestamp, np.ndarray]]] = {}
        for row in site_frame.sort_values("target_date").itertuples(index=False):
            row_dict = {"site_id": site_id, "target_date": row.target_date}
            profile = np.asarray([getattr(row, column) for column in load_columns], dtype=float)
            historical_matches = history.get(row.target_refined_date_type, [])
            last1 = historical_matches[-1] if len(historical_matches) >= 1 else None
            last2 = historical_matches[-2] if len(historical_matches) >= 2 else None

            row_dict["same_type_reference_count"] = int(min(len(historical_matches), 2))
            row_dict["same_type_last1_date"] = last1[0] if last1 is not None else pd.NaT
            row_dict["same_type_last2_date"] = last2[0] if last2 is not None else pd.NaT
            row_dict["same_type_last1_gap_days"] = (
                int((row.target_date - last1[0]).days) if last1 is not None else np.nan
            )
            row_dict["same_type_last2_gap_days"] = (
                int((row.target_date - last2[0]).days) if last2 is not None else np.nan
            )

            for hour in TARGET_HOURS:
                ref1_value = last1[1][hour] if last1 is not None else np.nan
                ref2_value = last2[1][hour] if last2 is not None else np.nan
                row_dict[f"same_type_last1_h{hour:02d}"] = ref1_value
                row_dict[f"same_type_last2_h{hour:02d}"] = ref2_value
                if np.isnan(ref1_value) and np.isnan(ref2_value):
                    row_dict[f"same_type_mean2_h{hour:02d}"] = np.nan
                elif np.isnan(ref2_value):
                    row_dict[f"same_type_mean2_h{hour:02d}"] = float(ref1_value)
                elif np.isnan(ref1_value):
                    row_dict[f"same_type_mean2_h{hour:02d}"] = float(ref2_value)
                else:
                    row_dict[f"same_type_mean2_h{hour:02d}"] = float((ref1_value + ref2_value) / 2.0)

            same_type_rows.append(row_dict)
            history.setdefault(row.target_refined_date_type, []).append((row.target_date, profile))

    return pd.DataFrame(same_type_rows)


def build_holiday_similarity_dataset(hourly_dataset: pd.DataFrame) -> pd.DataFrame:
    hourly = hourly_dataset.copy()
    hourly["date"] = pd.to_datetime(hourly["date"])

    refined_date_map = _build_refined_date_map(hourly)

    daily_load_profile = hourly.pivot_table(index=["site_id", "date"], columns="hour", values="load")
    daily_load_profile.columns = [f"target_load_h{int(hour):02d}" for hour in daily_load_profile.columns]
    daily_load_profile = daily_load_profile.reset_index().sort_values(["site_id", "date"]).reset_index(drop=True)

    daily_total = (
        hourly.groupby(["site_id", "date"], sort=True)
        .agg(target_daily_total=("load", "sum"))
        .reset_index()
    )

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
        target_calendar.merge(daily_load_profile.rename(columns={"date": "target_date"}), on=["site_id", "target_date"], how="left")
        .merge(daily_total.rename(columns={"date": "target_date"}), on=["site_id", "target_date"], how="left")
        .sort_values(["site_id", "target_date"])
        .reset_index(drop=True)
    )

    lag_feature_names: list[str] = []
    lag_source = daily_load_profile.rename(columns={"date": "source_date"})
    for lag_day in WEEKLY_LAG_DAYS:
        lag_frame = lag_source.copy()
        lag_frame["target_date"] = lag_frame["source_date"] + pd.Timedelta(days=lag_day)
        lag_frame = lag_frame.drop(columns=["source_date"])
        lag_frame = lag_frame.rename(
            columns={
                f"target_load_h{hour:02d}": f"lag{lag_day}_load_h{hour:02d}"
                for hour in TARGET_HOURS
            }
        )
        lag_feature_names.extend(
            [column for column in lag_frame.columns if column not in {"site_id", "target_date"}]
        )
        dataset = dataset.merge(lag_frame, on=["site_id", "target_date"], how="left")

    same_type_features = _make_same_type_reference_features(dataset)
    dataset = dataset.merge(same_type_features, on=["site_id", "target_date"], how="left")
    dataset["issue_date"] = dataset["target_date"] - pd.Timedelta(days=1)

    for hour in TARGET_HOURS:
        same_hour_columns = [f"lag{lag_day}_load_h{hour:02d}" for lag_day in WEEKLY_LAG_DAYS]
        dataset[f"weekly_lag_mean_h{hour:02d}"] = dataset[same_hour_columns].mean(axis=1)
        dataset[f"weekly_lag_std_h{hour:02d}"] = dataset[same_hour_columns].std(axis=1)

    dataset["weekly_lag_total_mean"] = dataset[
        [f"lag{lag_day}_load_h{hour:02d}" for lag_day in WEEKLY_LAG_DAYS for hour in TARGET_HOURS]
    ].to_numpy().reshape(len(dataset), len(WEEKLY_LAG_DAYS), len(TARGET_HOURS)).sum(axis=2).mean(axis=1)

    required_columns = lag_feature_names + [f"same_type_mean2_h{hour:02d}" for hour in TARGET_HOURS] + TARGET_COLUMNS
    dataset = dataset.dropna(subset=required_columns).sort_values(["target_date", "site_id"]).reset_index(drop=True)
    dataset["sample_id"] = dataset["site_id"] + "_" + dataset["target_date"].dt.strftime("%Y%m%d")
    return dataset


def holiday_similarity_dataset_summary(dataset: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(len(dataset)),
        "columns": int(dataset.shape[1]),
        "sites": sorted(dataset["site_id"].unique().tolist()),
        "target_date_start": str(dataset["target_date"].min()),
        "target_date_end": str(dataset["target_date"].max()),
        "issue_date_start": str(dataset["issue_date"].min()),
        "issue_date_end": str(dataset["issue_date"].max()),
        "weekly_lag_days": WEEKLY_LAG_DAYS,
        "target_hours": TARGET_HOURS,
        "refined_type_count": int(dataset["target_refined_date_type"].nunique()),
        "refined_type_examples": sorted(dataset["target_refined_date_type"].drop_duplicates().tolist())[:12],
    }


def split_holiday_similarity_dataset(
    dataset: pd.DataFrame,
    train_end: str = "2025-10-31",
    validation_end: str = "2025-12-31",
) -> SplitBundle:
    train_end_ts = pd.Timestamp(train_end)
    validation_end_ts = pd.Timestamp(validation_end)

    train = dataset[dataset["target_date"] <= train_end_ts].copy()
    validation = dataset[(dataset["target_date"] > train_end_ts) & (dataset["target_date"] <= validation_end_ts)].copy()
    test = dataset[dataset["target_date"] > validation_end_ts].copy()

    split_info = {
        "train_target_start": str(train["target_date"].min()),
        "train_target_end": str(train["target_date"].max()),
        "validation_target_start": str(validation["target_date"].min()),
        "validation_target_end": str(validation["target_date"].max()),
        "test_target_start": str(test["target_date"].min()),
        "test_target_end": str(test["target_date"].max()),
        "train_issue_start": str(train["issue_date"].min()),
        "train_issue_end": str(train["issue_date"].max()),
        "validation_issue_start": str(validation["issue_date"].min()),
        "validation_issue_end": str(validation["issue_date"].max()),
        "test_issue_start": str(test["issue_date"].min()),
        "test_issue_end": str(test["issue_date"].max()),
    }
    return SplitBundle(train=train, validation=validation, test=test, split_info=split_info)


def prediction_metrics_from_wide(actual: pd.DataFrame | np.ndarray, prediction: pd.DataFrame | np.ndarray) -> dict[str, float]:
    actual_values = actual.to_numpy() if isinstance(actual, pd.DataFrame) else np.asarray(actual)
    prediction_values = prediction.to_numpy() if isinstance(prediction, pd.DataFrame) else np.asarray(prediction)
    return regression_metrics(actual_values.ravel(), prediction_values.ravel())


def baseline_predictions(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    lag7 = frame[[f"lag7_load_h{hour:02d}" for hour in TARGET_HOURS]].to_numpy()
    mean721 = np.column_stack(
        [
            frame[[f"lag{lag_day}_load_h{hour:02d}" for lag_day in WEEKLY_LAG_DAYS]].mean(axis=1).to_numpy()
            for hour in TARGET_HOURS
        ]
    )
    same_type_last1 = frame[[f"same_type_last1_h{hour:02d}" for hour in TARGET_HOURS]].to_numpy()
    same_type_mean2 = frame[[f"same_type_mean2_h{hour:02d}" for hour in TARGET_HOURS]].to_numpy()
    return {
        "lag7_copy": lag7,
        "mean_7_14_21": mean721,
        "same_type_last1": same_type_last1,
        "same_type_mean2": same_type_mean2,
    }


def _feature_column_name(feature_key: str, hour: int) -> str:
    if feature_key.startswith("lag"):
        return f"{feature_key}_load_h{hour:02d}"
    return f"{feature_key}_h{hour:02d}"


def fit_weighted_linear_profile(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    feature_keys: list[str],
    model_name: str,
    alpha_grid: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    alpha_grid = alpha_grid or [0.0, 0.01, 0.1, 1.0, 10.0, 50.0]
    validation_predictions: list[np.ndarray] = []
    test_predictions: list[np.ndarray] = []
    weight_rows: list[dict[str, float | str]] = []

    for hour in TARGET_HOURS:
        feature_columns = [_feature_column_name(feature_key, hour) for feature_key in feature_keys]
        x_train = train[feature_columns].to_numpy(dtype=float)
        y_train = train[f"target_load_h{hour:02d}"].to_numpy(dtype=float)
        x_validation = validation[feature_columns].to_numpy(dtype=float)
        y_validation = validation[f"target_load_h{hour:02d}"].to_numpy(dtype=float)
        x_test = test[feature_columns].to_numpy(dtype=float)

        best_bundle: tuple[float, float, np.ndarray] | None = None
        for alpha in alpha_grid:
            if alpha == 0.0:
                coefficients = np.linalg.lstsq(x_train, y_train, rcond=None)[0]
            else:
                coefficients = np.linalg.solve(
                    x_train.T @ x_train + np.eye(x_train.shape[1]) * alpha,
                    x_train.T @ y_train,
                )
            candidate_validation = x_validation @ coefficients
            candidate_wape = prediction_metrics_from_wide(y_validation, candidate_validation)["wape_percent"]
            if best_bundle is None or candidate_wape < best_bundle[0]:
                best_bundle = (candidate_wape, alpha, coefficients)

        assert best_bundle is not None
        best_validation_wape, best_alpha, best_coefficients = best_bundle
        validation_predictions.append(x_validation @ best_coefficients)
        test_predictions.append(x_test @ best_coefficients)

        weight_row: dict[str, float | str] = {
            "model": model_name,
            "hour": hour,
            "validation_wape_percent": best_validation_wape,
            "alpha": best_alpha,
        }
        for feature_key, coefficient in zip(feature_keys, best_coefficients):
            weight_row[f"weight_{feature_key}"] = float(coefficient)
        weight_rows.append(weight_row)

    return np.column_stack(validation_predictions), np.column_stack(test_predictions), pd.DataFrame(weight_rows)


def make_daily_prediction_frame(
    frame: pd.DataFrame,
    best_model_name: str,
    best_prediction_matrix: np.ndarray,
    baseline_prediction_map: dict[str, np.ndarray],
) -> pd.DataFrame:
    base_columns = frame[
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
            "same_type_reference_count",
            "same_type_last1_date",
            "same_type_last2_date",
        ]
    ].reset_index(drop=True)

    value_columns: dict[str, object] = {"best_model_name": best_model_name}
    for hour in TARGET_HOURS:
        value_columns[f"actual_h{hour:02d}"] = frame[f"target_load_h{hour:02d}"].to_numpy()
        value_columns[f"pred_h{hour:02d}"] = best_prediction_matrix[:, hour]
        value_columns[f"lag7_h{hour:02d}"] = baseline_prediction_map["lag7_copy"][:, hour]
        value_columns[f"mean721_h{hour:02d}"] = baseline_prediction_map["mean_7_14_21"][:, hour]
        value_columns[f"same_type_mean2_h{hour:02d}"] = baseline_prediction_map["same_type_mean2"][:, hour]

    prediction_frame = pd.concat([base_columns, pd.DataFrame(value_columns)], axis=1)
    actual_matrix = prediction_frame[[f"actual_h{hour:02d}" for hour in TARGET_HOURS]].to_numpy()
    prediction_matrix = prediction_frame[[f"pred_h{hour:02d}" for hour in TARGET_HOURS]].to_numpy()
    prediction_frame["actual_daily_total"] = actual_matrix.sum(axis=1)
    prediction_frame["pred_daily_total"] = prediction_matrix.sum(axis=1)
    prediction_frame["pred_daily_mae"] = np.mean(np.abs(actual_matrix - prediction_matrix), axis=1)
    return prediction_frame


def make_long_prediction_frame(prediction_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in prediction_frame.iterrows():
        for hour in TARGET_HOURS:
            rows.append(
                {
                    "sample_id": row["sample_id"],
                    "site_id": row["site_id"],
                    "site_name": row["site_name"],
                    "city_cn": row["city_cn"],
                    "issue_date": row["issue_date"],
                    "target_date": row["target_date"],
                    "target_refined_date_type": row["target_refined_date_type"],
                    "hour": hour,
                    "actual_load": row[f"actual_h{hour:02d}"],
                    "predicted_load": row[f"pred_h{hour:02d}"],
                    "lag7_load": row[f"lag7_h{hour:02d}"],
                    "mean721_load": row[f"mean721_h{hour:02d}"],
                    "same_type_mean2_load": row[f"same_type_mean2_h{hour:02d}"],
                }
            )
    return pd.DataFrame(rows)


def per_site_metrics(prediction_frame: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for site_id, site_frame in prediction_frame.groupby("site_id"):
        actual = site_frame[[f"actual_h{hour:02d}" for hour in TARGET_HOURS]].to_numpy()
        predicted = site_frame[[f"pred_h{hour:02d}" for hour in TARGET_HOURS]].to_numpy()
        metric_row = prediction_metrics_from_wide(actual, predicted)
        metric_row.update(
            {
                "site_id": site_id,
                "site_name": site_frame["site_name"].iloc[0],
                "city_cn": site_frame["city_cn"].iloc[0],
                "days": int(len(site_frame)),
            }
        )
        rows.append(metric_row)
    return rows


def build_report_markdown(
    dataset_path: Path,
    dataset_summary: dict[str, object],
    split_info: dict[str, str],
    rolling_validation_rows: list[dict[str, object]],
    validation_rows: list[dict[str, object]],
    test_rows: list[dict[str, object]],
    per_site_rows: list[dict[str, object]],
    best_model_name: str,
    prediction_path: Path,
    weights_path: Path,
    summary_path: Path,
) -> str:
    return "\n".join(
        [
            "# 负荷预测报告（相似节日 + D-7/D-14/D-21 预测 D 日 24 点）",
            "",
            "## 任务定义",
            "- 预测对象：第 D 天 24 个小时负荷点。",
            "- 日期类型细化：把日期拆成节日本体、节前 1-3 天、节后 1-3 天、调休工作日、周末、普通工作日，并保留具体节日名称。",
            "- 参考曲线：使用同站点历史相同细分类日期的最近 1 次 / 最近 2 次平均曲线，以及 D-7、D-14、D-21 三个周周期滞后。",
            "- 防泄漏：测试和验证阶段都只允许使用目标日前已经发生的历史日期；严禁使用目标日负荷、目标日天气或更晚信息。",
            "",
            "## 数据与切分",
            f"- 数据集：`{dataset_path.name}`",
            f"- 数据规模：{dataset_summary['rows']} 行，{dataset_summary['columns']} 列",
            f"- 细化日期类型数量：{dataset_summary['refined_type_count']}",
            f"- 训练目标日区间：{split_info['train_target_start']} 到 {split_info['train_target_end']}",
            f"- 验证目标日区间：{split_info['validation_target_start']} 到 {split_info['validation_target_end']}",
            f"- 测试目标日区间：{split_info['test_target_start']} 到 {split_info['test_target_end']}",
            "",
            "## 滚动验证选型",
            "- 选型方式：2025 年 8 月到 12 月做扩窗滚动验证，按各折 WAPE 平均值选模型。",
            markdown_table(
                rolling_validation_rows,
                ["model", "rolling_mean_wape_percent", "rolling_std_wape_percent", "rolling_fold_count"],
            ),
            "",
            "## 验证集结果",
            markdown_table(
                validation_rows,
                ["model", "mae", "rmse", "wape_percent", "smape_percent", "mape_nonzero_percent", "r2"],
            ),
            "",
            "## 测试集结果",
            markdown_table(
                test_rows,
                ["model", "mae", "rmse", "wape_percent", "smape_percent", "mape_nonzero_percent", "r2"],
            ),
            "",
            f"最终选用模型：`{best_model_name}`",
            "",
            "## 分站点测试结果",
            markdown_table(
                per_site_rows,
                ["site_id", "site_name", "city_cn", "days", "mae", "rmse", "wape_percent", "smape_percent", "r2"],
            ),
            "",
            "## 产物",
            f"- 测试集逐日预测：`{prediction_path}`",
            f"- 线性模型权重：`{weights_path}`",
            f"- 模型摘要 JSON：`{summary_path}`",
        ]
    )


def save_summary_json(summary: dict[str, object], output_path: Path) -> None:
    write_json(summary, output_path)
