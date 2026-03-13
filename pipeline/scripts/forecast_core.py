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
from holiday_lib import _build_refined_date_map


ALL_LAG_DAYS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 21]
TARGET_HOURS = list(range(24))
TARGET_COLUMNS = [f"target_load_h{hour:02d}" for hour in TARGET_HOURS]


@dataclass
class SplitBundle:
    train: pd.DataFrame
    validation: pd.DataFrame
    development: pd.DataFrame
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


def build_high_accuracy_dataset(hourly_dataset: pd.DataFrame) -> pd.DataFrame:
    hourly = hourly_dataset.copy()
    hourly["date"] = pd.to_datetime(hourly["date"])

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

    lag_source = daily_load_profile.rename(columns={"date": "source_date"})
    lag_feature_names: list[str] = []
    for lag_day in ALL_LAG_DAYS:
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
            matches = history.get(row.target_refined_date_type, [])
            last1 = matches[-1] if len(matches) >= 1 else None
            last2 = matches[-2] if len(matches) >= 2 else None
            record["same_type_reference_count"] = int(min(len(matches), 2))
            record["same_type_last1_date"] = last1[0] if last1 is not None else pd.NaT
            record["same_type_last2_date"] = last2[0] if last2 is not None else pd.NaT
            for hour in TARGET_HOURS:
                ref1_value = last1[1][hour] if last1 is not None else np.nan
                ref2_value = last2[1][hour] if last2 is not None else np.nan
                record[f"same_type_mean2_h{hour:02d}"] = (
                    np.nan if np.isnan(ref1_value) and np.isnan(ref2_value) else float(np.nanmean([ref1_value, ref2_value]))
                )
            profile = np.asarray([getattr(row, column) for column in TARGET_COLUMNS], dtype=float)
            history.setdefault(row.target_refined_date_type, []).append((row.target_date, profile))
            same_type_rows.append(record)

    dataset = dataset.merge(pd.DataFrame(same_type_rows), on=["site_id", "target_date"], how="left")
    dataset["issue_date"] = dataset["target_date"] - pd.Timedelta(days=1)

    for hour in TARGET_HOURS:
        dataset[f"recent_lag_mean_h{hour:02d}"] = dataset[
            [f"lag{lag_day}_h{hour:02d}" for lag_day in [1, 2, 3, 4, 5, 6, 7]]
        ].mean(axis=1)

    required_columns = lag_feature_names + [f"same_type_mean2_h{hour:02d}" for hour in TARGET_HOURS] + TARGET_COLUMNS
    dataset = dataset.dropna(subset=required_columns).sort_values(["target_date", "site_id"]).reset_index(drop=True)
    dataset["sample_id"] = dataset["site_id"] + "_" + dataset["target_date"].dt.strftime("%Y%m%d")
    return dataset


def dataset_summary(dataset: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(len(dataset)),
        "columns": int(dataset.shape[1]),
        "sites": sorted(dataset["site_id"].unique().tolist()),
        "target_date_start": str(dataset["target_date"].min()),
        "target_date_end": str(dataset["target_date"].max()),
        "issue_date_start": str(dataset["issue_date"].min()),
        "issue_date_end": str(dataset["issue_date"].max()),
        "lag_days": ALL_LAG_DAYS,
        "target_hours": TARGET_HOURS,
        "refined_type_count": int(dataset["target_refined_date_type"].nunique()),
    }


def split_dataset(
    dataset: pd.DataFrame,
    train_end: str = "2025-10-31",
    validation_end: str = "2025-12-31",
    development_end: str | None = None,
) -> SplitBundle:
    train_end_ts = pd.Timestamp(train_end)
    validation_end_ts = pd.Timestamp(validation_end)
    development_end_ts = pd.Timestamp(development_end) if development_end is not None else validation_end_ts

    train = dataset[dataset["target_date"] <= train_end_ts].copy()
    validation = dataset[(dataset["target_date"] > train_end_ts) & (dataset["target_date"] <= validation_end_ts)].copy()
    development = dataset[dataset["target_date"] <= development_end_ts].copy()
    test = dataset[dataset["target_date"] > development_end_ts].copy()

    split_info = {
        "train_target_start": str(train["target_date"].min()),
        "train_target_end": str(train["target_date"].max()),
        "validation_target_start": str(validation["target_date"].min()),
        "validation_target_end": str(validation["target_date"].max()),
        "development_target_start": str(development["target_date"].min()),
        "development_target_end": str(development["target_date"].max()),
        "test_target_start": str(test["target_date"].min()),
        "test_target_end": str(test["target_date"].max()),
    }
    return SplitBundle(train=train, validation=validation, development=development, test=test, split_info=split_info)


def prediction_metrics_from_wide(actual: pd.DataFrame | np.ndarray, prediction: pd.DataFrame | np.ndarray) -> dict[str, float]:
    actual_values = actual.to_numpy() if isinstance(actual, pd.DataFrame) else np.asarray(actual)
    prediction_values = prediction.to_numpy() if isinstance(prediction, pd.DataFrame) else np.asarray(prediction)
    return regression_metrics(actual_values.ravel(), prediction_values.ravel())


def _fit_weighted_linear_for_group(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    feature_keys: list[str],
    alpha_grid: list[float],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    validation = validation.reset_index(drop=True).copy()
    test = test.reset_index(drop=True).copy()
    validation_predictions = np.full((len(validation), len(TARGET_HOURS)), np.nan)
    test_predictions = np.full((len(test), len(TARGET_HOURS)), np.nan)
    weight_rows: list[dict[str, float]] = []

    for hour in TARGET_HOURS:
        feature_columns = [f"{feature_key}_h{hour:02d}" for feature_key in feature_keys]
        train_group = train.dropna(subset=feature_columns + [f"target_load_h{hour:02d}"])
        validation_group = validation.dropna(subset=feature_columns + [f"target_load_h{hour:02d}"])
        test_group = test.dropna(subset=feature_columns)
        if len(train_group) < 20:
            continue

        x_train = train_group[feature_columns].to_numpy(dtype=float)
        y_train = train_group[f"target_load_h{hour:02d}"].to_numpy(dtype=float)
        x_validation = validation_group[feature_columns].to_numpy(dtype=float)
        y_validation = validation_group[f"target_load_h{hour:02d}"].to_numpy(dtype=float)

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

        if best_bundle is None:
            continue

        best_validation_wape, best_alpha, best_coefficients = best_bundle
        validation_predictions[validation_group.index.to_numpy(), hour] = x_validation @ best_coefficients
        if len(test_group) > 0:
            x_test = test_group[feature_columns].to_numpy(dtype=float)
            test_predictions[test_group.index.to_numpy(), hour] = x_test @ best_coefficients

        weight_row = {"hour": hour, "validation_wape_percent": best_validation_wape, "alpha": best_alpha}
        for feature_key, coefficient in zip(feature_keys, best_coefficients):
            weight_row[f"weight_{feature_key}"] = float(coefficient)
        weight_rows.append(weight_row)

    return validation_predictions, test_predictions, weight_rows


def fit_weighted_linear_profile(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    feature_keys: list[str],
    model_name: str,
    group_key: str | None = None,
    alpha_grid: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    alpha_grid = alpha_grid or [0.0, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0]
    validation = validation.reset_index(drop=True).copy()
    test = test.reset_index(drop=True).copy()
    validation_predictions = np.full((len(validation), len(TARGET_HOURS)), np.nan)
    test_predictions = np.full((len(test), len(TARGET_HOURS)), np.nan)
    weight_rows: list[dict[str, object]] = []

    if group_key is None:
        group_specs = [(None, train, validation, test)]
    else:
        group_specs = []
        for group_value in sorted(train[group_key].dropna().unique().tolist()):
            group_specs.append(
                (
                    group_value,
                    train[train[group_key].eq(group_value)].copy(),
                    validation[validation[group_key].eq(group_value)].copy(),
                    test[test[group_key].eq(group_value)].copy(),
                )
            )

    for group_value, train_group, validation_group, test_group in group_specs:
        group_validation_prediction, group_test_prediction, group_weight_rows = _fit_weighted_linear_for_group(
            train=train_group,
            validation=validation_group,
            test=test_group,
            feature_keys=feature_keys,
            alpha_grid=alpha_grid,
        )
        if len(validation_group) > 0:
            validation_predictions[validation_group.index.to_numpy(), :] = group_validation_prediction
        if len(test_group) > 0:
            test_predictions[test_group.index.to_numpy(), :] = group_test_prediction
        for row in group_weight_rows:
            row["model"] = model_name
            row["group_key"] = group_key or "all"
            row["group_value"] = group_value if group_value is not None else "all"
            weight_rows.append(row)

    return validation_predictions, test_predictions, pd.DataFrame(weight_rows)


def refit_and_predict_test(
    train_for_alpha: pd.DataFrame,
    development: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    feature_keys: list[str],
    model_name: str,
    group_key: str | None = None,
    alpha_grid: list[float] | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    alpha_grid = alpha_grid or [0.0, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0]
    test = test.reset_index(drop=True).copy()
    final_predictions = np.full((len(test), len(TARGET_HOURS)), np.nan)
    weight_rows: list[dict[str, object]] = []

    if group_key is None:
        group_specs = [(None, train_for_alpha, development, validation, test)]
    else:
        group_specs = []
        for group_value in sorted(development[group_key].dropna().unique().tolist()):
            group_specs.append(
                (
                    group_value,
                    train_for_alpha[train_for_alpha[group_key].eq(group_value)].copy(),
                    development[development[group_key].eq(group_value)].copy(),
                    validation[validation[group_key].eq(group_value)].copy(),
                    test[test[group_key].eq(group_value)].copy(),
                )
            )

    for group_value, train_group_raw, development_group, validation_group, test_group in group_specs:
        if len(train_group_raw) < 20 or len(development_group) < 20 or len(test_group) == 0:
            continue

        for hour in TARGET_HOURS:
            feature_columns = [f"{feature_key}_h{hour:02d}" for feature_key in feature_keys]
            train_group = train_group_raw.dropna(subset=feature_columns + [f"target_load_h{hour:02d}"])
            validation_group_hour = validation_group.dropna(subset=feature_columns + [f"target_load_h{hour:02d}"])
            development_group_hour = development_group.dropna(subset=feature_columns + [f"target_load_h{hour:02d}"])
            test_group_hour = test_group.dropna(subset=feature_columns)
            if len(train_group) < 20 or len(validation_group_hour) == 0 or len(development_group_hour) < 20 or len(test_group_hour) == 0:
                continue

            x_validation = validation_group_hour[feature_columns].to_numpy(dtype=float)
            y_validation = validation_group_hour[f"target_load_h{hour:02d}"].to_numpy(dtype=float)

            best_bundle: tuple[float, float] | None = None
            x_model_select = train_group[feature_columns].to_numpy(dtype=float)
            y_model_select = train_group[f"target_load_h{hour:02d}"].to_numpy(dtype=float)
            for alpha in alpha_grid:
                if alpha == 0.0:
                    coefficients = np.linalg.lstsq(x_model_select, y_model_select, rcond=None)[0]
                else:
                    coefficients = np.linalg.solve(
                        x_model_select.T @ x_model_select + np.eye(x_model_select.shape[1]) * alpha,
                        x_model_select.T @ y_model_select,
                    )
                candidate_validation = x_validation @ coefficients
                candidate_wape = prediction_metrics_from_wide(y_validation, candidate_validation)["wape_percent"]
                if best_bundle is None or candidate_wape < best_bundle[0]:
                    best_bundle = (candidate_wape, alpha)

            if best_bundle is None:
                continue

            _, best_alpha = best_bundle
            x_development = development_group_hour[feature_columns].to_numpy(dtype=float)
            y_development = development_group_hour[f"target_load_h{hour:02d}"].to_numpy(dtype=float)
            if best_alpha == 0.0:
                final_coefficients = np.linalg.lstsq(x_development, y_development, rcond=None)[0]
            else:
                final_coefficients = np.linalg.solve(
                    x_development.T @ x_development + np.eye(x_development.shape[1]) * best_alpha,
                    x_development.T @ y_development,
                )

            x_test = test_group_hour[feature_columns].to_numpy(dtype=float)
            final_predictions[test_group_hour.index.to_numpy(), hour] = x_test @ final_coefficients

            weight_row = {
                "model": model_name,
                "group_key": group_key or "all",
                "group_value": group_value if group_value is not None else "all",
                "hour": hour,
                "alpha": best_alpha,
            }
            for feature_key, coefficient in zip(feature_keys, final_coefficients):
                weight_row[f"weight_{feature_key}"] = float(coefficient)
            weight_rows.append(weight_row)

    return final_predictions, pd.DataFrame(weight_rows)


def make_daily_prediction_frame(
    frame: pd.DataFrame,
    best_model_name: str,
    best_prediction_matrix: np.ndarray,
) -> pd.DataFrame:
    base_column_names = [
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
    optional_column_names = [
        "holiday_context_tag",
        "holiday_context_family",
        "holiday_context_side",
        "holiday_context_bucket",
        "days_since_last_holiday",
        "days_to_next_holiday",
        "last_holiday_family",
        "next_holiday_family",
    ]
    selected_columns = base_column_names + [column for column in optional_column_names if column in frame.columns]
    base_columns = frame[selected_columns].reset_index(drop=True)

    value_columns: dict[str, object] = {"best_model_name": best_model_name}
    for hour in TARGET_HOURS:
        value_columns[f"actual_h{hour:02d}"] = frame[f"target_load_h{hour:02d}"].to_numpy()
        value_columns[f"pred_h{hour:02d}"] = best_prediction_matrix[:, hour]

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


def daily_total_metrics(prediction_frame: pd.DataFrame) -> dict[str, float]:
    actual = prediction_frame["actual_daily_total"].to_numpy(dtype=float)
    predicted = prediction_frame["pred_daily_total"].to_numpy(dtype=float)
    abs_error = np.abs(predicted - actual)
    return {
        "daily_total_wape_percent": float(abs_error.sum() / np.abs(actual).sum() * 100),
        "daily_total_accuracy_percent": float(100 - abs_error.sum() / np.abs(actual).sum() * 100),
        "daily_total_bias_percent": float((predicted.sum() - actual.sum()) / actual.sum() * 100),
        "daily_total_rmse": float(np.sqrt(np.mean((predicted - actual) ** 2))),
    }


def build_report_markdown(
    dataset_path: Path,
    dataset_summary_map: dict[str, object],
    split_info: dict[str, str],
    rolling_rows: list[dict[str, object]],
    test_rows: list[dict[str, object]],
    per_site_rows: list[dict[str, object]],
    best_model_name: str,
    prediction_path: Path,
    weights_path: Path,
    summary_path: Path,
) -> str:
    return "\n".join(
        [
            "# 负荷预测报告（高准确率版）",
            "",
            "## 任务定义",
            "- 预测对象：第 D 天 24 个小时负荷点。",
            "- 允许信息：目标日前已发生的历史负荷与细化节日类型相似日，不使用目标日和未来信息。",
            "- 为提升准确率，本版同时比较近邻滞后、周周期滞后和相似节日曲线，并允许按站点分别学习每小时权重。",
            "",
            "## 数据与切分",
            f"- 数据集：`{dataset_path.name}`",
            f"- 数据规模：{dataset_summary_map['rows']} 行，{dataset_summary_map['columns']} 列",
            f"- 训练目标日区间：{split_info['train_target_start']} 到 {split_info['train_target_end']}",
            f"- 验证目标日区间：{split_info['validation_target_start']} 到 {split_info['validation_target_end']}",
            f"- 测试目标日区间：{split_info['test_target_start']} 到 {split_info['test_target_end']}",
            "",
            "## 滚动验证选型",
            markdown_table(
                rolling_rows,
                ["model", "rolling_mean_wape_percent", "rolling_std_wape_percent", "rolling_fold_count"],
            ),
            "",
            "## 最终测试结果",
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
