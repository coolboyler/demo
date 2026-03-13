from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from china_holiday_official import build_official_holiday_frame


HOURLY_DATASET_NAME = "hourly_dataset.csv"
HOURLY_DATASET_LEGACY_NAMES = ("load_weather_calendar_hourly.csv",)
HOURLY_SUMMARY_NAME = "hourly_dataset_summary.json"
HOURLY_SUMMARY_LEGACY_NAMES = ("load_weather_calendar_hourly_summary.json",)
D5_DATASET_NAME = "d5_dataset.csv"
D5_DATASET_LEGACY_NAMES = ("d5_issuance_daily_dataset.csv",)
D5_OUTPUT_PREFIX = "d5"
WEEKLY_COMPARE_NAME = "weekly_compare.xlsx"
WEEKLY_COMPARE_LEGACY_NAMES = ("merged_total_fixed_weekly_vs_best.xlsx",)
MARCH_FORECAST_NAME = "forecast_2026_03_01.csv"
MARCH_FORECAST_LEGACY_NAMES = ("march1_2026_merged_total_tousimday_forecast_corrected.csv",)


SITE_METADATA = {
    "gz": {
        "site_id": "gz",
        "site_name": "广州风恒科技有限公司",
        "city_cn": "广州",
        "city_en": "Guangzhou",
        "province_cn": "广东",
        "latitude": 23.1291,
        "longitude": 113.2644,
        "raw_file": "gz.xlsx",
        "legacy_raw_files": ("广州风恒科技有限公司(1).xlsx",),
    },
    "fs": {
        "site_id": "fs",
        "site_name": "佛山市弘兴新能源有限公司",
        "city_cn": "佛山",
        "city_en": "Foshan",
        "province_cn": "广东",
        "latitude": 23.0215,
        "longitude": 113.1214,
        "raw_file": "fs.xlsx",
        "legacy_raw_files": ("佛山市弘兴新能源有限公司(1).xlsx",),
    },
}

WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "surface_pressure",
]


def file_candidates(directory: Path, preferred_name: str, legacy_names: Iterable[str] = ()) -> list[Path]:
    names = [preferred_name, *legacy_names]
    ordered_names = list(dict.fromkeys(names))
    return [directory / name for name in ordered_names]


def resolve_existing_path(directory: Path, preferred_name: str, legacy_names: Iterable[str] = ()) -> Path:
    candidates = file_candidates(directory, preferred_name, legacy_names)
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def tou_daily_name(case_name: str) -> str:
    return f"tou_{case_name}_daily.csv"


def legacy_tou_daily_name(case_name: str) -> str:
    return f"merged_total_d5_tousimday_{case_name}_predictions_daily.csv"


def tou_summary_name(case_name: str) -> str:
    return f"tou_{case_name}_summary.json"


def legacy_tou_summary_name(case_name: str) -> str:
    return f"merged_total_d5_tousimday_{case_name}_summary.json"


def tou_report_name(case_name: str) -> str:
    return f"tou_{case_name}.md"


def legacy_tou_report_name(case_name: str) -> str:
    return f"merged_total_d5_tousimday_{case_name}_report.md"


def parse_hourly_load(raw_file: Path, site_meta: dict[str, object]) -> pd.DataFrame:
    raw = pd.read_excel(raw_file, header=None)
    header = raw.iloc[1]
    daily = raw.iloc[2:].copy()
    daily.columns = header

    daily["日期"] = pd.to_datetime(daily["日期"].astype(str), format="%Y%m%d", errors="raise")
    hourly_columns = [column for column in daily.columns if isinstance(column, dt.time)]

    long = daily.melt(
        id_vars=["日期", "日电量"],
        value_vars=hourly_columns,
        var_name="hour_marker",
        value_name="load",
    )
    long["load"] = pd.to_numeric(long["load"], errors="coerce")
    long["daily_energy_reported"] = pd.to_numeric(long["日电量"], errors="coerce")
    long["timestamp"] = long["日期"] + pd.to_timedelta(long["hour_marker"].astype(str))
    long = long.sort_values("timestamp").reset_index(drop=True)

    long["date"] = long["timestamp"].dt.normalize()
    hourly_sum = long.groupby("date")["load"].transform("sum")
    long["daily_load_sum_from_hourly"] = hourly_sum
    long["daily_energy_gap_reported_minus_hourly"] = (
        long["daily_energy_reported"] - long["daily_load_sum_from_hourly"]
    )

    long["site_id"] = site_meta["site_id"]
    long["site_name"] = site_meta["site_name"]
    long["city_cn"] = site_meta["city_cn"]
    long["city_en"] = site_meta["city_en"]
    long["province_cn"] = site_meta["province_cn"]

    return long[
        [
            "site_id",
            "site_name",
            "city_cn",
            "city_en",
            "province_cn",
            "timestamp",
            "date",
            "load",
            "daily_energy_reported",
            "daily_load_sum_from_hourly",
            "daily_energy_gap_reported_minus_hourly",
        ]
    ]


def fetch_weather_archive(
    site_meta: dict[str, object],
    start_date: dt.date,
    end_date: dt.date,
    weather_dir: Path,
    refresh: bool = False,
) -> pd.DataFrame:
    weather_dir.mkdir(parents=True, exist_ok=True)
    cache_path = weather_dir / f"{site_meta['site_id']}_open_meteo_hourly.csv"

    if cache_path.exists() and not refresh:
        cached = pd.read_csv(cache_path, parse_dates=["timestamp"])
        cache_start = cached["timestamp"].min().date()
        cache_end = cached["timestamp"].max().date()
        if cache_start <= start_date and cache_end >= end_date:
            return cached.sort_values("timestamp").reset_index(drop=True)

    response = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": site_meta["latitude"],
            "longitude": site_meta["longitude"],
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "hourly": ",".join(WEATHER_VARIABLES),
            "timezone": "Asia/Shanghai",
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    hourly = pd.DataFrame(payload["hourly"])
    hourly["timestamp"] = pd.to_datetime(hourly.pop("time"))
    hourly["site_id"] = site_meta["site_id"]
    hourly["weather_latitude"] = payload.get("latitude")
    hourly["weather_longitude"] = payload.get("longitude")
    hourly["weather_timezone"] = payload.get("timezone")

    hourly.to_csv(cache_path, index=False)
    return hourly.sort_values("timestamp").reset_index(drop=True)


def add_calendar_features(dataset: pd.DataFrame) -> pd.DataFrame:
    frame = dataset.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["date"] = frame["timestamp"].dt.normalize()
    frame["year"] = frame["timestamp"].dt.year
    frame["month"] = frame["timestamp"].dt.month
    frame["quarter"] = frame["timestamp"].dt.quarter
    frame["weekofyear"] = frame["timestamp"].dt.isocalendar().week.astype(int)
    frame["dayofmonth"] = frame["timestamp"].dt.day
    frame["dayofweek"] = frame["timestamp"].dt.dayofweek
    frame["dayofyear"] = frame["timestamp"].dt.dayofyear
    frame["hour"] = frame["timestamp"].dt.hour
    frame["is_weekend"] = frame["dayofweek"].isin([5, 6]).astype(int)
    frame["is_month_start"] = frame["timestamp"].dt.is_month_start.astype(int)
    frame["is_month_end"] = frame["timestamp"].dt.is_month_end.astype(int)
    frame["season"] = frame["month"].map(
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
    frame["hour_bucket"] = pd.cut(
        frame["hour"],
        bins=[-1, 5, 11, 17, 23],
        labels=["overnight", "morning", "afternoon", "evening"],
    ).astype(str)

    frame["hour_sin"] = np.sin(2 * np.pi * frame["hour"] / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * frame["hour"] / 24)
    frame["dow_sin"] = np.sin(2 * np.pi * frame["dayofweek"] / 7)
    frame["dow_cos"] = np.cos(2 * np.pi * frame["dayofweek"] / 7)
    frame["doy_sin"] = np.sin(2 * np.pi * frame["dayofyear"] / 366)
    frame["doy_cos"] = np.cos(2 * np.pi * frame["dayofyear"] / 366)

    detail_frame = build_official_holiday_frame(frame["date"].drop_duplicates().sort_values())
    frame = frame.merge(detail_frame, on="date", how="left")
    frame["date_type_cn"] = np.select(
        [
            frame["is_holiday_cn"].eq(1),
            frame["is_makeup_workday"].eq(1),
            frame["is_weekend"].eq(1),
        ],
        ["holiday", "makeup_workday", "weekend"],
        default="workday",
    )

    frame["temp_dew_gap"] = frame["temperature_2m"] - frame["dew_point_2m"]
    frame["temp_apparent_gap"] = frame["apparent_temperature"] - frame["temperature_2m"]
    frame["wind_precip_interaction"] = frame["wind_speed_10m"] * frame["precipitation"]

    return frame.sort_values(["site_id", "timestamp"]).reset_index(drop=True)


def build_hourly_dataset(
    raw_dir: Path,
    weather_dir: Path,
    refresh_weather: bool = False,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for site_meta in SITE_METADATA.values():
        raw_file = resolve_existing_path(
            raw_dir,
            str(site_meta["raw_file"]),
            tuple(site_meta.get("legacy_raw_files", ())),
        )
        load_frame = parse_hourly_load(raw_file, site_meta)
        weather_frame = fetch_weather_archive(
            site_meta=site_meta,
            start_date=load_frame["timestamp"].min().date(),
            end_date=load_frame["timestamp"].max().date(),
            weather_dir=weather_dir,
            refresh=refresh_weather,
        )
        merged = load_frame.merge(weather_frame, on=["site_id", "timestamp"], how="left")
        frames.append(merged)

    dataset = pd.concat(frames, ignore_index=True)
    dataset = add_calendar_features(dataset)
    dataset["weather_missing_count"] = dataset[WEATHER_VARIABLES].isna().sum(axis=1)
    return dataset.sort_values(["site_id", "timestamp"]).reset_index(drop=True)


def _feature_plan_for_horizon(horizon_hours: int) -> tuple[list[int], list[int], list[int]]:
    if horizon_hours == 1:
        return [1, 2, 3, 6, 12, 24, 25, 48, 72, 168, 336], [3, 6, 12, 24, 48, 168], [1, 24]
    if horizon_hours == 24:
        return [24, 48, 72, 96, 168, 336, 504], [24, 48, 72, 168], [24, 48, 168]

    load_lags = sorted({horizon_hours, horizon_hours + 1, horizon_hours + 2, horizon_hours + 6, horizon_hours + 12, horizon_hours + 24, horizon_hours + 48, horizon_hours + 168, horizon_hours + 336})
    roll_windows = [24, 48, 168]
    weather_lags = sorted({horizon_hours, horizon_hours + 24, horizon_hours + 168})
    return load_lags, roll_windows, weather_lags


def make_supervised_frame(dataset: pd.DataFrame, horizon_hours: int) -> pd.DataFrame:
    if horizon_hours <= 0:
        raise ValueError("horizon_hours must be positive.")

    frame = dataset.sort_values(["site_id", "timestamp"]).reset_index(drop=True).copy()
    by_site = frame.groupby("site_id", sort=False)

    load_lags, roll_windows, weather_lags = _feature_plan_for_horizon(horizon_hours)

    for lag in load_lags:
        frame[f"load_lag_{lag}"] = by_site["load"].shift(lag)

    shifted_load = by_site["load"].shift(horizon_hours)
    for window in roll_windows:
        frame[f"load_roll_mean_{window}"] = shifted_load.groupby(frame["site_id"]).transform(
            lambda series: series.rolling(window).mean()
        )
        frame[f"load_roll_std_{window}"] = shifted_load.groupby(frame["site_id"]).transform(
            lambda series: series.rolling(window).std()
        )

    if horizon_hours == 24:
        frame["prev_day_total_load"] = shifted_load.groupby(frame["site_id"]).transform(
            lambda series: series.rolling(24).sum()
        )
        same_hour_lags = [24, 48, 72, 96, 168]
        frame["same_hour_mean_recent_days"] = frame[[f"load_lag_{lag}" for lag in same_hour_lags]].mean(axis=1)
        frame["same_hour_std_recent_days"] = frame[[f"load_lag_{lag}" for lag in same_hour_lags]].std(axis=1)

    for weather_column in WEATHER_VARIABLES + ["temp_dew_gap", "temp_apparent_gap", "wind_precip_interaction"]:
        for lag in weather_lags:
            frame[f"{weather_column}_lag_{lag}"] = by_site[weather_column].shift(lag)
        base_shifted_weather = by_site[weather_column].shift(horizon_hours)
        frame[f"{weather_column}_roll24"] = base_shifted_weather.groupby(frame["site_id"]).transform(
            lambda series: series.rolling(24).mean()
        )
        if horizon_hours >= 24:
            frame[f"{weather_column}_roll168"] = base_shifted_weather.groupby(frame["site_id"]).transform(
                lambda series: series.rolling(168).mean()
            )

    excluded_columns = {
        "daily_energy_reported",
        "daily_load_sum_from_hourly",
        "daily_energy_gap_reported_minus_hourly",
    }
    supervised = frame.drop(columns=[column for column in excluded_columns if column in frame.columns])
    supervised = supervised.dropna().reset_index(drop=True)
    return supervised


def split_by_time(
    frame: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    unique_dates = sorted(frame["date"].drop_duplicates())
    if len(unique_dates) < 10:
        raise ValueError("Not enough dates to create train/validation/test splits.")

    train_end_idx = max(0, int(len(unique_dates) * train_ratio) - 1)
    val_end_idx = max(train_end_idx + 1, int(len(unique_dates) * (train_ratio + val_ratio)) - 1)
    val_end_idx = min(val_end_idx, len(unique_dates) - 2)

    train_end = unique_dates[train_end_idx]
    val_end = unique_dates[val_end_idx]

    train_frame = frame[frame["date"] <= train_end].copy()
    validation_frame = frame[(frame["date"] > train_end) & (frame["date"] <= val_end)].copy()
    test_frame = frame[frame["date"] > val_end].copy()

    split_info = {
        "train_start": str(train_frame["timestamp"].min()),
        "train_end": str(train_frame["timestamp"].max()),
        "validation_start": str(validation_frame["timestamp"].min()),
        "validation_end": str(validation_frame["timestamp"].max()),
        "test_start": str(test_frame["timestamp"].min()),
        "test_end": str(test_frame["timestamp"].max()),
    }
    return train_frame, validation_frame, test_frame, split_info


def make_model_matrices(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    categorical_columns = ["site_id", "city_cn", "season", "hour_bucket", "holiday_name_cn", "date_type_cn"]
    leaky_or_unused_columns = {
        "site_name",
        "city_en",
        "province_cn",
        "weather_timezone",
        "weather_latitude",
        "weather_longitude",
        "weather_missing_count",
        "temp_dew_gap",
        "temp_apparent_gap",
        "wind_precip_interaction",
    }
    leaky_or_unused_columns.update(WEATHER_VARIABLES)
    feature_columns = [
        column
        for column in train_frame.columns
        if column not in {"timestamp", "date", "load"} | leaky_or_unused_columns
    ]

    x_train = pd.get_dummies(train_frame[feature_columns], columns=categorical_columns, drop_first=False)
    x_validation = pd.get_dummies(validation_frame[feature_columns], columns=categorical_columns, drop_first=False)
    x_test = pd.get_dummies(test_frame[feature_columns], columns=categorical_columns, drop_first=False)

    x_validation = x_validation.reindex(columns=x_train.columns, fill_value=0)
    x_test = x_test.reindex(columns=x_train.columns, fill_value=0)

    y_train = train_frame["load"].to_numpy()
    y_validation = validation_frame["load"].to_numpy()
    y_test = test_frame["load"].to_numpy()
    return x_train, x_validation, x_test, y_train, y_validation, y_test, x_train.columns.tolist()


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    abs_error = np.abs(y_true - y_pred)

    mae = float(abs_error.mean())
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    wape = float(abs_error.sum() / np.maximum(np.abs(y_true).sum(), 1e-9) * 100)
    nonzero_mask = np.abs(y_true) > 1e-9
    mape_nonzero = float(
        np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100
    ) if nonzero_mask.any() else float("nan")
    denominator = np.abs(y_true) + np.abs(y_pred)
    smape_ratio = np.divide(
        2 * abs_error,
        denominator,
        out=np.zeros_like(abs_error, dtype=float),
        where=denominator != 0,
    )
    smape = float(np.mean(smape_ratio) * 100)

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {
        "mae": mae,
        "rmse": rmse,
        "wape_percent": wape,
        "smape_percent": smape,
        "mape_nonzero_percent": mape_nonzero,
        "r2": r2,
    }


def metrics_by_site(frame: pd.DataFrame, prediction_column: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for site_id, site_frame in frame.groupby("site_id"):
        metric_row = regression_metrics(site_frame["load"].to_numpy(), site_frame[prediction_column].to_numpy())
        metric_row.update(
            {
                "site_id": site_id,
                "site_name": site_frame["site_name"].iloc[0],
                "city_cn": site_frame["city_cn"].iloc[0],
                "rows": int(len(site_frame)),
            }
        )
        rows.append(metric_row)
    return rows


def dataset_summary(dataset: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(len(dataset)),
        "sites": sorted(dataset["site_id"].unique().tolist()),
        "timestamp_start": str(dataset["timestamp"].min()),
        "timestamp_end": str(dataset["timestamp"].max()),
        "weather_missing_rows": int((dataset["weather_missing_count"] > 0).sum()),
        "load_min": float(dataset["load"].min()),
        "load_max": float(dataset["load"].max()),
        "load_mean": float(dataset["load"].mean()),
        "daily_energy_gap_abs_mean": float(dataset["daily_energy_gap_reported_minus_hourly"].abs().mean()),
        "daily_energy_gap_abs_max": float(dataset["daily_energy_gap_reported_minus_hourly"].abs().max()),
    }


def write_json(data: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def markdown_table(rows: Iterable[dict[str, object]], columns: list[str]) -> str:
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return ""
    frame = frame[columns]
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for _, row in frame.iterrows():
        body.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join([header, divider] + body)
