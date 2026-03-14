from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from china_holiday_official import build_official_holiday_frame
from forecast_d6 import compute_date_type_group, compute_refined_type
from train_d5 import build_issue_gap_dataset, get_issue_lag_config
from update_d6 import holiday_family_name, phase_group_name, season_name


RAW_GLOB = "*.xlsx"
BASELINE_COMPANY_START = pd.Timestamp("2026-01-01")
TRAIN_END = pd.Timestamp("2026-01-31")
VALIDATION_END = pd.Timestamp("2026-02-28")
SITE_ID = "tianlang_total"
SITE_NAME = "天朗"
CITY_CN = "广东多市"
CITY_EN = "Guangdong"
PROVINCE_CN = "广东"
ISSUE_GAP_DAYS = 6
HOUR_LABELS = [f"{hour:02d}:00" for hour in range(24)]
LOAD_COLUMNS = [f"load_h{hour:02d}" for hour in range(24)]
IMPUTE_TOP_K = 6

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Tianlang D-6 datasets from monthly company-level hourly Excel files.")
    parser.add_argument("--raw-dir", type=Path, default=Path(__file__).resolve().parents[1] / "tianlang" / "raw")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_company_hourly(raw_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(raw_dir.glob(RAW_GLOB)):
        frame = pd.read_excel(path)
        frame.columns = [str(column).strip() for column in frame.columns]
        required = {"电力用户编码", "电力用户名称", "日期", *HOUR_LABELS}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
        selected = frame[["电力用户编码", "电力用户名称", "日期", *HOUR_LABELS]].copy()
        selected["source_file"] = path.name
        frames.append(selected)

    if not frames:
        raise ValueError(f"No Excel files found under {raw_dir}")

    merged = pd.concat(frames, ignore_index=True)
    merged["date"] = pd.to_datetime(merged["日期"].astype(str), format="%Y%m%d", errors="raise")
    merged = merged.drop_duplicates(subset=["电力用户编码", "date"], keep="last").reset_index(drop=True)
    if merged.empty:
        raise ValueError(f"No rows found under {raw_dir}")

    long = merged.melt(
        id_vars=["电力用户编码", "电力用户名称", "date", "source_file"],
        value_vars=HOUR_LABELS,
        var_name="hour_label",
        value_name="load",
    )
    long["hour"] = long["hour_label"].str.slice(0, 2).astype(int)
    long["load"] = pd.to_numeric(long["load"], errors="coerce").fillna(0.0)
    long["timestamp"] = long["date"] + pd.to_timedelta(long["hour"], unit="h")
    return long.sort_values(["date", "电力用户编码", "hour"]).reset_index(drop=True)


def build_baseline_company_pool(company_hourly: pd.DataFrame) -> pd.DataFrame:
    latest_snapshot_date = pd.Timestamp(company_hourly["date"].max()).normalize()
    roster = (
        company_hourly[company_hourly["date"].eq(latest_snapshot_date)][["电力用户编码", "电力用户名称"]]
        .drop_duplicates()
        .sort_values(["电力用户编码"])
        .reset_index(drop=True)
    )
    if roster.empty:
        raise ValueError(f"No companies found on latest snapshot date {latest_snapshot_date.date()}")
    return roster


def build_company_daily_profiles(company_hourly: pd.DataFrame, baseline_companies: pd.DataFrame) -> pd.DataFrame:
    filtered = company_hourly[company_hourly["电力用户编码"].isin(baseline_companies["电力用户编码"])].copy()
    daily = filtered.pivot_table(
        index=["电力用户编码", "电力用户名称", "date"],
        columns="hour",
        values="load",
        aggfunc="sum",
        fill_value=0.0,
    ).reset_index()
    daily.columns = ["电力用户编码", "电力用户名称", "date", *LOAD_COLUMNS]
    daily["daily_total"] = daily[LOAD_COLUMNS].sum(axis=1)
    return daily.sort_values(["电力用户编码", "date"]).reset_index(drop=True)


def build_date_meta(dates: list[pd.Timestamp]) -> pd.DataFrame:
    frame = pd.DataFrame({"date": sorted(pd.to_datetime(pd.Index(dates)).normalize().unique().tolist())})
    holiday_frame = build_official_holiday_frame(frame["date"].tolist()).rename(columns={"date": "holiday_date"})
    frame["holiday_date"] = frame["date"]
    frame = frame.merge(holiday_frame, on="holiday_date", how="left")
    frame["year"] = frame["date"].dt.year
    frame["month"] = frame["date"].dt.month
    frame["quarter"] = frame["date"].dt.quarter
    frame["weekofyear"] = frame["date"].dt.isocalendar().week.astype(int)
    frame["dayofmonth"] = frame["date"].dt.day
    frame["dayofweek"] = frame["date"].dt.dayofweek
    frame["dayofyear"] = frame["date"].dt.dayofyear
    frame["is_weekend"] = frame["dayofweek"].isin([5, 6]).astype(int)
    frame["is_month_start"] = frame["date"].dt.is_month_start.astype(int)
    frame["is_month_end"] = frame["date"].dt.is_month_end.astype(int)
    frame["season"] = frame["month"].map(season_name)
    frame["date_type_cn"] = [
        _date_type_label(target_date, holiday_info)
        for target_date, holiday_info in zip(frame["date"], frame.to_dict(orient="records"))
    ]
    frame["refined_date_type"] = frame["date"].map(compute_refined_type)
    frame["date_type_group"] = frame["refined_date_type"].map(compute_date_type_group)
    frame["holiday_family"] = frame["refined_date_type"].map(holiday_family_name)
    frame["phase_group"] = frame["refined_date_type"].map(phase_group_name)
    return frame.drop(columns=["holiday_date"]).sort_values("date").reset_index(drop=True)


def _weighted_profile(candidate_frame: pd.DataFrame, target_date: pd.Timestamp, top_k: int = IMPUTE_TOP_K) -> np.ndarray | None:
    if candidate_frame.empty:
        return None
    scored = candidate_frame.copy()
    scored["distance_days"] = (scored["date"] - target_date).abs().dt.days.astype(float)
    scored = scored.sort_values(["distance_days", "date"]).head(top_k)
    weights = 1.0 / (1.0 + scored["distance_days"].to_numpy(dtype=float))
    matrix = scored[LOAD_COLUMNS].to_numpy(dtype=float)
    return np.average(matrix, axis=0, weights=weights)


def _weighted_total(candidate_frame: pd.DataFrame, target_date: pd.Timestamp, top_k: int = IMPUTE_TOP_K) -> float | None:
    if candidate_frame.empty:
        return None
    scored = candidate_frame.copy()
    scored["distance_days"] = (scored["date"] - target_date).abs().dt.days.astype(float)
    scored = scored.sort_values(["distance_days", "date"]).head(top_k)
    weights = 1.0 / (1.0 + scored["distance_days"].to_numpy(dtype=float))
    return float(np.average(scored["daily_total"].to_numpy(dtype=float), weights=weights))


def _select_candidate_frame(observed: pd.DataFrame, target_row: pd.Series) -> tuple[pd.DataFrame, str]:
    if observed.empty:
        raise ValueError("Observed frame is empty when selecting candidate rows.")
    selectors = [
        ("same_refined_type", observed["refined_date_type"].eq(target_row["refined_date_type"])),
        (
            "same_group_dow_season",
            observed["date_type_group"].eq(target_row["date_type_group"])
            & observed["dayofweek"].eq(target_row["dayofweek"])
            & observed["season"].eq(target_row["season"]),
        ),
        (
            "same_phase_dow",
            observed["phase_group"].eq(target_row["phase_group"]) & observed["dayofweek"].eq(target_row["dayofweek"]),
        ),
        (
            "same_group_dow",
            observed["date_type_group"].eq(target_row["date_type_group"]) & observed["dayofweek"].eq(target_row["dayofweek"]),
        ),
        ("same_group", observed["date_type_group"].eq(target_row["date_type_group"])),
        ("same_dow", observed["dayofweek"].eq(target_row["dayofweek"])),
        ("any_observed", pd.Series(True, index=observed.index)),
    ]
    for method_name, mask in selectors:
        candidate_frame = observed.loc[mask].copy()
        if not candidate_frame.empty:
            return candidate_frame, method_name
    raise ValueError(f"Unable to select candidates for {target_row['电力用户编码']} on {target_row['date'].date()}")


def _estimate_total_from_anchor(
    candidate_frame: pd.DataFrame,
    target_date: pd.Timestamp,
    target_day_observed: pd.DataFrame,
    observed_by_date: dict[pd.Timestamp, pd.DataFrame],
    top_k: int = IMPUTE_TOP_K,
) -> float | None:
    if candidate_frame.empty or target_day_observed.empty:
        return None
    target_codes = set(target_day_observed["电力用户编码"].tolist())
    target_anchor_total = float(target_day_observed["daily_total"].sum())
    if target_anchor_total <= 1e-9:
        return None

    scored = candidate_frame.copy()
    scored["distance_days"] = (scored["date"] - target_date).abs().dt.days.astype(float)
    scored = scored.sort_values(["distance_days", "date"]).head(top_k)

    ratios: list[float] = []
    weights: list[float] = []
    for row in scored.itertuples(index=False):
        history_day = observed_by_date.get(pd.Timestamp(row.date))
        if history_day is None or history_day.empty:
            continue
        overlap = history_day[history_day["电力用户编码"].isin(target_codes)]
        anchor_total = float(overlap["daily_total"].sum())
        if anchor_total <= 1e-9:
            continue
        ratios.append(float(row.daily_total) / anchor_total)
        weights.append(1.0 / (1.0 + abs((pd.Timestamp(row.date) - target_date).days)))

    if not ratios:
        return None
    return float(np.average(np.array(ratios, dtype=float), weights=np.array(weights, dtype=float)) * target_anchor_total)


def _rescale_profile(profile: np.ndarray, target_total: float | None) -> np.ndarray:
    profile = np.maximum(np.asarray(profile, dtype=float), 0.0)
    if target_total is None:
        return profile
    total = float(profile.sum())
    if total <= 1e-9:
        return np.full(24, float(target_total) / 24.0, dtype=float)
    return profile * (float(target_total) / total)


def _estimate_missing_profile(observed: pd.DataFrame, target_row: pd.Series, target_total: float | None = None) -> tuple[np.ndarray, str]:
    candidate_frame, method_name = _select_candidate_frame(observed, target_row)
    profile = _weighted_profile(candidate_frame, target_row["date"])
    if profile is None:
        raise ValueError(f"Unable to impute company profile for {target_row['电力用户编码']} on {target_row['date'].date()}")
    return _rescale_profile(profile, target_total=target_total), method_name


def complete_company_daily(company_daily: pd.DataFrame, baseline_companies: pd.DataFrame) -> pd.DataFrame:
    all_dates = pd.DataFrame({"date": sorted(company_daily["date"].drop_duplicates().tolist())})
    date_meta = build_date_meta(all_dates["date"].tolist())
    full_grid = (
        baseline_companies.assign(_key=1)
        .merge(all_dates.assign(_key=1), on="_key", how="inner")
        .drop(columns="_key")
        .merge(date_meta, on="date", how="left")
        .merge(company_daily, on=["电力用户编码", "电力用户名称", "date"], how="left")
        .sort_values(["电力用户编码", "date"])
        .reset_index(drop=True)
    )
    full_grid["has_observed"] = full_grid["daily_total"].notna().astype(int)
    full_grid["imputation_method"] = np.where(full_grid["has_observed"].eq(1), "observed", "")
    observed_all = full_grid[full_grid["has_observed"].eq(1)].copy()
    observed_median_total_by_date = observed_all.groupby("date")["daily_total"].median().to_dict()
    observed_by_date = {pd.Timestamp(date): frame.copy() for date, frame in observed_all.groupby("date", sort=False)}

    for column in LOAD_COLUMNS:
        full_grid[column] = pd.to_numeric(full_grid[column], errors="coerce")

    completed_groups: list[pd.DataFrame] = []
    for _, company_frame in full_grid.groupby(["电力用户编码", "电力用户名称"], sort=False):
        company_frame = company_frame.sort_values("date").reset_index(drop=True)
        observed = company_frame[company_frame["has_observed"].eq(1)].copy()
        if observed.empty:
            raise ValueError(f"No observed history for company {company_frame.loc[0, '电力用户编码']}")
        for index, row in company_frame[company_frame["has_observed"].eq(0)].iterrows():
            target_day_observed = observed_by_date.get(pd.Timestamp(row["date"]), observed_all.iloc[0:0].copy())
            own_history = observed[observed["date"].lt(row["date"])].copy()
            if not own_history.empty:
                candidate_frame, method_name = _select_candidate_frame(own_history, row)
                anchor_total = _estimate_total_from_anchor(
                    candidate_frame=candidate_frame,
                    target_date=pd.Timestamp(row["date"]),
                    target_day_observed=target_day_observed,
                    observed_by_date=observed_by_date,
                )
                if anchor_total is None:
                    anchor_total = _weighted_total(candidate_frame, pd.Timestamp(row["date"]))
                profile = _weighted_profile(candidate_frame, pd.Timestamp(row["date"]))
                if profile is None:
                    raise ValueError(f"Unable to build profile for {row['电力用户编码']} on {pd.Timestamp(row['date']).date()}")
                profile = _rescale_profile(profile, target_total=anchor_total)
            else:
                global_history = observed_all[observed_all["date"].le(row["date"])].copy()
                target_total = float(
                    observed_median_total_by_date.get(
                        row["date"],
                        global_history["daily_total"].median() if not global_history.empty else 0.0,
                    )
                )
                profile, method_name = _estimate_missing_profile(global_history, row, target_total=target_total)
                method_name = f"global_{method_name}"
            for hour, column in enumerate(LOAD_COLUMNS):
                company_frame.at[index, column] = float(profile[hour])
            company_frame.at[index, "daily_total"] = float(profile.sum())
            company_frame.at[index, "imputation_method"] = method_name
        completed_groups.append(company_frame)

    completed = pd.concat(completed_groups, ignore_index=True)
    completed["daily_total"] = completed[LOAD_COLUMNS].sum(axis=1)
    completed["actual_daily_total_company"] = np.where(completed["has_observed"].eq(1), completed["daily_total"], 0.0)
    completed["imputed_daily_total_company"] = np.where(completed["has_observed"].eq(1), 0.0, completed["daily_total"])
    return completed.sort_values(["date", "电力用户编码"]).reset_index(drop=True)


def expand_company_hourly(company_daily: pd.DataFrame) -> pd.DataFrame:
    long = company_daily.melt(
        id_vars=[
            "电力用户编码",
            "电力用户名称",
            "date",
            "has_observed",
            "imputation_method",
            "actual_daily_total_company",
            "imputed_daily_total_company",
        ],
        value_vars=LOAD_COLUMNS,
        var_name="load_column",
        value_name="load",
    )
    long["hour"] = long["load_column"].str.replace("load_h", "", regex=False).astype(int)
    long["timestamp"] = long["date"] + pd.to_timedelta(long["hour"], unit="h")
    return long.sort_values(["date", "电力用户编码", "hour"]).reset_index(drop=True)


def aggregate_site_hourly(company_hourly: pd.DataFrame, baseline_company_count: int) -> pd.DataFrame:
    daily_company = company_hourly[
        [
            "date",
            "电力用户编码",
            "has_observed",
            "actual_daily_total_company",
            "imputed_daily_total_company",
        ]
    ].drop_duplicates(["date", "电力用户编码"])
    company_counts = (
        daily_company.groupby("date", as_index=False)
        .agg(
            actual_company_count=("has_observed", "sum"),
            actual_daily_total=("actual_daily_total_company", "sum"),
            imputed_daily_total=("imputed_daily_total_company", "sum"),
        )
        .assign(
            actual_company_count=lambda frame: frame["actual_company_count"].astype(int),
            imputed_company_count=lambda frame: baseline_company_count - frame["actual_company_count"],
        )
    )
    aggregated = (
        company_hourly.groupby(["date", "hour", "timestamp"], as_index=False)["load"]
        .sum()
        .merge(company_counts, on="date", how="left")
        .sort_values(["timestamp"])
        .reset_index(drop=True)
    )
    aggregated["daily_total"] = aggregated.groupby("date")["load"].transform("sum")
    return aggregated


def _date_type_label(target_date: pd.Timestamp, holiday_info: pd.Series) -> str:
    if int(holiday_info["is_holiday_cn"]) == 1:
        return "holiday"
    if int(holiday_info["is_makeup_workday"]) == 1:
        return "makeup_workday"
    if int(target_date.dayofweek in [5, 6]) == 1:
        return "weekend"
    return "workday"


def build_history_hourly(aggregated: pd.DataFrame, baseline_company_count: int) -> pd.DataFrame:
    holiday_frame = build_official_holiday_frame(sorted(aggregated["date"].drop_duplicates().tolist()))
    holiday_frame = holiday_frame.rename(columns={"date": "holiday_date"})
    frame = aggregated.copy()
    frame["holiday_date"] = frame["date"]
    frame = frame.merge(holiday_frame, on="holiday_date", how="left")

    frame["site_id"] = SITE_ID
    frame["site_name"] = SITE_NAME
    frame["city_cn"] = CITY_CN
    frame["city_en"] = CITY_EN
    frame["province_cn"] = PROVINCE_CN
    frame["year"] = frame["date"].dt.year
    frame["month"] = frame["date"].dt.month
    frame["quarter"] = frame["date"].dt.quarter
    frame["weekofyear"] = frame["date"].dt.isocalendar().week.astype(int)
    frame["dayofmonth"] = frame["date"].dt.day
    frame["dayofweek"] = frame["date"].dt.dayofweek
    frame["dayofyear"] = frame["date"].dt.dayofyear
    frame["is_weekend"] = frame["dayofweek"].isin([5, 6]).astype(int)
    frame["is_month_start"] = frame["date"].dt.is_month_start.astype(int)
    frame["is_month_end"] = frame["date"].dt.is_month_end.astype(int)
    frame["season"] = frame["month"].map(season_name)
    frame["date_type_cn"] = [
        _date_type_label(target_date, holiday_info)
        for target_date, holiday_info in zip(frame["date"], frame.to_dict(orient="records"))
    ]
    frame["split"] = np.where(frame["date"].lt(BASELINE_COMPANY_START), "train", np.where(frame["date"].le(TRAIN_END), "validation", "test"))
    frame["is_fully_actual_5_company"] = frame["actual_company_count"].eq(baseline_company_count).astype(int)
    frame["日电量"] = frame["daily_total"]

    return frame[
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
            "split",
            "actual_company_count",
            "imputed_company_count",
            "is_fully_actual_5_company",
            "actual_daily_total",
            "imputed_daily_total",
            "日电量",
            "load",
        ]
    ].sort_values(["timestamp"]).reset_index(drop=True)


def build_history_daily(aggregated: pd.DataFrame, baseline_company_count: int) -> pd.DataFrame:
    wide = aggregated.pivot_table(index="date", columns="hour", values="load", aggfunc="sum").reset_index()
    wide.columns = ["date", *LOAD_COLUMNS]
    wide = wide.sort_values("date").reset_index(drop=True)

    company_counts = aggregated[
        [
            "date",
            "actual_company_count",
            "imputed_company_count",
            "actual_daily_total",
            "imputed_daily_total",
            "daily_total",
        ]
    ].drop_duplicates("date")
    frame = wide.merge(company_counts, on="date", how="left")
    holiday_frame = build_official_holiday_frame(sorted(frame["date"].tolist()))
    holiday_frame = holiday_frame.rename(columns={"date": "holiday_date"})
    frame["holiday_date"] = frame["date"]
    frame = frame.merge(holiday_frame, on="holiday_date", how="left")

    frame["year"] = frame["date"].dt.year
    frame["month"] = frame["date"].dt.month
    frame["quarter"] = frame["date"].dt.quarter
    frame["weekofyear"] = frame["date"].dt.isocalendar().week.astype(int)
    frame["dayofmonth"] = frame["date"].dt.day
    frame["dayofweek"] = frame["date"].dt.dayofweek
    frame["dayofyear"] = frame["date"].dt.dayofyear
    frame["is_weekend"] = frame["dayofweek"].isin([5, 6]).astype(int)
    frame["is_month_start"] = frame["date"].dt.is_month_start.astype(int)
    frame["is_month_end"] = frame["date"].dt.is_month_end.astype(int)
    frame["season"] = frame["month"].map(season_name)
    frame["date_type_cn"] = [
        _date_type_label(target_date, holiday_info)
        for target_date, holiday_info in zip(frame["date"], frame.to_dict(orient="records"))
    ]
    frame["refined_date_type"] = frame["date"].map(compute_refined_type)
    frame["date_type_group"] = frame["refined_date_type"].map(compute_date_type_group)
    frame["holiday_family"] = frame["refined_date_type"].map(holiday_family_name)
    frame["phase_group"] = frame["refined_date_type"].map(phase_group_name)
    frame["split"] = np.where(frame["date"].lt(BASELINE_COMPANY_START), "train", np.where(frame["date"].le(TRAIN_END), "validation", "test"))
    frame["is_fully_actual_5_company"] = frame["actual_company_count"].eq(baseline_company_count).astype(int)
    frame["is_actual_observation"] = 1

    ordered_columns = [
        "date",
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
        "refined_date_type",
        "date_type_group",
        "holiday_family",
        "phase_group",
        "split",
        "actual_company_count",
        "imputed_company_count",
        "is_fully_actual_5_company",
        "actual_daily_total",
        "imputed_daily_total",
        *LOAD_COLUMNS,
        "daily_total",
        "is_actual_observation",
    ]
    return frame[ordered_columns].sort_values("date").reset_index(drop=True)


def build_daily_total_dataset(history_daily: pd.DataFrame, baseline_company_count: int) -> pd.DataFrame:
    frame = history_daily.copy()
    frame["site_id"] = SITE_ID
    frame["site_name"] = SITE_NAME
    frame["baseline_company_count"] = baseline_company_count
    ordered_columns = [
        "site_id",
        "site_name",
        "date",
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
        "refined_date_type",
        "date_type_group",
        "holiday_family",
        "phase_group",
        "split",
        "baseline_company_count",
        "actual_company_count",
        "imputed_company_count",
        "daily_total",
        "actual_daily_total",
        "imputed_daily_total",
        "is_actual_observation",
    ]
    return frame[ordered_columns].sort_values("date").reset_index(drop=True)


def assign_split(dataset: pd.DataFrame) -> pd.DataFrame:
    frame = dataset.copy()
    frame["split"] = np.where(
        frame["target_date"].lt(BASELINE_COMPANY_START),
        "train",
        np.where(frame["target_date"].le(TRAIN_END), "validation", "test"),
    )
    return frame


def build_summary(
    company_daily: pd.DataFrame,
    baseline_companies: pd.DataFrame,
    history_daily: pd.DataFrame,
    baseline_d6: pd.DataFrame,
    raw_dir: Path,
    baseline_snapshot_date: pd.Timestamp,
) -> dict[str, object]:
    short_lags, weekly_lags, _ = get_issue_lag_config(ISSUE_GAP_DAYS)
    observed_company_count = company_daily.groupby("date")["has_observed"].sum().astype(int)
    return {
        "site_id": SITE_ID,
        "site_name": SITE_NAME,
        "baseline_company_snapshot_date": str(pd.Timestamp(baseline_snapshot_date).date()),
        "raw_dir": str(raw_dir),
        "raw_file_count": len(list(raw_dir.glob(RAW_GLOB))),
        "source_date_start": str(company_daily["date"].min().date()),
        "source_date_end": str(company_daily["date"].max().date()),
        "history_daily_rows": int(len(history_daily)),
        "history_hourly_rows": int(len(history_daily) * 24),
        "baseline_d6_rows": int(len(baseline_d6)),
        "baseline_company_count": int(len(baseline_companies)),
        "daily_company_count_min": int(observed_company_count.min()),
        "daily_company_count_max": int(observed_company_count.max()),
        "daily_total_min": round(float(history_daily["daily_total"].min()), 4),
        "daily_total_max": round(float(history_daily["daily_total"].max()), 4),
        "target_date_start": str(baseline_d6["target_date"].min().date()) if not baseline_d6.empty else None,
        "target_date_end": str(baseline_d6["target_date"].max().date()) if not baseline_d6.empty else None,
        "split_counts": baseline_d6["split"].value_counts().to_dict(),
        "issue_gap_days": ISSUE_GAP_DAYS,
        "short_lags": short_lags,
        "weekly_lags": weekly_lags,
    }


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir.resolve()
    output_dir = (args.output_dir or raw_dir / "new").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_company_hourly = load_company_hourly(raw_dir)
    baseline_snapshot_date = pd.Timestamp(raw_company_hourly["date"].max()).normalize()
    baseline_companies = build_baseline_company_pool(raw_company_hourly)
    observed_company_daily = build_company_daily_profiles(raw_company_hourly, baseline_companies)
    company_daily = complete_company_daily(observed_company_daily, baseline_companies)
    company_hourly = expand_company_hourly(company_daily)
    aggregated = aggregate_site_hourly(company_hourly, baseline_company_count=len(baseline_companies))
    history_hourly = build_history_hourly(aggregated, baseline_company_count=len(baseline_companies))
    history_daily = build_history_daily(aggregated, baseline_company_count=len(baseline_companies))
    daily_total = build_daily_total_dataset(history_daily, baseline_company_count=len(baseline_companies))
    baseline_d6 = assign_split(build_issue_gap_dataset(history_hourly, issue_gap_days=ISSUE_GAP_DAYS))

    baseline_company_path = output_dir / "baseline_company_roster.csv"
    history_hourly_path = output_dir / "history_hourly.csv"
    history_daily_path = output_dir / "history_daily.csv"
    daily_total_path = output_dir / "daily_total_dataset.csv"
    baseline_d6_path = output_dir / "baseline_d6_dataset.csv"
    summary_path = output_dir / "dataset_summary.json"

    baseline_companies.to_csv(baseline_company_path, index=False, encoding="utf-8-sig")
    history_hourly.to_csv(history_hourly_path, index=False, encoding="utf-8-sig")
    history_daily.to_csv(history_daily_path, index=False, encoding="utf-8-sig")
    daily_total.to_csv(daily_total_path, index=False, encoding="utf-8-sig")
    baseline_d6.to_csv(baseline_d6_path, index=False, encoding="utf-8-sig")

    summary = build_summary(
        company_daily,
        baseline_companies,
        history_daily,
        baseline_d6,
        raw_dir,
        baseline_snapshot_date=baseline_snapshot_date,
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
