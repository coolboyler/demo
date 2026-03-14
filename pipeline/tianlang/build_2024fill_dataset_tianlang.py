from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from china_holiday_official import build_official_holiday_frame  # noqa: E402
from forecast_d6 import compute_date_type_group, compute_refined_type  # noqa: E402
from train_d5 import build_issue_gap_dataset  # noqa: E402
from update_d6 import holiday_family_name, phase_group_name, season_name  # noqa: E402


LOAD_COLUMNS = [f"load_h{hour:02d}" for hour in range(24)]
BASELINE_COMPANY_START = pd.Timestamp("2026-01-01")
TRAIN_END = pd.Timestamp("2026-01-31")
SUPPORTED_SOURCE_YEAR = 2025
SYNTHETIC_YEAR = 2024
TOP_K = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Tianlang 2024 backfilled dataset from 2025 site-level history.")
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--history-daily-path", type=Path, default=Path("new/history_daily.csv"))
    parser.add_argument("--history-hourly-path", type=Path, default=Path("new/history_hourly.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("new"))
    return parser.parse_args()


def load_inputs(history_daily_path: Path, history_hourly_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    history_daily = pd.read_csv(history_daily_path, parse_dates=["date"])
    history_hourly = pd.read_csv(history_hourly_path, parse_dates=["timestamp", "date"])
    return history_daily, history_hourly


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
    frame["date_type_cn"] = np.where(
        frame["is_holiday_cn"].eq(1),
        "holiday",
        np.where(frame["is_makeup_workday"].eq(1), "makeup_workday", np.where(frame["is_weekend"].eq(1), "weekend", "workday")),
    )
    frame["refined_date_type"] = frame["date"].map(compute_refined_type)
    frame["date_type_group"] = frame["refined_date_type"].map(compute_date_type_group)
    frame["holiday_family"] = frame["refined_date_type"].map(holiday_family_name)
    frame["phase_group"] = frame["refined_date_type"].map(phase_group_name)
    return frame.drop(columns=["holiday_date"]).sort_values("date").reset_index(drop=True)


def add_holiday_shape_meta(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.sort_values("date").copy().reset_index(drop=True)
    holiday_segment: list[str] = []
    holiday_rel_key: list[str] = []
    for row in output.itertuples(index=False):
        refined = str(row.refined_date_type)
        if refined.startswith("holiday:"):
            holiday_segment.append("holiday")
            holiday_rel_key.append("holiday")
        elif refined.startswith("pre_"):
            holiday_segment.append("pre")
            holiday_rel_key.append(refined[4:])
        elif refined.startswith("post_"):
            holiday_segment.append("post")
            holiday_rel_key.append(refined[5:])
        else:
            holiday_segment.append("other")
            holiday_rel_key.append("")
    output["holiday_segment"] = holiday_segment
    output["holiday_rel_key"] = holiday_rel_key
    output["holiday_pos"] = np.nan
    output["holiday_len"] = np.nan

    holiday_rows = output[output["holiday_segment"].eq("holiday")].copy()
    if not holiday_rows.empty:
        for family_name in sorted(value for value in holiday_rows["holiday_family"].dropna().unique().tolist() if value):
            family_idx = output.index[output["holiday_family"].eq(family_name) & output["holiday_segment"].eq("holiday")].tolist()
            if not family_idx:
                continue
            block: list[int] = [family_idx[0]]
            previous_idx = family_idx[0]
            for current_idx in family_idx[1:]:
                if (output.loc[current_idx, "date"] - output.loc[previous_idx, "date"]).days == 1:
                    block.append(current_idx)
                else:
                    block_len = len(block)
                    for offset, target_idx in enumerate(block, start=1):
                        output.loc[target_idx, "holiday_pos"] = offset
                        output.loc[target_idx, "holiday_len"] = block_len
                    block = [current_idx]
                previous_idx = current_idx
            if block:
                block_len = len(block)
                for offset, target_idx in enumerate(block, start=1):
                    output.loc[target_idx, "holiday_pos"] = offset
                    output.loc[target_idx, "holiday_len"] = block_len
    return output


def interpolate_holiday_profile(source_rows: pd.DataFrame, target_pos: int, target_len: int) -> np.ndarray:
    ordered = source_rows.sort_values("holiday_pos")
    matrix = ordered[LOAD_COLUMNS].to_numpy(dtype=float)
    source_x = np.linspace(0.0, 1.0, len(matrix))
    target_x = (target_pos - 1) / max(target_len - 1, 1) if target_len > 1 else 0.0
    return np.array([np.interp(target_x, source_x, matrix[:, hour]) for hour in range(24)], dtype=float)


def interpolate_relative_segment_profile(segment_rows: pd.DataFrame, target_offset: int) -> np.ndarray | None:
    if segment_rows.empty:
        return None
    keyed = segment_rows.copy()
    keyed["relative_offset"] = keyed["holiday_rel_key"].astype(str).str.extract(r"_d(\d+)$").astype(float)
    keyed = keyed.dropna(subset=["relative_offset"]).copy()
    if keyed.empty:
        return None
    keyed["relative_offset"] = keyed["relative_offset"].astype(int)
    grouped = keyed.groupby("relative_offset", as_index=False)[LOAD_COLUMNS].mean().sort_values("relative_offset")
    source_x = grouped["relative_offset"].to_numpy(dtype=float)
    matrix = grouped[LOAD_COLUMNS].to_numpy(dtype=float)
    if len(source_x) == 1:
        return matrix[0]
    return np.array([np.interp(float(target_offset), source_x, matrix[:, hour]) for hour in range(24)], dtype=float)


def weighted_profile(candidate_frame: pd.DataFrame, target_row: pd.Series, top_k: int = TOP_K) -> np.ndarray:
    scored = candidate_frame.copy()
    month_distance = np.abs(scored["month"].to_numpy(dtype=int) - int(target_row["month"]))
    month_distance = np.minimum(month_distance, 12 - month_distance)
    day_distance = np.abs(scored["dayofmonth"].to_numpy(dtype=int) - int(target_row["dayofmonth"]))
    recency_distance = np.abs(scored["dayofyear"].to_numpy(dtype=int) - int(target_row["dayofyear"]))
    score = 12.0 - month_distance * 1.2 - day_distance / 6.0 - recency_distance / 45.0
    score += np.where(scored["dayofweek"].eq(target_row["dayofweek"]), 2.5, 0.0)
    score += np.where(scored["season"].eq(target_row["season"]), 1.5, 0.0)
    score += np.where(scored["phase_group"].eq(target_row["phase_group"]), 1.5, 0.0)
    scored["score"] = score
    scored = scored.sort_values(["score", "date"], ascending=[False, True]).head(top_k)
    weights = np.maximum(scored["score"].to_numpy(dtype=float), 0.1)
    return np.average(scored[LOAD_COLUMNS].to_numpy(dtype=float), axis=0, weights=weights)


def select_candidates(source_meta: pd.DataFrame, target_row: pd.Series) -> pd.DataFrame:
    selectors = [
        source_meta["refined_date_type"].eq(target_row["refined_date_type"]),
        source_meta["date_type_group"].eq(target_row["date_type_group"]) & source_meta["dayofweek"].eq(target_row["dayofweek"]) & source_meta["season"].eq(target_row["season"]),
        source_meta["phase_group"].eq(target_row["phase_group"]) & source_meta["dayofweek"].eq(target_row["dayofweek"]),
        source_meta["date_type_group"].eq(target_row["date_type_group"]) & source_meta["dayofweek"].eq(target_row["dayofweek"]),
        source_meta["date_type_group"].eq(target_row["date_type_group"]),
        source_meta["dayofweek"].eq(target_row["dayofweek"]),
    ]
    for mask in selectors:
        candidate_frame = source_meta.loc[mask].copy()
        if not candidate_frame.empty:
            return candidate_frame
    return source_meta.copy()


def synthesize_2024_daily(history_daily: pd.DataFrame) -> pd.DataFrame:
    source_daily = history_daily[history_daily["date"].dt.year.eq(SUPPORTED_SOURCE_YEAR)].copy().reset_index(drop=True)
    source_meta = add_holiday_shape_meta(source_daily)
    target_dates = pd.date_range(f"{SYNTHETIC_YEAR}-01-01", f"{SYNTHETIC_YEAR}-12-31", freq="D")
    target_meta = add_holiday_shape_meta(build_date_meta(target_dates.tolist()))

    baseline_company_count = int(history_daily["actual_company_count"].max())
    synthetic_rows: list[dict[str, object]] = []

    for row in target_meta.itertuples(index=False):
        profile: np.ndarray | None = None
        if row.holiday_segment == "holiday" and row.holiday_family:
            source_rows = source_meta[
                source_meta["holiday_family"].eq(row.holiday_family) & source_meta["holiday_segment"].eq("holiday")
            ].copy()
            if not source_rows.empty:
                profile = interpolate_holiday_profile(source_rows, int(row.holiday_pos), int(row.holiday_len))
        elif row.holiday_segment in {"pre", "post"} and row.holiday_family:
            source_rows = source_meta[
                source_meta["holiday_family"].eq(row.holiday_family) & source_meta["holiday_rel_key"].eq(row.holiday_rel_key)
            ].copy()
            if not source_rows.empty:
                profile = source_rows[LOAD_COLUMNS].mean().to_numpy(dtype=float)
            else:
                target_offset = int(str(row.holiday_rel_key).rsplit("_d", 1)[1])
                segment_rows = source_meta[
                    source_meta["holiday_family"].eq(row.holiday_family) & source_meta["holiday_segment"].eq(row.holiday_segment)
                ].copy()
                profile = interpolate_relative_segment_profile(segment_rows, target_offset)
        if profile is None:
            candidates = select_candidates(source_meta, pd.Series(row._asdict()))
            profile = weighted_profile(candidates, pd.Series(row._asdict()))

        profile = np.maximum(np.asarray(profile, dtype=float), 0.0)
        total = float(profile.sum())
        record = row._asdict()
        record.update(
            {
                "actual_company_count": baseline_company_count,
                "imputed_company_count": 0,
                "is_fully_actual_5_company": 1,
                "actual_daily_total": total,
                "imputed_daily_total": 0.0,
                "daily_total": total,
                "is_actual_observation": 0,
                "split": "train",
            }
        )
        for hour, column in enumerate(LOAD_COLUMNS):
            record[column] = float(profile[hour])
        synthetic_rows.append(record)

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
    return pd.DataFrame(synthetic_rows)[ordered_columns].sort_values("date").reset_index(drop=True)


def to_hourly(history_daily: pd.DataFrame, template_hourly: pd.DataFrame) -> pd.DataFrame:
    site_info = template_hourly.iloc[0]
    rows: list[dict[str, object]] = []
    for day in history_daily.itertuples(index=False):
        for hour in range(24):
            timestamp = pd.Timestamp(day.date) + pd.Timedelta(hours=hour)
            rows.append(
                {
                    "site_id": site_info.site_id,
                    "site_name": site_info.site_name,
                    "city_cn": site_info.city_cn,
                    "city_en": site_info.city_en,
                    "province_cn": site_info.province_cn,
                    "timestamp": timestamp,
                    "date": pd.Timestamp(day.date),
                    "hour": hour,
                    "year": int(day.year),
                    "month": int(day.month),
                    "quarter": int(day.quarter),
                    "weekofyear": int(day.weekofyear),
                    "dayofmonth": int(day.dayofmonth),
                    "dayofweek": int(day.dayofweek),
                    "dayofyear": int(day.dayofyear),
                    "is_weekend": int(day.is_weekend),
                    "is_month_start": int(day.is_month_start),
                    "is_month_end": int(day.is_month_end),
                    "season": day.season,
                    "is_holiday_cn": int(day.is_holiday_cn),
                    "holiday_name_cn": day.holiday_name_cn,
                    "is_workday_cn": int(day.is_workday_cn),
                    "is_makeup_workday": int(day.is_makeup_workday),
                    "date_type_cn": day.date_type_cn,
                    "split": day.split,
                    "actual_company_count": int(day.actual_company_count),
                    "imputed_company_count": int(day.imputed_company_count),
                    "is_fully_actual_5_company": int(day.is_fully_actual_5_company),
                    "actual_daily_total": float(day.actual_daily_total),
                    "imputed_daily_total": float(day.imputed_daily_total),
                    "日电量": float(day.daily_total),
                    "load": float(getattr(day, LOAD_COLUMNS[hour])),
                }
            )
    return pd.DataFrame(rows).sort_values(["timestamp"]).reset_index(drop=True)


def assign_split(dataset: pd.DataFrame) -> pd.DataFrame:
    frame = dataset.copy()
    frame["split"] = np.where(
        frame["target_date"].lt(BASELINE_COMPANY_START),
        "train",
        np.where(frame["target_date"].le(TRAIN_END), "validation", "test"),
    )
    return frame


def build_summary(synthetic_daily: pd.DataFrame, baseline_d6: pd.DataFrame, output_dir: Path) -> dict[str, object]:
    return {
        "task_definition": "Synthetic 2024 Tianlang site-level backfill built from 2025 site history.",
        "source_year": SUPPORTED_SOURCE_YEAR,
        "synthetic_year": SYNTHETIC_YEAR,
        "synthetic_date_start": str(synthetic_daily["date"].min().date()),
        "synthetic_date_end": str(synthetic_daily["date"].max().date()),
        "synthetic_days": int(len(synthetic_daily)),
        "synthetic_daily_total_min": float(synthetic_daily["daily_total"].min()),
        "synthetic_daily_total_max": float(synthetic_daily["daily_total"].max()),
        "baseline_d6_rows": int(len(baseline_d6)),
        "train_rows": int(baseline_d6["split"].eq("train").sum()),
        "validation_rows": int(baseline_d6["split"].eq("validation").sum()),
        "test_rows": int(baseline_d6["split"].eq("test").sum()),
        "output_dir": str(output_dir),
    }


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    history_daily_path = base_dir / args.history_daily_path if not args.history_daily_path.is_absolute() else args.history_daily_path
    history_hourly_path = base_dir / args.history_hourly_path if not args.history_hourly_path.is_absolute() else args.history_hourly_path
    output_dir = base_dir / args.output_dir if not args.output_dir.is_absolute() else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    history_daily, history_hourly = load_inputs(history_daily_path, history_hourly_path)
    synthetic_daily = synthesize_2024_daily(history_daily)
    synthetic_hourly = to_hourly(synthetic_daily, history_hourly)

    augmented_daily = pd.concat([synthetic_daily, history_daily], ignore_index=True).sort_values("date").reset_index(drop=True)
    augmented_hourly = pd.concat([synthetic_hourly, history_hourly], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    baseline_d6 = assign_split(build_issue_gap_dataset(augmented_hourly, issue_gap_days=6))

    daily_path = output_dir / "history_daily_2024fill.csv"
    hourly_path = output_dir / "history_hourly_2024fill.csv"
    d6_path = output_dir / "baseline_d6_dataset_2024fill.csv"
    summary_path = output_dir / "dataset_2024fill_summary.json"

    augmented_daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    augmented_hourly.to_csv(hourly_path, index=False, encoding="utf-8-sig")
    baseline_d6.to_csv(d6_path, index=False, encoding="utf-8-sig")

    summary = build_summary(synthetic_daily, baseline_d6, output_dir)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
