from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_core import daily_total_metrics, make_daily_prediction_frame, make_long_prediction_frame, prediction_metrics_from_wide, save_summary_json
from train_equivalent_5_total_makeup_special import generic_makeup_profile
from train_equivalent_5_total_spring_special import load_rule_params, rule_prediction


TARGET_COLUMNS = [f"target_load_h{hour:02d}" for hour in range(24)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a complete holiday-routing model and evaluate monthly holdout metrics.")
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset-path", type=Path, default=Path("new/baseline_d6_dataset.csv"))
    parser.add_argument("--rule-summary-path", type=Path, default=Path("results/baseline_d6_summary.json"))
    parser.add_argument("--output-prefix", default="best_d6")
    return parser.parse_args()


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        dataset_path,
        parse_dates=["issue_date", "target_date", "same_type_last1_date", "same_type_last2_date"],
    )


def add_holiday_meta(dataset: pd.DataFrame) -> pd.DataFrame:
    frame = dataset.sort_values("target_date").copy().reset_index(drop=True)

    holiday_family: list[str] = []
    holiday_segment: list[str] = []
    holiday_rel_key: list[str] = []

    for row in frame.itertuples(index=False):
        refined_type = str(row.target_refined_date_type)
        if refined_type.startswith("holiday:"):
            holiday_family.append(str(row.target_holiday_name_cn))
            holiday_segment.append("holiday")
            holiday_rel_key.append("holiday")
        elif refined_type.startswith("pre_"):
            family_name = refined_type[4:].rsplit("_d", 1)[0]
            offset = int(refined_type.rsplit("_d", 1)[1])
            holiday_family.append(family_name)
            holiday_segment.append("pre")
            holiday_rel_key.append(f"pre_d{offset}")
        elif refined_type.startswith("post_"):
            family_name = refined_type[5:].rsplit("_d", 1)[0]
            offset = int(refined_type.rsplit("_d", 1)[1])
            holiday_family.append(family_name)
            holiday_segment.append("post")
            holiday_rel_key.append(f"post_d{offset}")
        else:
            holiday_family.append("")
            holiday_segment.append("other")
            holiday_rel_key.append("")

    frame["holiday_family"] = holiday_family
    frame["holiday_segment"] = holiday_segment
    frame["holiday_rel_key"] = holiday_rel_key
    frame["holiday_pos"] = np.nan
    frame["holiday_len"] = np.nan

    for family_name in sorted([value for value in frame["holiday_family"].unique().tolist() if value]):
        mask = frame["holiday_family"].eq(family_name) & frame["holiday_segment"].eq("holiday")
        indices = np.flatnonzero(mask.to_numpy())
        if len(indices) == 0:
            continue
        block_start = indices[0]
        previous_idx = indices[0]
        block_indices: list[int] = [indices[0]]
        for current_idx in indices[1:]:
            if (frame.loc[current_idx, "target_date"] - frame.loc[previous_idx, "target_date"]).days == 1:
                block_indices.append(current_idx)
            else:
                block_len = len(block_indices)
                for offset, target_idx in enumerate(block_indices, start=1):
                    frame.loc[target_idx, "holiday_pos"] = offset
                    frame.loc[target_idx, "holiday_len"] = block_len
                block_indices = [current_idx]
                block_start = current_idx
            previous_idx = current_idx
        if block_indices:
            block_len = len(block_indices)
            for offset, target_idx in enumerate(block_indices, start=1):
                frame.loc[target_idx, "holiday_pos"] = offset
                frame.loc[target_idx, "holiday_len"] = block_len

    holiday_blocks: list[dict[str, object]] = []
    holiday_rows = frame[frame["holiday_segment"].eq("holiday")].sort_values("target_date")
    if not holiday_rows.empty:
        current_family = None
        current_dates: list[pd.Timestamp] = []
        for row in holiday_rows.itertuples(index=False):
            row_date = pd.Timestamp(row.target_date)
            if current_family is None:
                current_family = row.holiday_family
                current_dates = [row_date]
                continue
            if row.holiday_family == current_family and (row_date - current_dates[-1]).days == 1:
                current_dates.append(row_date)
            else:
                holiday_blocks.append(
                    {
                        "holiday_family": current_family,
                        "start_date": current_dates[0],
                        "end_date": current_dates[-1],
                    }
                )
                current_family = row.holiday_family
                current_dates = [row_date]
        if current_family is not None and current_dates:
            holiday_blocks.append(
                {
                    "holiday_family": current_family,
                    "start_date": current_dates[0],
                    "end_date": current_dates[-1],
                }
            )

    holiday_context_tags: list[str] = []
    holiday_context_family: list[str] = []
    holiday_context_side: list[str] = []
    holiday_context_bucket: list[float] = []
    days_since_last_holiday: list[float] = []
    days_to_next_holiday: list[float] = []
    last_holiday_family: list[str] = []
    next_holiday_family: list[str] = []

    for row in frame.itertuples(index=False):
        current_date = pd.Timestamp(row.target_date)
        if row.holiday_segment == "holiday":
            holiday_context_tags.append(f"holiday:{row.holiday_family}:p{int(row.holiday_pos)}of{int(row.holiday_len)}")
            holiday_context_family.append(str(row.holiday_family))
            holiday_context_side.append("holiday")
            holiday_context_bucket.append(float(row.holiday_pos))
            days_since_last_holiday.append(0.0)
            days_to_next_holiday.append(0.0)
            last_holiday_family.append(str(row.holiday_family))
            next_holiday_family.append(str(row.holiday_family))
            continue

        if row.holiday_segment == "pre":
            offset = int(str(row.holiday_rel_key).split("pre_d", 1)[1])
            holiday_context_tags.append(f"pre:{row.holiday_family}:d{offset}")
            holiday_context_family.append(str(row.holiday_family))
            holiday_context_side.append("pre")
            holiday_context_bucket.append(float(offset))
            days_since_last_holiday.append(np.nan)
            days_to_next_holiday.append(float(offset))
            last_holiday_family.append("")
            next_holiday_family.append(str(row.holiday_family))
            continue

        if row.holiday_segment == "post":
            offset = int(str(row.holiday_rel_key).split("post_d", 1)[1])
            holiday_context_tags.append(f"post:{row.holiday_family}:d{offset}")
            holiday_context_family.append(str(row.holiday_family))
            holiday_context_side.append("post")
            holiday_context_bucket.append(float(offset))
            days_since_last_holiday.append(float(offset))
            days_to_next_holiday.append(np.nan)
            last_holiday_family.append(str(row.holiday_family))
            next_holiday_family.append("")
            continue

        last_block = None
        next_block = None
        for block in holiday_blocks:
            if block["end_date"] < current_date:
                last_block = block
            elif block["start_date"] > current_date:
                next_block = block
                break

        since_days = float((current_date - last_block["end_date"]).days) if last_block is not None else np.nan
        to_days = float((next_block["start_date"] - current_date).days) if next_block is not None else np.nan
        last_family = str(last_block["holiday_family"]) if last_block is not None else ""
        next_family = str(next_block["holiday_family"]) if next_block is not None else ""

        tag = f"ordinary:{row.target_date_type_group}"
        context_family = ""
        context_side = "ordinary"
        context_bucket_value = np.nan
        if np.isfinite(since_days) and since_days <= 28 and (not np.isfinite(to_days) or since_days <= to_days):
            bucket = int((since_days - 1) // 7 + 1) if since_days >= 1 else 1
            bucket = max(1, min(bucket, 4))
            tag = f"after:{last_family}:w{bucket}"
            context_family = last_family
            context_side = "after"
            context_bucket_value = float(bucket)
        elif np.isfinite(to_days) and to_days <= 28:
            bucket = int((to_days - 1) // 7 + 1) if to_days >= 1 else 1
            bucket = max(1, min(bucket, 4))
            tag = f"before:{next_family}:w{bucket}"
            context_family = next_family
            context_side = "before"
            context_bucket_value = float(bucket)

        holiday_context_tags.append(tag)
        holiday_context_family.append(context_family)
        holiday_context_side.append(context_side)
        holiday_context_bucket.append(context_bucket_value)
        days_since_last_holiday.append(since_days)
        days_to_next_holiday.append(to_days)
        last_holiday_family.append(last_family)
        next_holiday_family.append(next_family)

    frame["holiday_context_tag"] = holiday_context_tags
    frame["holiday_context_family"] = holiday_context_family
    frame["holiday_context_side"] = holiday_context_side
    frame["holiday_context_bucket"] = holiday_context_bucket
    frame["days_since_last_holiday"] = days_since_last_holiday
    frame["days_to_next_holiday"] = days_to_next_holiday
    frame["last_holiday_family"] = last_holiday_family
    frame["next_holiday_family"] = next_holiday_family
    return frame


def interpolate_holiday_profile(previous_holidays: pd.DataFrame, target_pos: int, target_len: int) -> np.ndarray:
    source = previous_holidays.sort_values("holiday_pos")
    source_matrix = source[TARGET_COLUMNS].to_numpy(dtype=float)
    source_x = np.linspace(0.0, 1.0, len(source_matrix))
    target_x = (target_pos - 1) / max(target_len - 1, 1) if target_len > 1 else 0.0
    return np.array([np.interp(target_x, source_x, source_matrix[:, hour]) for hour in range(24)], dtype=float)


def holiday_special_profile(
    row: pd.Series | pd.core.frame.Pandas,
    history: pd.DataFrame,
    holiday_history_mode: str = "mean_all",
    shoulder_history_mode: str = "last1",
) -> np.ndarray | None:
    family_history = history[history["holiday_family"].eq(row.holiday_family)].copy()
    if family_history.empty or not row.holiday_family:
        return None

    if row.holiday_segment == "holiday":
        holiday_rows = family_history[family_history["holiday_segment"].eq("holiday")].copy()
        if holiday_rows.empty:
            return None
        available_years = sorted(holiday_rows["target_date"].dt.year.unique().tolist())
        if holiday_history_mode == "last1":
            available_years = available_years[-1:]
        profiles: list[np.ndarray] = []
        for target_year in available_years:
            previous_holidays = holiday_rows[holiday_rows["target_date"].dt.year.eq(target_year)].copy()
            profiles.append(interpolate_holiday_profile(previous_holidays, int(row.holiday_pos), int(row.holiday_len)))
        return np.mean(profiles, axis=0)

    same_key_rows = family_history[family_history["holiday_rel_key"].eq(row.holiday_rel_key)].sort_values("target_date")
    if same_key_rows.empty:
        return None
    if shoulder_history_mode == "last1":
        same_key_rows = same_key_rows.tail(1)
    else:
        same_key_rows = same_key_rows.tail(2)
    return same_key_rows[TARGET_COLUMNS].mean().to_numpy(dtype=float)


def daily_accuracy(actual: np.ndarray, pred: np.ndarray) -> float:
    actual_daily = actual.sum(axis=1)
    pred_daily = pred.sum(axis=1)
    daily_wape = np.abs(pred_daily - actual_daily).sum() / np.abs(actual_daily).sum() * 100
    return float(100 - daily_wape)


def ordinary_similarity_scores(
    row: pd.Series | pd.core.frame.Pandas,
    candidates: pd.DataFrame,
) -> np.ndarray:
    if candidates.empty:
        return np.array([], dtype=float)

    target_date = pd.Timestamp(row.target_date)
    previous_year_same_date = target_date - pd.DateOffset(years=1)
    previous_year_same_weekday = target_date - pd.Timedelta(days=364)

    candidate_dates = pd.to_datetime(candidates["target_date"])
    month_distance = np.abs(candidates["target_month"].to_numpy(dtype=int) - int(row.target_month))
    month_distance = np.minimum(month_distance, 12 - month_distance)
    dom_distance = np.abs(candidates["target_dayofmonth"].to_numpy(dtype=int) - int(row.target_dayofmonth))
    year_anchor_gap = np.abs((candidate_dates - previous_year_same_date).dt.days.to_numpy(dtype=float))
    weekday_anchor_gap = np.abs((candidate_dates - previous_year_same_weekday).dt.days.to_numpy(dtype=float))
    recency_gap = np.abs((pd.Timestamp(row.issue_date) - candidate_dates).dt.days.to_numpy(dtype=float))

    scores = np.zeros(len(candidates), dtype=float)
    scores += np.where(candidates["target_refined_date_type"].eq(row.target_refined_date_type), 10.0, 0.0)
    scores += np.where(candidates["holiday_context_tag"].eq(row.holiday_context_tag), 9.0, 0.0)
    scores += np.where(candidates["holiday_context_family"].eq(row.holiday_context_family), 4.0, 0.0)
    scores += np.where(candidates["holiday_context_side"].eq(row.holiday_context_side), 2.0, 0.0)
    scores += np.where(
        np.abs(candidates["holiday_context_bucket"].fillna(-999).to_numpy(dtype=float) - float(row.holiday_context_bucket if pd.notna(row.holiday_context_bucket) else -999)) < 0.1,
        2.5,
        0.0,
    )
    scores += np.where(candidates["target_dayofweek"].eq(row.target_dayofweek), 4.0, 0.0)
    scores += np.where(candidates["target_month"].eq(row.target_month), 4.0, np.maximum(0.0, 2.0 - month_distance * 0.75))
    scores += np.maximum(0.0, 3.0 - dom_distance / 2.0)
    scores += np.maximum(0.0, 6.0 - year_anchor_gap / 3.0)
    scores += np.maximum(0.0, 5.0 - weekday_anchor_gap / 4.0)
    scores += np.maximum(0.0, 1.5 - recency_gap / 180.0)
    return scores


def ordinary_similar_profile(
    row: pd.Series | pd.core.frame.Pandas,
    history: pd.DataFrame,
    top_k: int,
) -> np.ndarray | None:
    profile, _ = ordinary_similar_profile_with_details(row, history, top_k)
    return profile


def ordinary_similar_profile_with_details(
    row: pd.Series | pd.core.frame.Pandas,
    history: pd.DataFrame,
    top_k: int,
) -> tuple[np.ndarray | None, list[dict[str, object]]]:
    if str(row.holiday_segment) != "other":
        return None, []

    candidates = history[
        history["holiday_segment"].eq("other")
        & history["target_date_type_group"].eq(row.target_date_type_group)
    ].copy()
    if candidates.empty:
        return None, []

    # Prefer the same refined day type first, e.g. workday_w4 to workday_w4.
    same_refined_candidates = candidates[candidates["target_refined_date_type"].eq(row.target_refined_date_type)].copy()
    if len(same_refined_candidates) >= top_k:
        candidates = same_refined_candidates.reset_index(drop=True)

    scores = ordinary_similarity_scores(row, candidates)
    positive_mask = scores > 0
    if not positive_mask.any():
        return None, []

    candidate_index = np.flatnonzero(positive_mask)
    ranked_index = candidate_index[np.argsort(-scores[positive_mask])[:top_k]]
    selected_scores = scores[ranked_index]
    if selected_scores.sum() <= 0:
        return None, []
    weights = selected_scores / selected_scores.sum()
    selected_frame = candidates.iloc[ranked_index].copy().reset_index(drop=True)
    selected_profiles = selected_frame[TARGET_COLUMNS].to_numpy(dtype=float)
    details = []
    for idx, (_, candidate_row) in enumerate(selected_frame.iterrows()):
        details.append(
            {
                "reference_rank": idx + 1,
                "reference_date": pd.Timestamp(candidate_row["target_date"]).strftime("%Y-%m-%d"),
                "reference_context_tag": str(candidate_row.get("holiday_context_tag", "")),
                "reference_refined_type": str(candidate_row.get("target_refined_date_type", "")),
                "reference_score": float(selected_scores[idx]),
                "reference_weight": float(weights[idx]),
            }
        )
    return np.average(selected_profiles, axis=0, weights=weights), details


def train_ordinary_activation_map(dataset: pd.DataFrame, base_rule_prediction: np.ndarray) -> pd.DataFrame:
    train = dataset[dataset["split"].eq("train")].copy().reset_index(drop=True)
    base_train_prediction = base_rule_prediction[dataset["split"].eq("train").to_numpy()]
    rows: list[dict[str, object]] = []

    ordinary_groups = ["workday", "weekend"]
    top_k_grid = [3, 5, 7]
    alpha_grid = [0.2, 0.35, 0.5, 0.65]

    for ordinary_group in ordinary_groups:
        group_rows = train[
            train["holiday_segment"].eq("other")
            & train["target_date_type_group"].eq(ordinary_group)
            & train["target_is_makeup_workday"].eq(0)
            & train["target_is_holiday_cn"].eq(0)
        ].copy()
        if group_rows.empty:
            rows.append(
                {
                    "holiday_family": f"__ordinary_{ordinary_group}__",
                    "train_rows": 0,
                    "special_daily_accuracy_percent": np.nan,
                    "base_daily_accuracy_percent": np.nan,
                    "accuracy_lift_percent": np.nan,
                    "is_active": 0,
                    "activation_reason": "no_history",
                    "sample_dates": "",
                    "similar_top_k": np.nan,
                    "similar_alpha": np.nan,
                }
            )
            continue

        best_bundle: dict[str, object] | None = None
        for top_k in top_k_grid:
            actual_rows: list[np.ndarray] = []
            base_rows: list[np.ndarray] = []
            similar_rows: list[np.ndarray] = []
            successful_dates: list[pd.Timestamp] = []

            for row in group_rows.itertuples(index=False):
                history = train[train["target_date"].le(row.issue_date)].copy()
                similar_profile = ordinary_similar_profile(row, history, top_k=top_k)
                if similar_profile is None:
                    continue
                matching_idx = group_rows.index[group_rows["target_date"].eq(row.target_date)][0]
                actual_rows.append(np.array([getattr(row, column) for column in TARGET_COLUMNS], dtype=float))
                base_rows.append(base_train_prediction[matching_idx])
                similar_rows.append(similar_profile)
                successful_dates.append(row.target_date)

            if len(actual_rows) < 20:
                continue

            actual_matrix = np.vstack(actual_rows)
            base_matrix = np.vstack(base_rows)
            similar_matrix = np.vstack(similar_rows)
            base_accuracy = daily_accuracy(actual_matrix, base_matrix)

            for alpha in alpha_grid:
                blend_matrix = (1 - alpha) * base_matrix + alpha * similar_matrix
                special_accuracy = daily_accuracy(actual_matrix, blend_matrix)
                lift = special_accuracy - base_accuracy
                candidate = {
                    "holiday_family": f"__ordinary_{ordinary_group}__",
                    "train_rows": int(len(actual_rows)),
                    "special_daily_accuracy_percent": float(special_accuracy),
                    "base_daily_accuracy_percent": float(base_accuracy),
                    "accuracy_lift_percent": float(lift),
                    "is_active": int(lift > 0.2),
                    "activation_reason": "special_better" if lift > 0.2 else "base_better",
                    "sample_dates": ",".join(date_value.strftime("%Y-%m-%d") for date_value in successful_dates[-20:]),
                    "similar_top_k": int(top_k),
                    "similar_alpha": float(alpha),
                }
                if best_bundle is None or candidate["accuracy_lift_percent"] > best_bundle["accuracy_lift_percent"]:
                    best_bundle = candidate

        if best_bundle is None:
            rows.append(
                {
                    "holiday_family": f"__ordinary_{ordinary_group}__",
                    "train_rows": 0,
                    "special_daily_accuracy_percent": np.nan,
                    "base_daily_accuracy_percent": np.nan,
                    "accuracy_lift_percent": np.nan,
                    "is_active": 0,
                    "activation_reason": "no_valid_backtest",
                    "sample_dates": "",
                    "similar_top_k": np.nan,
                    "similar_alpha": np.nan,
                }
            )
        else:
            rows.append(best_bundle)

    return pd.DataFrame(rows).sort_values("holiday_family").reset_index(drop=True)


def train_family_activation_map(dataset: pd.DataFrame, base_rule_prediction: np.ndarray) -> pd.DataFrame:
    train = dataset[dataset["split"].eq("train")].copy().reset_index(drop=True)
    base_train_prediction = base_rule_prediction[dataset["split"].eq("train").to_numpy()]

    rows: list[dict[str, object]] = []
    families = sorted([value for value in train["holiday_family"].unique().tolist() if value])
    for family_name in families:
        family_rows = train[train["holiday_family"].eq(family_name) & train["holiday_segment"].ne("other")].copy()
        if family_rows.empty:
            continue
        special_rows: list[np.ndarray] = []
        base_rows: list[np.ndarray] = []
        actual_rows: list[np.ndarray] = []
        successful_dates: list[pd.Timestamp] = []
        for family_row in family_rows.itertuples(index=False):
            history = train[train["target_date"].dt.year.lt(family_row.target_date.year)].copy()
            special_profile = holiday_special_profile(family_row, history)
            if special_profile is None:
                continue
            matching_idx = family_rows.index[family_rows["target_date"].eq(family_row.target_date)][0]
            special_rows.append(special_profile)
            base_rows.append(base_train_prediction[matching_idx])
            actual_rows.append(np.array([getattr(family_row, column) for column in TARGET_COLUMNS], dtype=float))
            successful_dates.append(family_row.target_date)

        if len(special_rows) == 0:
            rows.append(
                {
                    "holiday_family": family_name,
                    "train_rows": 0,
                    "special_daily_accuracy_percent": np.nan,
                    "base_daily_accuracy_percent": np.nan,
                    "accuracy_lift_percent": np.nan,
                    "is_active": 0,
                    "activation_reason": "no_history",
                }
            )
            continue

        special_matrix = np.vstack(special_rows)
        base_matrix = np.vstack(base_rows)
        actual_matrix = np.vstack(actual_rows)
        special_accuracy = daily_accuracy(actual_matrix, special_matrix)
        base_accuracy = daily_accuracy(actual_matrix, base_matrix)
        lift = special_accuracy - base_accuracy
        is_active = int(len(actual_rows) >= 2 and lift > 0.5)
        rows.append(
            {
                "holiday_family": family_name,
                "train_rows": int(len(actual_rows)),
                "special_daily_accuracy_percent": float(special_accuracy),
                "base_daily_accuracy_percent": float(base_accuracy),
                "accuracy_lift_percent": float(lift),
                "is_active": is_active,
                "activation_reason": "special_better" if is_active else "base_better_or_insufficient_rows",
                "sample_dates": ",".join(date_value.strftime("%Y-%m-%d") for date_value in successful_dates),
            }
        )

    generic_makeup_rows = train[
        train["target_is_makeup_workday"].eq(1)
        & train["target_refined_date_type"].astype(str).str.startswith("makeup_workday")
    ].copy()
    makeup_special_rows: list[np.ndarray] = []
    makeup_base_rows: list[np.ndarray] = []
    makeup_actual_rows: list[np.ndarray] = []
    successful_makeup_dates: list[pd.Timestamp] = []

    for row in generic_makeup_rows.itertuples(index=False):
        history = train[train["target_date"].le(row.issue_date)].copy()
        special_profile = generic_makeup_profile(history, row)
        if special_profile is None:
            continue
        matching_idx = generic_makeup_rows.index[generic_makeup_rows["target_date"].eq(row.target_date)][0]
        makeup_special_rows.append(special_profile)
        makeup_base_rows.append(base_train_prediction[matching_idx])
        makeup_actual_rows.append(np.array([getattr(row, column) for column in TARGET_COLUMNS], dtype=float))
        successful_makeup_dates.append(row.target_date)

    if len(makeup_special_rows) > 0:
        special_matrix = np.vstack(makeup_special_rows)
        base_matrix = np.vstack(makeup_base_rows)
        actual_matrix = np.vstack(makeup_actual_rows)
        special_accuracy = daily_accuracy(actual_matrix, special_matrix)
        base_accuracy = daily_accuracy(actual_matrix, base_matrix)
        lift = special_accuracy - base_accuracy
        is_active = int(len(actual_matrix) >= 1 and lift > 0.0)
        rows.append(
            {
                "holiday_family": "__generic_makeup__",
                "train_rows": int(len(actual_matrix)),
                "special_daily_accuracy_percent": float(special_accuracy),
                "base_daily_accuracy_percent": float(base_accuracy),
                "accuracy_lift_percent": float(lift),
                "is_active": is_active,
                "activation_reason": "special_better" if is_active else "base_better",
                "sample_dates": ",".join(date_value.strftime("%Y-%m-%d") for date_value in successful_makeup_dates),
            }
        )
    else:
        rows.append(
            {
                "holiday_family": "__generic_makeup__",
                "train_rows": 0,
                "special_daily_accuracy_percent": np.nan,
                "base_daily_accuracy_percent": np.nan,
                "accuracy_lift_percent": np.nan,
                "is_active": 0,
                "activation_reason": "no_history",
                "sample_dates": "",
            }
        )

    return pd.DataFrame(rows).sort_values(["is_active", "holiday_family"], ascending=[False, True]).reset_index(drop=True)


def apply_holiday_router(
    frame: pd.DataFrame,
    full_history: pd.DataFrame,
    base_prediction: np.ndarray,
    active_families: set[str],
    makeup_active: bool,
    ordinary_config_map: dict[str, dict[str, float]] | None = None,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    prediction_matrix = base_prediction.copy()
    replacement_rows: list[dict[str, object]] = []
    ordinary_config_map = ordinary_config_map or {}

    for index, row in enumerate(frame.itertuples(index=False)):
        replacement_profile: np.ndarray | None = None
        route_name = "base_rule"
        similar_details: list[dict[str, object]] = []
        history = full_history[full_history["target_date"].le(row.issue_date)].copy()

        if (
            makeup_active
            and int(row.target_is_makeup_workday) == 1
            and str(row.target_refined_date_type).startswith("makeup_workday")
        ):
            replacement_profile = generic_makeup_profile(history, row)
            route_name = "generic_makeup"
        elif row.holiday_family in active_families and row.holiday_segment in {"holiday", "pre", "post"}:
            replacement_profile = holiday_special_profile(row, history[history["target_date"].dt.year.lt(row.target_date.year)].copy())
            route_name = f"holiday_family:{row.holiday_family}"
        elif row.holiday_segment == "other" and row.target_date_type_group in ordinary_config_map:
            config = ordinary_config_map[row.target_date_type_group]
            similar_profile, similar_details = ordinary_similar_profile_with_details(row, history, top_k=int(config["top_k"]))
            if similar_profile is not None:
                alpha = float(config["alpha"])
                replacement_profile = (1 - alpha) * prediction_matrix[index, :] + alpha * similar_profile
                route_name = f"ordinary_similar:{row.target_date_type_group}"

        if replacement_profile is not None:
            if all(hasattr(row, column) for column in TARGET_COLUMNS):
                actual_daily_total = float(np.array([getattr(row, column) for column in TARGET_COLUMNS], dtype=float).sum())
            else:
                actual_daily_total = np.nan
            prediction_matrix[index, :] = replacement_profile
            replacement_rows.append(
                {
                    "target_date": row.target_date,
                    "split": row.split,
                    "holiday_family": row.holiday_family,
                    "holiday_segment": row.holiday_segment,
                    "target_refined_date_type": row.target_refined_date_type,
                    "route_name": route_name,
                    "actual_daily_total": actual_daily_total,
                    "pred_daily_total": float(replacement_profile.sum()),
                    "similar_reference_dates": "|".join(detail["reference_date"] for detail in similar_details),
                    "similar_reference_tags": "|".join(detail["reference_context_tag"] for detail in similar_details),
                    "similar_reference_scores": "|".join(f"{detail['reference_score']:.6f}" for detail in similar_details),
                    "similar_reference_weights": "|".join(f"{detail['reference_weight']:.6f}" for detail in similar_details),
                }
            )

    return prediction_matrix, replacement_rows


def segment_metrics(frame: pd.DataFrame, segment_name: str) -> dict[str, object]:
    actual = frame[[f"actual_h{hour:02d}" for hour in range(24)]].to_numpy(dtype=float)
    predicted = frame[[f"pred_h{hour:02d}" for hour in range(24)]].to_numpy(dtype=float)
    hourly = prediction_metrics_from_wide(actual, predicted)
    daily = daily_total_metrics(frame)
    return {
        "segment": segment_name,
        "days": int(len(frame)),
        "hourly_accuracy_percent": float(100 - hourly["wape_percent"]),
        "hourly_wape_percent": float(hourly["wape_percent"]),
        "daily_accuracy_percent": float(daily["daily_total_accuracy_percent"]),
        "daily_wape_percent": float(daily["daily_total_wape_percent"]),
        "daily_bias_percent": float(daily["daily_total_bias_percent"]),
    }


def monthly_metric_rows(frame: pd.DataFrame, model_variant: str, split_name: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    month_frame = frame.copy()
    month_frame["year_month"] = month_frame["target_date"].dt.strftime("%Y-%m")
    for year_month, sub in month_frame.groupby("year_month"):
        rows.append(
            {
                "model_variant": model_variant,
                "split": split_name,
                "year_month": year_month,
                **segment_metrics(sub.copy(), "month"),
            }
        )
    return rows


def build_report_markdown(
    dataset_path: Path,
    issue_gap_days: int,
    activation_rows: list[dict[str, object]],
    segment_rows: list[dict[str, object]],
    monthly_rows: list[dict[str, object]],
    replacement_rows: list[dict[str, object]],
    prediction_path: Path,
    summary_path: Path,
) -> str:
    activation_cols = ["holiday_family", "train_rows", "special_daily_accuracy_percent", "base_daily_accuracy_percent", "accuracy_lift_percent", "is_active"]
    segment_cols = ["model_variant", "segment", "days", "hourly_accuracy_percent", "daily_accuracy_percent", "daily_bias_percent"]
    monthly_cols = ["model_variant", "split", "year_month", "days", "hourly_accuracy_percent", "daily_accuracy_percent", "daily_bias_percent"]
    replacement_cols = [
        "target_date",
        "split",
        "holiday_family",
        "holiday_segment",
        "route_name",
        "actual_daily_total",
        "pred_daily_total",
        "similar_reference_dates",
        "similar_reference_tags",
        "similar_reference_scores",
        "similar_reference_weights",
    ]

    def md_table(rows: list[dict[str, object]], columns: list[str]) -> list[str]:
        if not rows:
            return []
        header = "| " + " | ".join(columns) + " |"
        divider = "| " + " | ".join(["---"] * len(columns)) + " |"
        body = ["| " + " | ".join(str(row[column]) for column in columns) + " |" for row in rows]
        return [header, divider, *body]

    return "\n".join(
        [
            "# 假期分流模型评估",
            "",
            f"- 数据集：`{dataset_path}`",
            f"- 普通日走规则 D-{issue_gap_days}。",
            "- 节假日前后和节日本身按节日家族路由，只有训练回测优于基线的家族才激活。",
            "- 纯调休工作日 `makeup_workday_w*` 单独走调休专模。",
            "",
            "## 家族激活结果",
            *md_table(activation_rows, activation_cols),
            "",
            "## 总体分段评估",
            *md_table(segment_rows, segment_cols),
            "",
            "## 月度评估",
            *md_table(monthly_rows, monthly_cols),
            "",
            "## 被路由的日期",
            *md_table(replacement_rows, replacement_cols),
            "",
            f"- 预测输出：`{prediction_path}`",
            f"- 摘要输出：`{summary_path}`",
        ]
    )


def main() -> None:
    args = parse_args()
    results_dir = args.base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset = add_holiday_meta(load_dataset(args.dataset_path))
    rule_params = load_rule_params(args.rule_summary_path)
    rule_summary = json.loads(args.rule_summary_path.read_text(encoding="utf-8"))
    issue_gap_days = int(rule_summary.get("issue_gap_days", 5))
    base_rule_prediction = rule_prediction(dataset, rule_params, issue_gap_days=issue_gap_days)

    family_activation_frame = train_family_activation_map(dataset, base_rule_prediction)
    ordinary_activation_frame = train_ordinary_activation_map(dataset, base_rule_prediction)
    activation_frame = pd.concat([family_activation_frame, ordinary_activation_frame], ignore_index=True, sort=False)
    active_families = set(
        activation_frame.loc[
            activation_frame["is_active"].eq(1) & ~activation_frame["holiday_family"].astype(str).str.startswith("__"),
            "holiday_family",
        ]
    )
    makeup_active = bool(
        activation_frame.loc[activation_frame["holiday_family"].eq("__generic_makeup__"), "is_active"].max()
    )
    ordinary_config_map: dict[str, dict[str, float]] = {}
    for ordinary_group in ["workday", "weekend"]:
        ordinary_key = f"__ordinary_{ordinary_group}__"
        ordinary_rows = activation_frame[
            activation_frame["holiday_family"].eq(ordinary_key) & activation_frame["is_active"].eq(1)
        ]
        if ordinary_rows.empty:
            continue
        ordinary_row = ordinary_rows.iloc[0]
        ordinary_config_map[ordinary_group] = {
            "top_k": int(ordinary_row["similar_top_k"]),
            "alpha": float(ordinary_row["similar_alpha"]),
        }

    validation = dataset[dataset["split"].eq("validation")].copy().reset_index(drop=True)
    test = dataset[dataset["split"].eq("test")].copy().reset_index(drop=True)
    validation_base = base_rule_prediction[dataset["split"].eq("validation").to_numpy()]
    test_base = base_rule_prediction[dataset["split"].eq("test").to_numpy()]

    validation_router_pred, validation_replacements = apply_holiday_router(
        frame=validation,
        full_history=dataset,
        base_prediction=validation_base,
        active_families=active_families,
        makeup_active=makeup_active,
        ordinary_config_map=ordinary_config_map,
    )
    test_router_pred, test_replacements = apply_holiday_router(
        frame=test,
        full_history=dataset,
        base_prediction=test_base,
        active_families=active_families,
        makeup_active=makeup_active,
        ordinary_config_map=ordinary_config_map,
    )

    base_rule_name = f"base_rule_d{issue_gap_days}"
    validation_base_frame = make_daily_prediction_frame(validation, base_rule_name, validation_base)
    test_base_frame = make_daily_prediction_frame(test, base_rule_name, test_base)
    validation_router_frame = make_daily_prediction_frame(validation, "holiday_router", validation_router_pred)
    test_router_frame = make_daily_prediction_frame(test, "holiday_router", test_router_pred)
    final_test_long = make_long_prediction_frame(test_router_frame)

    segment_rows = [
        {"model_variant": base_rule_name, **segment_metrics(validation_base_frame.copy(), "validation")},
        {"model_variant": base_rule_name, **segment_metrics(test_base_frame.copy(), "test")},
        {"model_variant": "holiday_router", **segment_metrics(validation_router_frame.copy(), "validation")},
        {"model_variant": "holiday_router", **segment_metrics(test_router_frame.copy(), "test")},
    ]

    monthly_rows = []
    monthly_rows.extend(monthly_metric_rows(validation_base_frame.copy(), base_rule_name, "validation"))
    monthly_rows.extend(monthly_metric_rows(test_base_frame.copy(), base_rule_name, "test"))
    monthly_rows.extend(monthly_metric_rows(validation_router_frame.copy(), "holiday_router", "validation"))
    monthly_rows.extend(monthly_metric_rows(test_router_frame.copy(), "holiday_router", "test"))

    replacement_rows = validation_replacements + test_replacements

    prediction_path = results_dir / f"{args.output_prefix}_test_daily.csv"
    long_path = results_dir / f"{args.output_prefix}_test_long.csv"
    summary_path = results_dir / f"{args.output_prefix}_summary.json"
    report_path = results_dir / f"{args.output_prefix}_report.md"
    activation_path = results_dir / f"{args.output_prefix}_activation.csv"
    segment_path = results_dir / f"{args.output_prefix}_segment.csv"
    monthly_path = results_dir / f"{args.output_prefix}_monthly.csv"
    replacement_path = results_dir / f"{args.output_prefix}_replaced_days.csv"

    test_router_frame.to_csv(prediction_path, index=False)
    final_test_long.to_csv(long_path, index=False)
    activation_frame.to_csv(activation_path, index=False)
    pd.DataFrame(segment_rows).to_csv(segment_path, index=False)
    pd.DataFrame(monthly_rows).to_csv(monthly_path, index=False)
    pd.DataFrame(replacement_rows).to_csv(replacement_path, index=False)

    summary = {
        "task_definition": f"Complete holiday-routing model with family-level activation and monthly holdout evaluation for D-{issue_gap_days}.",
        "dataset_path": str(args.dataset_path),
        "rule_summary_path": str(args.rule_summary_path),
        "issue_gap_days": issue_gap_days,
        "base_rule_name": base_rule_name,
        "active_holiday_families": sorted(active_families),
        "makeup_active": makeup_active,
        "ordinary_similar_config": ordinary_config_map,
        "activation_summary": activation_frame.to_dict(orient="records"),
        "segment_metrics": segment_rows,
        "monthly_metrics": monthly_rows,
        "validation_daily_total_metrics": daily_total_metrics(validation_router_frame),
        "daily_total_metrics": daily_total_metrics(test_router_frame),
        "selected_model_validation_metrics": prediction_metrics_from_wide(
            validation[TARGET_COLUMNS].to_numpy(dtype=float),
            validation_router_pred,
        ),
        "selected_model_test_metrics": prediction_metrics_from_wide(
            test[TARGET_COLUMNS].to_numpy(dtype=float),
            test_router_pred,
        ),
        "prediction_outputs": {
            "daily": str(prediction_path),
            "long": str(long_path),
            "activation_summary": str(activation_path),
            "segment_metrics": str(segment_path),
            "monthly_metrics": str(monthly_path),
            "replaced_days": str(replacement_path),
            "report": str(report_path),
        },
    }
    save_summary_json(summary, summary_path)
    report_path.write_text(
        build_report_markdown(
            dataset_path=args.dataset_path,
            issue_gap_days=issue_gap_days,
            activation_rows=activation_frame.to_dict(orient="records"),
            segment_rows=segment_rows,
            monthly_rows=monthly_rows,
            replacement_rows=replacement_rows,
            prediction_path=prediction_path,
            summary_path=summary_path,
        ),
        encoding="utf-8",
    )

    print(summary)


if __name__ == "__main__":
    main()
