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

import train_best_d6 as shared  # noqa: E402


TARGET_COLUMNS = shared.TARGET_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Tianlang holiday router using only 2025-2026 data.")
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--dataset-path", type=Path, default=Path("new/baseline_d6_dataset_2024fill.csv"))
    parser.add_argument("--rule-summary-path", type=Path, default=Path("results/baseline_d6_tianlang_2024fill_shared_summary.json"))
    parser.add_argument("--output-prefix", default="best_d6_tianlang_2024fill_shared")
    return parser.parse_args()


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    dataset = shared.load_dataset(dataset_path)
    available_years = sorted(set(dataset["target_date"].dt.year.unique()).intersection(dataset["issue_date"].dt.year.unique()))
    mask = dataset["target_date"].dt.year.isin(available_years) & dataset["issue_date"].dt.year.isin(available_years)
    return dataset.loc[mask].copy().reset_index(drop=True)


def holiday_relative_offset(row: pd.Series | pd.core.frame.Pandas) -> int | None:
    if str(row.holiday_segment) not in {"pre", "post"}:
        return None
    rel_key = str(row.holiday_rel_key)
    if "_d" not in rel_key:
        return None
    return int(rel_key.rsplit("_d", 1)[1])


def interpolate_relative_segment_profile(segment_rows: pd.DataFrame, target_offset: int) -> np.ndarray | None:
    if segment_rows.empty:
        return None
    keyed = segment_rows.copy()
    keyed["relative_offset"] = keyed["holiday_rel_key"].astype(str).str.extract(r"_d(\d+)$").astype(float)
    keyed = keyed.dropna(subset=["relative_offset"]).copy()
    if keyed.empty:
        return None
    keyed["relative_offset"] = keyed["relative_offset"].astype(int)
    grouped = keyed.groupby("relative_offset", as_index=False)[TARGET_COLUMNS].mean().sort_values("relative_offset")
    source_x = grouped["relative_offset"].to_numpy(dtype=float)
    source_matrix = grouped[TARGET_COLUMNS].to_numpy(dtype=float)
    if len(source_x) == 1:
        return source_matrix[0]
    return np.array([np.interp(float(target_offset), source_x, source_matrix[:, hour]) for hour in range(24)], dtype=float)


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
            profiles.append(shared.interpolate_holiday_profile(previous_holidays, int(row.holiday_pos), int(row.holiday_len)))
        return np.mean(profiles, axis=0)

    same_key_rows = family_history[family_history["holiday_rel_key"].eq(row.holiday_rel_key)].sort_values("target_date")
    if same_key_rows.empty:
        target_offset = holiday_relative_offset(row)
        if target_offset is None:
            return None
        same_segment_rows = family_history[family_history["holiday_segment"].eq(row.holiday_segment)].copy()
        return interpolate_relative_segment_profile(same_segment_rows, target_offset)

    if shoulder_history_mode == "last1":
        same_key_rows = same_key_rows.tail(1)
    else:
        same_key_rows = same_key_rows.tail(2)
    return same_key_rows[TARGET_COLUMNS].mean().to_numpy(dtype=float)


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
            history = train[train["target_date"].ne(family_row.target_date)].copy()
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
        special_accuracy = shared.daily_accuracy(actual_matrix, special_matrix)
        base_accuracy = shared.daily_accuracy(actual_matrix, base_matrix)
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
        special_profile = shared.generic_makeup_profile(history, row)
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
        special_accuracy = shared.daily_accuracy(actual_matrix, special_matrix)
        base_accuracy = shared.daily_accuracy(actual_matrix, base_matrix)
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
            replacement_profile = shared.generic_makeup_profile(history, row)
            route_name = "generic_makeup"
        elif row.holiday_family in active_families and row.holiday_segment in {"holiday", "pre", "post"}:
            replacement_profile = holiday_special_profile(row, history)
            route_name = f"holiday_family:{row.holiday_family}"
        elif row.holiday_segment == "other" and row.target_date_type_group in ordinary_config_map:
            config = ordinary_config_map[row.target_date_type_group]
            similar_profile, similar_details = shared.ordinary_similar_profile_with_details(row, history, top_k=int(config["top_k"]))
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


def main() -> None:
    args = parse_args()
    results_dir = args.base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = args.base_dir / args.dataset_path if not args.dataset_path.is_absolute() else args.dataset_path
    rule_summary_path = args.base_dir / args.rule_summary_path if not args.rule_summary_path.is_absolute() else args.rule_summary_path

    dataset = shared.add_holiday_meta(load_dataset(dataset_path))
    rule_params = shared.load_rule_params(rule_summary_path)
    rule_summary = json.loads(rule_summary_path.read_text(encoding="utf-8"))
    issue_gap_days = int(rule_summary.get("issue_gap_days", 5))
    base_rule_prediction = shared.rule_prediction(dataset, rule_params, issue_gap_days=issue_gap_days)

    family_activation_frame = train_family_activation_map(dataset, base_rule_prediction)
    ordinary_activation_frame = shared.train_ordinary_activation_map(dataset, base_rule_prediction)
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
    validation_base_frame = shared.make_daily_prediction_frame(validation, base_rule_name, validation_base)
    test_base_frame = shared.make_daily_prediction_frame(test, base_rule_name, test_base)
    validation_router_frame = shared.make_daily_prediction_frame(validation, "holiday_router", validation_router_pred)
    test_router_frame = shared.make_daily_prediction_frame(test, "holiday_router", test_router_pred)
    final_test_long = shared.make_long_prediction_frame(test_router_frame)

    segment_rows = [
        {"model_variant": base_rule_name, **shared.segment_metrics(validation_base_frame.copy(), "validation")},
        {"model_variant": base_rule_name, **shared.segment_metrics(test_base_frame.copy(), "test")},
        {"model_variant": "holiday_router", **shared.segment_metrics(validation_router_frame.copy(), "validation")},
        {"model_variant": "holiday_router", **shared.segment_metrics(test_router_frame.copy(), "test")},
    ]

    monthly_rows: list[dict[str, object]] = []
    monthly_rows.extend(shared.monthly_metric_rows(validation_base_frame.copy(), base_rule_name, "validation"))
    monthly_rows.extend(shared.monthly_metric_rows(test_base_frame.copy(), base_rule_name, "test"))
    monthly_rows.extend(shared.monthly_metric_rows(validation_router_frame.copy(), "holiday_router", "validation"))
    monthly_rows.extend(shared.monthly_metric_rows(test_router_frame.copy(), "holiday_router", "test"))

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
        "task_definition": f"Tianlang holiday router for the retained 2024fill shared dataset at D-{issue_gap_days}.",
        "site_name": "天朗",
        "year_scope": sorted(dataset["target_date"].dt.year.unique().tolist()),
        "dataset_path": str(dataset_path),
        "rule_summary_path": str(rule_summary_path),
        "issue_gap_days": issue_gap_days,
        "base_rule_name": base_rule_name,
        "active_holiday_families": sorted(active_families),
        "makeup_active": makeup_active,
        "ordinary_similar_config": ordinary_config_map,
        "activation_summary": activation_frame.to_dict(orient="records"),
        "segment_metrics": segment_rows,
        "monthly_metrics": monthly_rows,
        "validation_daily_total_metrics": shared.daily_total_metrics(validation_router_frame),
        "daily_total_metrics": shared.daily_total_metrics(test_router_frame),
        "selected_model_validation_metrics": shared.prediction_metrics_from_wide(
            validation[TARGET_COLUMNS].to_numpy(dtype=float),
            validation_router_pred,
        ),
        "selected_model_test_metrics": shared.prediction_metrics_from_wide(
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
    shared.save_summary_json(summary, summary_path)
    report_path.write_text(
        shared.build_report_markdown(
            dataset_path=dataset_path,
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
