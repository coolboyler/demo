from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_core import daily_total_metrics, make_daily_prediction_frame, make_long_prediction_frame, prediction_metrics_from_wide, save_summary_json
from train_d5 import get_issue_lag_config, same_type_scaled_profile, weighted_profile


TARGET_COLUMNS = [f"target_load_h{hour:02d}" for hour in range(24)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an equivalent five-company total baseline for a configurable issue gap.")
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset-path", type=Path, default=Path("new/baseline_d6_dataset.csv"))
    parser.add_argument("--issue-gap-days", type=int, default=6)
    parser.add_argument("--output-prefix", default=None, help="Default is baseline_d{issue_gap_days}")
    return parser.parse_args()


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        dataset_path,
        parse_dates=["issue_date", "target_date", "same_type_last1_date", "same_type_last2_date"],
    )


def split_dataset(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    train = dataset[dataset["split"].eq("train")].copy().reset_index(drop=True)
    validation = dataset[dataset["split"].eq("validation")].copy().reset_index(drop=True)
    test = dataset[dataset["split"].eq("test")].copy().reset_index(drop=True)
    split_info = {
        "train_target_start": str(train["target_date"].min().date()),
        "train_target_end": str(train["target_date"].max().date()),
        "validation_target_start": str(validation["target_date"].min().date()),
        "validation_target_end": str(validation["target_date"].max().date()),
        "test_target_start": str(test["target_date"].min().date()),
        "test_target_end": str(test["target_date"].max().date()),
    }
    return train, validation, test, split_info


def search_best_model(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    short_lags: list[int],
    weekly_lags: list[int],
) -> tuple[str, dict[str, float], np.ndarray, list[dict[str, object]], list[dict[str, object]]]:
    validation_actual = validation[TARGET_COLUMNS].to_numpy()
    test_actual = test[TARGET_COLUMNS].to_numpy()

    validation_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    candidate_map: dict[str, tuple[np.ndarray, dict[str, float]]] = {}
    model_counter: dict[str, int] = {}

    for short_lambda in np.linspace(0.05, 0.60, 23):
        short_weights = np.exp(-short_lambda * np.arange(len(short_lags)))
        validation_short = weighted_profile(validation, short_lags, short_weights)
        test_short = weighted_profile(test, short_lags, short_weights)

        same_validation = same_type_scaled_profile(validation, short_weights, lag_days=short_lags)
        same_test = same_type_scaled_profile(test, short_weights, lag_days=short_lags)
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
            weekly_weights = np.exp(-weekly_lambda * np.arange(len(weekly_lags)))
            validation_weekly = weighted_profile(validation, weekly_lags, weekly_weights)
            test_weekly = weighted_profile(test, weekly_lags, weekly_weights)
            for same_alpha in np.linspace(0.0, 0.35, 8):
                for weekly_alpha in np.linspace(0.0, 0.50, 11):
                    if same_alpha + weekly_alpha > 0.70:
                        continue
                    short_alpha = 1.0 - same_alpha - weekly_alpha
                    blend_validation = short_alpha * validation_short + same_alpha * same_validation + weekly_alpha * validation_weekly
                    blend_test = short_alpha * test_short + same_alpha * same_test + weekly_alpha * test_weekly
                    model_counter["blend_short_same_weekly"] = model_counter.get("blend_short_same_weekly", 0) + 1
                    model_id = f"blend_short_same_weekly_{model_counter['blend_short_same_weekly']:03d}"
                    validation_rows.append({"model": model_id, **prediction_metrics_from_wide(validation_actual, blend_validation)})
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
    top_test_rows = sorted([test_rows_by_model[row["model"]] for row in validation_rows], key=lambda row: row["wape_percent"])
    return best_model_name, best_params, best_test_prediction, validation_rows, top_test_rows


def main() -> None:
    args = parse_args()
    results_dir = args.base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_prefix = args.output_prefix or f"baseline_d{args.issue_gap_days}"
    dataset_path = args.base_dir / args.dataset_path if not args.dataset_path.is_absolute() else args.dataset_path
    dataset = load_dataset(dataset_path)
    train, validation, test, split_info = split_dataset(dataset)
    short_lags, weekly_lags, _ = get_issue_lag_config(args.issue_gap_days)

    best_model_name, best_params, best_test_prediction, validation_rows, top_test_rows = search_best_model(
        validation=validation,
        test=test,
        short_lags=short_lags,
        weekly_lags=weekly_lags,
    )

    prediction_frame = make_daily_prediction_frame(test, best_model_name, best_test_prediction)
    prediction_frame["split"] = "test"
    prediction_frame["is_fully_actual_5_company"] = 1
    long_prediction_frame = make_long_prediction_frame(prediction_frame)
    daily_metrics = daily_total_metrics(prediction_frame)

    prediction_path = results_dir / f"{output_prefix}_test_daily.csv"
    long_path = results_dir / f"{output_prefix}_test_long.csv"
    summary_path = results_dir / f"{output_prefix}_summary.json"

    prediction_frame.to_csv(prediction_path, index=False)
    long_prediction_frame.to_csv(long_path, index=False)

    selected_test_metrics = next(row for row in top_test_rows if row["model"] == best_model_name)
    summary = {
        "task_definition": f"Equivalent five-company total D-{args.issue_gap_days} baseline using unified five-company total history.",
        "dataset_path": str(dataset_path),
        "rows": int(len(dataset)),
        "issue_gap_days": int(args.issue_gap_days),
        "short_lags": short_lags,
        "weekly_lags": weekly_lags,
        "split_info": split_info,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "selected_model": best_model_name,
        "selected_model_params": best_params,
        "selected_model_test_metrics": selected_test_metrics,
        "daily_total_metrics": daily_metrics,
        "validation_top_rows": validation_rows,
        "test_top_rows": top_test_rows,
        "prediction_outputs": {"daily": str(prediction_path), "long": str(long_path)},
    }
    save_summary_json(summary, summary_path)
    print(summary)


if __name__ == "__main__":
    main()
