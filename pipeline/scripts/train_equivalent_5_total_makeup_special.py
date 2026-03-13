from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from china_holiday_official import OFFICIAL_MAKEUP_MAP
from forecast_core import daily_total_metrics, make_daily_prediction_frame, make_long_prediction_frame, prediction_metrics_from_wide, save_summary_json


TARGET_COLUMNS = [f"target_load_h{hour:02d}" for hour in range(24)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a makeup-workday special model on top of the Spring Festival special baseline.")
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--base-prediction-path", type=Path, required=True)
    parser.add_argument("--output-prefix", required=True)
    return parser.parse_args()


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        dataset_path,
        parse_dates=["issue_date", "target_date", "same_type_last1_date", "same_type_last2_date"],
    )


def load_base_prediction_frame(prediction_path: Path) -> pd.DataFrame:
    return pd.read_csv(prediction_path, parse_dates=["issue_date", "target_date"])


def anchor_before(frame: pd.DataFrame, issue_date: pd.Timestamp) -> float | None:
    prior = frame[frame["target_date"] <= issue_date].sort_values("target_date")
    normal = prior[
        prior["target_date_type_cn"].eq("workday")
        & prior["target_is_makeup_workday"].eq(0)
        & prior["target_is_holiday_cn"].eq(0)
    ].tail(3)
    if len(normal) == 0:
        return None
    return float(normal[TARGET_COLUMNS].sum(axis=1).mean())


def donor_ratio(full_frame: pd.DataFrame, donor_date: pd.Timestamp) -> float | None:
    donor = full_frame[full_frame["target_date"].eq(donor_date)]
    if donor.empty:
        return None
    donor_row = donor.iloc[0]
    donor_anchor = anchor_before(full_frame[full_frame["target_date"] < donor_date], donor_row["issue_date"])
    if donor_anchor is None:
        return None
    return float(donor_row[TARGET_COLUMNS].to_numpy(dtype=float).sum() / donor_anchor)


def generic_makeup_profile(full_frame: pd.DataFrame, row: pd.Series | pd.core.frame.Pandas) -> np.ndarray | None:
    holiday_name = OFFICIAL_MAKEUP_MAP.get(pd.Timestamp(row.target_date).normalize())
    donors = full_frame[
        full_frame["target_is_makeup_workday"].eq(1) & full_frame["target_date"].le(row.issue_date)
    ].copy()
    if donors.empty:
        return None

    donors["makeup_holiday_name_cn"] = donors["target_date"].map(
        lambda date_value: OFFICIAL_MAKEUP_MAP.get(pd.Timestamp(date_value).normalize())
    )
    donors = donors[donors["target_refined_date_type"].astype(str).str.startswith("makeup_workday")].copy()
    if donors.empty:
        return None

    donor_group = donors[donors["makeup_holiday_name_cn"].eq(holiday_name)].copy()
    if donor_group.empty:
        donor_group = donors[donors["target_refined_date_type"].eq(row.target_refined_date_type)].copy()
    if donor_group.empty:
        donor_group = donors[donors["target_dayofweek"].eq(row.target_dayofweek)].copy()
    if donor_group.empty:
        return None

    target_anchor = anchor_before(full_frame, row.issue_date)
    if target_anchor is None:
        return None

    ratios: list[float] = []
    shapes: list[np.ndarray] = []
    for donor in donor_group.itertuples(index=False):
        ratio = donor_ratio(full_frame, donor.target_date)
        if ratio is None:
            continue
        profile = np.array([getattr(donor, column) for column in TARGET_COLUMNS], dtype=float)
        total = float(profile.sum())
        if total <= 0:
            continue
        ratios.append(ratio)
        shapes.append(profile / total)

    if not ratios or not shapes:
        return None

    pred_daily_total = target_anchor * float(np.median(np.array(ratios, dtype=float)))
    pred_shape = np.mean(np.vstack(shapes), axis=0)
    pred_shape = pred_shape / np.clip(pred_shape.sum(), 1e-9, None)
    return pred_daily_total * pred_shape


def replace_generic_makeup_predictions(
    frame: pd.DataFrame,
    full_history: pd.DataFrame,
    base_prediction_frame: pd.DataFrame,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    prediction_matrix = np.column_stack(
        [base_prediction_frame[f"pred_h{hour:02d}"].to_numpy(dtype=float) for hour in range(24)]
    )
    rows: list[dict[str, object]] = []

    generic_makeup_mask = (
        frame["target_is_makeup_workday"].eq(1)
        & frame["target_refined_date_type"].astype(str).str.startswith("makeup_workday")
    ).to_numpy()

    for index, row in enumerate(frame.itertuples(index=False)):
        if not generic_makeup_mask[index]:
            continue
        profile = generic_makeup_profile(full_history, row)
        if profile is None:
            continue
        prediction_matrix[index, :] = profile
        rows.append(
            {
                "target_date": row.target_date,
                "target_refined_date_type": row.target_refined_date_type,
                "issue_date": row.issue_date,
                "pred_daily_total": float(profile.sum()),
                "actual_daily_total": float(np.array([getattr(row, column) for column in TARGET_COLUMNS], dtype=float).sum()),
            }
        )

    return prediction_matrix, rows


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


def build_report_markdown(
    dataset_path: Path,
    base_prediction_path: Path,
    backtest_rows: list[dict[str, object]],
    segment_rows: list[dict[str, object]],
    replacement_rows: list[dict[str, object]],
    prediction_path: Path,
    summary_path: Path,
) -> str:
    backtest_cols = ["model_variant", "days", "hourly_accuracy_percent", "daily_accuracy_percent", "daily_bias_percent"]
    segment_cols = ["model_variant", "segment", "days", "hourly_accuracy_percent", "daily_accuracy_percent", "daily_bias_percent"]
    replacement_cols = ["target_date", "target_refined_date_type", "actual_daily_total", "pred_daily_total"]

    def md_table(rows: list[dict[str, object]], columns: list[str]) -> list[str]:
        if not rows:
            return []
        header = "| " + " | ".join(columns) + " |"
        divider = "| " + " | ".join(["---"] * len(columns)) + " |"
        body = ["| " + " | ".join(str(row[column]) for column in columns) + " |" for row in rows]
        return [header, divider, *body]

    return "\n".join(
        [
            "# 调休专模评估",
            "",
            f"- 数据集：`{dataset_path}`",
            f"- 基础预测：`{base_prediction_path}`",
            "- 只覆盖 `makeup_workday_w*` 纯调休工作日；春节前后调休继续由春节专模处理。",
            "- 日总量用最近 3 个正常工作日均值作锚，再乘以同节日家族调休日倍率中位数。",
            "",
            "## 2025 调休日回测",
            *md_table(backtest_rows, backtest_cols),
            "",
            "## 2026 测试分段对比",
            *md_table(segment_rows, segment_cols),
            "",
            "## 被替换的调休日",
            *md_table(replacement_rows, replacement_cols),
            "",
            f"- 预测输出：`{prediction_path}`",
            f"- 摘要输出：`{summary_path}`",
        ]
    )


def main() -> None:
    args = parse_args()
    new_dir = args.base_dir / "new"
    new_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset_path)
    base_prediction_frame = load_base_prediction_frame(args.base_prediction_path)
    test = dataset[dataset["split"].eq("test")].copy().reset_index(drop=True)

    test_prediction_matrix, replacement_rows = replace_generic_makeup_predictions(
        frame=test,
        full_history=dataset,
        base_prediction_frame=base_prediction_frame,
    )
    test_frame = make_daily_prediction_frame(test, "spring_plus_makeup_special", test_prediction_matrix)
    test_frame["special_segment"] = np.where(
        test["target_is_makeup_workday"].eq(1) & test["target_refined_date_type"].astype(str).str.startswith("makeup_workday"),
        "generic_makeup",
        "other",
    )
    long_prediction_frame = make_long_prediction_frame(test_frame)

    backtest_rows: list[dict[str, object]] = []
    generic_makeup_2025 = dataset[
        dataset["target_date"].dt.year.eq(2025)
        & dataset["target_is_makeup_workday"].eq(1)
        & dataset["target_refined_date_type"].astype(str).str.startswith("makeup_workday")
    ].copy().reset_index(drop=True)
    if len(generic_makeup_2025) > 0:
        base_backtest = make_daily_prediction_frame(
            generic_makeup_2025,
            "baseline_same_type",
            generic_makeup_2025[[f"same_type_mean2_h{hour:02d}" for hour in range(24)]].to_numpy(dtype=float),
        )
        special_rows: list[np.ndarray] = []
        for row in generic_makeup_2025.itertuples(index=False):
            profile = generic_makeup_profile(dataset[dataset["target_date"] < row.target_date].copy(), row)
            if profile is None:
                profile = np.array([getattr(row, f"same_type_mean2_h{hour:02d}") for hour in range(24)], dtype=float)
            special_rows.append(profile)
        special_backtest = make_daily_prediction_frame(
            generic_makeup_2025,
            "makeup_special",
            np.vstack(special_rows),
        )
        backtest_rows.extend(
            [
                {"model_variant": "baseline_same_type", **segment_metrics(base_backtest, "generic_makeup_2025")},
                {"model_variant": "makeup_special", **segment_metrics(special_backtest, "generic_makeup_2025")},
            ]
        )

    base_test_frame = base_prediction_frame.copy()
    base_test_frame["special_segment"] = np.where(
        test["target_is_makeup_workday"].eq(1) & test["target_refined_date_type"].astype(str).str.startswith("makeup_workday"),
        "generic_makeup",
        "other",
    )
    segment_rows = [
        {"model_variant": "spring_special", **segment_metrics(base_test_frame, "all_test")},
        {"model_variant": "spring_special", **segment_metrics(base_test_frame[base_test_frame["special_segment"].eq("generic_makeup")].copy(), "generic_makeup")},
        {"model_variant": "spring_plus_makeup_special", **segment_metrics(test_frame, "all_test")},
        {"model_variant": "spring_plus_makeup_special", **segment_metrics(test_frame[test_frame["special_segment"].eq("generic_makeup")].copy(), "generic_makeup")},
    ]

    prediction_path = new_dir / f"{args.output_prefix}_test_predictions_daily.csv"
    long_path = new_dir / f"{args.output_prefix}_test_predictions_long.csv"
    summary_path = new_dir / f"{args.output_prefix}_forecast_summary.json"
    report_path = new_dir / f"{args.output_prefix}_report.md"
    backtest_path = new_dir / f"{args.output_prefix}_backtest_metrics.csv"
    segment_path = new_dir / f"{args.output_prefix}_segment_metrics.csv"
    replacement_path = new_dir / f"{args.output_prefix}_replaced_days.csv"

    test_frame.to_csv(prediction_path, index=False)
    long_prediction_frame.to_csv(long_path, index=False)
    pd.DataFrame(backtest_rows).to_csv(backtest_path, index=False)
    pd.DataFrame(segment_rows).to_csv(segment_path, index=False)
    pd.DataFrame(replacement_rows).to_csv(replacement_path, index=False)

    summary = {
        "task_definition": "Generic makeup-workday special model layered on top of the Spring Festival special baseline.",
        "dataset_path": str(args.dataset_path),
        "base_prediction_path": str(args.base_prediction_path),
        "backtest_rows": backtest_rows,
        "test_segment_metrics": segment_rows,
        "daily_total_metrics": daily_total_metrics(test_frame),
        "selected_model_test_metrics": prediction_metrics_from_wide(
            test[TARGET_COLUMNS].to_numpy(dtype=float),
            test_prediction_matrix,
        ),
        "prediction_outputs": {
            "daily": str(prediction_path),
            "long": str(long_path),
            "backtest": str(backtest_path),
            "segments": str(segment_path),
            "replaced_days": str(replacement_path),
            "report": str(report_path),
        },
    }
    save_summary_json(summary, summary_path)
    report_path.write_text(
        build_report_markdown(
            dataset_path=args.dataset_path,
            base_prediction_path=args.base_prediction_path,
            backtest_rows=backtest_rows,
            segment_rows=segment_rows,
            replacement_rows=replacement_rows,
            prediction_path=prediction_path,
            summary_path=summary_path,
        ),
        encoding="utf-8",
    )

    print(summary)


if __name__ == "__main__":
    main()
