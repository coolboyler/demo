from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PIPELINE_DIR = BASE_DIR / "pipeline"
PIPELINE_SCRIPTS_DIR = PIPELINE_DIR / "scripts"
PIPELINE_TIANLANG_DIR = PIPELINE_DIR / "tianlang"
DISPLAY_START_DATE = pd.Timestamp("2026-01-01")
ISSUE_GAP_DAYS = 6
APP_TIMEZONE = ZoneInfo("Asia/Shanghai")
HOURLY_COLUMNS = [f"load_h{hour:02d}" for hour in range(24)]
RESULT_DAILY_PATTERN = re.compile(r"^forecast_d6_(\d{8})\.csv$")
RESULT_RANGE_PATTERN = re.compile(r"^forecast_d6_range_.*\.csv$")
DEFAULT_SITE_CODE = "huihua"


def _ensure_pipeline_imports() -> None:
    for path in [PIPELINE_SCRIPTS_DIR, PIPELINE_TIANLANG_DIR]:
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)


_ensure_pipeline_imports()

from forecast_core import TARGET_COLUMNS, make_long_prediction_frame, save_summary_json  # noqa: E402
from forecast_d6 import (  # noqa: E402
    add_lag_features,
    base_rule_prediction,
    build_future_calendar_row,
    build_prediction_output,
    get_max_actual_date,
    load_router_config,
    to_model_history,
)
from train_best_d6 import add_holiday_meta, apply_holiday_router as shared_apply_holiday_router, load_dataset as shared_load_dataset  # noqa: E402
from train_equivalent_5_total_spring_special import load_rule_params, rule_prediction  # noqa: E402
from train_best_d6_tianlang import apply_holiday_router as tianlang_apply_holiday_router, load_dataset as tianlang_load_dataset  # noqa: E402
from update_d6 import build_daily_history_row, build_hourly_history_rows  # noqa: E402


@dataclass(frozen=True)
class SiteConfig:
    code: str
    name: str
    new_dir: Path
    results_dir: Path
    template_path: Path
    dataset_filename: str
    best_summary_filename: str
    baseline_summary_filename: str
    replaced_days_filename: str
    forecast_daily_glob: str
    forecast_daily_pattern: re.Pattern[str]
    forecast_range_glob: str | None
    forecast_range_pattern: re.Pattern[str] | None
    forecast_stem_template: str
    forecast_month_subdirs: bool
    reference_test_daily_filename: str | None
    dataset_loader: Callable[[Path], pd.DataFrame]
    router_applier: Callable[..., tuple[np.ndarray, list[dict[str, object]]]]


SITE_CONFIGS: dict[str, SiteConfig] = {
    "huihua": SiteConfig(
        code="huihua",
        name="辉华",
        new_dir=PIPELINE_DIR / "new",
        results_dir=PIPELINE_DIR / "results",
        template_path=PIPELINE_DIR / "new" / "actual_input_template.csv",
        dataset_filename="baseline_d6_dataset.csv",
        best_summary_filename="best_d6_summary.json",
        baseline_summary_filename="baseline_d6_summary.json",
        replaced_days_filename="best_d6_replaced_days.csv",
        forecast_daily_glob="forecast_d6_*.csv",
        forecast_daily_pattern=RESULT_DAILY_PATTERN,
        forecast_range_glob="forecast_d6_range_*.csv",
        forecast_range_pattern=RESULT_RANGE_PATTERN,
        forecast_stem_template="forecast_d6_{target}",
        forecast_month_subdirs=False,
        reference_test_daily_filename=None,
        dataset_loader=shared_load_dataset,
        router_applier=shared_apply_holiday_router,
    ),
    "tianlang": SiteConfig(
        code="tianlang",
        name="天朗",
        new_dir=PIPELINE_TIANLANG_DIR / "new",
        results_dir=PIPELINE_TIANLANG_DIR / "results",
        template_path=PIPELINE_DIR / "new" / "actual_input_template.csv",
        dataset_filename="baseline_d6_dataset_2024fill.csv",
        best_summary_filename="best_d6_tianlang_2024fill_shared_summary.json",
        baseline_summary_filename="baseline_d6_tianlang_2024fill_shared_summary.json",
        replaced_days_filename="best_d6_tianlang_2024fill_shared_replaced_days.csv",
        forecast_daily_glob="**/forecast_d6_tianlang_*_2024fill_shared_*.csv",
        forecast_daily_pattern=re.compile(r"^forecast_d6_tianlang_(\d{8})_2024fill_shared_(\d{8})\.csv$"),
        forecast_range_glob=None,
        forecast_range_pattern=None,
        forecast_stem_template="forecast_d6_tianlang_{target}_2024fill_shared_{target}",
        forecast_month_subdirs=True,
        reference_test_daily_filename="best_d6_tianlang_2024fill_shared_test_daily.csv",
        dataset_loader=tianlang_load_dataset,
        router_applier=tianlang_apply_holiday_router,
    ),
}


def resolve_site_code(site_code: str | None) -> str:
    normalized = str(site_code or DEFAULT_SITE_CODE).strip().lower()
    if normalized not in SITE_CONFIGS:
        raise ValueError(f"unsupported site: {site_code}")
    return normalized


def get_site_config(site_code: str | None) -> SiteConfig:
    return SITE_CONFIGS[resolve_site_code(site_code)]


def list_site_options() -> list[dict[str, str]]:
    return [{"code": config.code, "name": config.name} for config in SITE_CONFIGS.values()]


def get_actual_template_path(site_code: str | None) -> Path:
    return get_site_config(site_code).template_path


def get_best_summary_path(site: SiteConfig) -> Path:
    return site.results_dir / site.best_summary_filename


def get_baseline_summary_path(site: SiteConfig) -> Path:
    return site.results_dir / site.baseline_summary_filename


def get_replaced_days_path(site: SiteConfig) -> Path:
    return site.results_dir / site.replaced_days_filename


def build_forecast_stem(site: SiteConfig, target_label: str) -> str:
    return site.forecast_stem_template.format(target=target_label)


def build_forecast_output_dir(site: SiteConfig, target_date: pd.Timestamp) -> Path:
    if not site.forecast_month_subdirs:
        return site.results_dir
    return site.results_dir / pd.Timestamp(target_date).strftime("%Y-%m")


def get_reference_test_daily_path(site: SiteConfig) -> Path | None:
    if not site.reference_test_daily_filename:
        return None
    return site.results_dir / site.reference_test_daily_filename


def normalize_hourly_values(values: list[float]) -> list[float]:
    if len(values) != 24:
        raise ValueError("values must contain exactly 24 hourly values")
    cleaned = [round(float(value), 4) for value in values]
    if any(value < 0 for value in cleaned):
        raise ValueError("hourly values must be non-negative")
    return cleaned


def _load_history_daily(site: SiteConfig) -> pd.DataFrame:
    history = pd.read_csv(site.new_dir / "history_daily.csv", parse_dates=["date"])
    if "is_actual_observation" not in history.columns:
        history["is_actual_observation"] = 1
    return history.sort_values("date").reset_index(drop=True)


def _load_history_hourly(site: SiteConfig) -> pd.DataFrame:
    return pd.read_csv(site.new_dir / "history_hourly.csv", parse_dates=["timestamp", "date"])


def _today_local() -> date:
    return datetime.now(APP_TIMEZONE).date()


def _upload_guard_path(site: SiteConfig) -> Path:
    return BASE_DIR / "data" / f"upload_guard_{site.code}.json"


def _read_upload_guard(site: SiteConfig) -> dict[str, Any] | None:
    path = _upload_guard_path(site)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_upload_guard(site: SiteConfig, actual_date: date, target_date: date) -> None:
    path = _upload_guard_path(site)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "lock_date": _today_local().isoformat(),
        "actual_date": pd.Timestamp(actual_date).strftime("%Y-%m-%d"),
        "target_date": pd.Timestamp(target_date).strftime("%Y-%m-%d"),
        "uploaded_at": datetime.now(APP_TIMEZONE).isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_upload_context(site: SiteConfig, max_actual_date: pd.Timestamp) -> dict[str, Any]:
    today = _today_local()
    business_target_date = pd.Timestamp(today) + pd.Timedelta(days=1)
    business_actual_date = business_target_date - pd.Timedelta(days=ISSUE_GAP_DAYS)
    expected_actual_ts = max_actual_date + pd.Timedelta(days=1)
    expected_actual_date = expected_actual_ts.strftime("%Y-%m-%d")
    generated_target_date = (expected_actual_ts + pd.Timedelta(days=ISSUE_GAP_DAYS)).strftime("%Y-%m-%d")
    guard = _read_upload_guard(site) or {}
    locked_by_guard = str(guard.get("lock_date", "")) == today.isoformat()
    locked_by_progress = expected_actual_ts > business_actual_date
    is_locked_today = locked_by_guard or locked_by_progress
    locked_actual_date = str(guard.get("actual_date", "")).strip() if locked_by_guard else None
    locked_target_date = str(guard.get("target_date", "")).strip() if locked_by_guard else None
    uploaded_at = str(guard.get("uploaded_at", "")).strip() if locked_by_guard else None
    if locked_by_guard and locked_actual_date and locked_target_date:
        status_text = f"今日上传任务已完成：目标日 {locked_target_date} 对应的 D-6 实际 {locked_actual_date} 已上传"
        helper_text = f"这张卡片表示今天的上传任务，与右侧当前选中目标日无关。明天再开放 {expected_actual_date} 的上传，届时生成 {generated_target_date} 的预测。"
        workflow_actual_date = locked_actual_date
        workflow_target_date = locked_target_date
        next_step_text = f"明天开放 {expected_actual_date} -> 生成 {generated_target_date}"
    elif locked_by_progress:
        finished_actual_date = business_actual_date.strftime("%Y-%m-%d")
        finished_target_date = business_target_date.strftime("%Y-%m-%d")
        next_open_actual_date = expected_actual_date
        next_open_target_date = generated_target_date
        status_text = f"今日上传任务已完成：目标日 {finished_target_date} 对应的 D-6 实际 {finished_actual_date} 已完成"
        helper_text = f"这张卡片表示今天的上传任务，与右侧当前选中目标日无关。今天不再显示上传表单，下一次开放为 {next_open_actual_date}，届时生成 {next_open_target_date} 的预测。"
        workflow_actual_date = finished_actual_date
        workflow_target_date = finished_target_date
        next_step_text = f"下次开放 {next_open_actual_date} -> 生成 {next_open_target_date}"
    else:
        status_text = f"今日待完成上传任务：请上传 {expected_actual_date} 的 D-6 实际"
        helper_text = f"这张卡片表示今天的上传任务，与右侧当前选中目标日无关。提交后会写入历史，并立即生成目标日 {generated_target_date} 的 D 日预测。"
        workflow_actual_date = expected_actual_date
        workflow_target_date = generated_target_date
        next_step_text = f"上传完成后生成 {generated_target_date}"
    return {
        "server_today": today.isoformat(),
        "expected_actual_date": expected_actual_date,
        "generated_target_date": generated_target_date,
        "is_locked_today": is_locked_today,
        "locked_by_guard": locked_by_guard,
        "locked_by_progress": locked_by_progress,
        "business_target_date": business_target_date.strftime("%Y-%m-%d"),
        "business_actual_date": business_actual_date.strftime("%Y-%m-%d"),
        "locked_actual_date": locked_actual_date,
        "locked_target_date": locked_target_date,
        "uploaded_at": uploaded_at,
        "workflow_actual_date": workflow_actual_date,
        "workflow_target_date": workflow_target_date,
        "next_step_text": next_step_text,
        "status_text": status_text,
        "helper_text": helper_text,
        "history_file": str((site.new_dir / "history_daily.csv").relative_to(BASE_DIR)),
        "hourly_file": str((site.new_dir / "history_hourly.csv").relative_to(BASE_DIR)),
    }


def _resolved_router_summary_path(site: SiteConfig) -> Path:
    source_path = get_best_summary_path(site)
    resolved_path = site.results_dir / "_best_d6_summary_resolved.json"
    summary = json.loads(source_path.read_text(encoding="utf-8"))
    rule_summary_path = Path(summary["rule_summary_path"])
    if not rule_summary_path.is_absolute():
        rule_summary_path = (site.results_dir.parent / rule_summary_path).resolve()
    summary["rule_summary_path"] = str(rule_summary_path)
    resolved_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return resolved_path


def _vector_from_row(row: pd.Series, prefix: str) -> list[float]:
    return [float(row[f"{prefix}{hour:02d}"]) for hour in range(24)]


def _series(values: list[float] | None) -> list[dict[str, Any]]:
    if not values:
        return []
    return [{"hour": f"{hour:02d}:00", "value": round(float(value), 4)} for hour, value in enumerate(values)]


def _safe_round(value: float | None) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), 4)


def _metrics(predicted: list[float] | None, actual: list[float] | None) -> dict[str, float] | None:
    if predicted is None or actual is None:
        return None
    predicted_array = np.asarray(predicted, dtype=float)
    actual_array = np.asarray(actual, dtype=float)
    abs_error = np.abs(predicted_array - actual_array)
    squared_error = (predicted_array - actual_array) ** 2
    nonzero_mask = actual_array > 0
    percent_error = np.abs(predicted_array[nonzero_mask] - actual_array[nonzero_mask]) / actual_array[nonzero_mask] * 100
    hourly_wape_percent = abs_error.sum() / max(actual_array.sum(), 1e-9) * 100
    hit_rate_5 = np.mean((abs_error[nonzero_mask] / actual_array[nonzero_mask]) <= 0.05) * 100 if nonzero_mask.any() else 0.0
    mape = float(percent_error.mean()) if percent_error.size else 0.0
    hourly_accuracy = max(0.0, 100.0 - hourly_wape_percent)
    actual_total = float(actual_array.sum())
    predicted_total = float(predicted_array.sum())
    day_total_error_percent = abs(predicted_total - actual_total) / max(actual_total, 1e-9) * 100
    day_total_accuracy = max(0.0, 100.0 - day_total_error_percent)
    return {
        "accuracy": round(hourly_accuracy, 2),
        "hourly_accuracy": round(hourly_accuracy, 2),
        "day_total_accuracy": round(day_total_accuracy, 2),
        "hourly_wape": round(float(hourly_wape_percent), 2),
        "mape": round(mape, 2),
        "mae": round(float(abs_error.mean()), 4),
        "rmse": round(float(np.sqrt(squared_error.mean())), 4),
        "hit_rate_5": round(float(hit_rate_5), 2),
        "max_abs_error": round(float(abs_error.max()), 4),
        "bias": round(float((predicted_array.sum() - actual_array.sum()) / max(actual_array.sum(), 1e-9) * 100), 2),
    }


def _build_date_navigation(selected_date: pd.Timestamp, available_dates: list[pd.Timestamp]) -> dict[str, Any]:
    if not available_dates:
        return {
            "available_dates": [],
            "prev_date": None,
            "next_date": None,
            "min_date": None,
            "max_date": None,
        }

    selected = pd.Timestamp(selected_date).normalize()
    normalized_dates = [pd.Timestamp(item).normalize() for item in available_dates]
    try:
        index = normalized_dates.index(selected)
    except ValueError:
        index = len(normalized_dates) - 1
    return {
        "available_dates": [item.strftime("%Y-%m-%d") for item in normalized_dates],
        "prev_date": normalized_dates[index - 1].strftime("%Y-%m-%d") if index > 0 else None,
        "next_date": normalized_dates[index + 1].strftime("%Y-%m-%d") if index < len(normalized_dates) - 1 else None,
        "min_date": normalized_dates[0].strftime("%Y-%m-%d"),
        "max_date": normalized_dates[-1].strftime("%Y-%m-%d"),
    }


def _fixed_prediction_from_dataset(frame: pd.DataFrame) -> np.ndarray:
    output = np.zeros((len(frame), 24), dtype=float)
    for hour in range(24):
        output[:, hour] = (
            0.5 * frame[f"lag7_h{hour:02d}"].to_numpy(dtype=float)
            + 0.3 * frame[f"lag14_h{hour:02d}"].to_numpy(dtype=float)
            + 0.2 * frame[f"lag21_h{hour:02d}"].to_numpy(dtype=float)
        )
    return output


def _fixed_prediction_from_history(history: pd.DataFrame, target_date: pd.Timestamp) -> list[float]:
    date_lookup = history.set_index(history["date"].dt.normalize())
    lag_values = []
    for lag_day in [7, 14, 21]:
        lag_date = pd.Timestamp(target_date).normalize() - pd.Timedelta(days=lag_day)
        if lag_date not in date_lookup.index:
            raise ValueError(f"missing lag profile for {lag_date.date()}")
        lag_values.append(date_lookup.loc[lag_date, HOURLY_COLUMNS].to_numpy(dtype=float))
    profile = 0.5 * lag_values[0] + 0.3 * lag_values[1] + 0.2 * lag_values[2]
    return [round(float(value), 4) for value in profile.tolist()]


def _load_route_map(site: SiteConfig) -> dict[str, str]:
    path = get_replaced_days_path(site)
    if not path.exists():
        return {}
    frame = pd.read_csv(path, parse_dates=["target_date"])
    return {row.target_date.strftime("%Y-%m-%d"): str(row.route_name) for row in frame.itertuples(index=False)}


def _load_reference_test_predictions(site: SiteConfig) -> dict[str, dict[str, Any]]:
    path = get_reference_test_daily_path(site)
    if path is None or not path.exists():
        return {}

    frame = pd.read_csv(path, parse_dates=["target_date", "issue_date"])
    rows: dict[str, dict[str, Any]] = {}
    for row in frame.itertuples(index=False):
        target_iso = pd.Timestamp(row.target_date).strftime("%Y-%m-%d")
        actual_values = [round(float(getattr(row, f"actual_h{hour:02d}")), 4) for hour in range(24)]
        model_values = [round(float(getattr(row, f"pred_h{hour:02d}")), 4) for hour in range(24)]
        rows[target_iso] = {
            "target_date": target_iso,
            "issue_date": pd.Timestamp(row.issue_date).strftime("%Y-%m-%d"),
            "target_refined_date_type": str(getattr(row, "target_refined_date_type", "")),
            "target_date_type_group": str(getattr(row, "target_date_type_group", "")),
            "actual_values": actual_values,
            "model_values": model_values,
            "actual_total": round(float(getattr(row, "actual_daily_total", np.sum(actual_values))), 4),
            "model_total": round(float(getattr(row, "pred_daily_total", np.sum(model_values))), 4),
        }
    return rows


def _build_reference_predictions(site: SiteConfig) -> pd.DataFrame:
    dataset = add_holiday_meta(site.dataset_loader(site.new_dir / site.dataset_filename))
    router_summary = json.loads(get_best_summary_path(site).read_text(encoding="utf-8"))
    rule_params = load_rule_params(get_baseline_summary_path(site))
    base_prediction = rule_prediction(dataset, rule_params, issue_gap_days=ISSUE_GAP_DAYS)
    active_families = set(router_summary.get("active_holiday_families", []))
    makeup_active = bool(router_summary.get("makeup_active", False))
    ordinary_config = {
        str(key): {"top_k": int(value["top_k"]), "alpha": float(value["alpha"])}
        for key, value in router_summary.get("ordinary_similar_config", {}).items()
    }

    validation = dataset[dataset["split"].eq("validation")].copy().reset_index(drop=True)
    test = dataset[dataset["split"].eq("test")].copy().reset_index(drop=True)
    validation_prediction, _ = site.router_applier(
        frame=validation,
        full_history=dataset,
        base_prediction=base_prediction[dataset["split"].eq("validation").to_numpy()],
        active_families=active_families,
        makeup_active=makeup_active,
        ordinary_config_map=ordinary_config,
    )
    test_prediction, _ = site.router_applier(
        frame=test,
        full_history=dataset,
        base_prediction=base_prediction[dataset["split"].eq("test").to_numpy()],
        active_families=active_families,
        makeup_active=makeup_active,
        ordinary_config_map=ordinary_config,
    )

    combined = pd.concat([validation, test], ignore_index=True)
    model_prediction = np.vstack([validation_prediction, test_prediction])
    fixed_prediction = _fixed_prediction_from_dataset(combined)
    route_map = _load_route_map(site)
    direct_test_rows = _load_reference_test_predictions(site)

    rows: list[dict[str, Any]] = []
    for index, row in enumerate(combined.itertuples(index=False)):
        actual_values = [float(getattr(row, column)) for column in TARGET_COLUMNS]
        fixed_values = [float(value) for value in fixed_prediction[index].tolist()]
        target_date = pd.Timestamp(row.target_date).normalize()
        target_iso = target_date.strftime("%Y-%m-%d")
        override = direct_test_rows.get(target_iso)
        if override is not None:
            actual_values = [float(value) for value in override["actual_values"]]
            model_values = [float(value) for value in override["model_values"]]
            issue_date = override["issue_date"]
            target_refined_date_type = override["target_refined_date_type"]
            target_date_type_group = override["target_date_type_group"]
            actual_total = override["actual_total"]
            model_total = override["model_total"]
        else:
            model_values = [float(value) for value in model_prediction[index].tolist()]
            issue_date = pd.Timestamp(row.issue_date).strftime("%Y-%m-%d")
            target_refined_date_type = str(row.target_refined_date_type)
            target_date_type_group = str(row.target_date_type_group)
            actual_total = round(float(np.sum(actual_values)), 4)
            model_total = round(float(np.sum(model_values)), 4)
        rows.append(
            {
                "target_date": target_iso,
                "issue_date": issue_date,
                "split": str(row.split),
                "data_status": "complete",
                "target_refined_date_type": target_refined_date_type,
                "target_date_type_group": target_date_type_group,
                "route_name": route_map.get(target_iso, "base_rule"),
                "forecast_mode": f"strict_d{ISSUE_GAP_DAYS}",
                "actual_values": [round(value, 4) for value in actual_values],
                "model_values": [round(value, 4) for value in model_values],
                "fixed_values": [round(value, 4) for value in fixed_values],
                "actual_total": actual_total,
                "model_total": model_total,
                "fixed_total": round(float(np.sum(fixed_values)), 4),
                "model_metrics": _metrics(model_values, actual_values),
                "fixed_metrics": _metrics(fixed_values, actual_values),
                "history_source": "validation_test_reference_direct" if override is not None else "validation_test_reference",
            }
        )
    return pd.DataFrame(rows).sort_values("target_date").reset_index(drop=True)


def _load_saved_forecasts(site: SiteConfig) -> dict[str, dict[str, Any]]:
    forecasts: dict[str, dict[str, Any]] = {}
    if not site.results_dir.exists():
        return forecasts

    for path in sorted(site.results_dir.glob(site.forecast_daily_glob)):
        if path.name.endswith("_long.csv") or path.name.endswith("_matches.csv"):
            continue
        match = site.forecast_daily_pattern.match(path.name)
        if not match:
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        row = frame.iloc[0]
        target_iso = str(row["target_date"])
        forecasts[target_iso] = {
            "target_date": target_iso,
            "issue_date": str(row["issue_date"]),
            "target_refined_date_type": str(row.get("target_refined_date_type", "")),
            "target_date_type_group": str(row.get("target_date_type_group", "")),
            "route_name": str(row.get("route_name", "base_rule")),
            "forecast_mode": str(row.get("forecast_mode", f"strict_d{ISSUE_GAP_DAYS}")),
            "model_values": [round(float(row[f"pred_h{hour:02d}"]), 4) for hour in range(24)],
            "model_total": round(float(row["pred_daily_total"]), 4),
            "history_source": str(path.relative_to(site.results_dir)),
        }

    if site.forecast_range_glob and site.forecast_range_pattern:
        for path in sorted(site.results_dir.glob(site.forecast_range_glob)):
            if not site.forecast_range_pattern.match(path.name):
                continue
            frame = pd.read_csv(path)
            for row in frame.itertuples(index=False):
                target_iso = str(row.target_date)
                if target_iso in forecasts:
                    continue
                forecasts[target_iso] = {
                    "target_date": target_iso,
                    "issue_date": str(row.issue_date),
                    "target_refined_date_type": str(getattr(row, "target_refined_date_type", "")),
                    "target_date_type_group": str(getattr(row, "target_date_type_group", "")),
                    "route_name": str(getattr(row, "route_name", "base_rule")),
                    "forecast_mode": str(getattr(row, "forecast_mode", f"strict_d{ISSUE_GAP_DAYS}")),
                    "model_values": [round(float(getattr(row, f"pred_h{hour:02d}")), 4) for hour in range(24)],
                    "model_total": round(float(getattr(row, "pred_daily_total")), 4),
                    "history_source": str(path.relative_to(site.results_dir)),
                }
    return forecasts


def _build_operational_history(history: pd.DataFrame, saved_forecasts: dict[str, dict[str, Any]], start_date: pd.Timestamp, min_target: pd.Timestamp) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    actual_history = history[history["date"].ge(start_date)].copy()
    for row in actual_history.itertuples(index=False):
        target_date = pd.Timestamp(row.date).normalize()
        if target_date <= min_target:
            continue
        target_iso = target_date.strftime("%Y-%m-%d")
        forecast_row = saved_forecasts.get(target_iso)
        if forecast_row is None:
            continue
        actual_values = [float(getattr(row, column)) for column in HOURLY_COLUMNS]
        fixed_values = _fixed_prediction_from_history(history, target_date)
        model_values = forecast_row["model_values"]
        rows.append(
            {
                "target_date": target_iso,
                "issue_date": forecast_row["issue_date"],
                "split": "operational",
                "data_status": "complete",
                "target_refined_date_type": forecast_row.get("target_refined_date_type", ""),
                "target_date_type_group": forecast_row.get("target_date_type_group", ""),
                "route_name": forecast_row.get("route_name", "base_rule"),
                "forecast_mode": forecast_row.get("forecast_mode", f"strict_d{ISSUE_GAP_DAYS}"),
                "actual_values": [round(value, 4) for value in actual_values],
                "model_values": model_values,
                "fixed_values": fixed_values,
                "actual_total": round(float(np.sum(actual_values)), 4),
                "model_total": round(float(np.sum(model_values)), 4),
                "fixed_total": round(float(np.sum(fixed_values)), 4),
                "model_metrics": _metrics(model_values, actual_values),
                "fixed_metrics": _metrics(fixed_values, actual_values),
                "history_source": forecast_row.get("history_source", "operational_results"),
            }
        )
    return pd.DataFrame(rows).sort_values("target_date").reset_index(drop=True) if rows else pd.DataFrame()


def _predict_target_date(site: SiteConfig, history: pd.DataFrame, target_date: pd.Timestamp) -> dict[str, Any]:
    model_history = to_model_history(history, issue_gap_days=ISSUE_GAP_DAYS)
    raw_target_row = build_future_calendar_row(target_date, issue_gap_days=ISSUE_GAP_DAYS)
    context_source_columns = [
        "target_date",
        "target_holiday_name_cn",
        "target_refined_date_type",
        "target_date_type_group",
    ]
    context_frame = pd.concat(
        [
            model_history[context_source_columns],
            raw_target_row[context_source_columns],
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["target_date"], keep="last")
    context_frame = add_holiday_meta(context_frame)
    context_row = context_frame[context_frame["target_date"].eq(target_date)].copy()
    target_row = raw_target_row.drop(
        columns=[
            column
            for column in [
                "holiday_family",
                "holiday_segment",
                "holiday_rel_key",
                "holiday_pos",
                "holiday_len",
                "holiday_context_tag",
                "holiday_context_family",
                "holiday_context_side",
                "holiday_context_bucket",
                "days_since_last_holiday",
                "days_to_next_holiday",
                "last_holiday_family",
                "next_holiday_family",
            ]
            if column in raw_target_row.columns
        ],
        errors="ignore",
    ).merge(context_row, on=context_source_columns, how="left")
    target_row = add_lag_features(target_row, model_history, issue_gap_days=ISSUE_GAP_DAYS)

    rule_params, active_families, makeup_active, ordinary_config = load_router_config(_resolved_router_summary_path(site))
    base_prediction = base_rule_prediction(target_row, rule_params, issue_gap_days=ISSUE_GAP_DAYS)
    routed_prediction, replacement_rows = site.router_applier(
        frame=target_row,
        full_history=model_history,
        base_prediction=base_prediction,
        active_families=active_families,
        makeup_active=makeup_active,
        ordinary_config_map=ordinary_config,
    )
    max_actual_date = get_max_actual_date(history)
    forecast_mode = f"strict_d{ISSUE_GAP_DAYS}" if target_date <= max_actual_date + pd.Timedelta(days=ISSUE_GAP_DAYS) else "planning_recursive"
    output_frame = build_prediction_output(
        target_row=target_row,
        routed_prediction=routed_prediction,
        replacement_rows=replacement_rows,
        best_model_name="holiday_router_future",
        forecast_mode=forecast_mode,
    )
    return {
        "output_frame": output_frame,
        "long_frame": make_long_prediction_frame(output_frame),
        "replacement_rows": replacement_rows,
        "max_actual_date": max_actual_date,
    }


def _write_operational_forecast(site: SiteConfig, history: pd.DataFrame, target_date: pd.Timestamp) -> dict[str, Any]:
    prediction_bundle = _predict_target_date(
        site=site,
        history=history,
        target_date=pd.Timestamp(target_date).normalize(),
    )
    output_frame = prediction_bundle["output_frame"]
    long_frame = prediction_bundle["long_frame"] if "long_frame" in prediction_bundle else make_long_prediction_frame(output_frame)
    normalized_target_date = pd.Timestamp(target_date).normalize()
    target_label = normalized_target_date.strftime("%Y%m%d")
    forecast_stem = build_forecast_stem(site, target_label)
    output_dir = build_forecast_output_dir(site, normalized_target_date)
    output_dir.mkdir(parents=True, exist_ok=True)
    daily_path = output_dir / f"{forecast_stem}.csv"
    long_path = output_dir / f"{forecast_stem}_long.csv"
    summary_path = output_dir / f"{forecast_stem}_summary.json"
    match_path = output_dir / f"{forecast_stem}_matches.csv"
    output_frame.to_csv(daily_path, index=False)
    long_frame.to_csv(long_path, index=False)
    pd.DataFrame(prediction_bundle.get("replacement_rows", [])).to_csv(match_path, index=False)
    summary = {
        "task_definition": f"Operational D-{ISSUE_GAP_DAYS} future forecast using the holiday-router model.",
        "site_name": site.name,
        "history_path": str(site.new_dir / "history_daily.csv"),
        "router_summary_path": str(get_best_summary_path(site)),
        "issue_gap_days": ISSUE_GAP_DAYS,
        "target_date": str(pd.Timestamp(target_date).date()),
        "issue_date": str((pd.Timestamp(target_date) - pd.Timedelta(days=ISSUE_GAP_DAYS)).date()),
        "max_actual_date": str(prediction_bundle["max_actual_date"].date()),
        "route_name": str(output_frame.loc[0, "route_name"]),
        "forecast_mode": str(output_frame.loc[0, "forecast_mode"]),
        "used_predicted_history": int(output_frame.loc[0, "used_predicted_history"]),
        "pred_daily_total": float(output_frame.loc[0, "pred_daily_total"]),
        "prediction_outputs": {
            "daily": str(daily_path),
            "long": str(long_path),
            "matches": str(match_path),
        },
    }
    save_summary_json(summary, summary_path)
    return {
        "target_date": normalized_target_date.strftime("%Y-%m-%d"),
        "daily_path": str(daily_path.relative_to(site.results_dir)),
        "long_path": str(long_path.relative_to(site.results_dir)),
        "summary_path": str(summary_path.relative_to(site.results_dir)),
        "match_path": str(match_path.relative_to(site.results_dir)),
    }


def _ensure_current_forecast(site: SiteConfig, history: pd.DataFrame, saved_forecasts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    max_actual_date = get_max_actual_date(history)
    first_future_target = max_actual_date + pd.Timedelta(days=1)
    current_target = max_actual_date + pd.Timedelta(days=ISSUE_GAP_DAYS)
    wrote_missing_forecasts = False
    for target_date in pd.date_range(first_future_target, current_target, freq="D"):
        target_iso = pd.Timestamp(target_date).strftime("%Y-%m-%d")
        if target_iso in saved_forecasts:
            continue
        _write_operational_forecast(site, history, pd.Timestamp(target_date))
        wrote_missing_forecasts = True

    if wrote_missing_forecasts:
        saved_forecasts.update(_load_saved_forecasts(site))
    current_target_iso = current_target.strftime("%Y-%m-%d")
    return saved_forecasts[current_target_iso]


def _build_future_rows(
    history: pd.DataFrame,
    saved_forecasts: dict[str, dict[str, Any]],
    max_actual_date: pd.Timestamp,
    current_target_date: pd.Timestamp,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target_iso, forecast_row in sorted(saved_forecasts.items()):
        target_date = pd.Timestamp(target_iso).normalize()
        if target_date <= max_actual_date or target_date > current_target_date:
            continue
        try:
            fixed_values = _fixed_prediction_from_history(history, target_date)
        except ValueError:
            continue
        rows.append(
            {
                "target_date": target_iso,
                "issue_date": forecast_row["issue_date"],
                "split": "future",
                "data_status": "forecast_only",
                "target_refined_date_type": forecast_row.get("target_refined_date_type", ""),
                "target_date_type_group": forecast_row.get("target_date_type_group", ""),
                "route_name": forecast_row.get("route_name", "base_rule"),
                "forecast_mode": forecast_row.get("forecast_mode", f"strict_d{ISSUE_GAP_DAYS}"),
                "actual_values": None,
                "model_values": forecast_row["model_values"],
                "fixed_values": fixed_values,
                "actual_total": None,
                "model_total": forecast_row["model_total"],
                "fixed_total": round(float(np.sum(fixed_values)), 4),
                "model_metrics": None,
                "fixed_metrics": None,
                "history_source": forecast_row.get("history_source", "future_forecast"),
            }
        )
    return rows


def _month_summary(rows: list[dict[str, Any]], month: str) -> dict[str, Any]:
    month_rows = [row for row in rows if row["target_date"].startswith(month)]
    matched_rows = [row for row in month_rows if row.get("model_metrics")]
    return {
        "month": month,
        "days_count": len(month_rows),
        "matched_days_count": len(matched_rows),
        "latest_target_date": month_rows[-1]["target_date"] if month_rows else None,
        "model_metrics": _aggregate_metrics(matched_rows, "model_values"),
        "fixed_metrics": _aggregate_metrics(matched_rows, "fixed_values"),
    }


def _average_metrics(metrics_rows: list[dict[str, float]]) -> dict[str, float] | None:
    if not metrics_rows:
        return None
    keys = ["accuracy", "hourly_accuracy", "day_total_accuracy", "hourly_wape", "mape", "mae", "rmse", "hit_rate_5", "max_abs_error", "bias"]
    return {key: round(sum(float(item[key]) for item in metrics_rows) / len(metrics_rows), 2) for key in keys}


def _aggregate_metrics(rows: list[dict[str, Any]], prediction_key: str) -> dict[str, float] | None:
    matched_rows = [row for row in rows if row.get("actual_values") and row.get(prediction_key)]
    if not matched_rows:
        return None

    actual_matrix = np.asarray([row["actual_values"] for row in matched_rows], dtype=float)
    predicted_matrix = np.asarray([row[prediction_key] for row in matched_rows], dtype=float)
    abs_error = np.abs(predicted_matrix - actual_matrix)
    squared_error = (predicted_matrix - actual_matrix) ** 2
    nonzero_mask = actual_matrix > 0
    percent_error = np.abs(predicted_matrix[nonzero_mask] - actual_matrix[nonzero_mask]) / actual_matrix[nonzero_mask] * 100
    hourly_wape_percent = abs_error.sum() / max(actual_matrix.sum(), 1e-9) * 100
    actual_daily = actual_matrix.sum(axis=1)
    predicted_daily = predicted_matrix.sum(axis=1)
    daily_error_percent = np.abs(predicted_daily - actual_daily).sum() / max(actual_daily.sum(), 1e-9) * 100
    bias_percent = (predicted_daily.sum() - actual_daily.sum()) / max(actual_daily.sum(), 1e-9) * 100
    hit_rate_5 = np.mean((abs_error[nonzero_mask] / actual_matrix[nonzero_mask]) <= 0.05) * 100 if nonzero_mask.any() else 0.0
    hourly_mape = float(percent_error.mean()) if percent_error.size else 0.0
    hourly_accuracy = max(0.0, 100.0 - hourly_wape_percent)
    day_total_accuracy = max(0.0, 100.0 - daily_error_percent)

    return {
        "accuracy": round(float(hourly_accuracy), 2),
        "hourly_accuracy": round(float(hourly_accuracy), 2),
        "day_total_accuracy": round(float(day_total_accuracy), 2),
        "hourly_wape": round(float(hourly_wape_percent), 2),
        "mape": round(float(hourly_mape), 2),
        "mae": round(float(abs_error.mean()), 4),
        "rmse": round(float(np.sqrt(squared_error.mean())), 4),
        "hit_rate_5": round(float(hit_rate_5), 2),
        "max_abs_error": round(float(abs_error.max()), 4),
        "bias": round(float(bias_percent), 2),
    }


def _decorate_row(row: dict[str, Any]) -> dict[str, Any]:
    selected_values = row["actual_values"] if row.get("actual_values") else row["model_values"]
    peak = None
    valley = None
    if selected_values:
        peak_value = max(selected_values)
        valley_value = min(selected_values)
        peak = {"hour": f"{selected_values.index(peak_value):02d}:00", "value": round(float(peak_value), 4)}
        valley = {"hour": f"{selected_values.index(valley_value):02d}:00", "value": round(float(valley_value), 4)}
    return {
        **row,
        "actual_series": _series(row.get("actual_values")),
        "model_series": _series(row.get("model_values")),
        "fixed_series": _series(row.get("fixed_values")),
        "has_actual": bool(row.get("actual_values")),
        "peak": peak,
        "valley": valley,
        "avg_load": round(float(np.mean(selected_values)), 4) if selected_values else None,
    }


def build_dashboard_payload(target_date: date | None = None, site_code: str = DEFAULT_SITE_CODE) -> dict[str, Any]:
    site = get_site_config(site_code)
    history = _load_history_daily(site)
    saved_forecasts = _load_saved_forecasts(site)
    current_forecast = _ensure_current_forecast(site, history, saved_forecasts)
    reference_rows = _build_reference_predictions(site)
    min_reference_date = pd.Timestamp(reference_rows["target_date"].max()) if not reference_rows.empty else DISPLAY_START_DATE
    operational_rows = _build_operational_history(history, saved_forecasts, DISPLAY_START_DATE, min_reference_date)
    historical_rows = reference_rows.to_dict(orient="records")
    if not operational_rows.empty:
        historical_rows.extend(operational_rows.to_dict(orient="records"))
    historical_rows = [row for row in historical_rows if pd.Timestamp(row["target_date"]) >= DISPLAY_START_DATE]
    historical_rows.sort(key=lambda item: item["target_date"])

    max_actual_date = get_max_actual_date(history)
    current_target_ts = max_actual_date + pd.Timedelta(days=ISSUE_GAP_DAYS)
    current_target_date = current_target_ts.strftime("%Y-%m-%d")
    future_rows = _build_future_rows(history, saved_forecasts, max_actual_date, current_target_ts)
    future_row = next((row for row in future_rows if row["target_date"] == current_target_date), None)
    if future_row is None:
        fixed_future_values = _fixed_prediction_from_history(history, current_target_ts)
        future_row = {
            "target_date": current_target_date,
            "issue_date": current_forecast["issue_date"],
            "split": "future",
            "data_status": "forecast_only",
            "target_refined_date_type": current_forecast.get("target_refined_date_type", ""),
            "target_date_type_group": current_forecast.get("target_date_type_group", ""),
            "route_name": current_forecast.get("route_name", "base_rule"),
            "forecast_mode": current_forecast.get("forecast_mode", f"strict_d{ISSUE_GAP_DAYS}"),
            "actual_values": None,
            "model_values": current_forecast["model_values"],
            "fixed_values": fixed_future_values,
            "actual_total": None,
            "model_total": current_forecast["model_total"],
            "fixed_total": round(float(np.sum(fixed_future_values)), 4),
            "model_metrics": None,
            "fixed_metrics": None,
            "history_source": current_forecast.get("history_source", "current_forecast"),
        }
        future_rows.append(future_row)

    future_rows.sort(key=lambda item: item["target_date"])
    all_rows = historical_rows + future_rows
    available_dates = [pd.Timestamp(row["target_date"]) for row in all_rows]
    upload_context = _build_upload_context(site, max_actual_date)
    default_selected_date = upload_context.get("workflow_target_date") or current_target_date
    selected_ts = pd.Timestamp(target_date).normalize() if target_date is not None else pd.Timestamp(default_selected_date)
    selected_iso = selected_ts.strftime("%Y-%m-%d")
    selected_row = next((row for row in all_rows if row["target_date"] == selected_iso), None)
    if selected_row is None:
        selected_ts = pd.Timestamp(current_target_date)
        selected_iso = selected_ts.strftime("%Y-%m-%d")
        selected_row = next((row for row in all_rows if row["target_date"] == selected_iso), future_row)

    selected_month = selected_iso[:7]
    historical_months = sorted({row["target_date"][:7] for row in historical_rows}, reverse=True)
    monthly = [_month_summary([row for row in all_rows if row["target_date"].startswith(month)], month) for month in historical_months]
    selected_month_summary = next((row for row in monthly if row["month"] == selected_month), {"month": selected_month, "days_count": 0, "matched_days_count": 0, "latest_target_date": None, "model_metrics": None, "fixed_metrics": None})
    selected_month_rows = [row for row in all_rows if row["target_date"].startswith(selected_month)]
    latest_seven_rows = historical_rows[-7:]

    history_rows = []
    for row in reversed(selected_month_rows):
        history_rows.append(
            {
                "target_date": row["target_date"],
                "issue_date": row["issue_date"],
                "data_status": row["data_status"],
                "actual_total": row["actual_total"],
                "model_total": row["model_total"],
                "fixed_total": row["fixed_total"],
                "target_refined_date_type": row["target_refined_date_type"],
                "route_name": row["route_name"],
                "forecast_mode": row["forecast_mode"],
                "model_metrics": row["model_metrics"],
                "fixed_metrics": row["fixed_metrics"],
            }
        )

    selected_decorated = _decorate_row(selected_row)
    return {
        "site_code": site.code,
        "site_name": site.name,
        "available_sites": list_site_options(),
        "formula_label": "0.5*D-7 + 0.3*D-14 + 0.2*D-21",
        "selected_date": selected_iso,
        "max_actual_date": max_actual_date.strftime("%Y-%m-%d"),
        "current_target_date": current_target_date,
        "next_upload_date": upload_context["expected_actual_date"],
        "date_navigation": _build_date_navigation(selected_ts, available_dates),
        "selected_record": selected_decorated,
        "accuracy": {
            "matched_days": len(historical_rows),
            "history_start": historical_rows[0]["target_date"] if historical_rows else None,
            "history_end": historical_rows[-1]["target_date"] if historical_rows else None,
            "rolling_7d": {
                "model_metrics": _aggregate_metrics(latest_seven_rows, "model_values"),
                "fixed_metrics": _aggregate_metrics(latest_seven_rows, "fixed_values"),
            },
            "selected_month": selected_month_summary,
            "selected_month_trend": [
                {
                    "target_date": row["target_date"],
                    "model_metrics": row["model_metrics"],
                    "fixed_metrics": row["fixed_metrics"],
                }
                for row in selected_month_rows
            ],
            "monthly": monthly,
        },
        "history": history_rows,
        "upload_context": upload_context,
        "pipeline_files": {
            "scripts_dir": str((PIPELINE_SCRIPTS_DIR).relative_to(BASE_DIR)),
            "history_daily": str((site.new_dir / "history_daily.csv").relative_to(BASE_DIR)),
            "history_hourly": str((site.new_dir / "history_hourly.csv").relative_to(BASE_DIR)),
            "router_summary": str(get_best_summary_path(site).relative_to(BASE_DIR)),
            "current_forecast": current_forecast.get("history_source"),
        },
    }


def append_actual_and_refresh(actual_date: date, values: list[float], site_code: str = DEFAULT_SITE_CODE) -> dict[str, Any]:
    site = get_site_config(site_code)
    guard = _read_upload_guard(site) or {}
    if str(guard.get("lock_date", "")) == _today_local().isoformat():
        locked_actual_date = str(guard.get("actual_date", "")).strip() or "今天"
        raise ValueError(f"今天已经上传过 {locked_actual_date} 的实际值，上传入口明天会自动恢复。")

    cleaned_values = normalize_hourly_values(values)
    target_ts = pd.Timestamp(actual_date).normalize()
    history_daily = _load_history_daily(site)
    max_actual_date = get_max_actual_date(history_daily)
    expected_date = max_actual_date + pd.Timedelta(days=1)
    business_actual_date = pd.Timestamp(_today_local()) - pd.Timedelta(days=ISSUE_GAP_DAYS)
    if expected_date > business_actual_date:
        raise ValueError(
            f"今天对应的 D-6 实际 {business_actual_date.strftime('%Y-%m-%d')} 已完成，今天不能再上传 {expected_date.strftime('%Y-%m-%d')}。"
        )
    if target_ts != expected_date:
        raise ValueError(f"actual date must be {expected_date.strftime('%Y-%m-%d')}")

    history_hourly = _load_history_hourly(site)
    profile = np.asarray(cleaned_values, dtype=float)
    daily_row = build_daily_history_row(target_ts, profile, history_daily.columns.tolist())
    hourly_rows = build_hourly_history_rows(target_ts, profile, history_hourly)

    updated_daily = pd.concat([history_daily, daily_row], ignore_index=True).sort_values("date").reset_index(drop=True)
    updated_hourly = pd.concat([history_hourly, hourly_rows], ignore_index=True).sort_values(["timestamp"]).reset_index(drop=True)
    updated_daily.to_csv(site.new_dir / "history_daily.csv", index=False)
    updated_hourly.to_csv(site.new_dir / "history_hourly.csv", index=False)

    new_max_actual = get_max_actual_date(updated_daily)
    next_target = new_max_actual + pd.Timedelta(days=ISSUE_GAP_DAYS)
    artifact_info = _write_operational_forecast(site, updated_daily, next_target)
    _write_upload_guard(site, target_ts.date(), next_target.date())

    return {
        "message": "actual stored and forecast refreshed",
        "site_code": site.code,
        "site_name": site.name,
        "actual_date": target_ts.strftime("%Y-%m-%d"),
        "next_target_date": next_target.strftime("%Y-%m-%d"),
        "forecast_files": artifact_info,
    }
