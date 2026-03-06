from __future__ import annotations

import json
import math
import random
import sqlite3
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Optional, Union
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_FILE = DATA_DIR / "store.json"
DB_FILE = DATA_DIR / "store.sqlite3"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
DEFAULT_SITE_NAME = "辉华"
LEGACY_SITE_NAME = "华海升园区"
SOURCE_NAME_MAP = {
    "manual": "手工录入",
    "manual-web": "网页录入",
    "manual-ems": "EMS回填",
    "api": "接口上传",
    "api-demo": "接口样例",
    "ems-demo": "EMS样例",
    "dispatch-center": "调度中心",
    "ems": "EMS系统",
}


def asset_version(path: Path) -> str:
    try:
        return str(int(path.stat().st_mtime))
    except OSError:
        return "1"


class ForecastPayload(BaseModel):
    target_date: date
    values: list[float] = Field(min_length=24, max_length=24)
    source: str = "manual"
    site_name: str = DEFAULT_SITE_NAME
    note: str = ""
    generated_at: Optional[datetime] = None

    @field_validator("values")
    @classmethod
    def validate_values(cls, values: list[float]) -> list[float]:
        return normalize_hourly_values(values, "load forecast")

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str) -> str:
        return normalize_source_name(value)


class ActualPayload(BaseModel):
    target_date: date
    values: list[float] = Field(min_length=24, max_length=24)
    source: str = "manual"

    @field_validator("values")
    @classmethod
    def validate_values(cls, values: list[float]) -> list[float]:
        return normalize_hourly_values(values, "actual load")

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str) -> str:
        return normalize_source_name(value)


class LoadForecastBlock(BaseModel):
    values: list[float] = Field(min_length=24, max_length=24)
    source: str = "api"
    site_name: str = DEFAULT_SITE_NAME
    note: str = ""
    generated_at: Optional[datetime] = None

    @field_validator("values")
    @classmethod
    def validate_values(cls, values: list[float]) -> list[float]:
        return normalize_hourly_values(values, "load forecast")

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str) -> str:
        return normalize_source_name(value)


class LoadActualBlock(BaseModel):
    values: list[float] = Field(min_length=24, max_length=24)
    source: str = "api"

    @field_validator("values")
    @classmethod
    def validate_values(cls, values: list[float]) -> list[float]:
        return normalize_hourly_values(values, "actual load")

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str) -> str:
        return normalize_source_name(value)


class LoadRecordPayload(BaseModel):
    target_date: date
    forecast: Optional[LoadForecastBlock] = None
    actual: Optional[LoadActualBlock] = None

    @model_validator(mode="after")
    def validate_data_presence(self) -> "LoadRecordPayload":
        if self.forecast is None and self.actual is None:
            raise ValueError("forecast or actual data must be provided")
        return self


class ForecastRecord(ForecastPayload):
    id: str
    generated_at: datetime
    created_by: str = "demo"

    model_config = ConfigDict(from_attributes=True)


class ActualRecord(ActualPayload):
    id: str
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


def normalize_hourly_values(values: list[float], label: str) -> list[float]:
    if len(values) != 24:
        raise ValueError(f"values must contain exactly 24 hourly {label} values")
    cleaned = [round(float(value), 2) for value in values]
    if any(value < 0 for value in cleaned):
        raise ValueError("hourly load values must be non-negative")
    return cleaned


def now_local() -> datetime:
    return datetime.now().replace(microsecond=0)


def date_to_iso(value: Union[date, datetime]) -> str:
    return value.isoformat()


def normalize_source_name(source: str) -> str:
    return SOURCE_NAME_MAP.get(source, source)


def build_seed_store() -> dict[str, Any]:
    today = date.today()
    base_site = DEFAULT_SITE_NAME
    forecasts: list[dict[str, Any]] = []
    actuals: list[dict[str, Any]] = []

    random.seed(11)

    for day_offset in range(-9, 2):
        target = today + timedelta(days=day_offset)
        seasonal = 780 + day_offset * 2
        actual_values: list[float] = []
        forecast_values: list[float] = []
        for hour in range(24):
            morning_wave = 95 * math.sin((hour - 6) / 24 * math.pi * 2)
            evening_wave = 60 * math.sin((hour - 14) / 24 * math.pi * 2)
            working_load = 115 if 8 <= hour <= 19 else 45
            actual = seasonal + morning_wave + evening_wave + working_load + random.uniform(-18, 18)
            actual_values.append(round(max(actual, 360), 2))

            bias = random.uniform(-0.045, 0.05)
            forecast = actual_values[-1] * (1 + bias)
            forecast_values.append(round(max(forecast, 340), 2))

        generated_at = datetime.combine(target - timedelta(days=1), time(16, 30))
        forecasts.append(
            ForecastRecord(
                id=str(uuid4()),
                target_date=target,
                values=forecast_values,
                source="接口样例",
                site_name=base_site,
                note="系统样例预测",
                generated_at=generated_at,
                created_by="seed",
            ).model_dump(mode="json")
        )
        if target <= today:
            actuals.append(
                ActualRecord(
                    id=str(uuid4()),
                    target_date=target,
                    values=actual_values,
                    source="EMS样例",
                    updated_at=datetime.combine(target, time(23, 50)),
                ).model_dump(mode="json")
            )

    return {
        "site_name": base_site,
        "forecasts": forecasts,
        "actuals": actuals,
    }


def get_db_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_FILE)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS forecasts (
            id TEXT PRIMARY KEY,
            target_date TEXT NOT NULL,
            generated_at TEXT NOT NULL,
            source TEXT NOT NULL,
            site_name TEXT NOT NULL,
            note TEXT NOT NULL DEFAULT '',
            created_by TEXT NOT NULL DEFAULT 'demo',
            values_json TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_forecasts_target_date_generated_at
            ON forecasts(target_date, generated_at);

        CREATE TABLE IF NOT EXISTS actuals (
            id TEXT PRIMARY KEY,
            target_date TEXT NOT NULL UNIQUE,
            updated_at TEXT NOT NULL,
            source TEXT NOT NULL,
            values_json TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_actuals_target_date
            ON actuals(target_date);
        """
    )


def database_has_data(connection: sqlite3.Connection) -> bool:
    settings_count = connection.execute("SELECT COUNT(*) FROM settings").fetchone()[0]
    forecast_count = connection.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0]
    actual_count = connection.execute("SELECT COUNT(*) FROM actuals").fetchone()[0]
    return settings_count > 0 or forecast_count > 0 or actual_count > 0


def normalize_store_payload(store: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    changed = False
    normalized_store = {
        "site_name": store.get("site_name", DEFAULT_SITE_NAME),
        "forecasts": list(store.get("forecasts", [])),
        "actuals": list(store.get("actuals", [])),
    }

    if normalized_store["site_name"] in {None, "", LEGACY_SITE_NAME}:
        normalized_store["site_name"] = DEFAULT_SITE_NAME
        changed = True

    for forecast in normalized_store["forecasts"]:
        if forecast.get("site_name") in {None, "", LEGACY_SITE_NAME}:
            forecast["site_name"] = DEFAULT_SITE_NAME
            changed = True
        normalized_source = normalize_source_name(forecast.get("source", ""))
        if forecast.get("source") != normalized_source:
            forecast["source"] = normalized_source
            changed = True

    for actual in normalized_store["actuals"]:
        normalized_source = normalize_source_name(actual.get("source", ""))
        if actual.get("source") != normalized_source:
            actual["source"] = normalized_source
            changed = True

    return normalized_store, changed


def write_store_to_db(store: dict[str, Any]) -> None:
    normalized_store, _ = normalize_store_payload(store)
    forecasts = [ForecastRecord.model_validate(item) for item in normalized_store.get("forecasts", [])]
    actuals = [ActualRecord.model_validate(item) for item in normalized_store.get("actuals", [])]

    with get_db_connection() as connection:
        initialize_database(connection)
        with connection:
            connection.execute("DELETE FROM settings")
            connection.execute("DELETE FROM forecasts")
            connection.execute("DELETE FROM actuals")
            connection.execute(
                "INSERT INTO settings(key, value) VALUES(?, ?)",
                ("site_name", normalized_store.get("site_name", DEFAULT_SITE_NAME)),
            )
            connection.executemany(
                """
                INSERT INTO forecasts(id, target_date, generated_at, source, site_name, note, created_by, values_json)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        row.id,
                        date_to_iso(row.target_date),
                        date_to_iso(row.generated_at),
                        row.source,
                        row.site_name,
                        row.note,
                        row.created_by,
                        json.dumps(row.values, ensure_ascii=False),
                    )
                    for row in forecasts
                ],
            )
            connection.executemany(
                """
                INSERT INTO actuals(id, target_date, updated_at, source, values_json)
                VALUES(?, ?, ?, ?, ?)
                """,
                [
                    (
                        row.id,
                        date_to_iso(row.target_date),
                        date_to_iso(row.updated_at),
                        row.source,
                        json.dumps(row.values, ensure_ascii=False),
                    )
                    for row in actuals
                ],
            )


def read_store_from_db() -> dict[str, Any]:
    with get_db_connection() as connection:
        initialize_database(connection)
        site_name_row = connection.execute(
            "SELECT value FROM settings WHERE key = ?",
            ("site_name",),
        ).fetchone()
        forecast_rows = connection.execute(
            """
            SELECT id, target_date, generated_at, source, site_name, note, created_by, values_json
            FROM forecasts
            ORDER BY target_date DESC, generated_at DESC
            """
        ).fetchall()
        actual_rows = connection.execute(
            """
            SELECT id, target_date, updated_at, source, values_json
            FROM actuals
            ORDER BY target_date DESC
            """
        ).fetchall()

    return {
        "site_name": site_name_row["value"] if site_name_row else DEFAULT_SITE_NAME,
        "forecasts": [
            {
                "id": row["id"],
                "target_date": row["target_date"],
                "generated_at": row["generated_at"],
                "source": row["source"],
                "site_name": row["site_name"],
                "note": row["note"],
                "created_by": row["created_by"],
                "values": json.loads(row["values_json"]),
            }
            for row in forecast_rows
        ],
        "actuals": [
            {
                "id": row["id"],
                "target_date": row["target_date"],
                "updated_at": row["updated_at"],
                "source": row["source"],
                "values": json.loads(row["values_json"]),
            }
            for row in actual_rows
        ],
    }


def ensure_storage() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with get_db_connection() as connection:
        initialize_database(connection)
        if database_has_data(connection):
            return

    if DATA_FILE.exists():
        with DATA_FILE.open("r", encoding="utf-8") as handle:
            seed_store = json.load(handle)
    else:
        seed_store = build_seed_store()

    write_store_to_db(seed_store)


def load_store() -> dict[str, Any]:
    ensure_storage()
    store = read_store_from_db()
    store, changed = normalize_store_payload(store)
    if changed:
        save_store(store)

    return store


def save_store(store: dict[str, Any]) -> None:
    write_store_to_db(store)


def parse_forecast_rows(store: dict[str, Any]) -> list[ForecastRecord]:
    rows = [ForecastRecord.model_validate(item) for item in store.get("forecasts", [])]
    return sorted(rows, key=lambda item: (item.target_date, item.generated_at), reverse=True)


def parse_actual_rows(store: dict[str, Any]) -> list[ActualRecord]:
    rows = [ActualRecord.model_validate(item) for item in store.get("actuals", [])]
    return sorted(rows, key=lambda item: item.target_date, reverse=True)


def latest_forecasts_by_date(forecasts: list[ForecastRecord]) -> dict[date, ForecastRecord]:
    latest: dict[date, ForecastRecord] = {}
    for forecast in forecasts:
        current = latest.get(forecast.target_date)
        if current is None or forecast.generated_at > current.generated_at:
            latest[forecast.target_date] = forecast
    return latest


def compute_metrics(forecast_values: list[float], actual_values: list[float]) -> dict[str, float]:
    abs_errors = [abs(forecast - actual) for forecast, actual in zip(forecast_values, actual_values)]
    squared_errors = [(forecast - actual) ** 2 for forecast, actual in zip(forecast_values, actual_values)]
    percent_errors = [
        abs(forecast - actual) / actual * 100 for forecast, actual in zip(forecast_values, actual_values) if actual > 0
    ]
    within_five = [
        1 for forecast, actual in zip(forecast_values, actual_values) if actual > 0 and abs(forecast - actual) / actual <= 0.05
    ]
    return {
        "mae": round(sum(abs_errors) / len(abs_errors), 2),
        "rmse": round(math.sqrt(sum(squared_errors) / len(squared_errors)), 2),
        "mape": round(sum(percent_errors) / len(percent_errors), 2) if percent_errors else 0.0,
        "hit_rate_5": round(sum(within_five) / 24 * 100, 2),
        "max_abs_error": round(max(abs_errors), 2),
    }


def find_actual_map(actuals: list[ActualRecord]) -> dict[date, ActualRecord]:
    return {row.target_date: row for row in actuals}


def build_source_label(forecast: Optional[ForecastRecord], actual: Optional[ActualRecord]) -> Optional[str]:
    sources = [source for source in [forecast.source if forecast else None, actual.source if actual else None] if source]
    if not sources:
        return None
    deduped = list(dict.fromkeys(sources))
    return " / ".join(deduped)


def build_data_status(forecast: Optional[ForecastRecord], actual: Optional[ActualRecord]) -> str:
    if forecast and actual:
        return "complete"
    if forecast:
        return "forecast_only"
    if actual:
        return "actual_only"
    return "empty"


def build_day_row(target_date: date, forecast: Optional[ForecastRecord], actual: Optional[ActualRecord]) -> dict[str, Any]:
    metrics = compute_metrics(forecast.values, actual.values) if forecast and actual else None
    return {
        "id": forecast.id if forecast else actual.id if actual else None,
        "target_date": date_to_iso(target_date),
        "generated_at": date_to_iso(forecast.generated_at) if forecast else None,
        "updated_at": date_to_iso(actual.updated_at) if actual else None,
        "source": build_source_label(forecast, actual) or "-",
        "forecast_source": forecast.source if forecast else None,
        "actual_source": actual.source if actual else None,
        "site_name": forecast.site_name if forecast else None,
        "note": forecast.note if forecast else "",
        "forecast_total": round(sum(forecast.values), 2) if forecast else None,
        "actual_total": round(sum(actual.values), 2) if actual else None,
        "metrics": metrics,
        "has_forecast": forecast is not None,
        "has_actual": actual is not None,
        "data_status": build_data_status(forecast, actual),
    }


def build_daily_rows(
    forecast_map: dict[date, ForecastRecord],
    actual_map: dict[date, ActualRecord],
) -> list[dict[str, Any]]:
    all_dates = sorted(set(forecast_map) | set(actual_map), reverse=True)
    return [build_day_row(target, forecast_map.get(target), actual_map.get(target)) for target in all_dates]


def average_metrics(rows: list[dict[str, Any]]) -> Optional[dict[str, float]]:
    metric_rows = [row for row in rows if row.get("metrics")]
    if not metric_rows:
        return None
    return {
        "mape": round(sum(row["metrics"]["mape"] for row in metric_rows) / len(metric_rows), 2),
        "mae": round(sum(row["metrics"]["mae"] for row in metric_rows) / len(metric_rows), 2),
        "rmse": round(sum(row["metrics"]["rmse"] for row in metric_rows) / len(metric_rows), 2),
        "hit_rate_5": round(sum(row["metrics"]["hit_rate_5"] for row in metric_rows) / len(metric_rows), 2),
    }


def build_monthly_rows(daily_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in daily_rows:
        if row.get("metrics"):
            grouped[row["target_date"][:7]].append(row)

    monthly_rows: list[dict[str, Any]] = []
    for month, rows in grouped.items():
        metrics = average_metrics(rows)
        latest_target_date = max(row["target_date"] for row in rows)
        monthly_rows.append(
            {
                "month": month,
                "days_count": len(rows),
                "forecast_total": round(sum(row["forecast_total"] or 0 for row in rows), 2),
                "actual_total": round(sum(row["actual_total"] or 0 for row in rows), 2),
                "latest_target_date": latest_target_date,
                "metrics": metrics,
            }
        )
    return sorted(monthly_rows, key=lambda row: row["month"], reverse=True)


def build_date_navigation(selected_date: date, available_dates: list[date]) -> dict[str, Any]:
    if not available_dates:
        return {
            "available_dates": [],
            "prev_date": None,
            "next_date": None,
            "min_date": None,
            "max_date": None,
        }

    prev_date = None
    next_date = None
    if selected_date in available_dates:
        index = available_dates.index(selected_date)
        prev_date = date_to_iso(available_dates[index - 1]) if index > 0 else None
        next_date = date_to_iso(available_dates[index + 1]) if index < len(available_dates) - 1 else None

    return {
        "available_dates": [date_to_iso(item) for item in available_dates],
        "prev_date": prev_date,
        "next_date": next_date,
        "min_date": date_to_iso(available_dates[0]),
        "max_date": date_to_iso(available_dates[-1]),
    }


def make_chart_series(values: list[float]) -> list[dict[str, Any]]:
    return [{"hour": f"{hour:02d}:00", "value": value} for hour, value in enumerate(values)]


def build_selected_record(
    selected_date: date,
    forecast_map: dict[date, ForecastRecord],
    actual_map: dict[date, ActualRecord],
) -> dict[str, Any]:
    forecast = forecast_map.get(selected_date)
    actual = actual_map.get(selected_date)
    preferred_values = forecast.values if forecast else actual.values if actual else []
    peak = None
    valley = None
    if preferred_values:
        peak_value = max(preferred_values)
        peak_hour = preferred_values.index(peak_value)
        valley_value = min(preferred_values)
        valley_hour = preferred_values.index(valley_value)
        peak = {"hour": f"{peak_hour:02d}:00", "value": peak_value}
        valley = {"hour": f"{valley_hour:02d}:00", "value": valley_value}

    return {
        "id": forecast.id if forecast else actual.id if actual else None,
        "target_date": date_to_iso(selected_date),
        "generated_at": date_to_iso(forecast.generated_at) if forecast else None,
        "actual_updated_at": date_to_iso(actual.updated_at) if actual else None,
        "source": build_source_label(forecast, actual),
        "forecast_source": forecast.source if forecast else None,
        "actual_source": actual.source if actual else None,
        "site_name": forecast.site_name if forecast else None,
        "note": forecast.note if forecast else "",
        "forecast_values": forecast.values if forecast else None,
        "forecast_series": make_chart_series(forecast.values) if forecast else [],
        "actual_values": actual.values if actual else None,
        "actual_series": make_chart_series(actual.values) if actual else [],
        "forecast_total": round(sum(forecast.values), 2) if forecast else None,
        "actual_total": round(sum(actual.values), 2) if actual else None,
        "avg": round(sum(preferred_values) / 24, 2) if preferred_values else None,
        "peak": peak,
        "valley": valley,
        "peak_gap": round(peak["value"] - valley["value"], 2) if peak and valley else None,
        "daily_metrics": compute_metrics(forecast.values, actual.values) if forecast and actual else None,
        "has_forecast": forecast is not None,
        "has_actual": actual is not None,
        "data_status": build_data_status(forecast, actual),
    }


def build_accuracy_summary(daily_rows: list[dict[str, Any]], selected_date: date) -> dict[str, Any]:
    matched_rows = [row for row in daily_rows if row.get("metrics")]
    recent_rows = matched_rows[:7]
    selected_date_iso = date_to_iso(selected_date)
    selected_day = next((row for row in daily_rows if row["target_date"] == selected_date_iso), None)
    selected_month = selected_date.strftime("%Y-%m")
    selected_month_rows = [row for row in matched_rows if row["target_date"].startswith(selected_month)]
    selected_month_history = [row for row in daily_rows if row["target_date"].startswith(selected_month)]
    monthly_rows = build_monthly_rows(daily_rows)
    selected_month_summary = next(
        (row for row in monthly_rows if row["month"] == selected_month),
        {
            "month": selected_month,
            "days_count": 0,
            "forecast_total": None,
            "actual_total": None,
            "latest_target_date": None,
            "metrics": None,
        },
    )

    return {
        "matched_days": len(matched_rows),
        "rolling_7d": average_metrics(recent_rows),
        "selected_day": selected_day,
        "selected_month": selected_month_summary,
        "selected_month_trend": list(reversed(selected_month_rows)),
        "selected_month_history": selected_month_history,
        "monthly": monthly_rows,
    }


def build_dashboard_payload(store: dict[str, Any], target_date: Optional[date] = None) -> dict[str, Any]:
    forecasts = parse_forecast_rows(store)
    actuals = parse_actual_rows(store)
    forecast_map = latest_forecasts_by_date(forecasts)
    actual_map = find_actual_map(actuals)
    daily_rows = build_daily_rows(forecast_map, actual_map)
    available_dates = sorted(set(forecast_map) | set(actual_map))

    if target_date is not None:
        selected_date = target_date
    elif available_dates:
        selected_date = available_dates[-1]
    else:
        selected_date = date.today()

    selected_record = build_selected_record(selected_date, forecast_map, actual_map)
    accuracy = build_accuracy_summary(daily_rows, selected_date)
    navigation = build_date_navigation(selected_date, available_dates)

    return {
        "site_name": store.get("site_name", DEFAULT_SITE_NAME),
        "selected_date": date_to_iso(selected_date),
        "date_navigation": navigation,
        "selected_record": selected_record,
        "accuracy": accuracy,
        "history": accuracy["selected_month_history"],
        "api_example": {
            "load_record": {
                "target_date": date_to_iso(selected_date),
                "forecast": {
                    "source": "调度中心",
                    "site_name": store.get("site_name", DEFAULT_SITE_NAME),
                    "note": "联合上传负荷预测与负荷实际",
                    "values": [730 + hour * 5 for hour in range(24)],
                },
                "actual": {
                    "source": "EMS系统",
                    "values": [710 + hour * 4 for hour in range(24)],
                },
            },
            "forecast": {
                "target_date": date_to_iso(selected_date + timedelta(days=1)),
                "source": "调度中心",
                "site_name": store.get("site_name", DEFAULT_SITE_NAME),
                "note": "外部系统推送",
                "values": [730 + hour * 5 for hour in range(24)],
            },
            "actual": {
                "target_date": date_to_iso(selected_date),
                "source": "EMS系统",
                "values": [710 + hour * 4 for hour in range(24)],
            },
        },
    }


def append_forecast_record(
    store: dict[str, Any],
    payload: ForecastPayload,
    *,
    created_by: str = "api",
) -> ForecastRecord:
    record = ForecastRecord(
        id=str(uuid4()),
        target_date=payload.target_date,
        values=payload.values,
        source=payload.source,
        site_name=payload.site_name,
        note=payload.note,
        generated_at=payload.generated_at or now_local(),
        created_by=created_by,
    )
    store["site_name"] = record.site_name
    store.setdefault("forecasts", []).append(record.model_dump(mode="json"))
    return record


def upsert_actual_record(store: dict[str, Any], payload: ActualPayload) -> ActualRecord:
    rows = parse_actual_rows(store)
    new_rows: list[dict[str, Any]] = []
    replaced = False
    record = ActualRecord(
        id=str(uuid4()),
        target_date=payload.target_date,
        values=payload.values,
        source=payload.source,
        updated_at=now_local(),
    )

    for row in rows:
        if row.target_date == payload.target_date:
            new_rows.append(record.model_dump(mode="json"))
            replaced = True
        else:
            new_rows.append(row.model_dump(mode="json"))

    if not replaced:
        new_rows.append(record.model_dump(mode="json"))

    store["actuals"] = new_rows
    return record


def verify_write_access(api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> None:
    expected = None
    try:
        import os

        expected = os.getenv("DEMO_API_KEY")
    except Exception:
        expected = None

    if expected and api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_storage()
    yield


app = FastAPI(title="负荷预测展示 Demo", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "page_title": "负荷预测展示 Demo",
            "style_version": asset_version(STATIC_DIR / "style.css"),
            "script_version": asset_version(STATIC_DIR / "app.js"),
        },
    )


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/dashboard")
async def get_dashboard(target_date: Optional[date] = Query(default=None)) -> dict[str, Any]:
    store = load_store()
    return build_dashboard_payload(store, target_date=target_date)


@app.get("/api/forecasts")
async def get_forecasts(limit: int = Query(default=20, ge=1, le=100)) -> dict[str, Any]:
    store = load_store()
    forecasts = parse_forecast_rows(store)[:limit]
    return {
        "items": [
            {
                "id": row.id,
                "target_date": date_to_iso(row.target_date),
                "generated_at": date_to_iso(row.generated_at),
                "source": row.source,
                "site_name": row.site_name,
                "note": row.note,
                "values": row.values,
            }
            for row in forecasts
        ]
    }


@app.post("/api/forecasts", dependencies=[Depends(verify_write_access)])
async def create_forecast(payload: ForecastPayload) -> dict[str, Any]:
    store = load_store()
    record = append_forecast_record(store, payload)
    save_store(store)
    return {
        "message": "forecast stored",
        "id": record.id,
        "target_date": date_to_iso(record.target_date),
        "generated_at": date_to_iso(record.generated_at),
    }


@app.post("/api/actuals", dependencies=[Depends(verify_write_access)])
async def upsert_actual(payload: ActualPayload) -> dict[str, Any]:
    store = load_store()
    record = upsert_actual_record(store, payload)
    save_store(store)
    return {
        "message": "actual stored",
        "id": record.id,
        "target_date": date_to_iso(record.target_date),
        "updated_at": date_to_iso(record.updated_at),
    }


@app.post("/api/load-records", dependencies=[Depends(verify_write_access)])
async def create_load_record(payload: LoadRecordPayload) -> dict[str, Any]:
    store = load_store()
    forecast_record: Optional[ForecastRecord] = None
    actual_record: Optional[ActualRecord] = None

    if payload.forecast is not None:
        forecast_record = append_forecast_record(
            store,
            ForecastPayload(
                target_date=payload.target_date,
                values=payload.forecast.values,
                source=payload.forecast.source,
                site_name=payload.forecast.site_name,
                note=payload.forecast.note,
                generated_at=payload.forecast.generated_at,
            ),
        )

    if payload.actual is not None:
        actual_record = upsert_actual_record(
            store,
            ActualPayload(
                target_date=payload.target_date,
                values=payload.actual.values,
                source=payload.actual.source,
            ),
        )

    save_store(store)
    return {
        "message": "load record stored",
        "target_date": date_to_iso(payload.target_date),
        "stored": {
            "forecast": forecast_record is not None,
            "actual": actual_record is not None,
        },
        "forecast_id": forecast_record.id if forecast_record else None,
        "actual_id": actual_record.id if actual_record else None,
        "forecast_generated_at": date_to_iso(forecast_record.generated_at) if forecast_record else None,
        "actual_updated_at": date_to_iso(actual_record.updated_at) if actual_record else None,
    }


@app.get("/api/history")
async def get_history(limit: int = Query(default=12, ge=1, le=90)) -> dict[str, Any]:
    store = load_store()
    forecast_map = latest_forecasts_by_date(parse_forecast_rows(store))
    actual_map = find_actual_map(parse_actual_rows(store))
    return {"items": build_daily_rows(forecast_map, actual_map)[:limit]}
