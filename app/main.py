from __future__ import annotations

import io
from datetime import date
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator

from app.pipeline_service import (
    DEFAULT_SITE_CODE,
    append_actual_and_refresh,
    build_dashboard_payload,
    get_actual_template_path,
    normalize_hourly_values,
    resolve_site_code,
)

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / 'static'
TEMPLATES_DIR = BASE_DIR / 'templates'


def asset_version(path: Path) -> str:
    try:
        return str(int(path.stat().st_mtime))
    except OSError:
        return '1'


def normalize_hour_column(column: str) -> str | None:
    value = str(column).strip().lower()
    if value.startswith('load_h') and len(value) == 8:
        return value
    if value.startswith('h') and len(value) == 3 and value[1:].isdigit():
        return f'load_{value}'
    if value.endswith(':00') and len(value) == 5 and value[:2].isdigit():
        return f'load_h{value[:2]}'
    if value.isdigit() and 0 <= int(value) <= 23:
        return f'load_h{int(value):02d}'
    return None


def parse_values_text(raw_text: str) -> list[float]:
    cleaned = raw_text.replace("\r", " ").replace("\t", " ").replace("，", " ").replace(",", " ")
    values = [float(part) for part in cleaned.split() if part.strip()]
    return normalize_hourly_values(values)



def parse_upload_csv(content: bytes, actual_date: date | None) -> tuple[date, list[float]]:
    frame = pd.read_csv(io.BytesIO(content))
    frame.columns = [str(column).strip() for column in frame.columns]

    if {'hour', 'load'}.issubset(frame.columns):
        resolved_date = actual_date
        if resolved_date is None and 'date' in frame.columns and pd.notna(frame.loc[0, 'date']):
            resolved_date = pd.Timestamp(frame.loc[0, 'date']).date()
        if resolved_date is None:
            raise ValueError('长表 CSV 需要填写实际日期，或在文件里提供 date 列。')
        hour_map: dict[int, float] = {}
        for row in frame.itertuples(index=False):
            hour_value = str(getattr(row, 'hour')).strip()
            hour_int = int(hour_value[:2]) if hour_value.endswith(':00') else int(hour_value)
            hour_map[hour_int] = float(getattr(row, 'load'))
        values = [hour_map[hour] for hour in range(24)]
        return resolved_date, normalize_hourly_values(values)

    renamed: dict[str, str] = {}
    for column in frame.columns:
        normalized = normalize_hour_column(column)
        if normalized is not None:
            renamed[column] = normalized
    wide = frame.rename(columns=renamed)
    if len(wide) != 1:
        raise ValueError('宽表 CSV 必须只有一行。')

    resolved_date = actual_date
    if resolved_date is None and 'date' in wide.columns and pd.notna(wide.loc[0, 'date']):
        resolved_date = pd.Timestamp(wide.loc[0, 'date']).date()
    if resolved_date is None:
        raise ValueError('请填写实际日期，或在模板 CSV 中保留 date 列。')

    values = [float(wide.loc[0, f'load_h{hour:02d}']) for hour in range(24)]
    return resolved_date, normalize_hourly_values(values)


class ActualUploadPayload(BaseModel):
    actual_date: date
    values: list[float] = Field(min_length=24, max_length=24)

    @field_validator('values')
    @classmethod
    def validate_values(cls, values: list[float]) -> list[float]:
        return normalize_hourly_values(values)


app = FastAPI(title='D-6 负荷预测展示', version='2.0.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)
app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')
templates = Jinja2Templates(directory=TEMPLATES_DIR)


@app.get('/', response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name='index.html',
        context={
            'page_title': 'D-6 负荷预测展示',
            'style_version': asset_version(STATIC_DIR / 'style.css'),
            'script_version': asset_version(STATIC_DIR / 'app.js'),
        },
    )


@app.get('/healthz')
async def healthz() -> dict[str, str]:
    return {'status': 'ok'}


@app.get('/api/dashboard')
async def get_dashboard(
    target_date: Optional[date] = Query(default=None),
    site: str = Query(default=DEFAULT_SITE_CODE),
) -> dict[str, Any]:
    try:
        return build_dashboard_payload(target_date=target_date, site_code=resolve_site_code(site))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get('/api/history')
async def get_history(
    target_date: Optional[date] = Query(default=None),
    site: str = Query(default=DEFAULT_SITE_CODE),
) -> dict[str, Any]:
    try:
        payload = build_dashboard_payload(target_date=target_date, site_code=resolve_site_code(site))
        return {'items': payload['history']}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get('/api/actual-template')
async def get_actual_template(site: str = Query(default=DEFAULT_SITE_CODE)) -> FileResponse:
    try:
        site_code = resolve_site_code(site)
        return FileResponse(
            get_actual_template_path(site_code),
            media_type='text/csv',
            filename=f'actual_input_template_{site_code}.csv',
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post('/api/actuals')
async def upload_actual(
    payload: ActualUploadPayload,
    site: str = Query(default=DEFAULT_SITE_CODE),
) -> dict[str, Any]:
    try:
        return append_actual_and_refresh(
            actual_date=payload.actual_date,
            values=payload.values,
            site_code=resolve_site_code(site),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post('/api/actual-upload')
async def upload_actual_form(
    site: str = Form(default=DEFAULT_SITE_CODE),
    actual_date: Optional[str] = Form(default=None),
    raw_values: str = Form(default=''),
    actual_file: Optional[UploadFile] = File(default=None),
) -> dict[str, Any]:
    try:
        site_code = resolve_site_code(site)
        resolved_date = date.fromisoformat(actual_date) if actual_date else None
        if actual_file is not None and actual_file.filename:
            upload_content = await actual_file.read()
            resolved_date, values = parse_upload_csv(upload_content, resolved_date)
        else:
            if resolved_date is None:
                raise ValueError('请先填写实际日期。')
            values = parse_values_text(raw_values)
        result = append_actual_and_refresh(actual_date=resolved_date, values=values, site_code=site_code)
        result['dashboard'] = build_dashboard_payload(site_code=site_code)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post('/api/d6-actuals')
async def upload_d6_actual(
    payload: ActualUploadPayload,
    site: str = Query(default=DEFAULT_SITE_CODE),
) -> dict[str, Any]:
    return await upload_actual(payload, site)
