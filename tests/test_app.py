import asyncio
import shutil
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, '/Users/cayron/work/demo')
import app.main as main_module
import app.pipeline_service as pipeline_service


@pytest.fixture()
def patched_pipeline(tmp_path, monkeypatch):
    source_root = Path('/Users/cayron/work/demo/pipeline')
    target_root = tmp_path / 'pipeline'
    shutil.copytree(source_root, target_root)

    monkeypatch.setattr(pipeline_service, 'BASE_DIR', tmp_path)
    monkeypatch.setattr(pipeline_service, 'PIPELINE_DIR', target_root)
    monkeypatch.setattr(pipeline_service, 'PIPELINE_SCRIPTS_DIR', target_root / 'scripts')
    monkeypatch.setattr(pipeline_service, 'PIPELINE_NEW_DIR', target_root / 'new')
    monkeypatch.setattr(pipeline_service, 'PIPELINE_RESULTS_DIR', target_root / 'results')
    monkeypatch.setattr(main_module, 'PIPELINE_NEW_DIR', target_root / 'new')
    return target_root


def request(app, method, url, **kwargs):
    async def run_request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url='http://testserver') as client:
            return await client.request(method, url, **kwargs)

    return asyncio.run(run_request())


def build_values(base: float) -> list[float]:
    return [round(base + index * 0.02, 4) for index in range(24)]


def test_dashboard_contains_comparison_metrics(patched_pipeline) -> None:
    response = request(main_module.app, 'GET', '/api/dashboard')
    assert response.status_code == 200
    payload = response.json()

    assert payload['selected_date'] == '2026-03-13'
    assert payload['formula_label'] == '0.5*D-7 + 0.3*D-14 + 0.2*D-21'
    assert payload['selected_record']['data_status'] == 'forecast_only'
    assert payload['selected_record']['model_total'] is not None
    assert payload['selected_record']['fixed_total'] is not None
    assert payload['date_navigation']['prev_date'] == '2026-03-12'
    assert payload['accuracy']['matched_days'] >= 66


def test_dashboard_can_show_january_accuracy(patched_pipeline) -> None:
    response = request(main_module.app, 'GET', '/api/dashboard', params={'target_date': '2026-01-10'})
    assert response.status_code == 200
    payload = response.json()

    assert payload['selected_date'] == '2026-01-10'
    assert payload['selected_record']['model_metrics']['day_total_accuracy'] is not None
    assert payload['selected_record']['fixed_metrics']['day_total_accuracy'] is not None
    assert payload['accuracy']['selected_month']['month'] == '2026-01'
    assert payload['accuracy']['selected_month']['model_metrics']['hourly_accuracy'] == 86.46
    assert payload['accuracy']['selected_month']['model_metrics']['day_total_accuracy'] == 92.71
    assert payload['accuracy']['selected_month']['days_count'] == 31
    assert len(payload['history']) == 31


def test_upload_actual_generates_next_forecast(patched_pipeline) -> None:
    response = request(
        main_module.app,
        'POST',
        '/api/actuals',
        json={
            'actual_date': '2026-03-08',
            'values': build_values(1.6),
        },
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload['actual_date'] == '2026-03-08'
    assert payload['next_target_date'] == '2026-03-14'
    assert (patched_pipeline / 'results' / 'forecast_d6_20260314.csv').exists()

    future_dashboard = request(main_module.app, 'GET', '/api/dashboard', params={'target_date': '2026-03-14'})
    assert future_dashboard.status_code == 200
    future_payload = future_dashboard.json()
    assert future_payload['selected_record']['data_status'] == 'forecast_only'
    assert future_payload['selected_record']['model_total'] is not None
    assert future_payload['date_navigation']['prev_date'] == '2026-03-13'

    history_dashboard = request(main_module.app, 'GET', '/api/dashboard', params={'target_date': '2026-03-08'})
    assert history_dashboard.status_code == 200
    history_payload = history_dashboard.json()
    assert history_payload['selected_record']['data_status'] == 'complete'
    assert history_payload['selected_record']['model_metrics']['day_total_accuracy'] is not None
    assert history_payload['accuracy']['matched_days'] >= 67
