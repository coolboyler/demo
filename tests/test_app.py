import asyncio
import shutil
import sys
from datetime import date
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
    monkeypatch.setattr(pipeline_service, 'PIPELINE_TIANLANG_DIR', target_root / 'tianlang')
    monkeypatch.setattr(
        pipeline_service,
        'SITE_CONFIGS',
        {
            'huihua': pipeline_service.SiteConfig(
                code='huihua',
                name='辉华',
                new_dir=target_root / 'new',
                results_dir=target_root / 'results',
                template_path=target_root / 'new' / 'actual_input_template.csv',
                dataset_filename='baseline_d6_dataset.csv',
                best_summary_filename='best_d6_summary.json',
                baseline_summary_filename='baseline_d6_summary.json',
                replaced_days_filename='best_d6_replaced_days.csv',
                forecast_daily_glob='forecast_d6_*.csv',
                forecast_daily_pattern=pipeline_service.RESULT_DAILY_PATTERN,
                forecast_range_glob='forecast_d6_range_*.csv',
                forecast_range_pattern=pipeline_service.RESULT_RANGE_PATTERN,
                forecast_stem_template='forecast_d6_{target}',
                forecast_month_subdirs=False,
                reference_test_daily_filename=None,
                dataset_loader=pipeline_service.shared_load_dataset,
                router_applier=pipeline_service.shared_apply_holiday_router,
            ),
            'tianlang': pipeline_service.SiteConfig(
                code='tianlang',
                name='天朗',
                new_dir=target_root / 'tianlang' / 'new',
                results_dir=target_root / 'tianlang' / 'results',
                template_path=target_root / 'new' / 'actual_input_template.csv',
                dataset_filename='baseline_d6_dataset_2024fill.csv',
                best_summary_filename='best_d6_tianlang_2024fill_shared_summary.json',
                baseline_summary_filename='baseline_d6_tianlang_2024fill_shared_summary.json',
                replaced_days_filename='best_d6_tianlang_2024fill_shared_replaced_days.csv',
                forecast_daily_glob='**/forecast_d6_tianlang_*_2024fill_shared_*.csv',
                forecast_daily_pattern=pipeline_service.re.compile(r'^forecast_d6_tianlang_(\d{8})_2024fill_shared_(\d{8})\.csv$'),
                forecast_range_glob=None,
                forecast_range_pattern=None,
                forecast_stem_template='forecast_d6_tianlang_{target}_2024fill_shared_{target}',
                forecast_month_subdirs=True,
                reference_test_daily_filename='best_d6_tianlang_2024fill_shared_test_daily.csv',
                dataset_loader=pipeline_service.tianlang_load_dataset,
                router_applier=pipeline_service.tianlang_apply_holiday_router,
            ),
        },
    )
    monkeypatch.setattr(pipeline_service, '_today_local', lambda: date(2026, 3, 15))
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

    assert payload['site_code'] == 'huihua'
    assert payload['site_name'] == '辉华'
    assert payload['selected_date'] == '2026-03-14'
    assert payload['formula_label'] == '0.5*D-7 + 0.3*D-14 + 0.2*D-21'
    assert payload['selected_record']['data_status'] == 'forecast_only'
    assert payload['selected_record']['model_total'] is not None
    assert payload['selected_record']['fixed_total'] is not None
    assert payload['date_navigation']['prev_date'] == '2026-03-13'
    assert payload['upload_context']['workflow_actual_date'] == '2026-03-09'
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


def test_dashboard_supports_tianlang_site(patched_pipeline) -> None:
    response = request(main_module.app, 'GET', '/api/dashboard', params={'site': 'tianlang'})
    assert response.status_code == 200
    payload = response.json()

    assert payload['site_code'] == 'tianlang'
    assert payload['site_name'] == '天朗'
    assert payload['selected_date'] == '2026-03-15'
    assert payload['current_target_date'] == '2026-03-15'
    assert payload['max_actual_date'] == '2026-03-09'
    assert payload['upload_context']['workflow_actual_date'] == '2026-03-10'
    assert payload['accuracy']['selected_month']['model_metrics']['day_total_accuracy'] == 87.39
    assert payload['pipeline_files']['router_summary'] == 'pipeline/tianlang/results/best_d6_tianlang_2024fill_shared_summary.json'
    assert payload['pipeline_files']['current_forecast'] == '2026-03/forecast_d6_tianlang_20260315_2024fill_shared_20260315.csv'
    assert '2026-03-11' in payload['date_navigation']['available_dates']
    assert '2026-03-12' in payload['date_navigation']['available_dates']
    assert '2026-03-14' in payload['date_navigation']['available_dates']
    assert (patched_pipeline / 'tianlang' / 'results' / '2026-03' / 'forecast_d6_tianlang_20260311_2024fill_shared_20260311.csv').exists()
    assert (patched_pipeline / 'tianlang' / 'results' / '2026-03' / 'forecast_d6_tianlang_20260312_2024fill_shared_20260312.csv').exists()
    assert (patched_pipeline / 'tianlang' / 'results' / '2026-03' / 'forecast_d6_tianlang_20260314_2024fill_shared_20260314.csv').exists()
    assert any(site['code'] == 'huihua' for site in payload['available_sites'])


def test_upload_actual_generates_next_forecast(patched_pipeline) -> None:
    response = request(
        main_module.app,
        'POST',
        '/api/actuals',
        params={'site': 'huihua'},
        json={
            'actual_date': '2026-03-09',
            'values': build_values(1.6),
        },
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload['site_code'] == 'huihua'
    assert payload['actual_date'] == '2026-03-09'
    assert payload['next_target_date'] == '2026-03-15'
    assert (patched_pipeline / 'results' / 'forecast_d6_20260315.csv').exists()

    future_dashboard = request(main_module.app, 'GET', '/api/dashboard', params={'site': 'huihua', 'target_date': '2026-03-15'})
    assert future_dashboard.status_code == 200
    future_payload = future_dashboard.json()
    assert future_payload['selected_record']['data_status'] == 'forecast_only'
    assert future_payload['selected_record']['model_total'] is not None
    assert future_payload['date_navigation']['prev_date'] == '2026-03-14'

    history_dashboard = request(main_module.app, 'GET', '/api/dashboard', params={'site': 'huihua', 'target_date': '2026-03-09'})
    assert history_dashboard.status_code == 200
    history_payload = history_dashboard.json()
    assert history_payload['selected_record']['data_status'] == 'complete'
    assert history_payload['selected_record']['model_metrics']['day_total_accuracy'] is not None
    assert history_payload['accuracy']['matched_days'] >= 67


def test_upload_actual_allows_multiple_backfills_same_day(patched_pipeline, monkeypatch) -> None:
    monkeypatch.setattr(pipeline_service, '_today_local', lambda: date(2026, 3, 16))

    first_response = request(
        main_module.app,
        'POST',
        '/api/actuals',
        params={'site': 'huihua'},
        json={
            'actual_date': '2026-03-09',
            'values': build_values(1.6),
        },
    )
    assert first_response.status_code == 200
    first_payload = first_response.json()
    assert first_payload['next_target_date'] == '2026-03-15'

    mid_dashboard = request(main_module.app, 'GET', '/api/dashboard', params={'site': 'huihua'})
    assert mid_dashboard.status_code == 200
    mid_payload = mid_dashboard.json()
    assert mid_payload['max_actual_date'] == '2026-03-09'
    assert mid_payload['upload_context']['is_locked_today'] is False
    assert mid_payload['upload_context']['uploaded_today'] is True
    assert mid_payload['upload_context']['workflow_actual_date'] == '2026-03-10'
    assert mid_payload['upload_context']['badge_text'] == '可继续补传'

    second_response = request(
        main_module.app,
        'POST',
        '/api/actuals',
        params={'site': 'huihua'},
        json={
            'actual_date': '2026-03-10',
            'values': build_values(1.8),
        },
    )
    assert second_response.status_code == 200
    second_payload = second_response.json()
    assert second_payload['next_target_date'] == '2026-03-16'
    assert (patched_pipeline / 'results' / 'forecast_d6_20260316.csv').exists()

    second_dashboard = request(main_module.app, 'GET', '/api/dashboard', params={'site': 'huihua'})
    assert second_dashboard.status_code == 200
    second_dashboard_payload = second_dashboard.json()
    assert second_dashboard_payload['max_actual_date'] == '2026-03-10'
    assert second_dashboard_payload['upload_context']['is_locked_today'] is False
    assert second_dashboard_payload['upload_context']['workflow_actual_date'] == '2026-03-11'
    assert second_dashboard_payload['upload_context']['badge_text'] == '可继续补传'

    third_response = request(
        main_module.app,
        'POST',
        '/api/actuals',
        params={'site': 'huihua'},
        json={
            'actual_date': '2026-03-11',
            'values': build_values(1.9),
        },
    )
    assert third_response.status_code == 200
    third_payload = third_response.json()
    assert third_payload['next_target_date'] == '2026-03-17'
    assert (patched_pipeline / 'results' / 'forecast_d6_20260317.csv').exists()

    final_dashboard = request(main_module.app, 'GET', '/api/dashboard', params={'site': 'huihua'})
    assert final_dashboard.status_code == 200
    final_payload = final_dashboard.json()
    assert final_payload['max_actual_date'] == '2026-03-11'
    assert final_payload['upload_context']['is_locked_today'] is True
    assert final_payload['upload_context']['workflow_actual_date'] == '2026-03-11'
    assert final_payload['upload_context']['badge_text'] == '今日上传已完成'


def test_upload_actual_is_isolated_by_site(patched_pipeline, monkeypatch) -> None:
    monkeypatch.setattr(pipeline_service, '_today_local', lambda: date(2026, 3, 16))

    response = request(
        main_module.app,
        'POST',
        '/api/actuals',
        params={'site': 'tianlang'},
        json={
            'actual_date': '2026-03-10',
            'values': build_values(2.1),
        },
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload['site_code'] == 'tianlang'
    assert payload['next_target_date'] == '2026-03-16'
    assert (patched_pipeline / 'tianlang' / 'results' / '2026-03' / 'forecast_d6_tianlang_20260316_2024fill_shared_20260316.csv').exists()

    huihua_dashboard = request(main_module.app, 'GET', '/api/dashboard', params={'site': 'huihua'})
    tianlang_dashboard = request(main_module.app, 'GET', '/api/dashboard', params={'site': 'tianlang'})
    assert huihua_dashboard.status_code == 200
    assert tianlang_dashboard.status_code == 200
    assert huihua_dashboard.json()['max_actual_date'] == '2026-03-08'
    assert tianlang_dashboard.json()['max_actual_date'] == '2026-03-10'
    assert tianlang_dashboard.json()['pipeline_files']['current_forecast'] == '2026-03/forecast_d6_tianlang_20260316_2024fill_shared_20260316.csv'
    assert '2026-03-14' in tianlang_dashboard.json()['date_navigation']['available_dates']
