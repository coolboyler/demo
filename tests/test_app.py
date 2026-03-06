import json
from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

import app.main as main_module


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(main_module, "DATA_FILE", tmp_path / "store.json")
    monkeypatch.setattr(main_module, "DB_FILE", tmp_path / "store.sqlite3")
    with TestClient(main_module.app) as test_client:
        yield test_client


def build_values(base: float) -> list[float]:
    return [round(base + index * 1.5, 2) for index in range(24)]


def test_dashboard_loads(client: TestClient) -> None:
    response = client.get("/api/dashboard")
    assert response.status_code == 200
    payload = response.json()
    assert "selected_record" in payload
    assert "accuracy" in payload


def test_can_create_forecast_and_actual(client: TestClient) -> None:
    target = date.today() + timedelta(days=3)

    forecast_response = client.post(
        "/api/forecasts",
        json={
            "target_date": target.isoformat(),
            "values": build_values(800),
            "source": "pytest",
            "site_name": "测试站点",
            "note": "test",
        },
    )
    assert forecast_response.status_code == 200

    actual_response = client.post(
        "/api/actuals",
        json={
            "target_date": target.isoformat(),
            "values": build_values(790),
            "source": "pytest",
        },
    )
    assert actual_response.status_code == 200

    history_response = client.get("/api/history")
    assert history_response.status_code == 200
    items = history_response.json()["items"]
    assert any(item["target_date"] == target.isoformat() for item in items)

    dashboard_response = client.get("/api/dashboard", params={"target_date": target.isoformat()})
    assert dashboard_response.status_code == 200
    dashboard_payload = dashboard_response.json()
    assert dashboard_payload["selected_date"] == target.isoformat()
    assert dashboard_payload["selected_record"]["forecast_total"] is not None
    assert dashboard_payload["accuracy"]["selected_day"]["metrics"]["mape"] is not None
    assert dashboard_payload["selected_record"]["forecast_source"] == "pytest"
    assert dashboard_payload["selected_record"]["actual_source"] == "pytest"


def test_dashboard_uses_latest_forecast_per_day_for_daily_and_monthly_accuracy(client: TestClient) -> None:
    next_month_anchor = date.today().replace(day=28) + timedelta(days=10)
    target = next_month_anchor.replace(day=5)

    first_forecast = client.post(
        "/api/forecasts",
        json={
            "target_date": target.isoformat(),
            "values": build_values(810),
            "source": "pytest-first",
            "site_name": "测试站点",
            "generated_at": f"{target.isoformat()}T01:00:00",
        },
    )
    assert first_forecast.status_code == 200

    second_forecast_values = build_values(860)
    second_forecast = client.post(
        "/api/forecasts",
        json={
            "target_date": target.isoformat(),
            "values": second_forecast_values,
            "source": "pytest-second",
            "site_name": "测试站点",
            "generated_at": f"{target.isoformat()}T09:00:00",
        },
    )
    assert second_forecast.status_code == 200

    actual_response = client.post(
        "/api/actuals",
        json={
            "target_date": target.isoformat(),
            "values": build_values(855),
            "source": "pytest-actual",
        },
    )
    assert actual_response.status_code == 200

    dashboard_response = client.get("/api/dashboard", params={"target_date": target.isoformat()})
    assert dashboard_response.status_code == 200
    payload = dashboard_response.json()

    assert payload["selected_record"]["forecast_total"] == round(sum(second_forecast_values), 2)
    assert payload["accuracy"]["selected_month"]["days_count"] == 1
    assert len(payload["accuracy"]["selected_month_trend"]) == 1


def test_can_upload_forecast_and_actual_in_one_request(client: TestClient) -> None:
    target = date.today() + timedelta(days=4)

    response = client.post(
        "/api/load-records",
        json={
            "target_date": target.isoformat(),
            "forecast": {
                "values": build_values(860),
                "source": "pytest-forecast",
                "site_name": "组合测试站点",
                "note": "combined",
            },
            "actual": {
                "values": build_values(852),
                "source": "pytest-actual",
            },
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["stored"] == {"forecast": True, "actual": True}
    assert payload["forecast_id"] is not None
    assert payload["actual_id"] is not None
    assert payload["forecast_generated_at"] is not None
    assert payload["actual_updated_at"] is not None

    dashboard_response = client.get("/api/dashboard", params={"target_date": target.isoformat()})
    assert dashboard_response.status_code == 200
    dashboard_payload = dashboard_response.json()
    selected_record = dashboard_payload["selected_record"]

    assert selected_record["forecast_source"] == "pytest-forecast"
    assert selected_record["actual_source"] == "pytest-actual"
    assert selected_record["data_status"] == "complete"
    assert selected_record["peak_gap"] is not None
    assert dashboard_payload["site_name"] == "组合测试站点"


def test_combined_upload_requires_forecast_or_actual(client: TestClient) -> None:
    response = client.post(
        "/api/load-records",
        json={
            "target_date": (date.today() + timedelta(days=2)).isoformat(),
        },
    )
    assert response.status_code == 422


def test_rejects_wrong_hour_count(client: TestClient) -> None:
    response = client.post(
        "/api/forecasts",
        json={
            "target_date": (date.today() + timedelta(days=1)).isoformat(),
            "values": [1, 2, 3],
        },
    )
    assert response.status_code == 422


def test_migrates_legacy_json_store_to_sqlite(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(main_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(main_module, "DATA_FILE", tmp_path / "store.json")
    monkeypatch.setattr(main_module, "DB_FILE", tmp_path / "store.sqlite3")

    legacy_target = date.today().isoformat()
    legacy_store = {
        "site_name": "华海升园区",
        "forecasts": [
            {
                "id": "legacy-forecast",
                "target_date": legacy_target,
                "generated_at": f"{legacy_target}T09:00:00",
                "source": "api-demo",
                "site_name": "华海升园区",
                "note": "legacy",
                "created_by": "seed",
                "values": build_values(700),
            }
        ],
        "actuals": [
            {
                "id": "legacy-actual",
                "target_date": legacy_target,
                "updated_at": f"{legacy_target}T23:00:00",
                "source": "ems-demo",
                "values": build_values(690),
            }
        ],
    }
    (tmp_path / "store.json").write_text(json.dumps(legacy_store, ensure_ascii=False), encoding="utf-8")

    store = main_module.load_store()

    assert (tmp_path / "store.sqlite3").exists()
    assert store["site_name"] == "辉华"
    assert store["forecasts"][0]["site_name"] == "辉华"
    assert store["forecasts"][0]["source"] == "接口样例"
    assert store["actuals"][0]["source"] == "EMS样例"
