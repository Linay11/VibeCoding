from __future__ import annotations

import importlib

from fastapi.testclient import TestClient


def _make_client(monkeypatch, tmp_path: str) -> TestClient:
    monkeypatch.setenv("BACKEND_ADAPTER_STORE", tmp_path)
    monkeypatch.setenv("BACKEND_ADAPTER_CORS", "*")
    import backend_adapter.main as main

    importlib.reload(main)
    return TestClient(main.app)


def _assert_run_contract(run: dict, scenario_id: str) -> None:
    assert isinstance(run.get("runId"), str) and run["runId"]
    assert run.get("scenarioId") == scenario_id
    assert isinstance(run.get("generatedAt"), str) and run["generatedAt"]

    metrics = run.get("metrics")
    assert isinstance(metrics, dict)
    assert isinstance(metrics.get("solveTimeMs"), (int, float))
    assert isinstance(metrics.get("infeasibilityRate"), (int, float))
    assert isinstance(metrics.get("suboptimality"), (int, float))

    assert isinstance(run.get("strategies"), list)
    assert isinstance(run.get("trend"), list)
    assert isinstance(run.get("comparison"), list)

    assert run.get("adapterMode") in {"real", "compat"}
    assert isinstance(run.get("adapterNote"), str)


def test_get_scenarios_returns_stable_shape(monkeypatch, tmp_path) -> None:
    client = _make_client(monkeypatch, str(tmp_path / "latest_runs.json"))
    response = client.get("/api/scenarios")
    assert response.status_code == 200
    payload = response.json()
    scenarios = payload.get("scenarios")
    assert isinstance(scenarios, list)
    assert len(scenarios) > 0
    first = scenarios[0]
    assert isinstance(first.get("id"), str) and first["id"]
    assert isinstance(first.get("name"), str) and first["name"]
    assert isinstance(first.get("description"), str)


def test_post_run_rejects_unknown_scenario(monkeypatch, tmp_path) -> None:
    client = _make_client(monkeypatch, str(tmp_path / "latest_runs.json"))
    response = client.post("/api/runs", json={"scenarioId": "not-a-real-scenario"})
    assert response.status_code == 400
    payload = response.json()
    assert payload.get("error", {}).get("code") == "INVALID_ARGUMENT"


def test_latest_returns_404_when_empty(monkeypatch, tmp_path) -> None:
    client = _make_client(monkeypatch, str(tmp_path / "latest_runs.json"))
    response = client.get("/api/runs/latest", params={"scenarioId": "portfolio"})
    assert response.status_code == 404
    payload = response.json()
    assert payload.get("error", {}).get("code") == "NOT_FOUND"


def test_latest_requires_scenario_id(monkeypatch, tmp_path) -> None:
    client = _make_client(monkeypatch, str(tmp_path / "latest_runs.json"))
    response = client.get("/api/runs/latest")
    assert response.status_code == 400
    payload = response.json()
    assert payload.get("error", {}).get("code") == "INVALID_ARGUMENT"


def test_post_run_returns_stable_run_contract(monkeypatch, tmp_path) -> None:
    client = _make_client(monkeypatch, str(tmp_path / "latest_runs.json"))
    response = client.post("/api/runs", json={"scenarioId": "portfolio"})
    assert response.status_code == 200
    payload = response.json()
    run = payload.get("run")
    assert isinstance(run, dict)
    _assert_run_contract(run, "portfolio")


def test_latest_returns_stable_run_contract_after_post(monkeypatch, tmp_path) -> None:
    client = _make_client(monkeypatch, str(tmp_path / "latest_runs.json"))

    post_response = client.post("/api/runs", json={"scenarioId": "portfolio"})
    assert post_response.status_code == 200

    latest_response = client.get("/api/runs/latest", params={"scenarioId": "portfolio"})
    assert latest_response.status_code == 200
    payload = latest_response.json()
    run = payload.get("run")
    assert isinstance(run, dict)
    _assert_run_contract(run, "portfolio")


def test_power118_is_listed_and_returns_stable_run_contract(monkeypatch, tmp_path) -> None:
    client = _make_client(monkeypatch, str(tmp_path / "latest_runs.json"))

    scenarios_response = client.get("/api/scenarios")
    assert scenarios_response.status_code == 200
    scenarios = scenarios_response.json().get("scenarios", [])
    assert any(item.get("id") == "power-118" for item in scenarios)

    run_response = client.post("/api/runs", json={"scenarioId": "power-118"})
    assert run_response.status_code == 200
    run = run_response.json().get("run")
    assert isinstance(run, dict)
    _assert_run_contract(run, "power-118")
    assert run.get("adapterMode") in {"real", "compat"}
    assert "power-118" in run.get("adapterNote", "")
