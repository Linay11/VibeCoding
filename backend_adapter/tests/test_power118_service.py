from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from backend_adapter.services import power118_service


def _preview_payload() -> dict:
    return {
        "summary": {
            "numBus": 118,
            "numLine": 186,
            "numGen": 54,
            "numLoad": 91,
            "peakLoad": 6109.99,
        },
        "totalLoadByHour": [100.0, 120.0, 140.0],
        "generatorCapacityPreview": [
            {"label": "Gen 27", "value": 420.0},
            {"label": "Gen 28", "value": 420.0},
            {"label": "Gen 11", "value": 350.0},
        ],
    }


def test_power118_service_returns_compat_payload_with_runtime_reason(monkeypatch) -> None:
    fake_module = SimpleNamespace(
        check_gurobi_runtime=lambda: {
            "available": False,
            "stage": "import",
            "reason": "gurobipy import failed: No module named 'gurobipy'",
        },
        load_power118_data=lambda data_path: _preview_payload(),
    )

    monkeypatch.setattr(power118_service, "_load_module", lambda: fake_module)
    monkeypatch.setattr(power118_service, "_power118_data", lambda: Path(__file__))

    run = power118_service.run_power118_once()

    assert run["scenarioId"] == "power-118"
    assert run["adapterMode"] == "compat"
    assert run["trend"] == [
        {"label": "H1", "value": 100.0},
        {"label": "H2", "value": 120.0},
        {"label": "H3", "value": 140.0},
    ]
    assert run["comparison"][0]["label"] == "Gen 27"
    assert "runtime blocked at import" in run["adapterNote"]
    assert "comparison is mapped from generator Pmax preview" in run["adapterNote"]


def test_power118_service_returns_real_payload_when_solver_succeeds(monkeypatch) -> None:
    fake_module = SimpleNamespace(
        check_gurobi_runtime=lambda: {
            "available": True,
            "stage": "ready",
            "reason": "gurobi runtime ready",
        },
        load_power118_data=lambda data_path: _preview_payload(),
        solve_scuc_118=lambda data_path, write_output=False: {
            "runtime": {
                "available": True,
                "stage": "ready",
                "reason": "gurobi runtime ready",
            },
            "statusName": "OPTIMAL",
            "feasible": True,
            "objective": 12345.67,
            "solveTimeMs": 456.7,
            "summary": _preview_payload()["summary"],
            "totalLoadByHour": [100.0, 120.0, 140.0],
            "topGenerators": [
                {"label": "Gen 27", "value": 1800.0},
                {"label": "Gen 28", "value": 1750.0},
                {"label": "Gen 11", "value": 1500.0},
            ],
            "peakLineFlowByHour": [90.0, 95.0, 91.0],
        },
    )

    monkeypatch.setattr(power118_service, "_load_module", lambda: fake_module)
    monkeypatch.setattr(power118_service, "_power118_data", lambda: Path(__file__))

    run = power118_service.run_power118_once()

    assert run["scenarioId"] == "power-118"
    assert run["adapterMode"] == "real"
    assert run["metrics"]["solveTimeMs"] == 456.7
    assert run["strategies"][0]["name"] == "SCUC-118 Schedule"
    assert run["strategies"][0]["cost"] == 12345.67
    assert run["comparison"][0]["value"] == 1800.0
    assert run["trend"][1]["value"] == 120.0
    assert "objective is mapped to strategies[0].cost" in run["adapterNote"]
    assert "comparison is mapped from solved generator dispatch totals" in run["adapterNote"]
