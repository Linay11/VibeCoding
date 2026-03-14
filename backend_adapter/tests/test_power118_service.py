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


def _loaded_artifacts(model_path=None, metadata_path=None) -> dict:
    return {
        "modelPath": "C:/tmp/power118_ml_model.joblib",
        "metadataPath": "C:/tmp/power118_ml_metadata.json",
        "loadSuccess": True,
        "loadFailureReason": None,
        "loadStatus": "loaded",
        "modelBundle": {"metrics": {}},
        "metadata": {
            "modelVersion": "power118-baseline-v1",
            "featureSchemaVersion": "power118-feature-schema-v1",
            "featureNames": ["f1"],
            "targetNames": ["t1"],
            "trainSampleCount": 12,
        },
        "modelVersion": "power118-baseline-v1",
        "featureSchemaVersion": "power118-feature-schema-v1",
    }


def test_power118_service_returns_compat_payload_with_runtime_reason(monkeypatch) -> None:
    fake_module = SimpleNamespace(
        check_gurobi_runtime=lambda: {
            "available": False,
            "stage": "import",
            "reason": "gurobipy import failed: No module named 'gurobipy'",
        },
        load_power118_data=lambda data_path, overrides=None: _preview_payload(),
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
    assert run["runtimeMs"] >= 0.0
    assert run["objectiveValue"] is None
    assert run["feasible"] is False


def test_power118_service_returns_real_payload_when_solver_succeeds(monkeypatch) -> None:
    fake_module = SimpleNamespace(
        check_gurobi_runtime=lambda: {
            "available": True,
            "stage": "ready",
            "reason": "gurobi runtime ready",
        },
        load_power118_data=lambda data_path, overrides=None: _preview_payload(),
        solve_scuc_118=lambda data_path, write_output=False, overrides=None, initial_unit_commitment=None, initial_dispatch=None, time_limit_s=None: {
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
    assert run["strategies"][0]["name"] == "SCUC-118 Exact Schedule"
    assert run["strategies"][0]["cost"] == 12345.67
    assert run["comparison"][0]["value"] == 1800.0
    assert run["trend"][1]["value"] == 120.0
    assert "objective is mapped to strategies[0].cost" in run["adapterNote"]
    assert "comparison is mapped from dispatch totals" in run["adapterNote"]
    assert run["solverModeUsed"] == "exact"
    assert run["runtimeMs"] == 456.7
    assert run["objectiveValue"] == 12345.67
    assert run["feasible"] is True
    assert run["modelLoadStatus"] == "not_requested"


def test_power118_service_returns_ml_payload_when_model_prediction_succeeds(monkeypatch) -> None:
    fake_module = SimpleNamespace(
        check_gurobi_runtime=lambda: {
            "available": False,
            "stage": "import",
            "reason": "gurobipy import failed: No module named 'gurobipy'",
        },
        load_power118_data=lambda data_path, overrides=None: _preview_payload(),
    )

    monkeypatch.setattr(power118_service, "_load_module", lambda: fake_module)
    monkeypatch.setattr(power118_service, "_power118_data", lambda: Path(__file__))
    monkeypatch.setattr(power118_service, "load_power118_model_artifacts", _loaded_artifacts)
    monkeypatch.setattr(
        power118_service,
        "_predict_with_model",
        lambda preview, model_artifacts: ({
            "feasible": True,
            "objective": 9876.5,
            "mlConfidence": 0.73,
            "repairApplied": True,
            "topGenerators": [
                {"label": "Gen 27", "value": 1800.0},
                {"label": "Gen 28", "value": 1750.0},
            ],
            "totalLoadByHour": [100.0, 120.0, 140.0],
            "modelVersion": "power118-baseline-v1",
            "featureSchemaVersion": "power118-feature-schema-v1",
        }, None),
    )

    run = power118_service.run_power118_once(run_mode="ml")

    assert run["adapterMode"] == "real"
    assert run["solverModeUsed"] == "ml"
    assert run["strategies"][0]["name"] == "SCUC-118 ML Schedule"
    assert run["mlConfidence"] == 0.73
    assert run["repairApplied"] is True
    assert run["modelVersion"] == "power118-baseline-v1"
    assert run["featureSchemaVersion"] == "power118-feature-schema-v1"
    assert run["modelLoadStatus"] == "loaded"
    assert run["fallbackReason"] is None
    assert run["objectiveValue"] == 9876.5
    assert run["feasible"] is True


def test_power118_service_returns_hybrid_payload_when_warm_start_succeeds(monkeypatch) -> None:
    fake_module = SimpleNamespace(
        check_gurobi_runtime=lambda: {
            "available": True,
            "stage": "ready",
            "reason": "gurobi runtime ready",
        },
        load_power118_data=lambda data_path, overrides=None: _preview_payload(),
        solve_scuc_118=lambda data_path, write_output=False, overrides=None, initial_unit_commitment=None, initial_dispatch=None, time_limit_s=None: {
            "runtime": {
                "available": True,
                "stage": "ready",
                "reason": "gurobi runtime ready",
            },
            "statusName": "OPTIMAL",
            "feasible": True,
            "objective": 123.4,
            "solveTimeMs": 22.2,
            "summary": _preview_payload()["summary"],
            "totalLoadByHour": [100.0, 120.0, 140.0],
            "topGenerators": [{"label": "Gen 27", "value": 1800.0}],
            "peakLineFlowByHour": [90.0, 95.0, 91.0],
            "warmStartUsed": initial_unit_commitment is not None or initial_dispatch is not None,
        },
    )

    monkeypatch.setattr(power118_service, "_load_module", lambda: fake_module)
    monkeypatch.setattr(power118_service, "_power118_data", lambda: Path(__file__))
    monkeypatch.setattr(power118_service, "load_power118_model_artifacts", _loaded_artifacts)
    monkeypatch.setattr(
        power118_service,
        "_predict_with_model",
        lambda preview, model_artifacts: ({
            "feasible": True,
            "objective": 120.0,
            "mlConfidence": 0.81,
            "repairApplied": False,
            "unitCommitmentByHour": [[1.0, 1.0, 1.0] for _ in range(54)],
            "generatorDispatchByHour": [[1.0, 1.0, 1.0] for _ in range(54)],
            "topGenerators": [{"label": "Gen 27", "value": 1800.0}],
            "totalLoadByHour": [100.0, 120.0, 140.0],
            "modelVersion": "power118-baseline-v1",
            "featureSchemaVersion": "power118-feature-schema-v1",
        }, None),
    )

    run = power118_service.run_power118_once(run_mode="hybrid")

    assert run["adapterMode"] == "real"
    assert run["solverModeUsed"] == "hybrid"
    assert run["mlConfidence"] == 0.81
    assert "solver received ML warm-start values" in run["adapterNote"]
    assert run["modelVersion"] == "power118-baseline-v1"
    assert run["featureSchemaVersion"] == "power118-feature-schema-v1"
    assert run["feasible"] is True


def test_power118_service_falls_back_when_model_schema_mismatches(monkeypatch) -> None:
    fake_module = SimpleNamespace(
        check_gurobi_runtime=lambda: {
            "available": False,
            "stage": "import",
            "reason": "gurobipy import failed: No module named 'gurobipy'",
        },
        load_power118_data=lambda data_path, overrides=None: _preview_payload(),
    )

    monkeypatch.setattr(power118_service, "_load_module", lambda: fake_module)
    monkeypatch.setattr(power118_service, "_power118_data", lambda: Path(__file__))
    monkeypatch.setattr(power118_service, "load_power118_model_artifacts", _loaded_artifacts)
    monkeypatch.setattr(
        power118_service,
        "_predict_with_model",
        lambda preview, model_artifacts: (None, "feature schema mismatch; missing=['hourlyLoad_1']"),
    )

    run = power118_service.run_power118_once(run_mode="ml", fallback_to_exact=False)

    assert run["adapterMode"] == "compat"
    assert run["solverModeUsed"] == "ml"
    assert "feature schema mismatch" in run["fallbackReason"]
    assert run["modelVersion"] == "power118-baseline-v1"
    assert run["featureSchemaVersion"] == "power118-feature-schema-v1"


def test_power118_service_falls_back_to_exact_when_model_missing(monkeypatch) -> None:
    fake_module = SimpleNamespace(
        check_gurobi_runtime=lambda: {
            "available": True,
            "stage": "ready",
            "reason": "gurobi runtime ready",
        },
        load_power118_data=lambda data_path, overrides=None: _preview_payload(),
        solve_scuc_118=lambda data_path, write_output=False, overrides=None, initial_unit_commitment=None, initial_dispatch=None, time_limit_s=None: {
            "runtime": {"available": True, "stage": "ready", "reason": "gurobi runtime ready"},
            "statusName": "OPTIMAL",
            "feasible": True,
            "objective": 111.0,
            "solveTimeMs": 12.0,
            "summary": _preview_payload()["summary"],
            "totalLoadByHour": [100.0, 120.0, 140.0],
            "topGenerators": [{"label": "Gen 27", "value": 1800.0}],
            "peakLineFlowByHour": [90.0, 95.0, 91.0],
        },
    )

    monkeypatch.setattr(power118_service, "_load_module", lambda: fake_module)
    monkeypatch.setattr(power118_service, "_power118_data", lambda: Path(__file__))
    monkeypatch.setattr(
        power118_service,
        "load_power118_model_artifacts",
        lambda model_path=None, metadata_path=None: {
            "modelPath": "C:/tmp/power118_ml_model.joblib",
            "metadataPath": "C:/tmp/power118_ml_metadata.json",
            "loadSuccess": False,
            "loadFailureReason": "model artifact not found: C:/tmp/power118_ml_model.joblib",
            "loadStatus": "failed",
            "modelBundle": None,
            "metadata": None,
            "modelVersion": None,
            "featureSchemaVersion": None,
        },
    )

    run = power118_service.run_power118_once(run_mode="hybrid", fallback_to_exact=True)

    assert run["adapterMode"] == "real"
    assert run["solverModeUsed"] == "exact"
    assert "model artifact not found" in run["fallbackReason"]
    assert run["modelLoadStatus"] == "failed"
