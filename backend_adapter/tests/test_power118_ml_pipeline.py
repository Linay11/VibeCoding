from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np

from backend_adapter.services.power118_dataset import build_power118_feature_record, build_power118_target_record
from backend_adapter.services.power118_dataset import (
    build_power118_constraint_label_record,
    build_power118_fixing_label_record,
)
from backend_adapter.services.power118_ml_model import (
    build_power118_metadata,
    load_power118_model_artifacts,
    predict_power118_constraints,
    predict_power118_schedule,
    write_power118_metadata_file,
    validate_power118_feature_schema,
)
from scripts import build_power118_ml_dataset as dataset_script
from scripts.eval_power118_modes import build_eval_record, summarize_eval_records


class _DummyModel:
    def __init__(self, output_vector):
        self._output_vector = np.asarray(output_vector, dtype=float)

    def predict(self, X):
        return np.repeat(self._output_vector.reshape(1, -1), repeats=len(X), axis=0)


def _power_data() -> dict:
    generators = []
    for gen_id in range(1, 3):
        generators.append(
            {
                "genId": gen_id,
                "busIndex": gen_id - 1,
                "pMin": 10.0,
                "pMax": 100.0,
                "a2": 0.01,
                "a1": 2.0,
                "a0": 5.0,
                "rampUp": 50.0,
                "rampDown": 50.0,
                "startCost": 30.0,
                "shutCost": 0.0,
                "minUpTime": 1,
                "minDownTime": 1,
            }
        )

    return {
        "dataPath": "dummy.xls",
        "timeHorizon": 3,
        "bus": [1, 2],
        "branches": [{"capacity": 100.0, "x": 0.1}, {"capacity": 150.0, "x": 0.2}],
        "loadRows": [[1, 1, 30.0, 35.0, 32.0], [2, 2, 20.0, 22.0, 21.0]],
        "loadAtBus": [[30.0, 20.0], [35.0, 22.0], [32.0, 21.0]],
        "generators": generators,
        "summary": {"numBus": 2, "numLine": 2, "numGen": 2, "numLoad": 2, "peakLoad": 57.0, "totalDailyLoad": 160.0},
        "totalLoadByHour": [50.0, 57.0, 53.0],
        "generatorCapacityPreview": [{"label": "Gen 1", "value": 100.0}, {"label": "Gen 2", "value": 100.0}],
    }


def _load_solver_module():
    solver_path = Path(__file__).resolve().parents[2] / "external" / "power118" / "SCUC_118_new.py"
    spec = importlib.util.spec_from_file_location("external.power118.scuc_118_new_test", solver_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_power118_feature_record_contains_hourly_and_generator_features() -> None:
    record = build_power118_feature_record(_power_data())

    assert record["peakLoad"] == 57.0
    assert record["hourlyLoad_1"] == 50.0
    assert record["hourlyLoad_3"] == 53.0
    assert record["gen1_pMax"] == 100.0
    assert record["gen2_startCost"] == 30.0


def test_build_power118_target_record_flattens_commitment_and_dispatch() -> None:
    target = build_power118_target_record(
        {
            "objective": 123.0,
            "solveTimeMs": 9.0,
            "feasible": True,
            "unitCommitmentByHour": [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
            "generatorDispatchByHour": [[20.0, 0.0, 30.0], [0.0, 25.0, 23.0]],
        }
    )

    assert target["objective"] == 123.0
    assert target["unitCommitment_g1_h1"] == 1.0
    assert target["unitCommitment_g2_h2"] == 1.0
    assert target["dispatch_g2_h3"] == 23.0


def test_build_power118_constraint_and_fixing_labels() -> None:
    result = {
        "unitCommitmentByHour": [[1.0, 1.0, 1.0], [0.0, 1.0, 1.0]],
        "generatorDispatchByHour": [[20.0, 25.0, 30.0], [0.0, 25.0, 23.0]],
        "constraintDiagnostics": {
            "bindingConstraintCounts": {"generatorLimit": 2, "ramp": 1, "line": 1, "balance": 6, "total": 10},
            "generatorLimitActiveIndices": ["genLimit:g1:h1:pMax"],
            "rampActiveIndices": ["ramp:g1:h1:up"],
            "lineActiveIndices": ["line:g1:h1:absCap"],
            "balanceActiveIndices": ["balance:b1:h1"],
            "topTightConstraints": {"generatorLimit": [{"constraintId": "genLimit:g1:h1:pMax", "slack": 0.0}]},
            "constraintSlackSummary": {"generatorLimit": {"min": 0.0, "mean": 0.1, "max": 0.2}},
        },
        "activeGeneratorLimitCount": 2,
        "activeRampConstraintCount": 1,
        "activeLineConstraintCount": 1,
        "activeBalanceConstraintCount": 6,
    }

    constraint_record = build_power118_constraint_label_record(result)
    fixing_record = build_power118_fixing_label_record(result)

    assert constraint_record["constraintLabelAvailable"] == 1.0
    assert constraint_record["constraint_activeGeneratorLimitCount"] == 2.0
    assert "genLimit:g1:h1:pMax" in constraint_record["constraint_generatorLimitActiveIndicesJson"]
    assert "fixCommitment_g1_h2" in fixing_record


def test_predict_power118_schedule_repairs_and_scores_prediction() -> None:
    power_data = _power_data()
    feature_record = build_power118_feature_record(power_data)
    feature_columns = list(feature_record.keys())
    commitment_output = [1.0, 1.0, 1.0, 0.0, 1.0, 0.0]
    dispatch_output = [25.0, 25.0, 25.0, 15.0, 15.0, 15.0]
    model_bundle = {
        "feature_columns": feature_columns,
        "metrics": {"commitment_train_r2": 0.7, "dispatch_train_r2": 0.6},
        "commitment_model": _DummyModel(commitment_output),
        "dispatch_model": _DummyModel(dispatch_output),
    }

    metadata = build_power118_metadata(feature_columns, ["target-a"], train_sample_count=12)
    prediction = predict_power118_schedule(power_data, model_bundle, metadata=metadata)

    assert prediction["feasible"] is True
    assert prediction["mlConfidence"] > 0.0
    assert len(prediction["unitCommitmentByHour"]) == 2
    assert len(prediction["unitCommitmentByHour"][0]) == 3
    assert prediction["objective"] > 0.0
    assert prediction["modelVersion"] == metadata["modelVersion"]
    assert prediction["featureSchemaVersion"] == metadata["featureSchemaVersion"]


def test_predict_power118_constraints_returns_fixing_and_summary_fields() -> None:
    power_data = _power_data()
    feature_record = build_power118_feature_record(power_data)
    feature_columns = list(feature_record.keys())
    metadata = build_power118_metadata(feature_columns, ["target-a"], train_sample_count=12)
    metadata["constraintTargetNames"] = [f"fixCommitment_g{g}_h{h}" for g in range(1, 3) for h in range(1, 4)]
    metadata["constraintSummaryTargetNames"] = [
        "constraint_totalActiveConstraintCount",
        "constraint_activeLineRatio",
    ]
    metadata["constraintPredictionMode"] = "fixing-mask"
    model_bundle = {
        "feature_columns": feature_columns,
        "metrics": {"constraint_summary_train_r2": 0.7, "constraint_fixing_train_r2": 0.8},
        "constraint_summary_model": _DummyModel([12.0, 0.2]),
        "constraint_fixing_model": _DummyModel([0.9, 0.1, 0.8, 0.2, 0.9, 0.3]),
    }

    prediction = predict_power118_constraints(power_data, model_bundle, metadata=metadata)

    assert prediction["constraintModelEnabled"] is True
    assert prediction["predictedActiveConstraintCount"] == 12
    assert prediction["constraintConfidence"] > 0.0
    assert len(prediction["predictedFixedCommitmentMaskScores"]) == 2


def test_validate_power118_feature_schema_reports_mismatch() -> None:
    power_data = _power_data()
    feature_record = build_power118_feature_record(power_data)
    expected_feature_names = list(feature_record.keys())[:-1]

    feature_array, schema_error = validate_power118_feature_schema(power_data, expected_feature_names)

    assert feature_array is None
    assert schema_error is not None
    assert "feature schema mismatch" in schema_error


def test_eval_helpers_build_records_and_summary() -> None:
    exact_run = {
        "adapterMode": "real",
        "solverModeUsed": "exact",
        "feasible": True,
        "hasIncumbent": True,
        "optimal": False,
        "terminatedByTimeLimit": True,
        "runtimeMs": 100.0,
        "objectiveValue": 50.0,
        "fallbackReason": None,
        "repairApplied": None,
        "mlConfidence": None,
        "modelVersion": None,
        "featureSchemaVersion": None,
        "modelLoadStatus": "not_requested",
    }
    ml_run = {
        "adapterMode": "real",
        "solverModeUsed": "ml",
        "feasible": True,
        "hasIncumbent": True,
        "optimal": False,
        "terminatedByTimeLimit": False,
        "runtimeMs": 20.0,
        "objectiveValue": 55.0,
        "fallbackReason": None,
        "repairApplied": True,
        "mlConfidence": 0.8,
        "modelVersion": "power118-baseline-v1",
        "featureSchemaVersion": "power118-feature-schema-v1",
        "modelLoadStatus": "loaded",
    }

    record_exact = build_eval_record("case-00000", "exact", exact_run, exact_run)
    record_ml = build_eval_record("case-00000", "ml", ml_run, exact_run)
    summary = summarize_eval_records([record_exact, record_ml])

    assert record_ml["objectiveGapVsExact"] == 0.1
    assert record_ml["usedModelVersion"] == "power118-baseline-v1"
    assert len(summary) == 2
    assert json.loads(json.dumps(summary))[0]["runCount"] == 1


def test_load_power118_model_artifacts_reads_joblib_and_metadata(tmp_path) -> None:
    feature_names = list(build_power118_feature_record(_power_data()).keys())
    metadata = build_power118_metadata(feature_names, ["dispatch_g1_h1"], train_sample_count=3)
    model_path = Path(tmp_path) / "power118_ml_model.joblib"
    metadata_path = Path(tmp_path) / "power118_ml_metadata.json"
    joblib.dump(
        {
            "feature_columns": feature_names,
            "commitment_columns": ["unitCommitment_g1_h1"],
            "dispatch_columns": ["dispatch_g1_h1"],
            "commitment_model": _DummyModel([1.0] * 6),
            "dispatch_model": _DummyModel([10.0] * 6),
            "metadata": metadata,
        },
        model_path,
    )
    write_power118_metadata_file(metadata, metadata_path=metadata_path)

    artifacts = load_power118_model_artifacts(model_path=model_path, metadata_path=metadata_path)

    assert artifacts["loadSuccess"] is True
    assert artifacts["loadStatus"] == "loaded"
    assert artifacts["modelVersion"] == metadata["modelVersion"]
    assert artifacts["featureSchemaVersion"] == metadata["featureSchemaVersion"]


def test_solution_diagnostics_marks_time_limit_with_incumbent_feasible() -> None:
    solver_module = _load_solver_module()

    diagnostics = solver_module._solution_diagnostics("TIME_LIMIT", 1)

    assert diagnostics["feasible"] is True
    assert diagnostics["hasIncumbent"] is True
    assert diagnostics["terminatedByTimeLimit"] is True
    assert diagnostics["optimal"] is False


def test_solution_diagnostics_marks_time_limit_without_incumbent_infeasible() -> None:
    solver_module = _load_solver_module()

    diagnostics = solver_module._solution_diagnostics("TIME_LIMIT", 0)

    assert diagnostics["feasible"] is False
    assert diagnostics["hasIncumbent"] is False
    assert diagnostics["terminatedByTimeLimit"] is True


def test_solution_diagnostics_keeps_optimal_and_suboptimal_feasible() -> None:
    solver_module = _load_solver_module()

    optimal_diagnostics = solver_module._solution_diagnostics("OPTIMAL", 1)
    suboptimal_diagnostics = solver_module._solution_diagnostics("SUBOPTIMAL", 1)

    assert optimal_diagnostics["feasible"] is True
    assert optimal_diagnostics["optimal"] is True
    assert suboptimal_diagnostics["feasible"] is True
    assert suboptimal_diagnostics["optimal"] is False


def test_dataset_builder_keeps_time_limited_incumbent_samples(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        dataset_script,
        "_load_power118_module",
        lambda: type(
            "FakeModule",
            (),
            {
                "check_gurobi_runtime": staticmethod(lambda: {"available": True, "stage": "ready", "reason": "ok"}),
                "solve_scuc_118": staticmethod(
                    lambda data_path=None, write_output=False, overrides=None, time_limit_s=None: {
                        "feasible": True,
                        "statusName": "TIME_LIMIT",
                        "hasIncumbent": True,
                        "objective": 123.0,
                        "solveTimeMs": 20000.0,
                        "unitCommitmentByHour": [[1.0, 1.0], [0.0, 1.0]],
                        "generatorDispatchByHour": [[20.0, 21.0], [0.0, 10.0]],
                    }
                ),
            },
        )(),
    )
    monkeypatch.setattr(
        dataset_script,
        "load_power118_data",
        lambda data_path=None, overrides=None: {
            "dataPath": "dummy.xls",
            "timeHorizon": 2,
            "bus": [1, 2],
            "branches": [],
            "loadRows": [[1, 1, 30.0, 31.0]],
            "loadAtBus": [[30.0, 0.0], [31.0, 0.0]],
            "generators": [
                {
                    "genId": 1,
                    "busIndex": 0,
                    "pMin": 10.0,
                    "pMax": 100.0,
                    "a2": 0.01,
                    "a1": 2.0,
                    "a0": 3.0,
                    "rampUp": 40.0,
                    "rampDown": 40.0,
                    "startCost": 10.0,
                    "shutCost": 0.0,
                    "minUpTime": 1,
                    "minDownTime": 1,
                },
                {
                    "genId": 2,
                    "busIndex": 1,
                    "pMin": 5.0,
                    "pMax": 60.0,
                    "a2": 0.01,
                    "a1": 1.5,
                    "a0": 2.0,
                    "rampUp": 30.0,
                    "rampDown": 30.0,
                    "startCost": 8.0,
                    "shutCost": 0.0,
                    "minUpTime": 1,
                    "minDownTime": 1,
                },
            ],
            "summary": {"numBus": 2, "numLine": 0, "numGen": 2, "numLoad": 1, "peakLoad": 31.0, "totalDailyLoad": 61.0},
            "totalLoadByHour": [30.0, 31.0],
            "generatorCapacityPreview": [{"label": "Gen 1", "value": 100.0}],
        },
    )
    monkeypatch.setattr(dataset_script, "generate_power118_override_set", lambda base_data, n_samples, seed: [{"a": 1}] * n_samples)

    dataset_path, summary_path, summary = dataset_script.build_dataset(
        num_samples=2,
        seed=7,
        output_dir=Path(tmp_path) / "dataset",
        dataset_filename="dataset.pkl",
        time_limit_s=20.0,
    )

    assert dataset_path.exists()
    assert summary_path.exists()
    assert summary["keptSampleCount"] == 2
    assert summary["droppedNoIncumbentCount"] == 0
    assert summary["droppedInfeasibleCount"] == 0
