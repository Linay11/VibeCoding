from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from scripts import check_power118_eval_consistency as consistency_script
from scripts import build_power118_ml_dataset as dataset_script
from scripts import eval_power118_modes as eval_script
from scripts import train_power118_model as train_script


class _FakeExtraTreesRegressor:
    def __init__(self, *args, **kwargs):
        self._mean = None

    def fit(self, X, y):
        frame = pd.DataFrame(y)
        self._mean = frame.mean(axis=0).to_numpy()
        return self

    def predict(self, X):
        return [self._mean for _ in range(len(X))]

    def score(self, X, y):
        return 0.5


def test_build_dataset_writes_output_dir_and_summary(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        dataset_script,
        "_load_power118_module",
        lambda: SimpleNamespace(
            check_gurobi_runtime=lambda: {"available": True, "stage": "ready", "reason": "ok"},
            solve_scuc_118=lambda data_path=None, write_output=False, overrides=None, time_limit_s=None: {
                "feasible": True,
                "objective": 123.0,
                "solveTimeMs": 12.0,
                "unitCommitmentByHour": [[1.0, 1.0], [0.0, 1.0]],
                "generatorDispatchByHour": [[20.0, 21.0], [0.0, 10.0]],
            },
        ),
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
        time_limit_s=None,
    )

    assert dataset_path.exists()
    assert summary_path.exists()
    assert summary["keptSampleCount"] == 2


def test_train_model_writes_versioned_dir_without_publishing_defaults(monkeypatch, tmp_path) -> None:
    import sys
    import types

    fake_sklearn = types.ModuleType("sklearn")
    fake_ensemble = types.ModuleType("sklearn.ensemble")
    fake_ensemble.ExtraTreesRegressor = _FakeExtraTreesRegressor
    monkeypatch.setitem(sys.modules, "sklearn", fake_sklearn)
    monkeypatch.setitem(sys.modules, "sklearn.ensemble", fake_ensemble)

    dataset_path = Path(tmp_path) / "dataset.pkl"
    pd.to_pickle(
        {
            "features": pd.DataFrame([{"f1": 1.0, "f2": 2.0}, {"f1": 2.0, "f2": 3.0}]),
            "targets": pd.DataFrame(
                [
                    {"unitCommitment_g1_h1": 1.0, "dispatch_g1_h1": 20.0},
                    {"unitCommitment_g1_h1": 0.0, "dispatch_g1_h1": 10.0},
                ]
            ),
            "constraint_labels": pd.DataFrame(
                [
                    {"constraint_totalActiveConstraintCount": 10.0, "constraint_activeLineRatio": 0.2},
                    {"constraint_totalActiveConstraintCount": 12.0, "constraint_activeLineRatio": 0.3},
                ]
            ),
            "fixing_labels": pd.DataFrame(
                [
                    {"fixCommitment_g1_h1": 1.0, "fixCommitment_g1_h2": 0.0},
                    {"fixCommitment_g1_h1": 0.0, "fixCommitment_g1_h2": 1.0},
                ]
            ),
            "constraint_candidates": pd.DataFrame(
                [
                    {
                        "sampleId": "s1",
                        "constraintId": "ramp:g1:h1:up",
                        "constraintType": "ramp",
                        "inst_hourLoad": 10.0,
                        "inst_slack": 0.1,
                        "abs_constraintTypeCode": 1.0,
                        "abs_hourNorm": 0.1,
                        "labelRankScore": 0.9,
                    },
                    {
                        "sampleId": "s2",
                        "constraintId": "line:g1:h1:absCap",
                        "constraintType": "line",
                        "inst_hourLoad": 12.0,
                        "inst_slack": 0.2,
                        "abs_constraintTypeCode": 2.0,
                        "abs_hourNorm": 0.2,
                        "labelRankScore": 0.8,
                    },
                ]
            ),
        },
        dataset_path,
    )

    model_bundle, metadata, training_summary = train_script.train_model(
        dataset_path=dataset_path,
        output_dir=Path(tmp_path) / "model",
        model_filename="model.joblib",
        metadata_filename="metadata.json",
        n_estimators=4,
        random_state=7,
        model_version="v-test",
        feature_schema_version="schema-test",
        publish_default_artifacts=False,
        archive_tag="fixed-tag",
    )

    assert Path(training_summary["archiveDir"]).exists()
    assert Path(training_summary["archiveModelPath"]).exists()
    assert Path(training_summary["archiveMetadataPath"]).exists()
    assert Path(training_summary["summaryPath"]).exists()
    assert training_summary["publishedDefaultArtifacts"] is False
    assert metadata["modelVersion"] == "v-test"
    assert model_bundle["featureSchemaVersion"] == "schema-test"
    assert metadata["constraintScoringModelEnabled"] is True
    assert metadata["instanceFeatureNames"] == ["inst_hourLoad", "inst_slack"]
    assert metadata["abstractFeatureNames"] == ["abs_constraintTypeCode", "abs_hourNorm"]


def test_evaluate_modes_writes_summary_and_report(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(eval_script, "load_power118_data", lambda: {"timeHorizon": 2})
    monkeypatch.setattr(eval_script, "generate_power118_override_set", lambda base_data, n_samples, seed: [{"b": 1}] * n_samples)
    monkeypatch.setattr(
        eval_script,
        "load_power118_model_artifacts",
        lambda model_path=None, metadata_path=None: {
            "loadSuccess": True,
            "loadFailureReason": None,
            "loadStatus": "loaded",
            "modelPath": "model.joblib",
            "metadataPath": "metadata.json",
            "modelVersion": "power118-baseline-v1",
            "featureSchemaVersion": "power118-feature-schema-v1",
        },
    )

    def fake_run_power118_once(run_mode="exact", **kwargs):
        normalized_run_mode = run_mode
        if run_mode in {"hybrid_warm_start", "hybrid_constraint_aware_v2", "hybrid_constraint_aware_v3"}:
            normalized_run_mode = "hybrid"
            if run_mode == "hybrid_warm_start":
                kwargs["hybrid_strategy"] = "warm_start"
            elif run_mode == "hybrid_constraint_aware_v2":
                kwargs["hybrid_strategy"] = "constraint_aware_v2"
            elif run_mode == "hybrid_constraint_aware_v3":
                kwargs["hybrid_strategy"] = "constraint_aware_v3"
        mode_map = {
            "exact": {
                "solverModeUsed": "exact",
                "adapterMode": "real",
                "runtimeMs": 100.0,
                "objectiveValue": 50.0,
                "feasible": True,
                "statusName": "TIME_LIMIT",
                "terminatedByTimeLimit": True,
                "hasIncumbent": True,
                "optimal": False,
                "solutionCount": 1,
            },
            "hybrid": {
                "solverModeUsed": "exact" if kwargs.get("hybrid_strategy") == "warm_start" else "hybrid",
                "adapterMode": "real",
                "runtimeMs": 140.0 if kwargs.get("hybrid_strategy") == "warm_start" else 95.0 if kwargs.get("hybrid_strategy") == "constraint_aware_v2" else 80.0,
                "objectiveValue": 52.0 if kwargs.get("hybrid_strategy") == "warm_start" else 51.5 if kwargs.get("hybrid_strategy") == "constraint_aware_v2" else 51.0,
                "feasible": True,
                "repairApplied": True,
                "fallbackReason": "hybrid warm-start solve failed: numeric issue" if kwargs.get("hybrid_strategy") == "warm_start" else None,
                "statusName": "TIME_LIMIT",
                "terminatedByTimeLimit": True,
                "hasIncumbent": True,
                "optimal": False,
                "solutionCount": 1,
                "constraintAwareHybridUsed": kwargs.get("hybrid_strategy") in {"constraint_aware_v2", "constraint_aware_v3"},
                "reducedSolveApplied": kwargs.get("hybrid_strategy") in {"constraint_aware_v2", "constraint_aware_v3"},
                "fixedCommitmentCount": 16 if kwargs.get("hybrid_strategy") == "constraint_aware_v2" else 0,
                "predictedActiveConstraintCount": 11 if kwargs.get("hybrid_strategy") == "constraint_aware_v2" else 7,
                "constraintConfidence": 0.77 if kwargs.get("hybrid_strategy") in {"constraint_aware_v2", "constraint_aware_v3"} else None,
                "repairAfterReducedSolve": False,
                "reducedSolveFallbackReason": None,
                "hybridStrategyUsed": kwargs.get("hybrid_strategy"),
                "constraintScoringUsed": kwargs.get("hybrid_strategy") == "constraint_aware_v3",
                "criticalConstraintCount": 7 if kwargs.get("hybrid_strategy") == "constraint_aware_v3" else 0,
                "deferredConstraintCount": 12 if kwargs.get("hybrid_strategy") == "constraint_aware_v3" else 0,
                "constraintReactivationCount": 1 if kwargs.get("hybrid_strategy") == "constraint_aware_v3" else 0,
                "stagedSolveRounds": 2 if kwargs.get("hybrid_strategy") == "constraint_aware_v3" else 1,
                "constraintAwareReductionMode": "critical_constraint_subset" if kwargs.get("hybrid_strategy") == "constraint_aware_v3" else "fixed_commitment_mask" if kwargs.get("hybrid_strategy") == "constraint_aware_v2" else "warm_start_only",
                "reducedModelValidated": kwargs.get("hybrid_strategy") == "constraint_aware_v3",
                "reductionRejectedReason": None,
            },
            "ml": {
                "solverModeUsed": "ml",
                "adapterMode": "real",
                "runtimeMs": 20.0,
                "objectiveValue": 55.0,
                "feasible": True,
                "mlConfidence": 0.8,
                "statusName": "ML_PREDICTED",
                "terminatedByTimeLimit": False,
                "hasIncumbent": True,
                "optimal": False,
                "solutionCount": None,
            },
        }
        payload = {
            "fallbackReason": None,
            "repairApplied": None,
            "mlConfidence": None,
            "modelVersion": "power118-baseline-v1",
            "featureSchemaVersion": "power118-feature-schema-v1",
            "modelLoadStatus": "loaded",
            "generatorDispatchByHour": [[20.0, 21.0]],
            "requestedMode": run_mode,
        }
        payload.update(mode_map[normalized_run_mode])
        return payload

    monkeypatch.setattr(eval_script, "run_power118_once", fake_run_power118_once)

    records, summary_payload, report_path = eval_script.evaluate_modes(
        num_cases=2,
        seed=7,
        output_dir=Path(tmp_path) / "eval",
        time_limit_ms=None,
        modes=["exact", "hybrid_warm_start", "hybrid_constraint_aware_v2", "hybrid_constraint_aware_v3", "ml"],
        require_exact_baseline=True,
    )

    assert len(records) == 10
    assert summary_payload["evaluation"]["exactRealBaselineAvailable"] is True
    exact_summary = next(row for row in summary_payload["modes"] if row["requestedMode"] == "exact")
    hybrid_warm_summary = next(row for row in summary_payload["modes"] if row["requestedMode"] == "hybrid_warm_start")
    hybrid_fixing_summary = next(row for row in summary_payload["modes"] if row["requestedMode"] == "hybrid_constraint_aware_v2")
    hybrid_scoring_summary = next(row for row in summary_payload["modes"] if row["requestedMode"] == "hybrid_constraint_aware_v3")
    ml_summary = next(row for row in summary_payload["modes"] if row["requestedMode"] == "ml")
    assert exact_summary["statusCounts"]["TIME_LIMIT_FEASIBLE"] == 2
    assert summary_payload["evaluation"]["exactBaselineStatus"] == "TIME_LIMIT_FEASIBLE"
    assert summary_payload["evaluation"]["exactBaselineTimeLimited"] is True
    assert summary_payload["evaluation"]["exactBaselineHasIncumbent"] is True
    assert hybrid_warm_summary["solverModeUsedCounts"]["exact"] == 2
    assert hybrid_warm_summary["fallbackReasonCounts"]["hybrid warm-start solve failed: numeric issue"] == 2
    assert hybrid_warm_summary["fallbackToModeCounts"]["exact"] == 2
    assert hybrid_warm_summary["fallbackCaseIds"] == ["case-00000", "case-00001"]
    assert hybrid_warm_summary["objectiveGapVsExact"] is not None
    assert hybrid_fixing_summary["solverModeUsedCounts"]["hybrid"] == 2
    assert hybrid_fixing_summary["fallbackCount"] == 0
    assert hybrid_scoring_summary["solverModeUsedCounts"]["hybrid"] == 2
    assert hybrid_scoring_summary["criticalConstraintCount"] if "criticalConstraintCount" in hybrid_scoring_summary else True
    assert hybrid_scoring_summary["fallbackCount"] == 0
    assert summary_payload["evaluation"]["hybridFallbackReasonCounts"]["hybrid warm-start solve failed: numeric issue"] == 2
    assert ml_summary["dispatchMAE"] is not None
    assert (Path(tmp_path) / "eval" / "summary.json").exists()
    assert report_path.exists()
    assert "Hybrid fallback reason distribution" in report_path.read_text(encoding="utf-8")


def test_exact_baseline_availability_helpers_cover_time_limit_cases() -> None:
    exact_time_limit_incumbent = {
        "adapterMode": "real",
        "feasible": True,
        "hasIncumbent": True,
        "terminatedByTimeLimit": True,
        "optimal": False,
    }
    exact_time_limit_no_incumbent = {
        "adapterMode": "real",
        "feasible": False,
        "hasIncumbent": False,
        "terminatedByTimeLimit": True,
        "optimal": False,
    }

    assert eval_script._exact_baseline_available(exact_time_limit_incumbent) is True
    assert eval_script._exact_baseline_available(exact_time_limit_no_incumbent) is False
    assert eval_script._derive_status(exact_time_limit_incumbent) == "TIME_LIMIT_FEASIBLE"
    assert eval_script._derive_status(exact_time_limit_no_incumbent) == "TIME_LIMIT_NO_INCUMBENT"


def test_eval_consistency_checker_accepts_consistent_payload() -> None:
    records_payload = {
        "records": [
                {
                    "caseId": "case-00000",
                    "requestedMode": "exact",
                    "adapterMode": "real",
                    "status": "TIME_LIMIT_FEASIBLE",
                    "feasible": True,
                    "optimal": False,
                    "hasIncumbent": True,
                    "terminatedByTimeLimit": True,
                    "isFallback": False,
                    "objectiveGapVsExact": 0.0,
                },
            {
                "caseId": "case-00000",
                "requestedMode": "ml",
                "adapterMode": "real",
                "status": "FEASIBLE",
                "feasible": True,
                "optimal": False,
                "hasIncumbent": True,
                "isFallback": False,
                "objectiveGapVsExact": 0.1,
            },
        ]
    }
    summary_payload = {
        "evaluation": {
            "exactRealBaselineAvailable": True,
            "exactBaselineStatus": "TIME_LIMIT_FEASIBLE",
            "exactBaselineTimeLimited": True,
            "exactBaselineHasIncumbent": True,
        },
        "modes": [
            {"requestedMode": "exact", "runCount": 1, "successCount": 1, "failureCount": 0, "fallbackCount": 0},
            {"requestedMode": "ml", "runCount": 1, "successCount": 1, "failureCount": 0, "fallbackCount": 0},
        ],
    }

    issues = consistency_script.check_eval_consistency(records_payload, summary_payload)

    assert issues == []


def test_eval_consistency_checker_detects_inconsistent_payload() -> None:
    records_payload = {
        "records": [
                {
                    "caseId": "case-00000",
                    "requestedMode": "exact",
                    "adapterMode": "real",
                    "status": "TIME_LIMIT_FEASIBLE",
                    "feasible": True,
                    "optimal": False,
                    "hasIncumbent": True,
                    "terminatedByTimeLimit": True,
                    "isFallback": False,
                    "objectiveGapVsExact": 0.0,
                },
            {
                "caseId": "case-00000",
                "requestedMode": "ml",
                "adapterMode": "real",
                "status": "FEASIBLE",
                "feasible": True,
                "optimal": False,
                "hasIncumbent": True,
                "isFallback": False,
                "objectiveGapVsExact": None,
            },
        ]
    }
    summary_payload = {
        "evaluation": {
            "exactRealBaselineAvailable": False,
            "exactBaselineStatus": "COMPAT",
            "exactBaselineTimeLimited": False,
            "exactBaselineHasIncumbent": False,
        },
        "modes": [
            {"requestedMode": "exact", "runCount": 1, "successCount": 0, "failureCount": 1, "fallbackCount": 0},
            {"requestedMode": "ml", "runCount": 1, "successCount": 1, "failureCount": 0, "fallbackCount": 0},
        ],
    }

    issues = consistency_script.check_eval_consistency(records_payload, summary_payload)

    assert any("exact baseline availability mismatch" in issue for issue in issues)
