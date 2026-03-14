from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

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
        mode_map = {
            "exact": {"solverModeUsed": "exact", "adapterMode": "real", "runtimeMs": 100.0, "objectiveValue": 50.0, "feasible": True},
            "hybrid": {"solverModeUsed": "hybrid", "adapterMode": "real", "runtimeMs": 40.0, "objectiveValue": 52.0, "feasible": True, "repairApplied": True},
            "ml": {"solverModeUsed": "ml", "adapterMode": "real", "runtimeMs": 20.0, "objectiveValue": 55.0, "feasible": True, "mlConfidence": 0.8},
        }
        payload = {
            "fallbackReason": None,
            "repairApplied": None,
            "mlConfidence": None,
            "modelVersion": "power118-baseline-v1",
            "featureSchemaVersion": "power118-feature-schema-v1",
            "modelLoadStatus": "loaded",
            "generatorDispatchByHour": [[20.0, 21.0]],
        }
        payload.update(mode_map[run_mode])
        return payload

    monkeypatch.setattr(eval_script, "run_power118_once", fake_run_power118_once)

    records, summary_payload, report_path = eval_script.evaluate_modes(
        num_cases=2,
        seed=7,
        output_dir=Path(tmp_path) / "eval",
        time_limit_ms=None,
        modes=["exact", "hybrid", "ml"],
        require_exact_baseline=True,
    )

    assert len(records) == 6
    assert summary_payload["evaluation"]["exactRealBaselineAvailable"] is True
    assert (Path(tmp_path) / "eval" / "summary.json").exists()
    assert report_path.exists()
