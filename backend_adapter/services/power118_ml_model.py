from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from backend_adapter.services.power118_dataset import build_power118_feature_frame


DEFAULT_MODEL_FILE = Path(__file__).resolve().parents[1] / "data" / "power118_ml_model.joblib"
DEFAULT_METADATA_FILE = Path(__file__).resolve().parents[1] / "data" / "power118_ml_metadata.json"
DEFAULT_MODEL_VERSION = "power118-baseline-v1"
DEFAULT_FEATURE_SCHEMA_VERSION = "power118-feature-schema-v1"
MODEL_PATH_ENV = "POWER118_ML_MODEL_PATH"
METADATA_PATH_ENV = "POWER118_ML_METADATA_PATH"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_power118_model_paths(
    model_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> tuple[Path, Path]:
    resolved_model_path = Path(model_path or os.getenv(MODEL_PATH_ENV) or DEFAULT_MODEL_FILE).resolve()
    resolved_metadata_path = Path(metadata_path or os.getenv(METADATA_PATH_ENV) or DEFAULT_METADATA_FILE).resolve()
    return resolved_model_path, resolved_metadata_path


def build_power118_metadata(
    feature_names: list[str],
    target_names: list[str],
    train_sample_count: int,
    model_version: str = DEFAULT_MODEL_VERSION,
    feature_schema_version: str = DEFAULT_FEATURE_SCHEMA_VERSION,
    trained_at: str | None = None,
) -> dict[str, Any]:
    return {
        "modelVersion": str(model_version),
        "featureSchemaVersion": str(feature_schema_version),
        "trainedAt": trained_at or _utc_now_iso(),
        "featureNames": list(feature_names),
        "targetNames": list(target_names),
        "trainSampleCount": int(train_sample_count),
    }


def write_power118_metadata_file(metadata: dict[str, Any], metadata_path: str | Path | None = None) -> Path:
    _, resolved_metadata_path = resolve_power118_model_paths(metadata_path=metadata_path)
    resolved_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return resolved_metadata_path


def _synthesized_metadata(model_bundle: dict[str, Any]) -> dict[str, Any]:
    metadata = model_bundle.get("metadata") if isinstance(model_bundle.get("metadata"), dict) else {}
    feature_names = metadata.get("featureNames") or list(model_bundle.get("feature_columns", []))
    target_names = metadata.get("targetNames") or (
        list(model_bundle.get("commitment_columns", [])) + list(model_bundle.get("dispatch_columns", []))
    )
    return {
        "modelVersion": metadata.get("modelVersion") or model_bundle.get("modelVersion") or None,
        "featureSchemaVersion": metadata.get("featureSchemaVersion") or model_bundle.get("featureSchemaVersion") or None,
        "trainedAt": metadata.get("trainedAt") or model_bundle.get("trainedAt") or None,
        "featureNames": list(feature_names),
        "targetNames": list(target_names),
        "trainSampleCount": int(metadata.get("trainSampleCount") or model_bundle.get("train_sample_count") or 0),
    }


def load_power118_model_artifacts(
    model_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> dict[str, Any]:
    resolved_model_path, resolved_metadata_path = resolve_power118_model_paths(model_path=model_path, metadata_path=metadata_path)
    artifacts = {
        "modelPath": str(resolved_model_path),
        "metadataPath": str(resolved_metadata_path),
        "loadSuccess": False,
        "loadFailureReason": None,
        "loadStatus": "failed",
        "modelBundle": None,
        "metadata": None,
        "modelVersion": None,
        "featureSchemaVersion": None,
    }

    if not resolved_model_path.exists():
        artifacts["loadFailureReason"] = f"model artifact not found: {resolved_model_path}"
        return artifacts

    try:
        model_bundle = joblib.load(resolved_model_path)
    except Exception as exc:
        artifacts["loadFailureReason"] = f"model artifact load failed: {exc}"
        return artifacts

    metadata: dict[str, Any] | None = None
    metadata_status = "loaded"
    if resolved_metadata_path.exists():
        try:
            metadata = json.loads(resolved_metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            artifacts["loadFailureReason"] = f"metadata load failed: {exc}"
            artifacts["modelBundle"] = model_bundle
            return artifacts
    else:
        metadata = _synthesized_metadata(model_bundle)
        metadata_status = "bundle_only"

    if not isinstance(model_bundle, dict):
        artifacts["loadFailureReason"] = "model artifact must deserialize to a dict bundle"
        return artifacts

    required_model_keys = {"commitment_model", "dispatch_model"}
    missing_keys = sorted(required_model_keys.difference(set(model_bundle.keys())))
    if missing_keys:
        artifacts["loadFailureReason"] = f"model bundle missing required keys: {', '.join(missing_keys)}"
        artifacts["modelBundle"] = model_bundle
        return artifacts

    feature_names = metadata.get("featureNames") if isinstance(metadata, dict) else None
    target_names = metadata.get("targetNames") if isinstance(metadata, dict) else None
    if not isinstance(feature_names, list) or not feature_names:
        artifacts["loadFailureReason"] = "metadata is missing featureNames"
        artifacts["modelBundle"] = model_bundle
        artifacts["metadata"] = metadata
        return artifacts
    if not isinstance(target_names, list) or not target_names:
        artifacts["loadFailureReason"] = "metadata is missing targetNames"
        artifacts["modelBundle"] = model_bundle
        artifacts["metadata"] = metadata
        return artifacts

    artifacts.update(
        {
            "loadSuccess": True,
            "loadFailureReason": None,
            "loadStatus": metadata_status,
            "modelBundle": model_bundle,
            "metadata": metadata,
            "modelVersion": metadata.get("modelVersion"),
            "featureSchemaVersion": metadata.get("featureSchemaVersion"),
        }
    )
    return artifacts


def load_power118_model_bundle(model_path: str | Path | None = None) -> dict[str, Any]:
    artifacts = load_power118_model_artifacts(model_path=model_path)
    if not artifacts["loadSuccess"]:
        raise FileNotFoundError(str(artifacts["loadFailureReason"]))
    return artifacts["modelBundle"]


def validate_power118_feature_schema(
    power_data: dict[str, Any],
    expected_feature_names: list[str],
) -> tuple[np.ndarray | None, str | None]:
    feature_frame = build_power118_feature_frame(power_data)
    actual_feature_names = list(feature_frame.columns)
    expected = list(expected_feature_names)

    if actual_feature_names != expected:
        missing = [name for name in expected if name not in actual_feature_names]
        unexpected = [name for name in actual_feature_names if name not in expected]
        order_mismatch = not missing and not unexpected and actual_feature_names != expected

        reason_parts = ["feature schema mismatch"]
        if missing:
            reason_parts.append(f"missing={missing[:5]}")
        if unexpected:
            reason_parts.append(f"unexpected={unexpected[:5]}")
        if order_mismatch:
            reason_parts.append("column order differs from training schema")
        return None, "; ".join(reason_parts)

    return feature_frame.to_numpy(dtype=float), None


def _reshape_prediction(values: np.ndarray, n_generators: int, horizon: int) -> list[list[float]]:
    reshaped = np.asarray(values, dtype=float).reshape(n_generators, horizon)
    return reshaped.tolist()


def _estimate_confidence(model_bundle: dict[str, Any]) -> float:
    metrics = model_bundle.get("metrics", {})
    commitment_score = max(0.0, float(metrics.get("commitment_train_r2", 0.0)))
    dispatch_score = max(0.0, float(metrics.get("dispatch_train_r2", 0.0)))
    confidence = 0.45 + 0.25 * min(commitment_score, 1.0) + 0.30 * min(dispatch_score, 1.0)
    return float(min(max(confidence, 0.05), 0.99))


def _repair_hourly_schedule(
    generators: list[dict[str, Any]],
    total_load_by_hour: list[float],
    unit_commitment: list[list[float]],
    dispatch: list[list[float]],
) -> tuple[list[list[float]], list[list[float]], bool, bool]:
    repaired_commitment = [row[:] for row in unit_commitment]
    repaired_dispatch = [row[:] for row in dispatch]
    repair_applied = False
    feasible = True

    num_generators = len(generators)
    horizon = len(total_load_by_hour)
    generators_by_capacity = sorted(range(num_generators), key=lambda idx: generators[idx]["pMax"], reverse=True)

    for hour in range(horizon):
        for gen_index, generator in enumerate(generators):
            repaired_commitment[gen_index][hour] = 1.0 if repaired_commitment[gen_index][hour] >= 0.5 else 0.0
            if repaired_commitment[gen_index][hour] < 0.5:
                if repaired_dispatch[gen_index][hour] != 0.0:
                    repair_applied = True
                repaired_dispatch[gen_index][hour] = 0.0
                continue

            min_output = float(generator["pMin"])
            max_output = float(generator["pMax"])
            clipped_value = float(np.clip(repaired_dispatch[gen_index][hour], min_output, max_output))
            if clipped_value != repaired_dispatch[gen_index][hour]:
                repair_applied = True
            repaired_dispatch[gen_index][hour] = clipped_value

        load = float(total_load_by_hour[hour])
        on_units = [idx for idx in range(num_generators) if repaired_commitment[idx][hour] >= 0.5]
        if not on_units and generators_by_capacity:
            largest = generators_by_capacity[0]
            repaired_commitment[largest][hour] = 1.0
            repaired_dispatch[largest][hour] = float(generators[largest]["pMin"])
            on_units = [largest]
            repair_applied = True

        total_dispatch = sum(repaired_dispatch[idx][hour] for idx in range(num_generators))
        if total_dispatch < load:
            remaining = load - total_dispatch
            for idx in generators_by_capacity:
                if remaining <= 1e-6:
                    break
                if repaired_commitment[idx][hour] < 0.5:
                    repaired_commitment[idx][hour] = 1.0
                    repaired_dispatch[idx][hour] = float(generators[idx]["pMin"])
                    on_units.append(idx)
                    remaining = max(0.0, remaining - repaired_dispatch[idx][hour])
                    repair_applied = True

            for idx in sorted(set(on_units), key=lambda item: generators[item]["pMax"], reverse=True):
                if remaining <= 1e-6:
                    break
                headroom = float(generators[idx]["pMax"]) - repaired_dispatch[idx][hour]
                if headroom <= 0:
                    continue
                added = min(headroom, remaining)
                repaired_dispatch[idx][hour] += added
                remaining -= added
                if added > 0:
                    repair_applied = True

            if remaining > 1e-3:
                feasible = False

        total_dispatch = sum(repaired_dispatch[idx][hour] for idx in range(num_generators))
        if total_dispatch > load:
            surplus = total_dispatch - load
            for idx in sorted(set(on_units), key=lambda item: repaired_dispatch[item][hour], reverse=True):
                if surplus <= 1e-6:
                    break
                min_output = float(generators[idx]["pMin"]) if repaired_commitment[idx][hour] >= 0.5 else 0.0
                reducible = repaired_dispatch[idx][hour] - min_output
                if reducible <= 0:
                    continue
                reduced = min(reducible, surplus)
                repaired_dispatch[idx][hour] -= reduced
                surplus -= reduced
                if reduced > 0:
                    repair_applied = True

            if surplus > 1e-3:
                feasible = False

        balanced_dispatch = sum(repaired_dispatch[idx][hour] for idx in range(num_generators))
        if abs(balanced_dispatch - load) > 1e-3:
            feasible = False

    return repaired_commitment, repaired_dispatch, repair_applied, feasible


def _estimate_objective(
    generators: list[dict[str, Any]],
    unit_commitment: list[list[float]],
    dispatch: list[list[float]],
) -> float:
    objective = 0.0
    num_generators = len(generators)
    horizon = len(dispatch[0]) if dispatch else 0
    for gen_index in range(num_generators):
        generator = generators[gen_index]
        previous_status = 0.0
        for hour in range(horizon):
            status = float(unit_commitment[gen_index][hour] >= 0.5)
            output = float(dispatch[gen_index][hour])
            objective += float(generator["a2"]) * output * output
            objective += float(generator["a1"]) * output
            objective += float(generator["a0"]) * status
            if status > previous_status:
                objective += float(generator["startCost"])
            elif previous_status > status:
                objective += float(generator["shutCost"])
            previous_status = status
    return float(objective)


def predict_power118_schedule(
    power_data: dict[str, Any],
    model_bundle: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = metadata or _synthesized_metadata(model_bundle)
    feature_array, schema_error = validate_power118_feature_schema(
        power_data=power_data,
        expected_feature_names=list(metadata.get("featureNames", [])),
    )
    if schema_error:
        raise ValueError(schema_error)

    commitment_model = model_bundle["commitment_model"]
    dispatch_model = model_bundle["dispatch_model"]

    raw_commitment = np.asarray(commitment_model.predict(feature_array)[0], dtype=float)
    raw_dispatch = np.asarray(dispatch_model.predict(feature_array)[0], dtype=float)

    generators = list(power_data.get("generators", []))
    horizon = int(power_data.get("timeHorizon", 24))
    num_generators = len(generators)

    unit_commitment = _reshape_prediction(raw_commitment, num_generators, horizon)
    dispatch = _reshape_prediction(raw_dispatch, num_generators, horizon)
    unit_commitment, dispatch, repair_applied, feasible = _repair_hourly_schedule(
        generators=generators,
        total_load_by_hour=list(power_data.get("totalLoadByHour", [])),
        unit_commitment=unit_commitment,
        dispatch=dispatch,
    )

    total_dispatch_by_generator = []
    for gen_index, generator in enumerate(generators):
        total_dispatch_by_generator.append(
            {
                "label": f"Gen {generator['genId']}",
                "value": float(sum(dispatch[gen_index])),
            }
        )

    return {
        "feasible": feasible,
        "objective": _estimate_objective(generators, unit_commitment, dispatch),
        "mlConfidence": _estimate_confidence(model_bundle),
        "repairApplied": repair_applied,
        "generatorDispatchByHour": dispatch,
        "unitCommitmentByHour": unit_commitment,
        "topGenerators": sorted(total_dispatch_by_generator, key=lambda item: item["value"], reverse=True)[:4],
        "totalLoadByHour": list(power_data.get("totalLoadByHour", [])),
        "summary": dict(power_data.get("summary", {})),
        "generatorCapacityPreview": list(power_data.get("generatorCapacityPreview", [])),
        "modelVersion": metadata.get("modelVersion"),
        "featureSchemaVersion": metadata.get("featureSchemaVersion"),
    }
