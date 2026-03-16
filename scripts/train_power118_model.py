from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend_adapter.services.power118_ml_model import (
    CONSTRAINT_LABEL_SCHEMA_VERSION,
    DEFAULT_FEATURE_SCHEMA_VERSION,
    DEFAULT_METADATA_FILE,
    DEFAULT_MODEL_FILE,
    DEFAULT_MODEL_VERSION,
    build_power118_metadata,
    write_power118_metadata_file,
)


DEFAULT_DATASET_PATH = ROOT_DIR / "backend_adapter" / "data" / "power118_dataset" / "power118_ml_dataset.pkl"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "backend_adapter" / "data" / "power118_model"
DEFAULT_MODEL_FILENAME = "power118_ml_model.joblib"
DEFAULT_METADATA_FILENAME = "power118_ml_metadata.json"
DEFAULT_SUMMARY_FILENAME = "training_summary.json"
DEFAULT_CONSTRAINT_TRAINING_OBJECTIVE = "auto"
DEFAULT_EXACT_PRIORITY_MIN_COVERAGE = 0.7
DEFAULT_EXACT_PRIORITY_WEIGHT = 2.0
DEFAULT_CRITICAL_CLASSIFICATION_THRESHOLD = 0.5
DEFAULT_MODEL_VARIANT = "default"
DEFAULT_FEATURE_ABLATION_MODE = "inst+abs"
DEFAULT_CONSTRAINT_FEATURE_NAN_FILL_STRATEGY = "fillna(0.0)"


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _build_archive_dir(output_dir: Path, archive_tag: str | None) -> Path:
    return output_dir / (archive_tag or _utc_ts())


def _write_training_summary(output_dir: Path, summary: dict[str, Any]) -> Path:
    summary_path = output_dir / DEFAULT_SUMMARY_FILENAME
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def _normalize_binary_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    return numeric.ge(0.5).astype(int)


def _resolve_constraint_training_targets(
    constraint_candidates: pd.DataFrame,
    requested_objective: str = DEFAULT_CONSTRAINT_TRAINING_OBJECTIVE,
    exact_priority_min_coverage: float = DEFAULT_EXACT_PRIORITY_MIN_COVERAGE,
    exact_priority_weight: float = DEFAULT_EXACT_PRIORITY_WEIGHT,
) -> dict[str, Any]:
    frame = constraint_candidates.copy()
    row_count = int(len(frame))
    zero_labels = np.zeros(row_count, dtype=int)
    unit_weights = np.ones(row_count, dtype=float)

    exact_available_series = (
        pd.to_numeric(frame["labelCriticalExactAvailable"], errors="coerce").fillna(0.0)
        if "labelCriticalExactAvailable" in frame.columns
        else pd.Series([0.0] * row_count, index=frame.index, dtype=float)
    )
    exact_available_mask = exact_available_series.ge(0.5).to_numpy(dtype=bool)
    exact_label_series = (
        _normalize_binary_series(frame["labelCriticalExact"])
        if "labelCriticalExact" in frame.columns
        else pd.Series([0] * row_count, index=frame.index, dtype=int)
    )
    if "labelCriticalProxy" in frame.columns:
        proxy_label_series = _normalize_binary_series(frame["labelCriticalProxy"])
        proxy_label_source = "labelCriticalProxy"
    elif "labelCritical" in frame.columns:
        proxy_label_series = _normalize_binary_series(frame["labelCritical"])
        proxy_label_source = "labelCritical"
    elif "labelRankScore" in frame.columns:
        proxy_label_series = pd.to_numeric(frame["labelRankScore"], errors="coerce").fillna(0.0).ge(0.75).astype(int)
        proxy_label_source = "labelRankScore"
    else:
        proxy_label_series = pd.Series([0] * row_count, index=frame.index, dtype=int)
        proxy_label_source = "unavailable"
    rank_fallback_series = (
        pd.to_numeric(frame["labelRankScore"], errors="coerce").fillna(0.0).ge(0.75).astype(int)
        if "labelRankScore" in frame.columns
        else pd.Series([0] * row_count, index=frame.index, dtype=int)
    )

    exact_coverage = float(exact_available_series.mean()) if row_count > 0 else 0.0
    proxy_coverage = float(proxy_label_series.notna().mean()) if row_count > 0 else 0.0
    objective = str(requested_objective or DEFAULT_CONSTRAINT_TRAINING_OBJECTIVE).strip().lower()
    supported_objectives = {"auto", "proxy-only", "mixed", "exact-priority"}
    if objective not in supported_objectives:
        raise ValueError(f"Unsupported constraint training objective: {requested_objective}")
    if objective == "auto":
        if exact_coverage <= 0.0:
            objective = "proxy-only"
        elif exact_coverage >= float(max(exact_priority_min_coverage, 0.0)):
            objective = "exact-priority"
        else:
            objective = "mixed"

    exact_priority_weight = float(max(exact_priority_weight, 1.0))
    if row_count == 0:
        return {
            "resolvedObjective": objective,
            "requestedObjective": requested_objective,
            "yCritical": zero_labels,
            "sampleWeights": unit_weights,
            "exactLabelCoverage": exact_coverage,
            "proxyLabelCoverage": proxy_coverage,
            "exactSampleCount": 0,
            "proxySampleCount": 0,
            "exactTrainingRatio": 0.0,
            "proxyTrainingRatio": 0.0,
            "exactPriorityWeight": exact_priority_weight,
            "labelPreferenceOrder": ["labelCriticalExact", "labelCriticalProxy", "labelCritical"],
            "trainingLabelSourceCounts": {},
            "proxyLabelSource": proxy_label_source,
        }

    y_proxy = proxy_label_series.to_numpy(dtype=int)
    y_exact = exact_label_series.to_numpy(dtype=int)
    y_rank_fallback = rank_fallback_series.to_numpy(dtype=int)
    y_final = np.array(y_proxy, copy=True)
    source_array = np.array([proxy_label_source] * row_count, dtype=object)
    sample_weights = np.ones(row_count, dtype=float)

    if objective in {"mixed", "exact-priority"}:
        y_final = np.array(y_proxy, copy=True)
        y_final[exact_available_mask] = y_exact[exact_available_mask]
        source_array[exact_available_mask] = "labelCriticalExact"
        if objective == "exact-priority":
            sample_weights[exact_available_mask] = exact_priority_weight

    if proxy_label_source == "unavailable":
        missing_mask = ~exact_available_mask
        y_final[missing_mask] = y_rank_fallback[missing_mask]
        source_array[missing_mask] = "labelRankScore"

    y_final = np.clip(y_final, 0, 1).astype(int)
    exact_sample_count = int(np.sum(source_array == "labelCriticalExact"))
    proxy_sample_count = int(row_count - exact_sample_count)
    source_counts = pd.Series(source_array).value_counts(dropna=False).to_dict()

    return {
        "resolvedObjective": objective,
        "requestedObjective": requested_objective,
        "yCritical": y_final,
        "sampleWeights": sample_weights,
        "exactLabelCoverage": exact_coverage,
        "proxyLabelCoverage": proxy_coverage,
        "exactSampleCount": exact_sample_count,
        "proxySampleCount": proxy_sample_count,
        "exactTrainingRatio": float(exact_sample_count / max(row_count, 1)),
        "proxyTrainingRatio": float(proxy_sample_count / max(row_count, 1)),
        "exactPriorityWeight": exact_priority_weight,
        "labelPreferenceOrder": ["labelCriticalExact", "labelCriticalProxy", "labelCritical"],
        "trainingLabelSourceCounts": {str(key): int(value) for key, value in source_counts.items()},
        "proxyLabelSource": proxy_label_source,
    }


def _predict_positive_probability(model: Any, feature_array: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(feature_array), dtype=float)
        classes = np.asarray(getattr(model, "classes_", [0, 1]))
        if probabilities.ndim == 2 and probabilities.shape[1] == 1:
            positive_value = 1.0 if int(classes[0]) == 1 else 0.0
            return np.full(probabilities.shape[0], positive_value, dtype=float)
        if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            positive_index = int(np.where(classes == 1)[0][0]) if np.any(classes == 1) else probabilities.shape[1] - 1
            return np.clip(probabilities[:, positive_index], 0.0, 1.0)
    raw_prediction = np.asarray(model.predict(feature_array), dtype=float).reshape(-1)
    return np.clip(raw_prediction, 0.0, 1.0)


def _resolve_constraint_feature_subset(
    instance_feature_names: list[str],
    abstract_feature_names: list[str],
    requested_mode: str = DEFAULT_FEATURE_ABLATION_MODE,
) -> dict[str, Any]:
    mode = str(requested_mode or DEFAULT_FEATURE_ABLATION_MODE).strip().lower()
    supported_modes = {"inst-only", "abs-only", "inst+abs"}
    if mode not in supported_modes:
        raise ValueError(f"Unsupported feature ablation mode: {requested_mode}")

    selected_instance = list(instance_feature_names)
    selected_abstract = list(abstract_feature_names)
    effective_mode = mode
    fallback_reason = None

    if mode == "inst-only":
        selected_abstract = []
        if not selected_instance and abstract_feature_names:
            # No instance features found; fallback to abs-only while marking it explicitly.
            selected_instance = []
            selected_abstract = list(abstract_feature_names)
            effective_mode = "abs-only"
            fallback_reason = "inst-only requested but no inst_ features; fell back to abs-only"
    elif mode == "abs-only":
        selected_instance = []
        if not selected_abstract and instance_feature_names:
            selected_instance = list(instance_feature_names)
            selected_abstract = []
            effective_mode = "inst-only"
            fallback_reason = "abs-only requested but no abs_ features; fell back to inst-only"
    else:
        if not selected_instance and selected_abstract:
            effective_mode = "abs-only"
            fallback_reason = "inst+abs requested but inst_ features unavailable; fell back to abs-only"
        elif not selected_abstract and selected_instance:
            effective_mode = "inst-only"
            fallback_reason = "inst+abs requested but abs_ features unavailable; fell back to inst-only"

    return {
        "requestedMode": mode,
        "effectiveMode": effective_mode,
        "fallbackReason": fallback_reason,
        "instanceFeatureNames": selected_instance,
        "abstractFeatureNames": selected_abstract,
        "featureNames": selected_instance + selected_abstract,
    }


def _prepare_constraint_feature_frame(
    constraint_candidates: pd.DataFrame,
    feature_names: list[str],
    fill_strategy: str = DEFAULT_CONSTRAINT_FEATURE_NAN_FILL_STRATEGY,
) -> dict[str, Any]:
    if not feature_names:
        empty_frame = pd.DataFrame(index=constraint_candidates.index)
        return {
            "featureFrame": empty_frame,
            "featureNames": [],
            "nanColumns": [],
            "droppedColumns": [],
            "nanCountBefore": 0,
            "nanCountAfter": 0,
            "fillStrategy": str(fill_strategy),
        }

    feature_frame = constraint_candidates[feature_names].copy()
    nan_mask = feature_frame.isna()
    nan_columns = [str(column) for column in feature_frame.columns if bool(nan_mask[column].any())]
    dropped_columns = [str(column) for column in feature_frame.columns if bool(nan_mask[column].all())]
    nan_count_before = int(nan_mask.sum().sum())

    if dropped_columns:
        feature_frame = feature_frame.drop(columns=dropped_columns, errors="ignore")
    feature_names_effective = [str(name) for name in feature_names if str(name) not in set(dropped_columns)]

    if fill_strategy == "fillna(0.0)":
        feature_frame = feature_frame.fillna(0.0)
    else:
        # Backward-compatible fallback for unknown strategy tokens.
        feature_frame = feature_frame.fillna(0.0)
        fill_strategy = "fillna(0.0)"

    nan_count_after = int(feature_frame.isna().sum().sum())
    return {
        "featureFrame": feature_frame,
        "featureNames": feature_names_effective,
        "nanColumns": nan_columns,
        "droppedColumns": dropped_columns,
        "nanCountBefore": nan_count_before,
        "nanCountAfter": nan_count_after,
        "fillStrategy": str(fill_strategy),
    }


def train_model(
    dataset_path: Path,
    output_dir: Path,
    model_filename: str,
    metadata_filename: str,
    n_estimators: int,
    random_state: int,
    model_version: str,
    feature_schema_version: str,
    publish_default_artifacts: bool = True,
    archive_tag: str | None = None,
    constraint_training_objective: str = DEFAULT_CONSTRAINT_TRAINING_OBJECTIVE,
    exact_priority_min_coverage: float = DEFAULT_EXACT_PRIORITY_MIN_COVERAGE,
    exact_priority_weight: float = DEFAULT_EXACT_PRIORITY_WEIGHT,
    critical_classification_threshold: float = DEFAULT_CRITICAL_CLASSIFICATION_THRESHOLD,
    model_variant: str = DEFAULT_MODEL_VARIANT,
    feature_ablation_mode: str = DEFAULT_FEATURE_ABLATION_MODE,
) -> tuple[dict, dict, dict[str, Any]]:
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

    dataset_bundle = pd.read_pickle(dataset_path)
    features = dataset_bundle["features"]
    targets = dataset_bundle["targets"]
    constraint_labels = dataset_bundle.get("constraint_labels")
    constraint_candidates = dataset_bundle.get("constraint_candidates")
    fixing_labels = dataset_bundle.get("fixing_labels")

    commitment_columns = [column for column in targets.columns if column.startswith("unitCommitment_")]
    dispatch_columns = [column for column in targets.columns if column.startswith("dispatch_")]
    target_names = list(targets.columns)

    if not commitment_columns or not dispatch_columns:
        raise ValueError("Dataset bundle is missing commitment or dispatch target columns")

    X = features.to_numpy(dtype=float)
    y_commitment = targets[commitment_columns].to_numpy(dtype=float)
    y_dispatch = targets[dispatch_columns].to_numpy(dtype=float)

    commitment_model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    dispatch_model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        random_state=random_state + 1,
        n_jobs=-1,
    )

    commitment_model.fit(X, y_commitment)
    dispatch_model.fit(X, y_dispatch)

    commitment_pred = commitment_model.predict(X)
    dispatch_pred = dispatch_model.predict(X)

    commitment_mae = float(abs(commitment_pred - y_commitment).mean())
    dispatch_mae = float(abs(dispatch_pred - y_dispatch).mean())
    dispatch_mean = float(abs(y_dispatch).mean() or 1.0)

    constraint_summary_model = None
    constraint_fixing_model = None
    constraint_scoring_model = None
    constraint_ranking_aux_model = None
    constraint_summary_columns: list[str] = []
    constraint_fixing_columns: list[str] = []
    instance_feature_names_all: list[str] = []
    abstract_feature_names_all: list[str] = []
    instance_feature_names: list[str] = []
    abstract_feature_names: list[str] = []
    feature_ablation_info = {
        "requestedMode": str(feature_ablation_mode),
        "effectiveMode": str(feature_ablation_mode),
        "fallbackReason": None,
        "featureNames": [],
    }
    constraint_feature_nan_info = {
        "nanColumns": [],
        "fillStrategy": DEFAULT_CONSTRAINT_FEATURE_NAN_FILL_STRATEGY,
        "droppedColumns": [],
        "nanCountBefore": 0,
        "nanCountAfter": 0,
    }
    constraint_metrics: dict[str, float] = {}
    constraint_scoring_target_name = "labelCritical"
    constraint_training_info = {
        "resolvedObjective": "proxy-only",
        "requestedObjective": str(constraint_training_objective),
        "exactLabelCoverage": 0.0,
        "proxyLabelCoverage": 0.0,
        "exactSampleCount": 0,
        "proxySampleCount": 0,
        "exactTrainingRatio": 0.0,
        "proxyTrainingRatio": 0.0,
        "exactPriorityWeight": float(max(exact_priority_weight, 1.0)),
        "labelPreferenceOrder": ["labelCriticalExact", "labelCriticalProxy", "labelCritical"],
        "trainingLabelSourceCounts": {},
        "proxyLabelSource": "unavailable",
    }
    critical_classification_threshold = float(min(max(critical_classification_threshold, 0.0), 1.0))

    if isinstance(constraint_labels, pd.DataFrame) and not constraint_labels.empty:
        constraint_summary_columns = [
            column
            for column in constraint_labels.columns
            if column.startswith("constraint_") and not column.endswith("Json")
        ]
        if constraint_summary_columns:
            y_constraint_summary = constraint_labels[constraint_summary_columns].to_numpy(dtype=float)
            constraint_summary_model = ExtraTreesRegressor(
                n_estimators=n_estimators,
                random_state=random_state + 2,
                n_jobs=-1,
            )
            constraint_summary_model.fit(X, y_constraint_summary)
            constraint_summary_pred = constraint_summary_model.predict(X)
            constraint_metrics["constraint_summary_train_r2"] = float(
                constraint_summary_model.score(X, y_constraint_summary)
            )
            constraint_metrics["constraint_summary_train_mae"] = float(
                abs(constraint_summary_pred - y_constraint_summary).mean()
            )

    if isinstance(fixing_labels, pd.DataFrame) and not fixing_labels.empty:
        constraint_fixing_columns = [
            column
            for column in fixing_labels.columns
            if column.startswith("fixCommitment_")
        ]
        if constraint_fixing_columns:
            y_constraint_fixing = fixing_labels[constraint_fixing_columns].to_numpy(dtype=float)
            constraint_fixing_model = ExtraTreesRegressor(
                n_estimators=n_estimators,
                random_state=random_state + 3,
                n_jobs=-1,
            )
            constraint_fixing_model.fit(X, y_constraint_fixing)
            constraint_fixing_pred = constraint_fixing_model.predict(X)
            constraint_metrics["constraint_fixing_train_r2"] = float(
                constraint_fixing_model.score(X, y_constraint_fixing)
            )
            constraint_metrics["constraint_fixing_train_mae"] = float(
                abs(constraint_fixing_pred - y_constraint_fixing).mean()
            )

    if isinstance(constraint_candidates, pd.DataFrame) and not constraint_candidates.empty:
        instance_feature_names_all = [
            column for column in constraint_candidates.columns if column.startswith("inst_")
        ]
        abstract_feature_names_all = [
            column for column in constraint_candidates.columns if column.startswith("abs_")
        ]
        feature_ablation_info = _resolve_constraint_feature_subset(
            instance_feature_names=instance_feature_names_all,
            abstract_feature_names=abstract_feature_names_all,
            requested_mode=feature_ablation_mode,
        )
        instance_feature_names = list(feature_ablation_info["instanceFeatureNames"])
        abstract_feature_names = list(feature_ablation_info["abstractFeatureNames"])
        constraint_feature_names = list(feature_ablation_info["featureNames"])
        if constraint_feature_names:
            prepared_features = _prepare_constraint_feature_frame(
                constraint_candidates=constraint_candidates,
                feature_names=constraint_feature_names,
                fill_strategy=DEFAULT_CONSTRAINT_FEATURE_NAN_FILL_STRATEGY,
            )
            constraint_feature_nan_info = {
                "nanColumns": list(prepared_features["nanColumns"]),
                "fillStrategy": str(prepared_features["fillStrategy"]),
                "droppedColumns": list(prepared_features["droppedColumns"]),
                "nanCountBefore": int(prepared_features["nanCountBefore"]),
                "nanCountAfter": int(prepared_features["nanCountAfter"]),
            }
            dropped_columns = set(constraint_feature_nan_info["droppedColumns"])
            constraint_feature_names = list(prepared_features["featureNames"])
            if dropped_columns:
                instance_feature_names = [
                    str(name) for name in instance_feature_names
                    if str(name) not in dropped_columns
                ]
                abstract_feature_names = [
                    str(name) for name in abstract_feature_names
                    if str(name) not in dropped_columns
                ]
            if not constraint_feature_names:
                constraint_metrics["constraint_feature_all_dropped"] = 1.0
            else:
                X_constraint = prepared_features["featureFrame"].to_numpy(dtype=float)
            constraint_training_info = _resolve_constraint_training_targets(
                constraint_candidates=constraint_candidates,
                requested_objective=constraint_training_objective,
                exact_priority_min_coverage=exact_priority_min_coverage,
                exact_priority_weight=exact_priority_weight,
            )
            if constraint_feature_names:
                y_constraint_critical = np.asarray(constraint_training_info["yCritical"], dtype=int)
                sample_weights = np.asarray(constraint_training_info["sampleWeights"], dtype=float)
                constraint_scoring_model = ExtraTreesClassifier(
                    n_estimators=n_estimators,
                    random_state=random_state + 4,
                    n_jobs=-1,
                )
                constraint_scoring_model.fit(X_constraint, y_constraint_critical, sample_weight=sample_weights)
                critical_probability = _predict_positive_probability(constraint_scoring_model, X_constraint)
                critical_prediction = (critical_probability >= critical_classification_threshold).astype(int)
                constraint_metrics["constraint_critical_train_accuracy"] = float(
                    np.mean(critical_prediction == y_constraint_critical)
                )
                constraint_metrics["constraint_critical_train_brier"] = float(
                    np.mean((critical_probability - y_constraint_critical) ** 2)
                )
                constraint_metrics["constraint_critical_positive_rate"] = float(
                    np.mean(y_constraint_critical)
                )
                constraint_metrics["constraint_scoring_train_r2"] = float(
                    constraint_scoring_model.score(X_constraint, y_constraint_critical)
                )
                if "labelRankScore" in constraint_candidates.columns:
                    y_constraint_rank = pd.to_numeric(
                        constraint_candidates["labelRankScore"],
                        errors="coerce",
                    ).fillna(0.0).to_numpy(dtype=float)
                    constraint_ranking_aux_model = ExtraTreesRegressor(
                        n_estimators=n_estimators,
                        random_state=random_state + 5,
                        n_jobs=-1,
                    )
                    constraint_ranking_aux_model.fit(X_constraint, y_constraint_rank)
                    rank_pred = np.asarray(constraint_ranking_aux_model.predict(X_constraint), dtype=float)
                    constraint_metrics["constraint_rank_aux_train_r2"] = float(
                        constraint_ranking_aux_model.score(X_constraint, y_constraint_rank)
                    )
                    constraint_metrics["constraint_rank_aux_train_mae"] = float(
                        np.mean(np.abs(rank_pred - y_constraint_rank))
                    )

    metadata = build_power118_metadata(
        feature_names=list(features.columns),
        target_names=target_names,
        train_sample_count=len(features),
        model_version=model_version,
        feature_schema_version=feature_schema_version,
    )
    metadata["constraintModelEnabled"] = bool(constraint_summary_model is not None or constraint_fixing_model is not None)
    metadata["constraintLabelSchemaVersion"] = CONSTRAINT_LABEL_SCHEMA_VERSION
    metadata["constraintTargetNames"] = constraint_fixing_columns
    metadata["constraintSummaryTargetNames"] = constraint_summary_columns
    metadata["constraintPredictionMode"] = "fixing-mask"
    metadata["constraintRepresentationVersion"] = "power118-constraint-repr-v3"
    metadata["constraintScoringModelEnabled"] = bool(constraint_scoring_model is not None)
    metadata["constraintScoringMode"] = "critical-first-classification"
    metadata["constraintScoringTargetName"] = constraint_scoring_target_name
    metadata["modelVariant"] = str(model_variant or DEFAULT_MODEL_VARIANT)
    metadata["featureAblationMode"] = str(feature_ablation_info["requestedMode"])
    metadata["featureAblationModeEffective"] = str(feature_ablation_info["effectiveMode"])
    metadata["featureAblationFallbackReason"] = feature_ablation_info["fallbackReason"]
    metadata["constraintTrainingObjective"] = str(constraint_training_info["resolvedObjective"])
    metadata["constraintTrainingObjectiveRequested"] = str(constraint_training_info["requestedObjective"])
    metadata["constraintLabelPreferenceOrder"] = list(constraint_training_info["labelPreferenceOrder"])
    metadata["exactLabelCoverage"] = float(constraint_training_info["exactLabelCoverage"])
    metadata["proxyLabelCoverage"] = float(constraint_training_info["proxyLabelCoverage"])
    metadata["exactLabelSampleCount"] = int(constraint_training_info["exactSampleCount"])
    metadata["proxyLabelSampleCount"] = int(constraint_training_info["proxySampleCount"])
    metadata["exactTrainingRatio"] = float(constraint_training_info["exactTrainingRatio"])
    metadata["proxyTrainingRatio"] = float(constraint_training_info["proxyTrainingRatio"])
    metadata["constraintTrainingLabelSourceCounts"] = dict(constraint_training_info["trainingLabelSourceCounts"])
    metadata["constraintProxyLabelSource"] = str(constraint_training_info["proxyLabelSource"])
    metadata["constraintExactPriorityWeight"] = float(constraint_training_info["exactPriorityWeight"])
    metadata["criticalClassificationThreshold"] = float(critical_classification_threshold)
    metadata["constraintCriticalFirstModel"] = bool(constraint_scoring_model is not None)
    metadata["constraintAuxRankingModelEnabled"] = bool(constraint_ranking_aux_model is not None)
    metadata["constraintAuxRankingTargetName"] = "labelRankScore"
    metadata["instanceFeatureNamesAll"] = list(instance_feature_names_all)
    metadata["abstractFeatureNamesAll"] = list(abstract_feature_names_all)
    metadata["instanceFeatureNames"] = instance_feature_names
    metadata["abstractFeatureNames"] = abstract_feature_names
    metadata["constraintFeatureNaNColumns"] = list(constraint_feature_nan_info["nanColumns"])
    metadata["constraintFeatureNaNFillStrategy"] = str(constraint_feature_nan_info["fillStrategy"])
    metadata["constraintFeatureDroppedColumns"] = list(constraint_feature_nan_info["droppedColumns"])
    metadata["constraintFeatureNaNCountBefore"] = int(constraint_feature_nan_info["nanCountBefore"])
    metadata["constraintFeatureNaNCountAfter"] = int(constraint_feature_nan_info["nanCountAfter"])
    model_bundle = {
        "feature_columns": list(features.columns),
        "commitment_columns": commitment_columns,
        "dispatch_columns": dispatch_columns,
        "commitment_model": commitment_model,
        "dispatch_model": dispatch_model,
        "constraint_summary_model": constraint_summary_model,
        "constraint_fixing_model": constraint_fixing_model,
        "constraint_scoring_model": constraint_scoring_model,
        "constraint_ranking_aux_model": constraint_ranking_aux_model,
        "constraint_summary_columns": constraint_summary_columns,
        "constraint_fixing_columns": constraint_fixing_columns,
        "model_variant": metadata["modelVariant"],
        "feature_ablation_mode": metadata["featureAblationMode"],
        "feature_ablation_mode_effective": metadata["featureAblationModeEffective"],
        "instance_feature_names": instance_feature_names,
        "abstract_feature_names": abstract_feature_names,
        "instance_feature_names_all": instance_feature_names_all,
        "abstract_feature_names_all": abstract_feature_names_all,
        "critical_classification_threshold": critical_classification_threshold,
        "constraint_training_objective": metadata["constraintTrainingObjective"],
        "constraint_feature_nan_columns": metadata["constraintFeatureNaNColumns"],
        "constraint_feature_nan_fill_strategy": metadata["constraintFeatureNaNFillStrategy"],
        "constraint_feature_dropped_columns": metadata["constraintFeatureDroppedColumns"],
        "constraint_feature_nan_count_before": metadata["constraintFeatureNaNCountBefore"],
        "constraint_feature_nan_count_after": metadata["constraintFeatureNaNCountAfter"],
        "metrics": {
            "commitment_train_r2": float(commitment_model.score(X, y_commitment)),
            "dispatch_train_r2": float(dispatch_model.score(X, y_dispatch)),
            "commitment_train_mae": commitment_mae,
            "dispatch_train_mae": dispatch_mae,
            "dispatch_train_mae_ratio": float(dispatch_mae / dispatch_mean),
            **constraint_metrics,
        },
        "dataset_path": str(dataset_path),
        "metadata": metadata,
        "modelVersion": model_version,
        "featureSchemaVersion": feature_schema_version,
        "train_sample_count": len(features),
        "trainedAt": metadata["trainedAt"],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = _build_archive_dir(output_dir, archive_tag)
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_model_path = archive_dir / model_filename
    archive_metadata_path = archive_dir / metadata_filename

    joblib.dump(model_bundle, archive_model_path)
    write_power118_metadata_file(metadata, metadata_path=archive_metadata_path)

    published_model_path = DEFAULT_MODEL_FILE.resolve()
    published_metadata_path = DEFAULT_METADATA_FILE.resolve()
    if publish_default_artifacts:
        published_model_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(archive_model_path, published_model_path)
        shutil.copy2(archive_metadata_path, published_metadata_path)

    training_summary = {
        "datasetPath": str(dataset_path),
        "archiveDir": str(archive_dir),
        "archiveModelPath": str(archive_model_path),
        "archiveMetadataPath": str(archive_metadata_path),
        "publishedModelPath": str(published_model_path) if publish_default_artifacts else None,
        "publishedMetadataPath": str(published_metadata_path) if publish_default_artifacts else None,
        "publishedDefaultArtifacts": bool(publish_default_artifacts),
        "modelVersion": model_version,
        "modelVariant": metadata["modelVariant"],
        "featureAblationMode": metadata["featureAblationMode"],
        "featureAblationModeEffective": metadata["featureAblationModeEffective"],
        "featureSchemaVersion": feature_schema_version,
        "trainSampleCount": len(features),
        "seed": random_state,
        "nEstimators": n_estimators,
        "constraintTrainingObjective": metadata["constraintTrainingObjective"],
        "constraintTrainingObjectiveRequested": metadata["constraintTrainingObjectiveRequested"],
        "exactLabelCoverage": metadata["exactLabelCoverage"],
        "proxyLabelCoverage": metadata["proxyLabelCoverage"],
        "exactTrainingRatio": metadata["exactTrainingRatio"],
        "proxyTrainingRatio": metadata["proxyTrainingRatio"],
        "constraintFeatureNaNColumns": metadata["constraintFeatureNaNColumns"],
        "constraintFeatureNaNFillStrategy": metadata["constraintFeatureNaNFillStrategy"],
        "constraintFeatureDroppedColumns": metadata["constraintFeatureDroppedColumns"],
        "constraintFeatureNaNCountBefore": metadata["constraintFeatureNaNCountBefore"],
        "constraintFeatureNaNCountAfter": metadata["constraintFeatureNaNCountAfter"],
        "metrics": model_bundle["metrics"],
    }
    summary_path = _write_training_summary(archive_dir, training_summary)
    training_summary["summaryPath"] = str(summary_path)
    return model_bundle, metadata, training_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a baseline ML model for power-118 SCUC.")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH, help="Input dataset pickle path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where versioned training artifacts are written.",
    )
    parser.add_argument("--model-filename", type=str, default=DEFAULT_MODEL_FILENAME, help="Model filename inside the output directory.")
    parser.add_argument(
        "--metadata-filename",
        type=str,
        default=DEFAULT_METADATA_FILENAME,
        help="Metadata filename inside the output directory.",
    )
    parser.add_argument("--n-estimators", type=int, default=64, help="Number of trees for the baseline regressor.")
    parser.add_argument("--random-state", type=int, default=7, help="Random seed.")
    parser.add_argument("--model-version", type=str, default=DEFAULT_MODEL_VERSION, help="Model version string.")
    parser.add_argument(
        "--feature-schema-version",
        type=str,
        default=DEFAULT_FEATURE_SCHEMA_VERSION,
        help="Feature schema version string.",
    )
    parser.add_argument(
        "--archive-tag",
        type=str,
        default=None,
        help="Optional versioned subdirectory name. Defaults to a UTC timestamp.",
    )
    parser.add_argument(
        "--no-publish-default-artifacts",
        action="store_true",
        help="Skip copying the trained artifacts to the default service load paths.",
    )
    parser.add_argument(
        "--constraint-training-objective",
        type=str,
        default=DEFAULT_CONSTRAINT_TRAINING_OBJECTIVE,
        choices=["auto", "proxy-only", "mixed", "exact-priority"],
        help="Critical-first objective: auto/proxy-only/mixed/exact-priority.",
    )
    parser.add_argument(
        "--exact-priority-min-coverage",
        type=float,
        default=DEFAULT_EXACT_PRIORITY_MIN_COVERAGE,
        help="When objective=auto, exact label coverage above this threshold switches to exact-priority.",
    )
    parser.add_argument(
        "--exact-priority-weight",
        type=float,
        default=DEFAULT_EXACT_PRIORITY_WEIGHT,
        help="Sample weight multiplier for exact labels in exact-priority objective.",
    )
    parser.add_argument(
        "--critical-classification-threshold",
        type=float,
        default=DEFAULT_CRITICAL_CLASSIFICATION_THRESHOLD,
        help="Threshold used to convert critical probability to binary critical label.",
    )
    parser.add_argument(
        "--model-variant",
        type=str,
        default=DEFAULT_MODEL_VARIANT,
        help="Variant identifier written into model metadata for ablation tracking.",
    )
    parser.add_argument(
        "--feature-ablation-mode",
        type=str,
        default=DEFAULT_FEATURE_ABLATION_MODE,
        choices=["inst-only", "abs-only", "inst+abs"],
        help="Constraint feature subset for ablation: inst-only/abs-only/inst+abs.",
    )
    args = parser.parse_args()

    model_bundle, metadata, training_summary = train_model(
        dataset_path=args.dataset_path.resolve(),
        output_dir=args.output_dir.resolve(),
        model_filename=args.model_filename,
        metadata_filename=args.metadata_filename,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        model_version=args.model_version,
        feature_schema_version=args.feature_schema_version,
        publish_default_artifacts=not args.no_publish_default_artifacts,
        archive_tag=args.archive_tag,
        constraint_training_objective=args.constraint_training_objective,
        exact_priority_min_coverage=args.exact_priority_min_coverage,
        exact_priority_weight=args.exact_priority_weight,
        critical_classification_threshold=args.critical_classification_threshold,
        model_variant=args.model_variant,
        feature_ablation_mode=args.feature_ablation_mode,
    )
    print("Power118 model training")
    print(f"- Input dataset: {args.dataset_path.resolve()}")
    print(f"- Archive dir: {training_summary['archiveDir']}")
    print(f"- Published default artifacts: {'YES' if training_summary['publishedDefaultArtifacts'] else 'NO'}")
    print(f"- Published model path: {training_summary['publishedModelPath']}")
    print(f"- Published metadata path: {training_summary['publishedMetadataPath']}")
    print(f"- Train sample count: {metadata['trainSampleCount']}")
    print(f"- Seed: {args.random_state}")
    print(f"- commitment_train_r2={model_bundle['metrics']['commitment_train_r2']:.4f}")
    print(f"- dispatch_train_r2={model_bundle['metrics']['dispatch_train_r2']:.4f}")
    print(f"- modelVariant={metadata.get('modelVariant')}")
    print(
        "- featureAblationMode="
        f"{metadata.get('featureAblationMode')} (effective={metadata.get('featureAblationModeEffective')})"
    )
    print(f"- constraintTrainingObjective={metadata.get('constraintTrainingObjective')}")
    print(
        "- exact/proxy coverage="
        f"{metadata.get('exactLabelCoverage', 0.0):.3f}/{metadata.get('proxyLabelCoverage', 0.0):.3f}"
    )
    print(f"- modelVersion={metadata['modelVersion']}")
    print(f"- featureSchemaVersion={metadata['featureSchemaVersion']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
