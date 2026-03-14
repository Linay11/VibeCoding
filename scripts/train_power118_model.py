from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
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


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _build_archive_dir(output_dir: Path, archive_tag: str | None) -> Path:
    return output_dir / (archive_tag or _utc_ts())


def _write_training_summary(output_dir: Path, summary: dict[str, Any]) -> Path:
    summary_path = output_dir / DEFAULT_SUMMARY_FILENAME
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


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
) -> tuple[dict, dict, dict[str, Any]]:
    from sklearn.ensemble import ExtraTreesRegressor

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
    constraint_summary_columns: list[str] = []
    constraint_fixing_columns: list[str] = []
    instance_feature_names: list[str] = []
    abstract_feature_names: list[str] = []
    constraint_metrics: dict[str, float] = {}

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
        instance_feature_names = [column for column in constraint_candidates.columns if column.startswith("inst_")]
        abstract_feature_names = [column for column in constraint_candidates.columns if column.startswith("abs_")]
        constraint_feature_names = instance_feature_names + abstract_feature_names
        if constraint_feature_names:
            X_constraint = constraint_candidates[constraint_feature_names].to_numpy(dtype=float)
            y_constraint_score = constraint_candidates["labelRankScore"].to_numpy(dtype=float)
            constraint_scoring_model = ExtraTreesRegressor(
                n_estimators=n_estimators,
                random_state=random_state + 4,
                n_jobs=-1,
            )
            constraint_scoring_model.fit(X_constraint, y_constraint_score)
            constraint_score_pred = constraint_scoring_model.predict(X_constraint)
            constraint_metrics["constraint_scoring_train_r2"] = float(
                constraint_scoring_model.score(X_constraint, y_constraint_score)
            )
            constraint_metrics["constraint_scoring_train_mae"] = float(
                abs(constraint_score_pred - y_constraint_score).mean()
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
    metadata["constraintScoringMode"] = "ranking-regression"
    metadata["instanceFeatureNames"] = instance_feature_names
    metadata["abstractFeatureNames"] = abstract_feature_names
    model_bundle = {
        "feature_columns": list(features.columns),
        "commitment_columns": commitment_columns,
        "dispatch_columns": dispatch_columns,
        "commitment_model": commitment_model,
        "dispatch_model": dispatch_model,
        "constraint_summary_model": constraint_summary_model,
        "constraint_fixing_model": constraint_fixing_model,
        "constraint_scoring_model": constraint_scoring_model,
        "constraint_summary_columns": constraint_summary_columns,
        "constraint_fixing_columns": constraint_fixing_columns,
        "instance_feature_names": instance_feature_names,
        "abstract_feature_names": abstract_feature_names,
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
        "featureSchemaVersion": feature_schema_version,
        "trainSampleCount": len(features),
        "seed": random_state,
        "nEstimators": n_estimators,
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
    print(f"- modelVersion={metadata['modelVersion']}")
    print(f"- featureSchemaVersion={metadata['featureSchemaVersion']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
