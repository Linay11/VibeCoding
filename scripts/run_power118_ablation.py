#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = ROOT_DIR / "backend_adapter" / "data" / "power118_dataset" / "power118_ml_dataset.pkl"
DEFAULT_RUNS_ROOT = ROOT_DIR / "backend_adapter" / "data" / "power118_ablation_runs"
DEFAULT_TRAIN_OBJECTIVES = ("proxy-only", "mixed", "exact-priority")
DEFAULT_FEATURE_ABLATION_MODES = ("inst+abs",)
DEFAULT_EVAL_MODES = (
    "exact",
    "hybrid_warm_start",
    "hybrid_constraint_aware_v2",
    "hybrid_constraint_aware_v3",
    "ml",
)
DEFAULT_MODEL_FILENAME = "power118_ml_model.joblib"
DEFAULT_METADATA_FILENAME = "power118_ml_metadata.json"


def _utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _slugify(value: str) -> str:
    normalized = []
    for ch in str(value or "variant").lower():
        if ch.isalnum() or ch in {"-", "_"}:
            normalized.append(ch)
        elif ch == "+":
            normalized.append("-plus-")
        else:
            normalized.append("-")
    normalized = "".join(normalized)
    normalized = "-".join(part for part in normalized.split("-") if part)
    return normalized or "variant"


def _cmd_text(command: list[str]) -> str:
    return shlex.join(command)


def _run_command(command: list[str], dry_run: bool = False) -> None:
    print(f"[run] {_cmd_text(command)}")
    if dry_run:
        return
    subprocess.run(command, check=True, cwd=str(ROOT_DIR))


def _write_command_file(path: Path, commands: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
    ]
    lines.extend(_cmd_text(command) for command in commands)
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _build_train_command(
    python_bin: str,
    dataset_path: Path,
    models_root: Path,
    archive_tag: str,
    objective: str,
    feature_mode: str,
    model_variant: str,
    n_estimators: int,
    random_state: int,
    model_version: str,
    feature_schema_version: str,
    model_filename: str,
    metadata_filename: str,
) -> list[str]:
    return [
        python_bin,
        str(ROOT_DIR / "scripts" / "train_power118_model.py"),
        "--dataset-path",
        str(dataset_path),
        "--output-dir",
        str(models_root),
        "--archive-tag",
        archive_tag,
        "--model-filename",
        model_filename,
        "--metadata-filename",
        metadata_filename,
        "--constraint-training-objective",
        objective,
        "--feature-ablation-mode",
        feature_mode,
        "--model-variant",
        model_variant,
        "--n-estimators",
        str(n_estimators),
        "--random-state",
        str(random_state),
        "--model-version",
        model_version,
        "--feature-schema-version",
        feature_schema_version,
        "--no-publish-default-artifacts",
    ]


def _variant_slug(objective: str, feature_mode: str) -> str:
    return _slugify(f"{objective}__{feature_mode}")


def _build_eval_command(
    python_bin: str,
    output_dir: Path,
    variant_config_path: Path,
    num_cases: int,
    seed: int,
    modes: list[str],
    time_limit_ms: int | None,
    require_exact_baseline: bool,
) -> list[str]:
    command = [
        python_bin,
        str(ROOT_DIR / "scripts" / "eval_power118_modes.py"),
        "--num-cases",
        str(num_cases),
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
        "--variant-config-path",
        str(variant_config_path),
        "--modes",
        *modes,
    ]
    if time_limit_ms is not None:
        command.extend(["--time-limit-ms", str(time_limit_ms)])
    if require_exact_baseline:
        command.append("--require-exact-baseline")
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description="Run minimal power118 ablation batch training + evaluation.")
    parser.add_argument("--python-bin", type=str, default=sys.executable or "python", help="Python interpreter used for child scripts.")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH, help="Dataset pickle path from build_power118_ml_dataset.py.")
    parser.add_argument("--run-root", type=Path, default=None, help="Root output directory for this ablation run.")
    parser.add_argument("--run-tag", type=str, default=None, help="Optional run tag used when --run-root is omitted.")
    parser.add_argument(
        "--objectives",
        nargs="+",
        default=list(DEFAULT_TRAIN_OBJECTIVES),
        choices=["proxy-only", "mixed", "exact-priority"],
        help="Constraint training objectives to train.",
    )
    parser.add_argument(
        "--feature-modes",
        nargs="+",
        default=list(DEFAULT_FEATURE_ABLATION_MODES),
        choices=["inst-only", "abs-only", "inst+abs"],
        help="Feature ablation modes to train.",
    )
    parser.add_argument("--n-estimators", type=int, default=64, help="Number of trees for train_power118_model.py.")
    parser.add_argument("--train-seed", type=int, default=7, help="Training random seed.")
    parser.add_argument("--eval-seed", type=int, default=7, help="Evaluation random seed.")
    parser.add_argument("--num-cases", type=int, default=8, help="Case count for eval_power118_modes.py.")
    parser.add_argument("--time-limit-ms", type=int, default=None, help="Optional exact/hybrid solver time limit in milliseconds.")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=list(DEFAULT_EVAL_MODES),
        choices=[
            "exact",
            "hybrid_warm_start",
            "hybrid_constraint_aware",
            "hybrid_constraint_aware_v2",
            "hybrid_constraint_aware_v3",
            "ml",
        ],
        help="Modes passed to eval_power118_modes.py.",
    )
    parser.add_argument("--model-version", type=str, default="power118-ablation-batch-v1", help="Model version written into metadata.")
    parser.add_argument("--feature-schema-version", type=str, default="power118-feature-schema-v1", help="Feature schema version for training.")
    parser.add_argument("--model-filename", type=str, default=DEFAULT_MODEL_FILENAME, help="Model file name inside each variant archive.")
    parser.add_argument("--metadata-filename", type=str, default=DEFAULT_METADATA_FILENAME, help="Metadata file name inside each variant archive.")
    parser.add_argument("--variant-config-out", type=Path, default=None, help="Path to write generated variant-config.json.")
    parser.add_argument("--skip-train", action="store_true", help="Skip training; expect model files already present.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation command execution.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands and generated files without executing train/eval.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing model files under run root.")
    parser.add_argument("--require-exact-baseline", action="store_true", help="Forwarded to eval_power118_modes.py.")
    args = parser.parse_args()

    run_tag = args.run_tag or _utc_tag()
    run_root = args.run_root.resolve() if args.run_root is not None else (DEFAULT_RUNS_ROOT / run_tag).resolve()
    dataset_path = args.dataset_path.resolve()
    models_root = (run_root / "models").resolve()
    eval_root = (run_root / "eval").resolve()
    commands_root = (run_root / "commands").resolve()
    variant_config_path = (
        args.variant_config_out.resolve()
        if args.variant_config_out is not None
        else (run_root / "variant-config.json").resolve()
    )

    if not args.skip_train and not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    train_commands: list[list[str]] = []
    variant_specs: list[dict[str, Any]] = []
    for objective in args.objectives:
        for feature_mode in args.feature_modes:
            variant_key = f"{objective}__{feature_mode}"
            variant_slug = _variant_slug(objective=objective, feature_mode=feature_mode)
            archive_tag = variant_slug
            model_path = models_root / archive_tag / args.model_filename
            metadata_path = models_root / archive_tag / args.metadata_filename

            if not args.skip_train and not args.overwrite and (model_path.exists() or metadata_path.exists()):
                raise FileExistsError(
                    f"Variant artifact exists already: model={model_path} metadata={metadata_path}. "
                    "Use --overwrite or a new --run-root."
                )

            train_commands.append(
                _build_train_command(
                    python_bin=args.python_bin,
                    dataset_path=dataset_path,
                    models_root=models_root,
                    archive_tag=archive_tag,
                    objective=objective,
                    feature_mode=feature_mode,
                    model_variant=variant_slug,
                    n_estimators=args.n_estimators,
                    random_state=args.train_seed,
                    model_version=args.model_version,
                    feature_schema_version=args.feature_schema_version,
                    model_filename=args.model_filename,
                    metadata_filename=args.metadata_filename,
                )
            )
            variant_specs.append(
                {
                    "modelVariant": variant_slug,
                    "modelPath": str(model_path),
                    "metadataPath": str(metadata_path),
                    "constraintTrainingObjective": objective,
                    "featureAblationMode": feature_mode,
                    "runEnabled": True,
                    "note": f"auto-generated by run_power118_ablation.py ({variant_key})",
                }
            )

    eval_command = _build_eval_command(
        python_bin=args.python_bin,
        output_dir=eval_root,
        variant_config_path=variant_config_path,
        num_cases=args.num_cases,
        seed=args.eval_seed,
        modes=list(args.modes),
        time_limit_ms=args.time_limit_ms,
        require_exact_baseline=args.require_exact_baseline,
    )

    run_root.mkdir(parents=True, exist_ok=True)
    models_root.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)
    commands_root.mkdir(parents=True, exist_ok=True)

    variant_config_path.parent.mkdir(parents=True, exist_ok=True)
    variant_config_path.write_text(json.dumps({"variants": variant_specs}, indent=2), encoding="utf-8")
    train_commands_path = commands_root / "train_commands.sh"
    eval_commands_path = commands_root / "eval_commands.sh"
    _write_command_file(train_commands_path, train_commands)
    _write_command_file(eval_commands_path, [eval_command])

    print("[power118-ablation] run root:", run_root)
    print("[power118-ablation] variant config:", variant_config_path)
    print("[power118-ablation] train command list:", train_commands_path)
    print("[power118-ablation] eval command list:", eval_commands_path)

    if not args.skip_train:
        for command in train_commands:
            _run_command(command, dry_run=args.dry_run)
    else:
        for spec in variant_specs:
            model_path = Path(spec["modelPath"])
            metadata_path = Path(spec["metadataPath"])
            if not model_path.exists() or not metadata_path.exists():
                raise FileNotFoundError(
                    f"skip-train enabled but artifact missing: model={model_path} metadata={metadata_path}"
                )

    if not args.skip_eval:
        _run_command(eval_command, dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
