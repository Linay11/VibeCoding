from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend_adapter.services.power118_data_augment import generate_power118_override_set
from backend_adapter.services.power118_dataset import (
    build_power118_feature_record,
    build_power118_metadata_record,
    build_power118_target_record,
    load_power118_data,
)


DEFAULT_OUTPUT_DIR = ROOT_DIR / "backend_adapter" / "data" / "power118_dataset"
DEFAULT_DATASET_FILENAME = "power118_ml_dataset.pkl"
DEFAULT_SUMMARY_FILENAME = "dataset_summary.json"


def _load_power118_module():
    from backend_adapter.services.power118_dataset import _load_power118_module  # type: ignore[attr-defined]

    return _load_power118_module()


def _write_dataset_summary(
    output_dir: Path,
    summary: dict[str, Any],
    summary_filename: str = DEFAULT_SUMMARY_FILENAME,
) -> Path:
    summary_path = output_dir / summary_filename
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def build_dataset(
    num_samples: int,
    seed: int,
    output_dir: Path,
    dataset_filename: str,
    time_limit_s: float | None,
    data_path: str | Path | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    external_module = _load_power118_module()
    runtime = external_module.check_gurobi_runtime()
    if not runtime.get("available"):
        raise RuntimeError(f"Cannot build dataset without exact solver runtime: {runtime.get('reason')}")

    base_data = load_power118_data(data_path=data_path)
    overrides_list = generate_power118_override_set(base_data=base_data, n_samples=num_samples, seed=seed)

    feature_rows: list[dict] = []
    target_rows: list[dict] = []
    metadata_rows: list[dict] = []

    for sample_index, overrides in enumerate(overrides_list, start=1):
        power_data = load_power118_data(data_path=data_path, overrides=overrides)
        result = external_module.solve_scuc_118(
            data_path=data_path,
            write_output=False,
            overrides=overrides,
            time_limit_s=time_limit_s,
        )
        if not result.get("feasible"):
            continue

        sample_id = f"power118-{sample_index:05d}"
        feature_rows.append(build_power118_feature_record(power_data))
        target_rows.append(build_power118_target_record(result))
        metadata_rows.append(
            build_power118_metadata_record(
                power_data=power_data,
                overrides=overrides,
                sample_id=sample_id,
                split="train",
            )
        )

    dataset_bundle = {
        "features": pd.DataFrame(feature_rows),
        "targets": pd.DataFrame(target_rows),
        "metadata": pd.DataFrame(metadata_rows),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / dataset_filename
    pd.to_pickle(dataset_bundle, dataset_path)

    dataset_summary = {
        "datasetPath": str(dataset_path),
        "summaryPath": str(output_dir / DEFAULT_SUMMARY_FILENAME),
        "inputDataPath": str(data_path or base_data.get("dataPath") or ""),
        "seed": seed,
        "requestedSampleCount": int(num_samples),
        "keptSampleCount": int(len(feature_rows)),
        "timeLimitS": float(time_limit_s) if time_limit_s is not None else None,
        "exactBaselineUsed": bool(runtime.get("available")),
        "runtimeStage": runtime.get("stage"),
        "runtimeReason": runtime.get("reason"),
    }
    summary_path = _write_dataset_summary(output_dir, dataset_summary)
    return dataset_path, summary_path, dataset_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build supervised dataset for power-118 SCUC ML experiments.")
    parser.add_argument("--num-samples", type=int, default=64, help="Number of perturbed SCUC samples to generate.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for data augmentation.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Optional override for the source power118 workbook path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for dataset artifacts.",
    )
    parser.add_argument(
        "--dataset-filename",
        type=str,
        default=DEFAULT_DATASET_FILENAME,
        help="Dataset pickle filename written into the output directory.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Deprecated direct output path. If provided, it overrides output-dir and dataset-filename.",
    )
    parser.add_argument(
        "--time-limit-s",
        type=float,
        default=None,
        help="Optional exact SCUC solver time limit in seconds.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    dataset_filename = args.dataset_filename
    if args.output_path is not None:
        output_dir = args.output_path.resolve().parent
        dataset_filename = args.output_path.name

    dataset_path, summary_path, dataset_summary = build_dataset(
        num_samples=args.num_samples,
        seed=args.seed,
        output_dir=output_dir,
        dataset_filename=dataset_filename,
        time_limit_s=args.time_limit_s,
        data_path=args.data_path.resolve() if args.data_path is not None else None,
    )
    print("Power118 dataset build")
    print(f"- Input source: {dataset_summary['inputDataPath']}")
    print(f"- Output dataset: {dataset_path}")
    print(f"- Output summary: {summary_path}")
    print(f"- Requested samples: {args.num_samples}")
    print(f"- Kept samples: {dataset_summary['keptSampleCount']}")
    print(f"- Seed: {args.seed}")
    print(f"- Exact baseline runtime available: {'YES' if dataset_summary['exactBaselineUsed'] else 'NO'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
