from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pandas as pd


GENERATOR_FEATURE_KEYS = (
    "busIndex",
    "pMin",
    "pMax",
    "a2",
    "a1",
    "a0",
    "rampUp",
    "rampDown",
    "startCost",
    "shutCost",
    "minUpTime",
    "minDownTime",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _power118_script() -> Path:
    return _repo_root() / "external" / "power118" / "SCUC_118_new.py"


def _load_power118_module():
    script_path = _power118_script()
    if not script_path.exists():
        raise FileNotFoundError(f"power118 script not found: {script_path}")

    spec = importlib.util.spec_from_file_location("external.power118.scuc_118_new", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module spec from: {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_power118_data(
    data_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    module = _load_power118_module()
    if not hasattr(module, "load_power118_data"):
        raise AttributeError("external power118 module is missing load_power118_data")
    return module.load_power118_data(data_path=data_path, overrides=overrides)


def build_power118_feature_record(power_data: dict[str, Any]) -> dict[str, float]:
    generators = power_data.get("generators", [])
    branches = power_data.get("branches", [])
    total_load_by_hour = power_data.get("totalLoadByHour", [])
    summary = power_data.get("summary", {})

    feature_row: dict[str, float] = {
        "timeHorizon": float(power_data.get("timeHorizon", 0)),
        "numBus": float(summary.get("numBus", 0)),
        "numLine": float(summary.get("numLine", 0)),
        "numGen": float(summary.get("numGen", 0)),
        "numLoad": float(summary.get("numLoad", 0)),
        "peakLoad": float(summary.get("peakLoad", 0.0)),
        "totalDailyLoad": float(summary.get("totalDailyLoad", 0.0)),
        "minHourlyLoad": float(min(total_load_by_hour) if total_load_by_hour else 0.0),
        "avgHourlyLoad": float(sum(total_load_by_hour) / len(total_load_by_hour) if total_load_by_hour else 0.0),
        "totalGeneratorPMax": float(sum(generator["pMax"] for generator in generators)),
        "totalGeneratorPMin": float(sum(generator["pMin"] for generator in generators)),
        "totalStartCost": float(sum(generator["startCost"] for generator in generators)),
        "totalShutCost": float(sum(generator["shutCost"] for generator in generators)),
        "avgRampUp": float(sum(generator["rampUp"] for generator in generators) / len(generators) if generators else 0.0),
        "avgRampDown": float(
            sum(generator["rampDown"] for generator in generators) / len(generators) if generators else 0.0
        ),
        "avgBranchCapacity": float(sum(branch["capacity"] for branch in branches) / len(branches) if branches else 0.0),
        "avgBranchReactance": float(sum(branch["x"] for branch in branches) / len(branches) if branches else 0.0),
    }

    peak_load = feature_row["peakLoad"] or 1.0
    feature_row["reserveMarginPeak"] = max(
        0.0,
        (feature_row["totalGeneratorPMax"] - peak_load) / peak_load,
    )

    for hour_index, load_value in enumerate(total_load_by_hour, start=1):
        feature_row[f"hourlyLoad_{hour_index}"] = float(load_value)

    for gen_index, generator in enumerate(generators, start=1):
        feature_row[f"gen{gen_index}_genId"] = float(generator["genId"])
        for key in GENERATOR_FEATURE_KEYS:
            feature_row[f"gen{gen_index}_{key}"] = float(generator[key])

    return feature_row


def build_power118_feature_frame(power_data: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([build_power118_feature_record(power_data)])


def build_power118_target_record(result: dict[str, Any]) -> dict[str, float]:
    target_row: dict[str, float] = {
        "objective": float(result.get("objective") or 0.0),
        "solveTimeMs": float(result.get("solveTimeMs") or 0.0),
        "feasible": 1.0 if result.get("feasible") else 0.0,
    }

    unit_commitment = result.get("unitCommitmentByHour", [])
    generator_dispatch = result.get("generatorDispatchByHour", [])

    for gen_index, hourly_values in enumerate(unit_commitment, start=1):
        for hour_index, value in enumerate(hourly_values, start=1):
            target_row[f"unitCommitment_g{gen_index}_h{hour_index}"] = float(value)

    for gen_index, hourly_values in enumerate(generator_dispatch, start=1):
        for hour_index, value in enumerate(hourly_values, start=1):
            target_row[f"dispatch_g{gen_index}_h{hour_index}"] = float(value)

    return target_row


def build_power118_target_frame(result: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([build_power118_target_record(result)])


def build_power118_metadata_record(
    power_data: dict[str, Any],
    overrides: dict[str, Any] | None = None,
    sample_id: str | None = None,
    split: str = "train",
) -> dict[str, Any]:
    summary = power_data.get("summary", {})
    total_load_by_hour = power_data.get("totalLoadByHour", [])
    return {
        "sampleId": sample_id or "",
        "split": split,
        "dataPath": str(power_data.get("dataPath") or ""),
        "numBus": int(summary.get("numBus", 0)),
        "numLine": int(summary.get("numLine", 0)),
        "numGen": int(summary.get("numGen", 0)),
        "numLoad": int(summary.get("numLoad", 0)),
        "peakLoad": float(summary.get("peakLoad", 0.0)),
        "totalDailyLoad": float(summary.get("totalDailyLoad", 0.0)),
        "totalLoadByHourJson": json.dumps(total_load_by_hour),
        "overridesJson": json.dumps(overrides or {}),
    }
