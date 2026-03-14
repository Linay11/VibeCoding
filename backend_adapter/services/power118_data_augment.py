from __future__ import annotations

from typing import Any

import numpy as np


def make_random_power118_overrides(
    base_data: dict[str, Any],
    rng: np.random.Generator,
    load_profile_scale_range: tuple[float, float] = (0.88, 1.12),
    hourly_noise_std: float = 0.03,
    generator_scale_range: tuple[float, float] = (0.94, 1.06),
    generator_cost_scale_range: tuple[float, float] = (0.92, 1.08),
    generator_fraction: float = 0.2,
) -> dict[str, Any]:
    time_horizon = int(base_data.get("timeHorizon", 24))
    generators = list(base_data.get("generators", []))
    if time_horizon <= 0:
        raise ValueError("time horizon must be positive")

    base_profile_scale = float(rng.uniform(*load_profile_scale_range))
    hourly_scale = base_profile_scale * rng.normal(loc=1.0, scale=hourly_noise_std, size=time_horizon)
    hourly_scale = np.clip(hourly_scale, 0.75, 1.25)

    n_generators = len(generators)
    n_selected = max(1, int(round(n_generators * generator_fraction))) if generators else 0
    selected_indices = set(rng.choice(n_generators, size=n_selected, replace=False).tolist()) if generators else set()

    generator_pmax_scale: dict[int, float] = {}
    generator_cost_scale: dict[int, float] = {}
    for index, generator in enumerate(generators):
        if index not in selected_indices:
            continue
        generator_id = int(generator["genId"])
        generator_pmax_scale[generator_id] = float(rng.uniform(*generator_scale_range))
        generator_cost_scale[generator_id] = float(rng.uniform(*generator_cost_scale_range))

    return {
        "hourlyLoadScale": hourly_scale.tolist(),
        "generatorPMaxScale": generator_pmax_scale,
        "generatorCostScale": generator_cost_scale,
    }


def generate_power118_override_set(
    base_data: dict[str, Any],
    n_samples: int,
    seed: int = 7,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    return [make_random_power118_overrides(base_data, rng) for _ in range(n_samples)]
