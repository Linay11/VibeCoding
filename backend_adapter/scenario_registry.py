from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ScenarioDef:
    id: str
    name: str
    description: str


SCENARIOS: List[ScenarioDef] = [
    ScenarioDef(
        id="portfolio",
        name="Portfolio Optimization",
        description="Single-run adapter over online_optimization.portfolio problem builder.",
    ),
    ScenarioDef(
        id="control",
        name="Control Setcover",
        description="Single-run adapter over online_optimization.control problem builder.",
    ),
    ScenarioDef(
        id="obstacle",
        name="Obstacle Avoidance",
        description="Single-run adapter over online_optimization.obstacle problem builder.",
    ),
]

SCENARIO_MAP: Dict[str, ScenarioDef] = {s.id: s for s in SCENARIOS}


def list_scenarios() -> List[dict]:
    return [
        {"id": s.id, "name": s.name, "description": s.description}
        for s in SCENARIOS
    ]


def get_scenario(scenario_id: str) -> Optional[ScenarioDef]:
    return SCENARIO_MAP.get(scenario_id)

