from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class ScenarioOut(BaseModel):
    id: str = Field(..., description="Scenario identifier")
    name: str = Field(..., description="Scenario display name")
    description: str = Field(..., description="Scenario description")


class MetricOut(BaseModel):
    solveTimeMs: float
    infeasibilityRate: float
    suboptimality: float


class StrategyOut(BaseModel):
    id: str
    name: str
    feasible: bool
    cost: float
    rank: int


class TrendPointOut(BaseModel):
    label: str
    value: float


class ComparisonPointOut(BaseModel):
    label: str
    value: float


class RunOut(BaseModel):
    runId: str
    scenarioId: str
    generatedAt: str
    metrics: MetricOut
    strategies: List[StrategyOut]
    trend: List[TrendPointOut] = Field(default_factory=list)
    comparison: List[ComparisonPointOut] = Field(default_factory=list)
    adapterMode: str = Field(
        default="compat",
        description="Adapter execution mode: real or compat",
    )
    adapterNote: str = Field(
        default="",
        description="Execution reason/details for diagnostics",
    )


class RunCreateRequest(BaseModel):
    scenarioId: str


class ScenarioListResponse(BaseModel):
    scenarios: List[ScenarioOut]


class RunResponse(BaseModel):
    run: RunOut


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody
