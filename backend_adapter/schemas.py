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
    requestedRunMode: str | None = Field(
        default=None,
        description="Requested execution mode for the run before any fallback or downgrade",
    )
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
    solverModeUsed: str = Field(
        default="",
        description="Requested/actual solver mode for the run: exact, hybrid, or ml",
    )
    mlConfidence: float | None = Field(
        default=None,
        description="Optional ML confidence estimate for hybrid or ml mode",
    )
    repairApplied: bool | None = Field(
        default=None,
        description="Whether lightweight ML schedule repair was applied",
    )
    fallbackReason: str | None = Field(
        default=None,
        description="Reason for any runtime fallback or mode downgrade",
    )
    modelVersion: str | None = Field(
        default=None,
        description="Loaded model version when ML artifacts are involved",
    )
    featureSchemaVersion: str | None = Field(
        default=None,
        description="Feature schema version used for ML inference",
    )
    runtimeMs: float | None = Field(
        default=None,
        description="Top-level runtime metric mirrored from the selected execution path",
    )
    objectiveValue: float | None = Field(
        default=None,
        description="Top-level objective value for exact, hybrid, or ml execution",
    )
    feasible: bool = Field(
        default=False,
        description="Top-level feasibility indicator for the returned run payload",
    )
    modelPath: str | None = Field(
        default=None,
        description="Resolved model artifact path when ML loading is attempted",
    )
    modelLoadStatus: str | None = Field(
        default=None,
        description="Model load status such as loaded, bundle_only, failed, or not_requested",
    )


class RunCreateRequest(BaseModel):
    scenarioId: str
    runMode: str | None = None
    timeLimitMs: int | None = None
    fallbackToExact: bool = True


class ScenarioListResponse(BaseModel):
    scenarios: List[ScenarioOut]


class RunResponse(BaseModel):
    run: RunOut


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody
