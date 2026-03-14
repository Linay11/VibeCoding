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
    requestedMode: str | None = Field(
        default=None,
        description="Alias of requestedRunMode for evaluation-oriented consumers",
    )
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
    statusName: str | None = Field(
        default=None,
        description="Underlying solver or inference status name",
    )
    statusCode: int | None = Field(
        default=None,
        description="Underlying solver status code when available",
    )
    solutionCount: int | None = Field(
        default=None,
        description="Number of solver solutions or incumbents found when available",
    )
    terminatedByTimeLimit: bool | None = Field(
        default=None,
        description="Whether the solve terminated because of a time limit",
    )
    optimal: bool | None = Field(
        default=None,
        description="Whether the returned result is proven optimal",
    )
    hasIncumbent: bool | None = Field(
        default=None,
        description="Whether the returned result contains an incumbent solution",
    )
    hybridStrategyRequested: str | None = Field(
        default=None,
        description="Requested internal hybrid strategy, if applicable",
    )
    hybridStrategyUsed: str | None = Field(
        default=None,
        description="Actual hybrid strategy used after fallback or downgrade",
    )
    constraintAwareHybridUsed: bool | None = Field(
        default=None,
        description="Whether the constraint-aware hybrid path was attempted",
    )
    reducedSolveApplied: bool | None = Field(
        default=None,
        description="Whether reduced solve with fixing or reduction was applied",
    )
    fixedCommitmentCount: int | None = Field(
        default=None,
        description="Number of commitment binaries fixed during reduced solve",
    )
    predictedActiveConstraintCount: int | None = Field(
        default=None,
        description="Predicted active constraint count from the constraint model",
    )
    constraintConfidence: float | None = Field(
        default=None,
        description="Confidence for constraint-aware predictions",
    )
    repairAfterReducedSolve: bool | None = Field(
        default=None,
        description="Whether a repair or alternate solve path was used after reduced solve",
    )
    reducedSolveFallbackReason: str | None = Field(
        default=None,
        description="Reason the reduced solve path was abandoned or repaired",
    )
    fixedBinaryRatio: float | None = Field(
        default=None,
        description="Fraction of commitment binaries fixed during reduced solve",
    )
    constraintReductionRatio: float | None = Field(
        default=None,
        description="Approximate reduction ratio induced by the constraint-aware strategy",
    )
    constraintScoringUsed: bool | None = Field(
        default=None,
        description="Whether the constraint scoring model was used",
    )
    criticalConstraintCount: int | None = Field(
        default=None,
        description="Number of predicted critical constraints kept in the reduced model",
    )
    deferredConstraintCount: int | None = Field(
        default=None,
        description="Number of predicted deferred constraints omitted from the reduced model",
    )
    constraintReactivationCount: int | None = Field(
        default=None,
        description="Number of deferred constraints reactivated during staged solve",
    )
    stagedSolveRounds: int | None = Field(
        default=None,
        description="Number of staged solve rounds used by the hybrid strategy",
    )
    constraintAwareReductionMode: str | None = Field(
        default=None,
        description="Reduction mode used for the constraint-aware hybrid strategy",
    )
    reducedModelValidated: bool | None = Field(
        default=None,
        description="Whether the reduced model result passed validation",
    )
    reductionRejectedReason: str | None = Field(
        default=None,
        description="Reason the reduced model result was rejected",
    )


class RunCreateRequest(BaseModel):
    scenarioId: str
    runMode: str | None = None
    hybridStrategy: str | None = None
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
