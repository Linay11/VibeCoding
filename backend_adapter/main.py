from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend_adapter.errors import APIError
from backend_adapter.run_contract import normalize_run_payload
from backend_adapter.run_store import RunStore
from backend_adapter.runners.scenario_runners import run_scenario_once
from backend_adapter.scenario_registry import get_scenario, list_scenarios
from backend_adapter.schemas import ErrorResponse, RunCreateRequest, RunResponse, ScenarioListResponse


def _error_payload(code: str, message: str) -> dict:
    body = ErrorResponse(error={"code": code, "message": message})
    return body.model_dump()


def _build_app() -> FastAPI:
    app = FastAPI(
        title="mlopt backend adapter",
        version="0.1.0",
        description="Minimal FastAPI adapter for frontend API contract.",
    )

    origins = os.getenv("BACKEND_ADAPTER_CORS", "*")
    allow_origins = [o.strip() for o in origins.split(",") if o.strip()]
    allow_all = "*" in allow_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=([] if allow_all else (allow_origins or ["http://localhost:5173"])),
        allow_origin_regex=(".*" if allow_all else None),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    store_path = Path(
        os.getenv(
            "BACKEND_ADAPTER_STORE",
            str(Path(__file__).resolve().parent / "data" / "latest_runs.json"),
        )
    )
    app.state.run_store = RunStore(store_path)

    @app.exception_handler(APIError)
    async def api_error_handler(_: Request, exc: APIError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(exc.code, exc.message),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content=_error_payload("INVALID_ARGUMENT", str(exc)),
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content=_error_payload("RUN_FAILED", f"internal error: {exc}"),
        )

    @app.get("/api/scenarios", response_model=ScenarioListResponse)
    async def get_api_scenarios() -> Any:
        scenarios = list_scenarios()
        return {"scenarios": scenarios}

    @app.post("/api/runs", response_model=RunResponse)
    async def post_api_runs(req: RunCreateRequest) -> Any:
        scenario_id = (req.scenarioId or "").strip()
        run_mode = (req.runMode or "exact").strip().lower()
        if not scenario_id:
            raise APIError(400, "INVALID_ARGUMENT", "scenarioId is required")

        if get_scenario(scenario_id) is None:
            raise APIError(400, "INVALID_ARGUMENT", f"unknown scenarioId: {scenario_id}")

        if run_mode not in {"exact", "hybrid", "ml"}:
            raise APIError(400, "INVALID_ARGUMENT", f"unsupported runMode: {run_mode}")

        run_payload = run_scenario_once(
            scenario_id,
            run_mode=run_mode,
            time_limit_ms=req.timeLimitMs,
            fallback_to_exact=req.fallbackToExact,
        )
        if not isinstance(run_payload, dict):
            raise APIError(500, "RUN_FAILED", "runner did not return a valid payload")

        normalized = normalize_run_payload(run_payload, scenario_id=scenario_id)

        app.state.run_store.save_latest(scenario_id, normalized)
        return {"run": normalized}

    @app.get("/api/runs/latest", response_model=RunResponse)
    async def get_api_runs_latest(
        scenarioId: str = Query(default="", description="Scenario identifier"),
    ) -> Any:
        scenario_id = (scenarioId or "").strip()
        if not scenario_id:
            raise APIError(400, "INVALID_ARGUMENT", "scenarioId query is required")

        if get_scenario(scenario_id) is None:
            raise APIError(400, "INVALID_ARGUMENT", f"unknown scenarioId: {scenario_id}")

        payload = app.state.run_store.get_latest(scenario_id)
        if payload is None:
            raise APIError(404, "NOT_FOUND", f"no latest run for scenarioId: {scenario_id}")

        normalized = normalize_run_payload(payload, scenario_id=scenario_id)
        app.state.run_store.save_latest(scenario_id, normalized)
        return {"run": normalized}

    return app


app = _build_app()
