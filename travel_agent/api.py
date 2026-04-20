from __future__ import annotations

import json
import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from travel_agent.config import refresh_config
from travel_agent.errors import classify_exception
from travel_agent.logger import log_api_request, log_api_response
from travel_agent.schemas import (
    BusinessEvent,
    CandidateRequest,
    CandidateSetResponse,
    LocationValidationRequest,
    LocationValidationResponse,
    ModelFetchPayload,
    ModelListResponse,
    ModelOption,
    PlanRequest,
    SettingsPayload,
)
from travel_agent.services.llm import list_models
from travel_agent.workflow import build_initial_state, generate_waypoint_candidates, travel_workflow


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api = FastAPI(title="Fatigue-Aware Slow Travel Agent")
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _mask(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}{'*' * (len(value) - 8)}{value[-4:]}"


@api.get("/api/settings")
async def get_settings():
    config = refresh_config()
    return {
        "amap_api_key": _mask(config.amap_api_key),
        "amap_api_key_set": bool(config.amap_api_key),
        "llm_api_key": _mask(config.llm_api_key),
        "llm_api_key_set": bool(config.llm_api_key),
        "llm_base_url": config.llm_base_url,
        "llm_model": config.llm_model,
        "llm_temperature": config.llm_temperature,
    }


@api.get("/api/settings/map")
async def get_map_settings():
    config = refresh_config()
    return {
        "amap_api_key": config.amap_api_key,
        "amap_api_key_set": bool(config.amap_api_key),
    }


@api.post("/api/settings")
async def update_settings(payload: SettingsPayload):
    updated: list[str] = []
    if payload.amap_api_key is not None:
        os.environ["AMAP_API_KEY"] = payload.amap_api_key
        updated.append("AMAP_API_KEY")
    if payload.llm_api_key is not None:
        os.environ["LLM_API_KEY"] = payload.llm_api_key
        updated.append("LLM_API_KEY")
    elif payload.dashscope_api_key is not None:
        os.environ["LLM_API_KEY"] = payload.dashscope_api_key
        updated.append("LLM_API_KEY")
    if payload.llm_base_url is not None:
        os.environ["LLM_BASE_URL"] = payload.llm_base_url
        updated.append("LLM_BASE_URL")
    if payload.llm_model is not None:
        os.environ["LLM_MODEL"] = payload.llm_model
        updated.append("LLM_MODEL")
    if payload.llm_temperature is not None:
        os.environ["LLM_TEMPERATURE"] = str(payload.llm_temperature)
        updated.append("LLM_TEMPERATURE")
    refresh_config()
    return {"status": "ok", "updated": updated}


@api.post("/api/settings/models")
async def fetch_models(payload: ModelFetchPayload):
    ids = list_models(api_key=payload.llm_api_key, base_url=payload.llm_base_url)
    return ModelListResponse(models=[ModelOption(id=model_id) for model_id in ids])


@api.post("/api/validate-location")
async def validate_location(payload: LocationValidationRequest):
    """Validate that a location exists within the specified city."""
    from travel_agent.services.amap import AMapClient

    result = AMapClient().validate_location(payload.city, payload.location)
    return LocationValidationResponse(
        valid=result["valid"],
        city=payload.city,
        location=payload.location,
        formatted_address=result.get("formatted_address"),
        message=result.get("message"),
    )


@api.post("/api/plans/candidates")
async def generate_candidates(payload: CandidateRequest):
    log_api_request("POST", "/api/plans/candidates", payload.model_dump())
    candidates = generate_waypoint_candidates(
        origin=payload.origin,
        destination=payload.destination,
        interests=payload.interests,
    )
    return CandidateSetResponse(
        mode=payload.mode,
        origin=payload.origin,
        destination=payload.destination,
        interests=payload.interests,
        candidates=candidates,
    )


@api.post("/api/plans/stream")
async def stream_plan(payload: PlanRequest):
    log_api_request("POST", "/api/plans/stream", payload.model_dump())
    initial_state = build_initial_state(payload)

    async def event_generator():
        started = BusinessEvent(
            event="planning_started",
            payload={
                "mode": payload.mode.value,
                "destination": payload.destination,
                "origin": payload.origin,
                "selected_waypoint_count": len(payload.selected_waypoints),
            },
        )
        yield _sse_dict(started.event, started.payload)

        try:
            async for chunk in travel_workflow.astream(initial_state, stream_mode="updates"):
                for updates in chunk.values():
                    for event in updates.get("emitted_events", []):
                        yield _sse_dict(event["event"], event["payload"])
        except Exception as exc:
            logger.exception("Planning failed")
            yield _sse_dict("error", classify_exception(exc))

    return EventSourceResponse(event_generator())


def _sse_dict(event: str, payload: dict):
    return {"event": event, "data": json.dumps(payload, ensure_ascii=False)}


frontend_dist = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "dist")
if os.path.isdir(frontend_dist):
    api.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
