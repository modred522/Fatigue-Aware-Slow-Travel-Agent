import json
import os
import traceback
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from agent import app as travel_app

api = FastAPI(title="Fatigue-Aware Slow Travel Agent")

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# Settings API
# ==========================================

class SettingsPayload(BaseModel):
    amap_api_key: str | None = None
    llm_model: str | None = None
    dashscope_api_key: str | None = None


@api.get("/api/settings")
async def get_settings():
    """Return current API configuration (keys are masked)."""
    amap_key = os.getenv("AMAP_API_KEY", "")
    dashscope_key = os.getenv("DASHSCOPE_API_KEY", "")
    llm_model = os.getenv("LLM_MODEL", "qwen-turbo")

    def mask(key: str) -> str:
        if not key or len(key) < 8:
            return "*" * len(key) if key else ""
        return key[:4] + "*" * (len(key) - 8) + key[-4:]

    return {
        "amap_api_key": mask(amap_key),
        "amap_api_key_set": bool(amap_key),
        "dashscope_api_key": mask(dashscope_key),
        "dashscope_api_key_set": bool(dashscope_key),
        "llm_model": llm_model,
    }


@api.post("/api/settings")
async def update_settings(payload: SettingsPayload):
    """Update API keys and model at runtime."""
    updated = []
    if payload.amap_api_key is not None:
        os.environ["AMAP_API_KEY"] = payload.amap_api_key
        updated.append("AMAP_API_KEY")
    if payload.dashscope_api_key is not None:
        os.environ["DASHSCOPE_API_KEY"] = payload.dashscope_api_key
        updated.append("DASHSCOPE_API_KEY")
    if payload.llm_model is not None:
        os.environ["LLM_MODEL"] = payload.llm_model
        updated.append("LLM_MODEL")
    return {"status": "ok", "updated": updated}


# ==========================================
# Planning Stream API
# ==========================================

def classify_error(e: Exception) -> dict:
    """Classify an exception into a user-friendly error message."""
    msg = str(e)

    if "AMAP_API_KEY" in msg and ("not configured" in msg or "not set" in msg):
        return {
            "type": "config_error",
            "message": msg,
            "field": "amap_api_key",
        }
    if "AMAP_API_KEY" in msg or "API key is invalid" in msg:
        return {
            "type": "api_key_error",
            "message": msg,
            "field": "amap_api_key",
        }
    if "DASHSCOPE_API_KEY" in msg or "dashscope" in msg.lower():
        return {
            "type": "config_error",
            "message": "DashScope API key is not configured or invalid. Please set it in Settings.",
            "field": "dashscope_api_key",
        }
    if "quota exceeded" in msg.lower():
        return {
            "type": "quota_error",
            "message": msg,
        }
    if "Cannot connect" in msg or "ConnectionError" in type(e).__name__:
        return {
            "type": "network_error",
            "message": "Network connection failed. Please check your internet connection.",
        }
    if "timed out" in msg.lower() or "Timeout" in type(e).__name__:
        return {
            "type": "timeout_error",
            "message": "Request timed out. The service may be slow, please try again.",
        }
    if "Could not find coordinates" in msg:
        return {
            "type": "location_error",
            "message": msg,
        }

    # Fallback: include the exception type for debugging
    return {
        "type": "unknown_error",
        "message": f"{type(e).__name__}: {msg}",
    }


@api.get("/api/plan/stream")
async def stream_plan(
    destination: str = Query(..., description="目的地城市名"),
    origin: str = Query("", description="出发地点（可选）"),
):
    # Pre-flight checks
    if not os.getenv("AMAP_API_KEY"):
        return EventSourceResponse(
            _single_error_event(
                "config_error",
                "AMap API key is not configured. Please go to Settings to set it up.",
                "amap_api_key",
            )
        )

    initial_state = {
        "destination": destination,
        "origin": origin,
        "itinerary": [],
        "cumulative_distance": 0,
        "current_location": "",
        "needs_rest": False,
    }

    async def event_generator():
        try:
            async for chunk in travel_app.astream(
                initial_state, stream_mode="updates"
            ):
                for node_name, updates in chunk.items():
                    event_data = {"node": node_name, "updates": updates}
                    yield {
                        "event": "node_update",
                        "data": json.dumps(event_data, ensure_ascii=False),
                    }
            yield {"event": "done", "data": json.dumps({"status": "complete"})}
        except Exception as e:
            traceback.print_exc()
            error_info = classify_error(e)
            yield {
                "event": "error",
                "data": json.dumps(error_info, ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())


async def _single_error_event(error_type: str, message: str, field: str | None = None):
    """Yield a single error event and close."""
    error_info = {"type": error_type, "message": message}
    if field:
        error_info["field"] = field
    yield {
        "event": "error",
        "data": json.dumps(error_info, ensure_ascii=False),
    }


# ==========================================
# Static file serving (production)
# ==========================================
frontend_dist = os.path.join(os.path.dirname(__file__), "frontend", "dist")
if os.path.isdir(frontend_dist):
    api.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(api, host="0.0.0.0", port=8000)
