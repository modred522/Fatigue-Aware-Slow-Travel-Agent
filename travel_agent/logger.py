"""Operation logger for debugging."""

import json
import logging
from datetime import datetime
from pathlib import Path

# Create logs directory
LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Setup file handler with rotation by date
log_file = LOGS_DIR / f"travel_agent_{datetime.now().strftime('%Y%m%d')}.log"

# Configure logger
logger = logging.getLogger("travel_agent.ops")
logger.setLevel(logging.DEBUG)
logger.propagate = False

formatter = logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

if not logger.handlers:
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Also log to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def log_operation(operation_type: str, details: dict | None = None):
    """Log an operation with details."""
    if details:
        logger.info(f"[{operation_type}] {json.dumps(details, ensure_ascii=False, default=str)}")
    else:
        logger.info(f"[{operation_type}]")


def log_api_request(method: str, path: str, payload: dict | None = None):
    """Log API request."""
    # Mask sensitive fields
    safe_payload = _mask_sensitive(payload) if payload else None
    log_operation("API_REQUEST", {
        "method": method,
        "path": path,
        "payload": safe_payload
    })


def log_api_response(path: str, status: str, data: dict | None = None):
    """Log API response."""
    log_operation("API_RESPONSE", {
        "path": path,
        "status": status,
        "data": data
    })


def log_geocode(address: str, city: str | None, result: str | None, error: str | None = None):
    """Log geocoding operation."""
    log_operation("GEOCODE", {
        "address": address,
        "city": city,
        "result": result,
        "error": error
    })


def log_route(
    origin: str,
    destination: str,
    city: str,
    mode: str,
    distance: int | None = None,
    error: str | None = None,
    fallback_used: bool = False
):
    """Log route calculation."""
    log_operation("ROUTE", {
        "origin": origin,
        "destination": destination,
        "city": city,
        "mode": mode,
        "distance_meters": distance,
        "error": error,
        "fallback_to_driving": fallback_used
    })


def log_llm_request(prompt: str, model: str, response: str | None = None, error: str | None = None):
    """Log LLM request/response."""
    # Truncate long prompts
    prompt_preview = prompt[:200] + "..." if len(prompt) > 200 else prompt
    response_preview = response[:200] + "..." if response and len(response) > 200 else response

    log_operation("LLM", {
        "model": model,
        "prompt": prompt_preview,
        "response": response_preview,
        "error": error
    })


def log_workflow_state(stage: str, state: dict):
    """Log workflow state at a specific stage."""
    safe_state = {
        "mode": state.get("mode"),
        "anchor_location": state.get("anchor_location"),
        "destination": state.get("destination"),
        "transport_mode": state.get("transport_mode"),
        "itinerary_count": len(state.get("itinerary_items", [])),
        "cumulative_distance": state.get("cumulative_distance_meters"),
    }
    log_operation("WORKFLOW", {"stage": stage, "state": safe_state})


def _mask_sensitive(data: dict) -> dict:
    """Mask sensitive fields in payload."""
    if not isinstance(data, dict):
        return data

    masked = {}
    sensitive_fields = {"amap_api_key", "llm_api_key", "api_key", "key", "token"}

    for k, v in data.items():
        if k in sensitive_fields and isinstance(v, str) and v:
            masked[k] = f"{v[:4]}****{v[-4:]}" if len(v) > 8 else "****"
        elif isinstance(v, dict):
            masked[k] = _mask_sensitive(v)
        elif isinstance(v, list):
            masked[k] = [_mask_sensitive(i) if isinstance(i, dict) else i for i in v]
        else:
            masked[k] = v

    return masked
