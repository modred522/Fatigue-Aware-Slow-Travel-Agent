from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AppError(Exception):
    type: str
    message: str
    field: str | None = None
    status_code: int = 400

    def __str__(self) -> str:
        return self.message


class ConfigError(AppError):
    def __init__(self, message: str, field: str | None = None):
        super().__init__("config_error", message, field, 400)


class ValidationError(AppError):
    def __init__(self, message: str, field: str | None = None):
        super().__init__("validation_error", message, field, 422)


class ExternalServiceError(AppError):
    def __init__(self, message: str, field: str | None = None):
        super().__init__("external_service_error", message, field, 502)


def classify_exception(exc: Exception) -> dict:
    if isinstance(exc, AppError):
        payload = {"type": exc.type, "message": exc.message}
        if exc.field:
            payload["field"] = exc.field
        return payload

    msg = str(exc)
    lowered = msg.lower()

    if "amap" in lowered and "key" in lowered:
        return {"type": "api_key_error", "message": msg, "field": "amap_api_key"}
    if "dashscope" in lowered or "llm api key" in lowered or "api key" in lowered:
        return {
            "type": "api_key_error",
            "message": msg,
            "field": "llm_api_key",
        }
    if "timeout" in lowered:
        return {"type": "timeout_error", "message": msg}
    if "connect" in lowered or "network" in lowered:
        return {"type": "network_error", "message": msg}
    if "coordinate" in lowered or "location" in lowered:
        return {"type": "location_error", "message": msg}
    if "distance exceeds limit" in lowered or "over_direction_range" in lowered:
        return {
            "type": "distance_limit_error",
            "message": msg,
            "field": "transport_mode",
        }

    return {"type": "unknown_error", "message": f"{type(exc).__name__}: {msg}"}
