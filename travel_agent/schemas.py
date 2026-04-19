from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class PlanningMode(str, Enum):
    LOCAL_EXPLORE = "local_explore"
    POINT_TO_POINT = "point_to_point"


class TransportMode(str, Enum):
    WALKING = "walking"
    TRANSIT = "transit"
    DRIVING = "driving"


class PlanningStage(str, Enum):
    CANDIDATE_SELECTION = "candidate_selection"
    ITINERARY_PLANNING = "itinerary_planning"
    COMPLETED = "completed"


class ItineraryItemKind(str, Enum):
    ORIGIN = "origin"
    SPOT = "spot"
    WAYPOINT = "waypoint"
    REST_STOP = "rest_stop"
    DESTINATION = "destination"


class SettingsPayload(BaseModel):
    amap_api_key: str | None = None
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_temperature: float | None = Field(default=None, ge=0, le=2)
    dashscope_api_key: str | None = None


class ModelFetchPayload(BaseModel):
    llm_api_key: str | None = None
    llm_base_url: str | None = None


class ModelOption(BaseModel):
    id: str


class ModelListResponse(BaseModel):
    models: list[ModelOption]


class CandidateWaypoint(BaseModel):
    id: str
    name: str
    reason: str
    role: str


class CandidateRequest(BaseModel):
    mode: Literal["point_to_point"] = "point_to_point"
    origin: str = Field(min_length=1)
    destination: str = Field(min_length=1)
    interests: list[str] = Field(default_factory=list)
    trip_duration_days: int = Field(default=1, ge=1, le=14)


class CandidateSetResponse(BaseModel):
    mode: PlanningMode
    origin: str
    destination: str
    interests: list[str]
    candidates: list[CandidateWaypoint]


class LocationValidationRequest(BaseModel):
    city: str = Field(min_length=1)
    location: str = Field(min_length=1)


class LocationValidationResponse(BaseModel):
    valid: bool
    city: str
    location: str
    formatted_address: str | None = None
    message: str | None = None


class PlanRequest(BaseModel):
    mode: PlanningMode
    city: str = Field(min_length=1)  # City name for geocoding
    destination: str = Field(min_length=1)  # Specific location within city
    origin: str = ""
    trip_duration_days: int = Field(default=1, ge=1, le=14)
    interests: list[str] = Field(default_factory=list)
    fatigue_threshold_meters: int = Field(default=3000, ge=500, le=20000)
    max_spots: int = Field(default=3, ge=1, le=12)
    selected_waypoints: list[CandidateWaypoint] = Field(default_factory=list)
    transport_mode: TransportMode = Field(default=TransportMode.WALKING)

    @model_validator(mode="after")
    def validate_mode_requirements(self) -> "PlanRequest":
        if self.mode == PlanningMode.POINT_TO_POINT and not self.origin.strip():
            raise ValueError("Origin is required for point-to-point mode.")
        return self


class ItineraryItem(BaseModel):
    id: str
    name: str
    kind: ItineraryItemKind
    sequence: int
    anchor_segment: str
    distance_from_previous_meters: int
    cumulative_distance_meters: int
    reason: str
    confirmed: bool = False
    transport_mode: TransportMode = Field(default=TransportMode.WALKING)


class TravelSummary(BaseModel):
    mode: PlanningMode
    origin: str
    destination: str
    total_distance_meters: int
    fatigue_threshold_meters: int
    rest_stop_count: int
    itinerary_items: list[ItineraryItem]
    transport_mode: TransportMode = Field(default=TransportMode.WALKING)


class BusinessEvent(BaseModel):
    event: str
    payload: dict
