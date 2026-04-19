from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from travel_agent.prompts import build_local_spot_prompt, build_rest_stop_prompt
from travel_agent.schemas import (
    BusinessEvent,
    CandidateWaypoint,
    ItineraryItem,
    ItineraryItemKind,
    PlanningMode,
    PlanningStage,
    PlanRequest,
    TravelSummary,
)
from travel_agent.errors import ExternalServiceError
from travel_agent.services.amap import AMapClient
from travel_agent.services.llm import invoke_text
from travel_agent.utils import extract_json, slugify


class TravelState(TypedDict):
    mode: str
    planning_stage: str
    origin: str
    destination: str
    city: str  # City name for geocoding (e.g., "杭州" from "杭州西湖")
    anchor_location: str
    trip_duration_days: int
    interests: list[str]
    fatigue_threshold_meters: int
    max_spots: int
    transport_mode: str
    candidate_waypoints: list[dict[str, Any]]
    selected_waypoints: list[dict[str, Any]]
    route_targets: list[dict[str, Any]]
    next_target_index: int
    itinerary_items: list[dict[str, Any]]
    cumulative_distance_meters: int
    current_segment_distance_meters: int
    needs_rest: bool
    rest_stop_count: int  # Track rest stops to prevent infinite loops
    event_log: list[dict[str, Any]]
    emitted_events: list[dict[str, Any]]
    summary: dict[str, Any] | None


def build_initial_state(request: PlanRequest) -> TravelState:
    itinerary_items: list[dict[str, Any]] = []
    route_targets: list[dict[str, Any]] = []
    anchor_location = request.destination
    city = request.city

    if request.mode == PlanningMode.POINT_TO_POINT:
        anchor_location = request.origin
        # For point-to-point, validate both origin and destination are in the same city
        # If city is not explicitly provided, use the one from request
        origin_item = ItineraryItem(
            id=f"origin-{slugify(request.origin)}",
            name=request.origin,
            kind=ItineraryItemKind.ORIGIN,
            sequence=1,
            anchor_segment="route-start",
            distance_from_previous_meters=0,
            cumulative_distance_meters=0,
            reason="Trip origin set by the traveler.",
            confirmed=True,
        )
        itinerary_items.append(origin_item.model_dump())
        for waypoint in request.selected_waypoints:
            route_targets.append(
                {
                    "id": waypoint.id,
                    "name": waypoint.name,
                    "reason": waypoint.reason,
                    "role": waypoint.role,
                    "kind": ItineraryItemKind.WAYPOINT.value,
                }
            )
        route_targets.append(
            {
                "id": f"destination-{slugify(request.destination)}",
                "name": request.destination,
                "reason": "Final destination selected by the traveler.",
                "role": "destination",
                "kind": ItineraryItemKind.DESTINATION.value,
            }
        )

    return {
        "mode": request.mode.value,
        "planning_stage": PlanningStage.ITINERARY_PLANNING.value,
        "origin": request.origin,
        "destination": request.destination,
        "city": city,
        "anchor_location": anchor_location,
        "trip_duration_days": request.trip_duration_days,
        "interests": request.interests,
        "fatigue_threshold_meters": request.fatigue_threshold_meters,
        "max_spots": request.max_spots,
        "transport_mode": request.transport_mode.value,
        "candidate_waypoints": [candidate.model_dump() for candidate in request.selected_waypoints],
        "selected_waypoints": [candidate.model_dump() for candidate in request.selected_waypoints],
        "route_targets": route_targets,
        "next_target_index": 0,
        "itinerary_items": itinerary_items,
        "cumulative_distance_meters": 0,
        "current_segment_distance_meters": 0,
        "needs_rest": False,
        "rest_stop_count": 0,
        "event_log": [],
        "emitted_events": [],
        "summary": None,
    }


def _append_event(state: TravelState, event: BusinessEvent) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    item = event.model_dump()
    return [*state["event_log"], item], [item]


def _spot_count(state: TravelState) -> int:
    return sum(
        1
        for item in state["itinerary_items"]
        if item["kind"] in {ItineraryItemKind.SPOT.value, ItineraryItemKind.WAYPOINT.value}
    )


def planner_node(state: TravelState) -> dict[str, Any]:
    itinerary = [*state["itinerary_items"]]
    sequence = len(itinerary) + 1

    if state["mode"] == PlanningMode.LOCAL_EXPLORE.value:
        existing_names = [item["name"] for item in itinerary]
        payload = extract_json(
            invoke_text(
                build_local_spot_prompt(
                    destination=state["destination"],
                    anchor_location=state["anchor_location"],
                    interests=state["interests"],
                    existing_names=existing_names,
                )
            )
        )
        item = ItineraryItem(
            id=f"spot-{slugify(payload['name'])}-{sequence}",
            name=payload["name"],
            kind=ItineraryItemKind.SPOT,
            sequence=sequence,
            anchor_segment=state["anchor_location"],
            distance_from_previous_meters=0,
            cumulative_distance_meters=state["cumulative_distance_meters"],
            reason=payload["reason"],
            confirmed=False,
        )
    else:
        target = state["route_targets"][state["next_target_index"]]
        item = ItineraryItem(
            id=target["id"],
            name=target["name"],
            kind=ItineraryItemKind(target["kind"]),
            sequence=sequence,
            anchor_segment=state["anchor_location"],
            distance_from_previous_meters=0,
            cumulative_distance_meters=state["cumulative_distance_meters"],
            reason=target["reason"],
            confirmed=True,
        )

    itinerary.append(item.model_dump())
    event_log, emitted_events = _append_event(
        state,
        BusinessEvent(
            event="itinerary_item_added",
            payload={"item": item.model_dump(), "mode": state["mode"]},
        ),
    )

    updates: dict[str, Any] = {
        "itinerary_items": itinerary,
        "event_log": event_log,
        "emitted_events": emitted_events,
    }
    if state["mode"] == PlanningMode.POINT_TO_POINT.value:
        updates["next_target_index"] = state["next_target_index"] + 1
    return updates


def distance_calculator_node(state: TravelState) -> dict[str, Any]:
    itinerary = [*state["itinerary_items"]]
    latest = dict(itinerary[-1])
    origin_name = state["anchor_location"]
    destination_name = latest["name"]
    transport_mode = state.get("transport_mode", "walking")

    # Validate location names
    if not origin_name or not origin_name.strip():
        raise ExternalServiceError(
            f"Origin location is empty. This may be a planning logic error. "
            f"Please check your input and try again."
        )
    if not destination_name or not destination_name.strip():
        raise ExternalServiceError(
            f"Destination location is empty. The LLM may have returned an invalid place name. "
            f"Please try again with different interests or destination."
        )

    # Use the extracted city name for geocoding
    city = state.get("city", state["destination"])

    try:
        route = AMapClient().route(origin_name, destination_name, city, mode=transport_mode)
    except ExternalServiceError as e:
        # Add context about which locations caused the error
        raise ExternalServiceError(
            f"Failed to calculate route from '{origin_name}' to '{destination_name}'. "
            f"Error: {e}"
        ) from e
    distance = route["distance_meters"]
    cumulative = state["cumulative_distance_meters"] + distance
    latest["distance_from_previous_meters"] = distance
    latest["cumulative_distance_meters"] = cumulative
    latest["transport_mode"] = transport_mode
    itinerary[-1] = latest

    event_log, emitted_events = _append_event(
        state,
        BusinessEvent(
            event="segment_distance_updated",
            payload={
                "item_id": latest["id"],
                "from": origin_name,
                "to": destination_name,
                "segment_distance_meters": distance,
                "cumulative_distance_meters": cumulative,
                "transport_mode": transport_mode,
            },
        ),
    )

    return {
        "itinerary_items": itinerary,
        "anchor_location": destination_name,
        "current_segment_distance_meters": distance,
        "cumulative_distance_meters": cumulative,
        "event_log": event_log,
        "emitted_events": emitted_events,
    }


def fatigue_router_node(state: TravelState) -> dict[str, Any]:
    needs_rest = state["cumulative_distance_meters"] >= state["fatigue_threshold_meters"]
    payload = {
        "needs_rest": needs_rest,
        "cumulative_distance_meters": state["cumulative_distance_meters"],
        "fatigue_threshold_meters": state["fatigue_threshold_meters"],
        "segment_distance_meters": state["current_segment_distance_meters"],
    }
    event_log, emitted_events = _append_event(
        state,
        BusinessEvent(event="fatigue_status_updated", payload=payload),
    )
    return {"needs_rest": needs_rest, "event_log": event_log, "emitted_events": emitted_events}


def rest_stop_finder_node(state: TravelState) -> dict[str, Any]:
    itinerary = [*state["itinerary_items"]]
    sequence = len(itinerary) + 1
    payload = extract_json(
        invoke_text(build_rest_stop_prompt(state["destination"], state["anchor_location"]))
    )
    item = ItineraryItem(
        id=f"rest-{slugify(payload['name'])}-{sequence}",
        name=payload["name"],
        kind=ItineraryItemKind.REST_STOP,
        sequence=sequence,
        anchor_segment=state["anchor_location"],
        distance_from_previous_meters=0,
        cumulative_distance_meters=state["cumulative_distance_meters"],
        reason=payload["reason"],
        confirmed=False,
    )
    itinerary.append(item.model_dump())
    event_log, emitted_events = _append_event(
        state,
        BusinessEvent(
            event="rest_stop_added",
            payload={
                "item": item.model_dump(),
                "cumulative_distance_meters": state["cumulative_distance_meters"],
            },
        ),
    )
    # Reset fatigue counter after rest stop to allow continuing the journey
    # Also increment rest_stop_count to prevent infinite loops
    rest_stop_count = state.get("rest_stop_count", 0) + 1
    return {
        "itinerary_items": itinerary,
        "cumulative_distance_meters": 0,
        "needs_rest": False,
        "rest_stop_count": rest_stop_count,
        "event_log": event_log,
        "emitted_events": emitted_events,
    }


def summary_builder_node(state: TravelState) -> dict[str, Any]:
    from travel_agent.schemas import TransportMode

    summary = TravelSummary(
        mode=PlanningMode(state["mode"]),
        origin=state["origin"],
        destination=state["destination"],
        total_distance_meters=state["cumulative_distance_meters"],
        fatigue_threshold_meters=state["fatigue_threshold_meters"],
        rest_stop_count=sum(
            1 for item in state["itinerary_items"] if item["kind"] == ItineraryItemKind.REST_STOP.value
        ),
        itinerary_items=[ItineraryItem.model_validate(item) for item in state["itinerary_items"]],
        transport_mode=TransportMode(state.get("transport_mode", "walking")),
    ).model_dump()

    event_log, emitted_events = _append_event(
        state,
        BusinessEvent(event="planning_completed", payload=summary),
    )
    return {
        "planning_stage": PlanningStage.COMPLETED.value,
        "summary": summary,
        "event_log": event_log,
        "emitted_events": emitted_events,
    }


def route_from_fatigue(state: TravelState) -> str:
    # Check for infinite loop protection - max 5 rest stops
    rest_stop_count = state.get("rest_stop_count", 0)
    if rest_stop_count >= 5:
        # Force completion if too many rest stops
        return "summary_builder"

    if state["needs_rest"]:
        return "rest_stop_finder"

    if state["mode"] == PlanningMode.LOCAL_EXPLORE.value:
        if _spot_count(state) >= state["max_spots"]:
            return "summary_builder"
        return "planner"

    if state["next_target_index"] >= len(state["route_targets"]):
        return "summary_builder"
    return "planner"


workflow = StateGraph(TravelState)
workflow.add_node("planner", planner_node)
workflow.add_node("distance_calculator", distance_calculator_node)
workflow.add_node("fatigue_router", fatigue_router_node)
workflow.add_node("rest_stop_finder", rest_stop_finder_node)
workflow.add_node("summary_builder", summary_builder_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "distance_calculator")
workflow.add_edge("distance_calculator", "fatigue_router")
workflow.add_conditional_edges(
    "fatigue_router",
    route_from_fatigue,
    {
        "planner": "planner",
        "rest_stop_finder": "rest_stop_finder",
        "summary_builder": "summary_builder",
    },
)
# After rest stop, reset fatigue and continue planning instead of ending
workflow.add_edge("rest_stop_finder", "planner")
workflow.add_edge("summary_builder", END)

travel_workflow = workflow.compile()


def generate_waypoint_candidates(origin: str, destination: str, interests: list[str]) -> list[CandidateWaypoint]:
    from travel_agent.prompts import build_candidate_prompt

    raw = invoke_text(build_candidate_prompt(origin, destination, interests))
    parsed = extract_json(raw)
    candidates: list[CandidateWaypoint] = []
    for index, item in enumerate(parsed[:5], start=1):
        candidate = CandidateWaypoint(
            id=f"candidate-{slugify(item['name'])}-{index}",
            name=item["name"],
            reason=item["reason"],
            role=item["role"],
        )
        candidates.append(candidate)
    return candidates
