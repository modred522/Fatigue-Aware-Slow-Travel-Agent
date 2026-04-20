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
    total_distance_meters: int
    current_segment_distance_meters: int
    needs_rest: bool
    rest_stop_count: int  # Track rest stops to prevent infinite loops
    replan_required: bool
    distance_replan_count: int
    distance_guard_exhausted: bool
    fallback_attempted: bool
    pending_destination_item: dict[str, Any] | None
    pending_remaining_distance_meters: int
    pending_full_segment_distance_meters: int
    pending_midpoint_coords: str
    rejected_spot_names: list[str]
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
            location_coords=_safe_geocode_coords(request.origin, city),
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
        "total_distance_meters": 0,
        "current_segment_distance_meters": 0,
        "needs_rest": False,
        "rest_stop_count": 0,
        "replan_required": False,
        "distance_replan_count": 0,
        "distance_guard_exhausted": False,
        "fallback_attempted": False,
        "pending_destination_item": None,
        "pending_remaining_distance_meters": 0,
        "pending_full_segment_distance_meters": 0,
        "pending_midpoint_coords": "",
        "rejected_spot_names": [],
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


def _safe_geocode_coords(name: str, city: str) -> str | None:
    try:
        return AMapClient().geocode(name, city)
    except ExternalServiceError:
        return None


def _max_segment_distance_meters(state: TravelState) -> int:
    """Per-hop budget derived from fatigue threshold.

    Rule: a traveler can rest once between two POIs, so the distance between
    neighboring POIs should stay within roughly two fatigue thresholds.
    """
    threshold = state["fatigue_threshold_meters"]
    return max(1200, threshold * 2)


def _midpoint_coords(origin_coords: str, destination_coords: str) -> str:
    origin_lng, origin_lat = [float(value) for value in origin_coords.split(",")]
    destination_lng, destination_lat = [float(value) for value in destination_coords.split(",")]
    return f"{(origin_lng + destination_lng) / 2:.6f},{(origin_lat + destination_lat) / 2:.6f}"


def planner_node(state: TravelState) -> dict[str, Any]:
    itinerary = [*state["itinerary_items"]]
    sequence = len(itinerary) + 1

    if state["mode"] == PlanningMode.LOCAL_EXPLORE.value:
        existing_names = [item["name"] for item in itinerary]
        max_segment_distance = _max_segment_distance_meters(state)
        payload = extract_json(
            invoke_text(
                build_local_spot_prompt(
                    destination=state["destination"],
                    anchor_location=state["anchor_location"],
                    interests=state["interests"],
                    existing_names=existing_names,
                    rejected_names=state.get("rejected_spot_names", []),
                    max_segment_distance_meters=max_segment_distance,
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
            location_coords=None,
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
            location_coords=None,
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
    else:
        updates["replan_required"] = False
        updates["distance_guard_exhausted"] = False
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
    max_segment_distance = _max_segment_distance_meters(state)
    fatigue_threshold = state["fatigue_threshold_meters"]

    if distance > fatigue_threshold and distance <= max_segment_distance:
        itinerary.pop()
        latest["location_coords"] = route["destination_coords"]
        midpoint_coords = _midpoint_coords(route["origin_coords"], route["destination_coords"])
        remaining_distance = max(0, distance - fatigue_threshold)
        event_log, emitted_events = _append_event(
            state,
            BusinessEvent(
                event="mid_segment_rest_required",
                payload={
                    "from": origin_name,
                    "to": destination_name,
                    "segment_distance_meters": distance,
                    "fatigue_threshold_meters": fatigue_threshold,
                    "remaining_distance_meters": remaining_distance,
                },
            ),
        )
        return {
            "itinerary_items": itinerary,
            "current_segment_distance_meters": 0,
            "pending_destination_item": latest,
            "pending_remaining_distance_meters": remaining_distance,
            "pending_full_segment_distance_meters": distance,
            "pending_midpoint_coords": midpoint_coords,
            "event_log": event_log,
            "emitted_events": emitted_events,
        }

    if state["mode"] == PlanningMode.LOCAL_EXPLORE.value and distance > max_segment_distance:
        retry_count = state.get("distance_replan_count", 0) + 1
        itinerary.pop()
        rejected = [*state.get("rejected_spot_names", [])]
        if destination_name not in rejected:
            rejected.append(destination_name)
        event_log, emitted_events = _append_event(
            state,
            BusinessEvent(
                event="itinerary_item_rejected",
                payload={
                    "name": destination_name,
                    "reason": "too_far_from_anchor",
                    "segment_distance_meters": distance,
                    "max_segment_distance_meters": max_segment_distance,
                    "distance_policy": "single_rest_reachable_within_two_thresholds",
                    "retry_count": retry_count,
                    "exhausted": retry_count >= 3,
                },
            ),
        )
        exhausted = retry_count >= 3
        return {
            "itinerary_items": itinerary,
            "current_segment_distance_meters": 0,
            "replan_required": not exhausted,
            "distance_replan_count": retry_count,
            "distance_guard_exhausted": exhausted,
            "rejected_spot_names": rejected,
            "event_log": event_log,
            "emitted_events": emitted_events,
        }

    cumulative = state["cumulative_distance_meters"] + distance
    total = state["total_distance_meters"] + distance
    latest["distance_from_previous_meters"] = distance
    latest["cumulative_distance_meters"] = cumulative
    latest["transport_mode"] = transport_mode
    latest["location_coords"] = route["destination_coords"]
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
                "total_distance_meters": total,
                "transport_mode": transport_mode,
            },
        ),
    )

    return {
        "itinerary_items": itinerary,
        "anchor_location": destination_name,
        "current_segment_distance_meters": distance,
        "cumulative_distance_meters": cumulative,
        "total_distance_meters": total,
        "replan_required": False,
        "distance_replan_count": 0,
        "distance_guard_exhausted": False,
        "pending_destination_item": None,
        "pending_remaining_distance_meters": 0,
        "pending_full_segment_distance_meters": 0,
        "pending_midpoint_coords": "",
        "event_log": event_log,
        "emitted_events": emitted_events,
    }


def fatigue_router_node(state: TravelState) -> dict[str, Any]:
    needs_rest = state["cumulative_distance_meters"] >= state["fatigue_threshold_meters"]
    payload = {
        "needs_rest": needs_rest,
        "cumulative_distance_meters": state["cumulative_distance_meters"],
        "total_distance_meters": state["total_distance_meters"],
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
        location_coords=_safe_geocode_coords(payload["name"], state["destination"]),
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
        "current_segment_distance_meters": 0,
        "needs_rest": False,
        "rest_stop_count": rest_stop_count,
        "replan_required": False,
        "distance_replan_count": 0,
        "distance_guard_exhausted": False,
        "event_log": event_log,
        "emitted_events": emitted_events,
    }


def mid_segment_rest_node(state: TravelState) -> dict[str, Any]:
    itinerary = [*state["itinerary_items"]]
    sequence = len(itinerary) + 1
    anchor = state["anchor_location"]
    city = state.get("city", state["destination"])
    fatigue_threshold = state["fatigue_threshold_meters"]
    pending_item = state.get("pending_destination_item")
    midpoint_coords = state.get("pending_midpoint_coords", "")
    excluded = {*(item["name"] for item in itinerary)}

    nearby = AMapClient().nearby_pois_by_location(
        location_coords=midpoint_coords,
        city=city,
        radius_meters=max(300, fatigue_threshold // 2),
        limit=12,
    )
    candidate = next((poi for poi in nearby if poi["name"] not in excluded), None)

    if candidate is None:
        nearby_destination = AMapClient().nearby_pois(
            anchor_name=str(pending_item["name"]),
            city=city,
            radius_meters=max(300, fatigue_threshold // 2),
            limit=8,
        )
        candidate = next((poi for poi in nearby_destination if poi["name"] not in excluded), None)

    if candidate is None:
        event_log, emitted_events = _append_event(
            state,
            BusinessEvent(
                event="fallback_spot_failed",
                payload={
                    "anchor_location": anchor,
                    "reason": "no_mid_segment_rest_stop_found",
                    "max_segment_distance_meters": fatigue_threshold,
                },
            ),
        )
        return {
            "distance_guard_exhausted": True,
            "event_log": event_log,
            "emitted_events": emitted_events,
        }

    item = ItineraryItem(
        id=f"rest-midway-{slugify(candidate['name'])}-{sequence}",
        name=candidate["name"],
        kind=ItineraryItemKind.REST_STOP,
        sequence=sequence,
        anchor_segment=anchor,
        distance_from_previous_meters=fatigue_threshold,
        cumulative_distance_meters=fatigue_threshold,
        reason=f"Mid-route rest before continuing to {pending_item['name']}.",
        confirmed=True,
        location_coords=candidate.get("location_coords"),
    )
    itinerary.append(item.model_dump())
    total_after_rest = state["total_distance_meters"] + fatigue_threshold

    event_log, emitted_events = _append_event(
        state,
        BusinessEvent(
            event="rest_stop_added",
            payload={
                "item": item.model_dump(),
                "cumulative_distance_meters": fatigue_threshold,
                "rest_kind": "mid_segment",
            },
        ),
    )
    return {
        "itinerary_items": itinerary,
        "total_distance_meters": total_after_rest,
        "cumulative_distance_meters": 0,
        "current_segment_distance_meters": 0,
        "needs_rest": False,
        "rest_stop_count": state.get("rest_stop_count", 0) + 1,
        "event_log": event_log,
        "emitted_events": emitted_events,
    }


def complete_pending_segment_node(state: TravelState) -> dict[str, Any]:
    itinerary = [*state["itinerary_items"]]
    pending_item = dict(state["pending_destination_item"])
    remaining_distance = state.get("pending_remaining_distance_meters", 0)
    total = state["total_distance_meters"] + remaining_distance

    pending_item["sequence"] = len(itinerary) + 1
    pending_item["distance_from_previous_meters"] = remaining_distance
    pending_item["cumulative_distance_meters"] = remaining_distance
    pending_item["transport_mode"] = state.get("transport_mode", "walking")
    itinerary.append(pending_item)

    event_log, emitted_events = _append_event(
        state,
        BusinessEvent(
            event="segment_distance_updated",
            payload={
                "item_id": pending_item["id"],
                "from": itinerary[-2]["name"] if len(itinerary) > 1 else state["anchor_location"],
                "to": pending_item["name"],
                "segment_distance_meters": remaining_distance,
                "cumulative_distance_meters": remaining_distance,
                "total_distance_meters": total,
                "transport_mode": state.get("transport_mode", "walking"),
            },
        ),
    )

    return {
        "itinerary_items": itinerary,
        "anchor_location": pending_item["name"],
        "current_segment_distance_meters": remaining_distance,
        "cumulative_distance_meters": remaining_distance,
        "total_distance_meters": total,
        "pending_destination_item": None,
        "pending_remaining_distance_meters": 0,
        "pending_full_segment_distance_meters": 0,
        "pending_midpoint_coords": "",
        "event_log": event_log,
        "emitted_events": emitted_events,
    }


def fallback_local_spot_node(state: TravelState) -> dict[str, Any]:
    itinerary = [*state["itinerary_items"]]
    sequence = len(itinerary) + 1
    anchor = state["anchor_location"]
    city = state.get("city", state["destination"])
    max_segment_distance = _max_segment_distance_meters(state)
    excluded = {
        *(item["name"] for item in itinerary),
        *state.get("rejected_spot_names", []),
    }

    nearby = AMapClient().nearby_pois(
        anchor_name=anchor,
        city=city,
        radius_meters=max_segment_distance,
        limit=12,
        keywords=state.get("interests", []),
    )
    candidate = next((poi for poi in nearby if poi["name"] not in excluded), None)

    if candidate is None:
        event_log, emitted_events = _append_event(
            state,
            BusinessEvent(
                event="fallback_spot_failed",
                payload={
                    "anchor_location": anchor,
                    "reason": "no_nearby_candidate_within_budget",
                    "max_segment_distance_meters": max_segment_distance,
                },
            ),
        )
        return {
            "distance_guard_exhausted": True,
            "fallback_attempted": True,
            "event_log": event_log,
            "emitted_events": emitted_events,
        }

    item = ItineraryItem(
        id=f"spot-fallback-{slugify(candidate['name'])}-{sequence}",
        name=candidate["name"],
        kind=ItineraryItemKind.SPOT,
        sequence=sequence,
        anchor_segment=anchor,
        distance_from_previous_meters=0,
        cumulative_distance_meters=state["cumulative_distance_meters"],
        reason="Fallback nearby POI selected after repeated long-distance rejections.",
        confirmed=True,
        location_coords=candidate.get("location_coords"),
    )
    itinerary.append(item.model_dump())
    event_log, emitted_events = _append_event(
        state,
        BusinessEvent(
            event="fallback_spot_selected",
            payload={
                "item": item.model_dump(),
                "anchor_location": anchor,
                "candidate_distance_meters": candidate["distance_meters"],
                "max_segment_distance_meters": max_segment_distance,
            },
        ),
    )
    return {
        "itinerary_items": itinerary,
        "replan_required": False,
        "distance_replan_count": 0,
        "distance_guard_exhausted": False,
        "fallback_attempted": True,
        "event_log": event_log,
        "emitted_events": emitted_events,
    }


def summary_builder_node(state: TravelState) -> dict[str, Any]:
    from travel_agent.schemas import TransportMode

    summary = TravelSummary(
        mode=PlanningMode(state["mode"]),
        origin=state["origin"],
        destination=state["destination"],
        total_distance_meters=state["total_distance_meters"],
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
    if state.get("distance_guard_exhausted", False):
        return "summary_builder"

    if state.get("replan_required", False):
        return "planner"

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


def route_from_distance(state: TravelState) -> str:
    if state.get("pending_destination_item"):
        return "mid_segment_rest"
    if state.get("distance_guard_exhausted", False):
        if (
            state["mode"] == PlanningMode.LOCAL_EXPLORE.value
            and not state.get("fallback_attempted", False)
            and _spot_count(state) < state["max_spots"]
        ):
            return "fallback_local_spot"
        return "summary_builder"
    if state.get("replan_required", False):
        return "planner"
    return "fatigue_router"


workflow = StateGraph(TravelState)
workflow.add_node("planner", planner_node)
workflow.add_node("distance_calculator", distance_calculator_node)
workflow.add_node("fatigue_router", fatigue_router_node)
workflow.add_node("rest_stop_finder", rest_stop_finder_node)
workflow.add_node("mid_segment_rest", mid_segment_rest_node)
workflow.add_node("complete_pending_segment", complete_pending_segment_node)
workflow.add_node("fallback_local_spot", fallback_local_spot_node)
workflow.add_node("summary_builder", summary_builder_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "distance_calculator")
workflow.add_conditional_edges(
    "distance_calculator",
    route_from_distance,
    {
        "planner": "planner",
        "fatigue_router": "fatigue_router",
        "mid_segment_rest": "mid_segment_rest",
        "fallback_local_spot": "fallback_local_spot",
        "summary_builder": "summary_builder",
    },
)
workflow.add_edge("mid_segment_rest", "complete_pending_segment")
workflow.add_edge("complete_pending_segment", "fatigue_router")
workflow.add_edge("fallback_local_spot", "distance_calculator")
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
