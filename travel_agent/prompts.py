from __future__ import annotations


def build_candidate_prompt(origin: str, destination: str, interests: list[str]) -> str:
    interest_text = ", ".join(interests) if interests else "local culture, food, parks"
    return f"""
You are planning a slow travel route in China.
Route origin: {origin}
Route destination: {destination}
User interests: {interest_text}

Recommend 4 waypoint candidates between the origin and destination.
They should be realistic, pleasant, and suitable for a fatigue-aware itinerary.

Return only JSON as an array with this shape:
[
  {{
    "name": "place name",
    "reason": "short reason under 18 words",
    "role": "midpoint stop | scenic detour | food break | cultural stop"
  }}
]
""".strip()


def build_local_spot_prompt(
    destination: str,
    anchor_location: str,
    interests: list[str],
    existing_names: list[str],
) -> str:
    interest_text = ", ".join(interests) if interests else "parks, streets, cafes, local culture"
    existing = ", ".join(existing_names) if existing_names else "none"
    return f"""
You are a slow-travel planner for {destination}.
Current anchor location: {anchor_location}
Traveler interests: {interest_text}
Already planned places: {existing}

Recommend one nearby place for a relaxed walking itinerary.
Prioritize calm, human-scale places over rushed tourism.

Return only JSON:
{{
  "name": "place name",
  "reason": "short reason under 18 words"
}}
""".strip()


def build_rest_stop_prompt(destination: str, current_location: str) -> str:
    return f"""
You are planning a fatigue-aware route in {destination}.
The traveler needs a rest stop near {current_location}.

Recommend one cafe, teahouse, or quiet rest place nearby.

Return only JSON:
{{
  "name": "place name",
  "reason": "short reason under 18 words"
}}
""".strip()
