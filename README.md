# Fatigue-Aware Slow Travel Agent

A dual-mode travel planning app for slow travel. The system combines an LLM, AMap routing/geocoding, a LangGraph workflow, and a React dashboard to generate itineraries that respect fatigue thresholds, insert rest stops, and stream planning progress in real time.

## What It Does

- `Local Explore`
  Around one anchor location, recommend nearby POIs and build a slow-travel itinerary.
- `Origin to Destination`
  Generate waypoint candidates first, let the user confirm them, then plan the full route.
- `Fatigue-aware routing`
  Track continuous walking distance, insert rest stops when fatigue exceeds the threshold, and support explicit midway rest on long but still reachable segments.
- `Map dashboard`
  Render itinerary markers and route lines on an AMap panel.
- `Fatigue curve`
  Visualize fatigue buildup, peaks, and reset points from the generated itinerary.
- `OpenAI-compatible LLM providers`
  Supports Qwen, Kimi, and any provider exposing compatible `chat.completions` plus model listing.

## Current Product Shape

### 1. Local Explore

Input:

- city
- destination / anchor location
- interests
- fatigue threshold
- max spots
- transport mode

Behavior:

- The planner recommends nearby POIs around the anchor.
- Distances are validated with AMap, not trusted from the LLM.
- If a POI is too far away, the workflow rejects it and replans.
- If a segment is longer than one fatigue threshold but still reachable within two thresholds, the workflow inserts a real midway rest stop and then completes the remaining segment.
- If repeated LLM recommendations are unusable, the workflow falls back to a nearby POI discovered from AMap.

### 2. Origin to Destination

Input:

- city
- origin
- destination
- interests
- fatigue threshold
- max spots
- transport mode

Behavior:

- `/api/plans/candidates` generates 3-5 waypoint candidates.
- The user confirms selected candidates in the frontend.
- `/api/plans/stream` plans the final itinerary using the confirmed waypoints.

## Tech Stack

| Layer | Technology |
| --- | --- |
| Frontend | React 19, TypeScript, Vite |
| Backend | FastAPI, Uvicorn |
| Workflow | LangGraph |
| Map / Geocoding | AMap Web Service + AMap JS SDK |
| LLM | OpenAI-compatible provider |
| Streaming | Server-Sent Events |

## Project Structure

```text
.
|-- travel_agent/
|   |-- api.py
|   |-- config.py
|   |-- errors.py
|   |-- logger.py
|   |-- prompts.py
|   |-- schemas.py
|   |-- workflow.py
|   `-- services/
|       |-- amap.py
|       `-- llm.py
|-- frontend/
|   |-- src/
|   |   |-- components/
|   |   |-- lib/
|   |   |-- i18n.ts
|   |   `-- types.ts
|   `-- package.json
|-- tests/
|-- requirements.txt
|-- .env.example
`-- README.md
```

## Requirements

- Python 3.10+
- Node.js 18+
- `uv`
- AMap API key
- An OpenAI-compatible LLM API key

## Configuration

Create a `.env` file in the project root.

Example:

```env
AMAP_API_KEY=your_amap_key
LLM_API_KEY=your_provider_key
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-turbo
LLM_TEMPERATURE=0.3
```

Example provider settings:

- Qwen / DashScope
  - `LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1`
  - `LLM_MODEL=qwen-turbo`
- Kimi / Moonshot
  - `LLM_BASE_URL=https://api.moonshot.cn/v1`
  - `LLM_MODEL=kimi-k2-0711-preview`

You can also configure all of these in the web UI:

- AMap API key
- LLM API key
- LLM base URL
- fetch model list
- select model
- temperature

## Installation

### Backend

```bash
uv venv
uv pip install -r requirements.txt
```

### Frontend

```bash
cd frontend
npm install
cd ..
```

## Run Locally

### Backend

```bash
uv run uvicorn travel_agent.api:api --reload --host 127.0.0.1 --port 8000
```

### Frontend

```bash
cd frontend
npm run dev
```

Frontend default URL:

- [http://localhost:5173](http://localhost:5173)

Backend default URL:

- [http://127.0.0.1:8000](http://127.0.0.1:8000)

## Dashboard Features

The current frontend includes:

- dual-mode planning form
- candidate confirmation flow
- streaming planning timeline
- structured itinerary summary
- fatigue curve panel
- live AMap panel with itinerary markers and route line

Map notes:

- The map panel uses the browser-side AMap JS SDK.
- The frontend reads the AMap key from `/api/settings/map`.
- If the key is missing, the map panel stays disabled and shows a hint.

## Workflow Notes

The planning workflow is centered in `travel_agent/workflow.py`.

Key behaviors:

- LLM suggestions are always validated by AMap route distance.
- Fatigue threshold controls continuous walking, not total trip distance.
- Neighboring POIs are allowed to be farther than one threshold if the route can be completed with one rest stop in between.
- Long-but-reachable segments are split into:
  1. midpoint rest stop
  2. remaining segment to the destination POI
- Repeated unusable local recommendations can fall back to nearby AMap-discovered POIs.

## API Overview

### `GET /api/settings`

Returns runtime settings with masked secrets.

### `GET /api/settings/map`

Returns AMap map settings for the frontend map panel.

Response:

```json
{
  "amap_api_key": "your_amap_key",
  "amap_api_key_set": true
}
```

### `POST /api/settings`

Update runtime settings.

### `POST /api/settings/models`

Fetch available models from the configured OpenAI-compatible provider.

### `POST /api/validate-location`

Validate that a location exists in the specified city.

### `POST /api/plans/candidates`

Generate waypoint candidates for `point_to_point`.

### `POST /api/plans/stream`

Run itinerary planning and consume streamed business events.

Important SSE events:

- `planning_started`
- `candidate_generated`
- `candidate_set_ready`
- `itinerary_item_added`
- `segment_distance_updated`
- `mid_segment_rest_required`
- `fatigue_status_updated`
- `rest_stop_added`
- `fallback_spot_selected`
- `planning_completed`
- `error`

## Tests

Frontend type check:

```bash
cd frontend
npx tsc -b
```

Backend unit tests:

```bash
.venv\Scripts\python.exe -m unittest tests.test_settings_api tests.test_llm_service tests.test_workflow_distance_guard
```

## Known Limitations

- The map panel currently draws a straight polyline between itinerary points, not a true AMap navigation polyline.
- The fatigue curve is derived from itinerary data; it is not yet a backend-generated analytics object.
- No database is used yet. Settings and plans are runtime-only.
- Candidate quality still depends on provider/model quality even though routing validation and fallback logic are in place.

## Next Recommended Improvements

- true route polyline rendering from AMap path results
- marker-to-itinerary hover/highlight linkage
- richer map popups with segment distance and reason
- backend-produced fatigue analytics object instead of frontend-only derivation
- persistent storage for settings, plans, and session history
