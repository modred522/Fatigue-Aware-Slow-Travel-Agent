# Fatigue-Aware Slow Travel Agent

An AI-powered travel planning app for slow travel with fatigue-aware routing. It supports:

- **Dual Planning Modes**:
  - `Local Explore`: plan nearby places around a single anchor location
  - `Origin to Destination`: generate waypoint candidates first, confirm them, then plan a fatigue-aware route
- **Location Validation**: Verify locations exist within specified cities before planning
- **Multiple Transport Modes**: Walking, public transit, and driving with automatic fallback
- **Fatigue-Aware Routing**: Automatically inserts rest stops when fatigue threshold is exceeded, continues planning until all waypoints are visited
- **Real-time Updates**: Planning events streamed to frontend via Server-Sent Events
- **Flexible LLM Support**: OpenAI-compatible providers including Qwen, Kimi, and others
- **Real distances and times** via AMap (高德地图) API

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | React 19, TypeScript, Vite |
| Backend | FastAPI, Uvicorn |
| Agent Workflow | LangGraph |
| LLM | OpenAI-compatible chat API |
| Map API | AMap |
| Streaming | Server-Sent Events |

## Project Structure

```text
.
├── api.py                 # Legacy FastAPI entry point
├── agent.py               # Legacy agent implementation
├── tools.py               # Legacy tools
├── travel_agent/          # Main package
│   ├── api.py             # FastAPI application
│   ├── config.py          # Configuration management
│   ├── errors.py          # Custom exceptions
│   ├── logger.py          # Logging utilities
│   ├── prompts.py         # LLM prompts
│   ├── schemas.py         # Pydantic models
│   ├── workflow.py        # LangGraph workflow
│   └── services/
│       ├── amap.py        # AMap API client
│       ├── llm.py         # LLM service with multi-provider support
│       └── settings.py    # Settings management
├── frontend/              # React + TypeScript frontend
│   └── src/
│       ├── components/    # React components
│       ├── lib/           # API clients and utilities
│       ├── i18n.ts        # Internationalization
│       └── types.ts       # TypeScript types
├── tests/                 # Unit tests
└── logs/                  # Application logs
```

## Prerequisites

- Python 3.10+
- Node.js 18+
- AMap API key
- An OpenAI-compatible LLM provider:
  - Qwen / DashScope
  - Kimi / Moonshot
  - any other provider exposing OpenAI-compatible `chat.completions` and `models.list`

## Configuration

Create `.env` in the project root:

```env
AMAP_API_KEY=your_amap_api_key_here
LLM_API_KEY=your_openai_compatible_api_key_here
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-turbo
```

Examples:

- Qwen / DashScope
  - `LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1`
  - `LLM_MODEL=qwen-turbo`
- Kimi / Moonshot
  - `LLM_BASE_URL=https://api.moonshot.cn/v1`
  - `LLM_MODEL=kimi-k2-0711-preview`

You can also configure these values in the web UI:

- AMap API key
- LLM API key
- LLM base URL
- fetch model list from provider
- select one available model
- temperature (0-2, Kimi models fixed at 1)

## Getting Started

### 1. Install dependencies

```bash
git clone <repository-url>
cd project-repo

uv venv
uv pip install -r requirements.txt

cd frontend
npm install
cd ..
```

### 2. Run development servers

Backend:

```bash
uv run uvicorn api:api --reload --host 0.0.0.0 --port 8000
```

Frontend:

```bash
cd frontend
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

## Usage

1. Open `Settings`
2. Enter:
   - AMap API key
   - LLM API key
   - LLM Base URL
3. Click `Fetch Models`
4. Choose one available model
5. Set temperature (optional, 0-2)
6. Pick a planning mode:
   - `Local Explore`
   - `Origin to Destination`

### Local Explore

- enter city (e.g., "杭州") and location (e.g., "西湖")
- validate the location exists in the specified city
- optionally enter interests, threshold, duration, max spots
- select transport mode (walking, transit, or driving)
- start planning directly

### Origin to Destination

- enter city, origin and destination
- generate waypoint candidates
- confirm selected waypoint candidates
- select transport mode
- start planning

The dashboard shows:

- planning timeline
- structured itinerary summary
- placeholder panels for map and fatigue visualization

### Transport Modes

- **Walking**: Real walking distance and time, with automatic fallback to driving for long distances
- **Transit**: Public transportation routes (bus/subway)
- **Driving**: Car/motorcycle routes

### Fatigue-Aware Planning

The system tracks cumulative travel distance and automatically inserts rest stops when the fatigue threshold is exceeded. Planning continues after rest stops until all waypoints are visited.

## API Reference

### `GET /api/settings`

Returns current runtime settings with masked secrets.

### `POST /api/settings`

Update runtime settings.

```json
{
  "amap_api_key": "your_key",
  "llm_api_key": "your_llm_key",
  "llm_base_url": "https://api.moonshot.cn/v1",
  "llm_model": "kimi-k2-0711-preview",
  "llm_temperature": "1.0"
}
```

### `POST /api/settings/models`

Fetch models from an OpenAI-compatible provider.

```json
{
  "llm_api_key": "your_llm_key",
  "llm_base_url": "https://api.moonshot.cn/v1"
}
```

### `POST /api/plans/candidates`

Generate waypoint candidates for `point_to_point` mode.

```json
{
  "mode": "point_to_point",
  "origin": "Shanghai",
  "destination": "Suzhou",
  "interests": ["gardens", "cafes"],
  "trip_duration_days": 1
}
```

### `POST /api/validate-location`

Validate that a location exists within a specific city.

```json
{
  "city": "杭州",
  "location": "西湖"
}
```

Response:

```json
{
  "valid": true,
  "formatted_address": "浙江省杭州市西湖区西湖风景区"
}
```

### `POST /api/plans/stream`

Start planning and receive SSE business events.

```json
{
  "mode": "point_to_point",
  "city": "杭州",
  "origin": "杭州东站",
  "destination": "西湖",
  "trip_duration_days": 1,
  "interests": ["gardens", "cafes"],
  "fatigue_threshold_meters": 3000,
  "max_spots": 3,
  "transport_mode": "walking",
  "selected_waypoints": [
    {
      "id": "candidate-zhouzhuang-1",
      "name": "Zhouzhuang",
      "reason": "classic canal town stop",
      "role": "scenic detour"
    }
  ]
}
```

Transport modes: `walking`, `transit`, `driving`

SSE event types:

- `planning_started`
- `itinerary_item_added`
- `segment_distance_updated`
- `fatigue_status_updated`
- `rest_stop_added`
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
.venv\Scripts\python.exe -m unittest tests.test_llm_service tests.test_settings_api
```
