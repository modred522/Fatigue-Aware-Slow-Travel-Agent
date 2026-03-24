# Fatigue-Aware Slow Travel Agent / 疲劳感知慢旅行助手

An AI-powered travel planning web application that recommends scenic spots while monitoring your physical fatigue level. The agent intelligently plans walking routes and suggests rest stops when you've walked too far.

一款基于 AI 的旅行规划 Web 应用，在推荐景点的同时监测你的体力疲劳程度。智能规划步行路线，当累计步行距离过长时自动推荐休息点。

---

## Table of Contents / 目录

- [Architecture / 系统架构](#architecture--系统架构)
- [Tech Stack / 技术栈](#tech-stack--技术栈)
- [Project Structure / 项目结构](#project-structure--项目结构)
- [Prerequisites / 环境要求](#prerequisites--环境要求)
- [Configuration / 配置](#configuration--配置)
- [Getting Started / 快速开始](#getting-started--快速开始)
- [Usage / 使用方法](#usage--使用方法)
- [API Reference / API 参考](#api-reference--api-参考)

---

## Architecture / 系统架构

```
┌─────────────────────────────────┐
│         React Frontend          │
│  (Vite + TypeScript)            │
│                                 │
│  DestinationInput → StepTimeline│
│  Settings Panel   → Itinerary   │
│         ▲                       │
│         │ SSE (EventSource)     │
└─────────┼───────────────────────┘
          │
┌─────────▼───────────────────────┐
│       FastAPI Backend           │
│  GET /api/plan/stream           │
│  GET/POST /api/settings         │
│         │                       │
│         ▼                       │
│  ┌─────────────────────────┐    │
│  │   LangGraph Workflow    │    │
│  │                         │    │
│  │  Planner ──► Calculator │    │
│  │    ▲            │       │    │
│  │    │      ┌─────┴─────┐ │    │
│  │    └──────┤  Router   │ │    │
│  │           └─────┬─────┘ │    │
│  │                 ▼       │    │
│  │           Rest Stop     │    │
│  └─────────────────────────┘    │
│         │              │        │
│   ChatTongyi      AMap API      │
│   (Qwen LLM)    (Walking Dist) │
└─────────────────────────────────┘
```

**Workflow / 工作流程：**

1. **Planner** — LLM recommends the next scenic spot based on current location / LLM 根据当前位置推荐下一个景点
2. **Calculator** — Calls AMap API to get real walking distance, checks fatigue (threshold: 3000m) / 调用高德 API 获取真实步行距离，检查疲劳度（阈值 3000m）
3. **Router** — Decides next action / 决定下一步操作：
   - Fatigue threshold exceeded → Rest Stop / 超过疲劳阈值 → 休息推荐
   - 3+ spots planned → End / 已规划 3 个以上景点 → 结束
   - Otherwise → Loop back to Planner / 否则 → 继续规划
4. **Rest Stop** — LLM recommends a nearby cafe or teahouse / LLM 推荐附近的咖啡馆或茶馆

**Streaming / 流式推送：** Each node's output is streamed to the frontend via Server-Sent Events (SSE) in real time, so users can watch the planning process unfold step by step.

每个节点的输出通过 SSE 实时推送到前端，用户可以逐步观看规划过程。

---

## Tech Stack / 技术栈

| Layer | Technology |
|-------|-----------|
| Frontend | React 19, TypeScript, Vite |
| Backend | FastAPI, Uvicorn |
| Agent Framework | LangGraph, LangChain |
| LLM | Alibaba Qwen (ChatTongyi) via DashScope |
| Map API | AMap (高德地图) — geocoding + walking directions |
| Streaming | Server-Sent Events (SSE) via `sse-starlette` |

---

## Project Structure / 项目结构

```
.
├── agent.py                  # LangGraph workflow (Planner → Calculator → Router → Rest Stop)
├── tools.py                  # AMap API integration (geocoding, walking distance)
├── api.py                    # FastAPI server (SSE streaming, settings API)
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (not committed)
├── Documents/                # Project specification & slides
└── frontend/                 # React application
    ├── index.html
    ├── vite.config.ts        # Vite config with API proxy
    ├── package.json
    └── src/
        ├── main.tsx          # Entry point
        ├── App.tsx           # Root component, SSE consumption
        ├── App.css           # Styles
        ├── types.ts          # TypeScript type definitions
        ├── i18n.ts           # Chinese/English translations
        └── components/
            ├── DestinationInput.tsx  # Origin + destination form
            ├── StepTimeline.tsx      # Real-time step display
            ├── ItinerarySummary.tsx  # Final itinerary view
            ├── Settings.tsx         # API key configuration panel
            └── ErrorBanner.tsx      # Structured error display
```

---

## Prerequisites / 环境要求

- **Python** 3.10+
- **Node.js** 18+
- **AMap API Key** — Register at [AMap Open Platform](https://lbs.amap.com/) / 注册 [高德开放平台](https://lbs.amap.com/) 获取 Key
- **DashScope API Key** — Register at [Alibaba Cloud DashScope](https://dashscope.aliyun.com/) / 注册 [阿里云 DashScope](https://dashscope.aliyun.com/) 获取 Key

---

## Configuration / 配置

### Option A: `.env` file / 通过 `.env` 文件

Create a `.env` file in the project root:

在项目根目录创建 `.env` 文件：

```env
AMAP_API_KEY=your_amap_api_key_here
DASHSCOPE_API_KEY=your_dashscope_api_key_here
LLM_MODEL=qwen-turbo
```

### Option B: Web UI / 通过 Web 界面

After starting the application, click the **Settings** button in the top-right corner to configure API keys and select the LLM model directly from the browser.

启动应用后，点击右上角的 **设置** 按钮，可在浏览器中直接配置 API Key 和选择 LLM 模型。

> **Note / 注意：** Keys configured via Web UI are stored in memory and will be lost when the server restarts. For persistent configuration, use the `.env` file.
>
> 通过 Web 界面配置的 Key 仅保存在内存中，服务重启后会丢失。如需持久化，请使用 `.env` 文件。

### Available Models / 可用模型

| Model | Description |
|-------|-------------|
| `qwen-turbo` | Fast and cost-effective (default) / 快速且经济（默认） |
| `qwen-plus` | Balanced performance / 性能均衡 |
| `qwen-max` | Highest quality / 最高质量 |

---

## Getting Started / 快速开始

### 1. Clone and install / 克隆并安装

```bash
git clone <repository-url>
cd project-repo

# Python dependencies
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Frontend dependencies
cd frontend
npm install
cd ..
```

### 2. Configure environment / 配置环境

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Run in development / 开发模式运行

Start backend and frontend in two separate terminals:

在两个终端分别启动后端和前端：

```bash
# Terminal 1: Backend
uvicorn api:api --reload

# Terminal 2: Frontend dev server
cd frontend
npm run dev
```

Open http://localhost:5173 in your browser.

在浏览器中打开 http://localhost:5173 。

### 4. Production build (optional) / 生产构建（可选）

```bash
cd frontend && npm run build && cd ..
# Now api.py will serve the frontend from frontend/dist/
uvicorn api:api --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 — the backend serves both the API and the frontend.

打开 http://localhost:8000 — 后端同时提供 API 和前端页面。

---

## Usage / 使用方法

1. **Set language** — Toggle between 中文 and English using the switch in the top-left corner / 通过左上角开关切换中英文
2. **Configure API keys** — Click Settings (⚙️) to enter your AMap and DashScope keys / 点击设置 (⚙️) 输入 API Key
3. **Enter trip details** — Fill in the destination city (required) and optionally a starting point / 输入目的地城市（必填），可选填出发地点
4. **Start planning** — Click "Start Planning" and watch the agent work in real time / 点击 "开始规划"，实时观看 Agent 工作过程
5. **View results** — The final itinerary appears with total walking distance once planning is complete / 规划完成后显示最终行程和总步行距离

---

## API Reference / API 参考

### `GET /api/plan/stream`

Stream the travel planning process via SSE.

通过 SSE 流式传输旅行规划过程。

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `destination` | string | Yes | Destination city name / 目的地城市 |
| `origin` | string | No | Starting point / 出发地点 |

**SSE Events:**

| Event | Data | Description |
|-------|------|-------------|
| `node_update` | `{node, updates}` | A workflow node has completed / 工作流节点执行完成 |
| `done` | `{status: "complete"}` | Planning finished / 规划结束 |
| `error` | `{type, message, field?}` | An error occurred / 发生错误 |

### `GET /api/settings`

Returns current configuration (API keys are masked).

返回当前配置（API Key 已脱敏）。

### `POST /api/settings`

Update API keys and model selection at runtime.

运行时更新 API Key 和模型选择。

```json
{
  "amap_api_key": "your_key",
  "dashscope_api_key": "your_key",
  "llm_model": "qwen-turbo"
}
```
