# agentic_friend — local house simulation (V1)

A learning project with one simulated avatar and three software agents: Coordinator,
Explorer and Critic. Python owns the world. Three.js visualizes it. All model calls
go to your configured local Ollama server; there are no API keys or cloud model dependencies.

## Start on Windows

Prerequisites: Python **3.12+**, Node.js **20.19+ or 22.12+** with npm, and
[Ollama](https://ollama.com/download). Install dependencies once while online.
After dependencies and the model are downloaded, the application works offline.

Run these commands from the project directory unless a `cd` is shown.

**Terminal 1 — Ollama**

```powershell
ollama serve
```

If the Ollama desktop app is already serving port 11434, leave it running instead.
In another terminal, download the default model once:

```powershell
ollama pull qwen3:8b
```

**Terminal 2 — backend**

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Environment activation is optional because these commands invoke the virtual
environment's Python directly. Do not overwrite an existing `.env` when updating.

**Terminal 3 — frontend**

```powershell
cd frontend
npm install
npm run dev
```

Open **[http://127.0.0.1:5173](http://127.0.0.1:5173)**. The development server proxies
`/api` and `/ws` to FastAPI on port 8000. Keep both processes running. Three.js is
served from the installed package, with no CDN or external fonts.

On macOS/Linux, use `python3 -m venv .venv` and `.venv/bin/python` in place of the
Windows Python paths. Other commands are the same.

### Already-installed environment in this workspace

The implementation session created a **root** `.venv` and `frontend/node_modules`.
On this machine, Python's normal launcher and npm were unavailable on PATH, so
the available bundled Python and pnpm were used to install dependencies. You can
use these commands from the project root with that existing installation:

```powershell
# Backend, terminal 1
.\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000

# Frontend, terminal 2
cd frontend
node node_modules/vite/bin/vite.js --host 127.0.0.1
```

The standard setup above remains the portable way to recreate the environment.
The included pnpm lockfile records the frontend versions used here; npm is also supported.

## Use it

1. Check that the sidebar says `qwen3:8b · ready locally`.
2. Enter a goal, such as **Find out who needs the charger and deliver it.**
3. Click **Run goal**. The feed shows short public agent summaries, tool results,
   NPC responses and Critic reviews. The active role is highlighted.
4. Watch the avatar move and objects change ownership. Inventory and location
   reflect the backend immediately; movement animation catches up visually.
5. **Cancel** stops inference and further actions but retains completed actions.
   **Reset** cancels the active task and restores the original world.

Only one task runs at a time. A second request receives HTTP 409. The world persists
between goals until Reset or a backend restart. Room labels, people and objects
are visible to you, but the agents must discover them with tools.

## How the code fits together

```text
backend/app/
  world/        deterministic models, initial state, simulator, validated tool registry
  llm/          small LLM protocol, JSON validation/retry, Ollama HTTP adapter
  agents/       distinct role prompts, typed decisions, role-specific model calls
  tasks/        task-scoped memory and explicit bounded asyncio agent loop
  events/       bounded event history and WebSocket subscriber queues
  main.py       FastAPI lifecycle, endpoints, WebSocket stream
  check_model.py  live structured-output smoke check
  evaluate.py   four live evaluation missions and programmatic success predicates
frontend/src/
  scene.js      primitive Three.js house, labels, avatar, people and objects
  animations.js queued room-to-room interpolation
  websocket.js  connection retry and authoritative snapshot resynchronization
  api.js        HTTP requests
  ui.js         task state and safe text-only activity feed
  main.js       interface wiring
```

`TaskManager._loop()` is the orchestration. Each Coordinator decision delegates,
requests a review, proposes completion, or fails. Explorer returns a single
allowlisted tool command or a report. A delegated task has an action budget,
after which control returns to Coordinator. Critic reviews plans on request and
every proposed completion. Reports from a delegation with no tool progress also
receive a review, allowing Critic feedback to correct unsupported claims. It has
a separate review budget to prevent debate loops. Repeated identical commands
return control early for Critic feedback and a revised Coordinator plan.

Coordinator and Critic receive **no simulator or tool handles**. Explorer's decision
class receives only the LLM client and tool schemas. The manager dispatches its
validated command through `WorldTools`; no role can execute Python or dynamically
resolve a function name. All three roles share the same configured model, with
distinct prompts and role-specific context. NPCs use deterministic keyword matching.

Explorer chooses a schema-constrained `command_id` such as `move_to:hall` or
`pick_up:charger`, with a free-text `message` for conversation. `agents/commands.py`
constructs these tool choices **only from observed rooms, inventory and successful
actions**. It never reads the simulator or examines the goal. The choice maps to
an ordinary `{tool, arguments}` call, which is validated again by the tool registry.
This avoids requiring a small local model to invent valid identifiers and arguments
on every step. The model still selects the route, questions, object actions and
when to report. Static rooms are remembered, and known object transfers update
that remembered view, so unchanged rooms do not require repeated `look` calls.

### World and evidence

The graph is `kitchen ↔ hall ↔ bedroom`, with `hall ↔ study`. The avatar starts in
hall. Mom and keys start in kitchen; Neave and laptop in bedroom; Dad and charger
in study. Every object has exactly one tagged location: room, robot or person.
Inventory is derived from those locations, never maintained as a second list.

`look` reveals the current room, exits, people, floor objects and visible held
objects. `get_status` reveals only the avatar's room and inventory. Tool failures
return structured observations and events, so the next model decision can recover.
Movement cannot skip the hall. Give/talk require the person to be in the room.

Task memory retains tool evidence, latest room/object observations, conversation
results, failed actions, the current plan, Explorer reports and Critic feedback.
Model prompts receive bounded recent history and compact discoveries, not the
entire event stream. Explorer reports are explicitly unverified; facts come from tools.
Completion must cite existing successful tool evidence IDs (a status call alone
does not qualify), and Critic must approve its match to the original goal.
This is evidence grounding, not a formal proof of arbitrary natural-language goals.

### Model interface

The adapter uses [Ollama's chat API](https://docs.ollama.com/api/chat) with a
[JSON schema in `format`](https://docs.ollama.com/capabilities/structured-outputs),
`stream: false`, `think: false`, low temperature and bounded generation/context.
Pydantic validates the result. Invalid JSON or a schema mismatch gets one retry,
then a safe task failure. HTTP failures and timeouts also become task failures.
The optional Ollama `thinking` field is ignored. Invalid raw output, model prompts
and hidden reasoning are never stored in task history, events or application logs.
Only brief explicitly requested public summaries and validated tool results are shown.

### Realtime consistency

The browser fetches `/api/world` at startup and on every reconnect. The WebSocket
also begins with an authoritative snapshot after subscribing, closing the
HTTP-to-WebSocket race. Events have monotonically increasing sequence numbers;
the UI ignores events already included in a snapshot. Slow subscribers receive
a resync snapshot instead of silently losing world changes. A reset/reconnect
discards pending visual movement and places the avatar at the authoritative position.

## Configuration

Copy `backend/.env.example` to `backend/.env`. Environment variables override it.

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama endpoint |
| `OLLAMA_MODEL` | `qwen3:8b` | One model shared by all roles |
| `LLM_TIMEOUT_SECONDS` | `120` | Per model HTTP request |
| `TASK_TIMEOUT_SECONDS` | `1800` | Entire task, including retries and reviews; allows CPU-only inference |
| `TEMPERATURE` | `0.1` | Conservative structured generation |
| `CONTEXT_TOKENS` | `8192` | Ollama context size |
| `MAX_COORDINATOR_CYCLES` | `20` | Maximum planning iterations |
| `MAX_EXPLORER_ACTIONS` | `8` | Maximum decisions per delegation |
| `MAX_TOOL_CALLS` | `50` | Includes bootstrap status and failed calls |
| `MAX_CRITIC_REVIEWS` | `6` | Prevents repeated review loops |
| `LOG_LEVEL` | `INFO` | Use `DEBUG` for application diagnostics |

The application uses in-process memory; use **one Uvicorn worker**. Bind it to
loopback as shown. There is intentionally no authentication or remote deployment setup.

## API

| Endpoint | Result |
|---|---|
| `GET /api/health` | Backend health plus Ollama connectivity/model availability |
| `GET /api/world` | `{world, sequence, task}` for visualization |
| `POST /api/world/reset` | Cancel, reset world, clear session tasks and event history |
| `POST /api/tasks` | `{ "goal": "..." }` → 202 with task state |
| `GET /api/tasks/{id}` | Task state, observations and reviews |
| `POST /api/tasks/{id}/cancel` | Cancel active task; terminal tasks unchanged |
| `WS /ws` | Initial snapshot, then world/agent/tool/task events |

Interactive API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).
Task history is bounded to the latest 50 tasks. Event history retains 300 events.
Neither survives a process restart. The UI keeps at most 200 activity entries.

## Tests and evaluation

From `backend`, with its virtual environment installed:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m app.check_model
.\.venv\Scripts\python.exe -m app.evaluate
# Run just one live mission:
.\.venv\Scripts\python.exe -m app.evaluate --mission 3
```

If using this workspace's root virtual environment, substitute
`..\.venv\Scripts\python.exe` for those commands while in `backend`.

`pytest` uses scripted/mock model responses and does **not** require Ollama.
The smoke check and evaluation commands intentionally call your local model.
Each evaluation mission starts a fresh independent world and exits nonzero if
the task fails or its programmatic success condition is unmet.

| Mission | Success condition |
|---|---|
| Find the charger | Successful `look` observed the charger |
| Find Neave and tell him dinner is ready | Successful `talk_to(neave, ...)` conveyed dinner-ready message |
| Find out who needs the charger and deliver it | Conversation established need and charger location is person/neave |
| Find the missing keys | Successful `look` observed keys |

The checks are intentionally simple predicates for these four missions. They
are not a general semantic goal evaluator and do not steer the production agents.

From `frontend`:

```powershell
npm test
npm run build
```

The JavaScript test checks interpolation order and resynchronization. The
production build checks module resolution and bundling. See `VALIDATION.md` for
the actual validation performed during implementation.

## Troubleshooting and V1 limits

- **Model missing:** run `ollama pull qwen3:8b`. **Ollama offline:** start its server.
- **Slow inference:** CPU-only 8B inference may take tens of seconds per decision.
  The role indicator stays active while waiting. Use a supported smaller local
  model via `OLLAMA_MODEL`, or increase the task timeout. No fallback to cloud occurs.
- **Failure or cancellation:** completed actions are not rolled back. Inspect the
  activity feed, then Reset for a clean attempt.
- **Disconnected UI:** it retries automatically and fetches a fresh world snapshot.
- **Blank 3D view:** enable WebGL/hardware acceleration in the browser.
- **Port conflict:** stop the existing process on 8000 or 5173. If changing ports,
  update both Vite proxy targets and the backend launch command.
- This is a four-room deterministic simulator with static NPC locations, keyword
  conversation, simple room interpolation, no pathfinding or physics, no persistent
  memory, and no physical hardware. NPCs cannot hand held objects back to the robot.
- Model decisions can be inefficient or wrong. Invalid actions are rejected,
  completion is reviewed, and hard budgets stop stalled tasks. These checks cannot
  guarantee that a local model will solve every arbitrary goal.

A useful V2 step is a larger regression mission suite with evidence-based goal
predicates and measured success/latency across local models, before adding a
larger world or more autonomous capabilities.
