# Validation record

Environment: Windows, Python 3.12.14, Node.js 24.19.0, FastAPI 0.141.1,
Pydantic 2.13.5, Vite 7.3.6, Three.js 0.180.0. Default local model:
`qwen3:8b` through Ollama. Ollama reported CPU-only inference (`size_vram: 0`).

## Automated checks

- Backend: **30 tests passed** with `python -m pytest -q`.
- Python source compilation passed with `python -m compileall -q app`
  before the final command-menu change; the final modules were imported/executed by pytest.
- Frontend animation test: **1 passed**, covering room-to-room queue order,
  interpolation and clearing stale animation on resynchronization.
- Production frontend build passed. Its only build warning was the approximately
  507 kB Three.js bundle exceeding Vite's default 500 kB warning threshold.
- After the small HTTP/WebSocket task-state race fix, `node --check` passed for
  `main.js`, `scene.js`, `websocket.js` and `ui.js`. The browser loaded this change.
  A final production rebuild was not executed: automatic approval review hit a
  usage limit. This was an approval-service failure, not a safety rejection.

The Python run emitted two upstream Starlette deprecation warnings concerning
httpx and AnyIO. They did not cause test failures.

Tests use mocked/scripted LLM responses, not a live model. They cover all four
mission predicates, world invariants, partial-observation isolation, valid and
invalid tool calls, pickup/drop/give, NPC conversation, reset, invalid JSON repair,
schema validation, model transport failures, execution budgets, timeout, cancellation,
completion evidence rejection, Critic rejection/recovery, duplicate-command recovery,
observation-derived command choices, cached object ownership, API lifecycle,
single-task enforcement and WebSocket snapshot/overflow behavior.

## Live checks

- FastAPI launched on `127.0.0.1:8000`; health and world endpoints responded.
- Vite launched on `127.0.0.1:5173`; the browser rendered the Three.js house,
  avatar, people, objects and labels.
- The browser submitted goals, showed live agent/tool events and world movement,
  and disabled Run while a task was active.
- Cancel stopped a running model task and preserved completed world actions.
- Restarting the backend caused automatic WebSocket reconnection and a fresh
  initial-world snapshot in the already-open browser.
- The browser console reported no errors or warnings in the inspected runs.
- Real `qwen3:8b` returned a validated structured-output smoke response.
- Focused live Explorer checks chose a valid movement after observing the hall,
  and chose `pick_up(charger)` when given real tool observations establishing
  the charger was visible and delivery was requested.

Earlier full live attempts exposed repeated reads/questions and an unsupported
delivery report. Those attempts were cancelled during development, not counted
as successes. They motivated observation-derived command choices, bounded
Critic recovery for repetition, and keeping unverified reports out of the UI's
confirmed action feed. A scripted mission is not evidence of live-model reliability.
