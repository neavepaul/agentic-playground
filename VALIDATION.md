# Recipient-question loop recovery

- 51 backend tests passed on the final run. Regression coverage includes normalized
  duplicate-message rejection, successful continuation through search/pickup/give,
  and bounded failure when Explorer ignores the correction. Existing follow-up
  notification coverage still passes.
- A focused live qwen3:8b probe used real tool observations from the master bedroom,
  a quoted recipient interpretation, empty inventory, and the misleading Critic
  advice from the reported loop. Before the final prompt adjustment it chose to
  announce its plan via talk_to. After the adjustment it selected
  move_to:bedroom_corridor. No full live mission was run for this change.
- An initial full test run hit an intermittent WebSocket TestClient teardown
  CancelledError. A targeted rerun and the final full suite passed; the WebSocket
  implementation was not modified here.
- Graphify update remains blocked: the Python runtime has no graphify module.

---

# Current JSON-world change validation

- 49 backend tests passed, including the original regression fixture and new
  scenario loading, reference validation, information boundary, reset isolation,
  transcript grounding, and an unfamiliar medicine/Alice workshop delivery plus
  follow-up notification using a scripted model.
- Frontend animation test passed; production build passed (Three.js bundle size warning).
- Browser visual inspection confirmed the sketch layout and labels; inspected
  browser console had no errors or warnings.
- A live qwen3:8b transcript interpretation check correctly returned Alice / medicine
  with the exact supporting quote. No simulator facts were supplied.
- Graph refresh attempted with `python -m graphify update .`; the workspace runtime
  reported `No module named graphify`. The saved graph may therefore be stale.
- Live model planning on this new house has not been validated. The historical
  evaluation below applies to the earlier four-room version.

---

# Validation record

Environment: Windows, Python 3.12.14, Node.js 24.19.0, FastAPI 0.141.1,
Pydantic 2.13.5, Vite 7.3.6, Three.js 0.180.0. Default local model:
`qwen3:8b` through Ollama. Ollama reported CPU-only inference (`size_vram: 0`).

## Automated checks

- Backend: **35 tests passed** with `python -m pytest -q`.
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
Additional regressions cover a laptop delivery failing a charger condition,
incomplete broadcasts, holding/placing objects, completion with valid but
insufficient evidence IDs, and a Critic veto preventing a proposed transfer.

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
as successes. One model-only completion review also incorrectly approved a
laptop transfer as a charger delivery; the independent mission predicate failed
that run. These findings motivated observation-derived command choices, fixed
typed goal conditions checked by Python, object-action scoping, transfer reviews,
bounded Critic recovery for repetition, and keeping unverified reports out of
the UI's confirmed action feed. A scripted mission is not evidence of live-model reliability.
