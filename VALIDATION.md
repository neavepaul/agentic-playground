# Digital tools — 2026-10-25

- Current implementation: removed the direct-command parser, execution path,
  configuration flag and shortcut-specific tests at the user's request. The
  earlier 0.295-second result below is historical and does NOT describe the
  current model-led workflow. Every digital request now requires model planning.
- Independent tools chosen in GoalPlan or a later batch_capabilities decision
  run concurrently. Full-batch validation/budget checks occur before execution;
  individual failures remain failures, and cancellation preserves completed
  evidence. The model chooses all queries, URLs, code and answer text.
- Digital prompts use bounded page excerpts with continuation offsets, shorter
  snippets, short evidence aliases resolved back to original IDs, concise tool
  instructions and schemas without annotation overhead. Full audit data remains.
- All 247 backend tests pass. Regression coverage verifies a literal URL/Python
  request requires model planning and can complete in two calls, later model-led
  batches, concurrent execution, cancellation, budget checks, evidence integrity,
  schema-property preservation, and existing robot behavior. Metrics now include
  initial GoalPlan calls; older call counts below excluded them.
- Live Colibri verification with the bypass removed: the example.com/Python
  request completed successfully in 294.382 seconds (4m54s), with two actual
  model requests, a model-selected initial read/Python batch, real stdout `42`,
  and a model-written answer citing both original evidence IDs. GoalPlan took
  172.165 seconds; the answer call took 121.982 seconds. Model inference accounted
  for essentially all wall time. Full audit: `artifacts/model-batching-result.json`;
  request measurements: `artifacts/model-batching-calls.json`.
- The earlier four-request model-led probe took about 399 seconds. This new
  two-request probe was faster, but the runs are not a controlled benchmark and
  do not establish a stable speedup. Current Colibri inference still takes minutes.
- Prompt excerpts used short evidence labels; the final answer's stored
  citations were restored to the actual task-local UUIDs and validated. The
  isolated test used no persistent personal memory; simulated time stayed 08:00.

- Historical shortcut experiment (removed): the exact user request “Read https://example.com and use
  Python to calculate 6 × 7. Tell me both results.” completed through the running
  backend API in 0.295 seconds, with a real page fetch, `print(6 * 7)` execution,
  both evidence IDs, and zero model requests. Audit:
  `artifacts/digital-tools-fast-result.json`. This is one warm-service functional
  timing; network conditions and cold starts can change the latency.
- In that experiment, fully specified literal URL/arithmetic requests used a bounded direct
  executor. Other digital goals can batch independent initial tools from their
  GoalPlan, avoiding separate model selection calls. A scripted integration test
  verifies a nonliteral read/calculate request uses two model calls (plan/answer).
  This batching improvement has not been separately timed through the real model.
- All 258 backend tests pass, covering alternate URLs/expressions, rejection of
  extra clauses/code/expensive arithmetic, actual restricted Python execution,
  failure evidence, batching, and the existing robot tests. The direct path does
  not require the local model and cannot emit an answer after a tool failure.

- Follow-up user run `b414cab9-90df-4071-8840-8ffd16b147f4` read example.com,
  then attempted a final answer claiming Python execution without having run it.
  The evidence guard rejected that attempt. The existing task recovered, ran
  `print(6 * 7)` with observed stdout `42`, and completed with both evidence IDs.
  Its completed audit is saved in `artifacts/digital-tools-user-run.json`.
- Completion choices now omit finish_answer until the required tools have
  successful observations. The model sees an explicit remaining-tool list and
  successful evidence IDs. Missing-evidence errors name only the missing tools;
  rejected answer prose is not replayed as factual context. Tool failures update
  public status and decision feedback. Regression coverage brings the suite to
  238 passing tests. The fix was applied after the user's run completed.

- Added public internet search (DDGS), bounded HTML/plain-text webpage reading,
  and small Python programs in Monty's restricted interpreter. No MCP service
  or search API key is needed. Colibri remains the configured local model.
- All 236 backend tests pass, including original robot behavior, digital-only
  tasks, a combined webpage-reading/physical-delivery task, failure recovery,
  cancellation propagation, output/execution limits, public-address checking,
  redirect validation, and task-local answer evidence. The frontend builds.
- Live public search returned Raspberry Pi documentation links; the real reader
  fetched example.com with TLS verification and a checked, pinned destination.
- An isolated task through the actual Colibri model completed: “Read
  https://example.com and use Python to calculate 6 * 7. Tell me the page title
  and the calculation result.” It chose read_webpage, run_python with
  `print(6 * 7)`, and finish_answer with both evidence IDs. It returned “Example
  Domain” and 42 with the source link. No tool failures or output repairs occurred.
- This live test used a separate in-memory robot world, no persistent memory,
  and action-driven time. The avatar did not move and simulated time remained
  08:00. Detailed results are in `artifacts/digital-tools-live-result.json`.
- The four model requests took about 399 seconds total: 112.56 seconds for goal
  interpretation and 286.33 seconds for the three Explorer calls. Existing task
  metrics omit the initial GoalPlan request, so their three-call count is not
  the total. This is a functional smoke test, not a performance benchmark.
- Python is a restricted subset, not arbitrary CPython/package execution. Web
  content is untrusted; provenance checks do not prove every answer factually
  correct. Existing skills/personality behavior has not been expanded here.

# Colibri execution and latency — 2026-10-25

- The live goal "Neave needs the medicine. Find it and bring it to him." completed
  with all conditions met after 12 tool calls. Its result is saved locally in
  `artifacts/medicine-delivery-completed-20261025.json`. This run preceded the speed
  changes and hit the local eight-action delegation limit before its final scan/give.
- Local `.env` now uses 40 actions per delegation. Direct initial delegation gives
  Explorer the full original goal after GoalPlan interpretation, avoiding the extra
  initial Coordinator decision. Later replanning and existing tool/stall limits remain.
- Model prompts use compact JSON and action/evidence references instead of repeating
  raw observations. The full action history remains available for verification.
  A representative initial Coordinator input shrank from 12,423 to 9,896 characters.
- Colibri's saved runtime configuration keeps 32 expert-cache slots per layer and a
  256 MiB GPU expert tier; maximum prompt batches increased from 128 to 1024 rows.
  A live goal-interpretation probe returned correct medicine/Neave conditions in
  65.0 seconds, versus 185.3 seconds for the earlier medicine/Paul probe. These were
  separate runs, not a controlled benchmark or a full-mission speed measurement.
- All 209 backend tests passed, including direct-delegation delivery without a
  redundant Coordinator call and subsequent Coordinator replanning when needed.
  Legacy scripted cases ran with documented execution limits and direct initial
  delegation disabled; new cases explicitly exercise the enabled path.
- Graphify was updated after the code changes. No full live mission has yet been
  timed using the new execution settings.

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
