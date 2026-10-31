# Kip cognitive architecture implementation

Status: initial safe migration verified, 31 October 2026. Larger migrations remain
explicitly deferred below; this is not completion of the full cognitive architecture.
Preserve FastAPI,
Colibri/Qwen, the world engine, evidence checks and existing agent roles.

## Verified starting point

Source and Graphify agree that main.py owns one ChatSession, TaskManager and world.
Chat already makes one streaming call, independently of the Coordinator/Explorer/
Critic loop. Critic review is already selective. Memory already provides FTS,
optional vectors, supersession and deterministic persona appraisal. These are assets
to retain, not duplicate. Fixed eight-turn/400-character history, destructive speech
presentation, volatile conversation IDs and lossy task identity are confirmed gaps.

## Ordered phases and acceptance criteria

1. **Completed and tested: correctness boundaries.** `chat/consent.py`, `chat/prompt.py`,
   `llm/colibri.py` and regression tests: preserve changed recipients, numbers,
   negations and directions in task identity; Markdown headings/code must not become
   action trailers; interrupted streams cannot execute actions. Preserve offer-ID
   binding and existing permission/evidence checks. No schema migration required.
2. **Completed and tested: smallest persistent conversation migration.** Add durable transcript
   storage using existing SQLite state transactions; restore a stable default
   session and allow explicit IDs at ChatSession boundary. Persist before publishing
   turns; never restore consent or silently replay actions. Tests reopen a file DB,
   isolate IDs, and verify no replay. Full multi-session HTTP/WebSocket/task ownership
   is a subsequent migration, requiring scoped events, memory and physical arbitration.
3. **Completed and tested: budgeted context foundation.** Add a reusable context selector and configurable
   prompt budget/reserved output; retain complete recent turns, relevant older turns
   and exact correction/commitment excerpts without model calls. Preserve status and
   current request; reject impossible budgets honestly. Test long/unicode context,
   corrections, follow-ups and tool output. Token estimates must be labeled estimates.
4. **Completed and tested: response representation and conversational defaults.** Preserve canonical
   and screen text alongside legacy spoken text; safe frontend text rendering; natural
   1–3 sentence default with complex-answer exception. Add metrics for queue wait,
   calls and estimated context. Validate frontend tests/build and backend streaming.
5. **Partially implemented:** task audit, restart interruption reporting, durable
   restrictions, dispatch markers for uncertain effects, immediate cancellation and
   cancellation epochs, queued-plan preservation, and per-attempt model budgets.
   Failed model attempts and repairs count toward the task limit. Terminal tasks
   cannot dispatch more tools. These are covered by regression tests.
   **Deferred:** explicit CHAT/THINK/ACT model routing with
   ambiguity resolution in the existing response call; adaptive generation budgets;
   restart eligibility (read-only/idempotent steps only), resumable queues, and a
   transactional outbox for effect uncertainty. The current journal records evidence
   and uncertainty, but cannot make physical effects atomic with SQLite.
   Do not auto-replay robot actions against a freshly initialized world.
6. **Deferred:** memory deletion UI/API with summary/transcript provenance policy,
   scoped retrieval, commitment lifecycle, world-independent digital task service,
   AgentMind suggestion cooldown/quiet schedules, request-wide inference budgets,
   request-local serving metrics, measured prompt cache/warm-up tuning.

## Validation and migration

Run focused tests after each phase, then the entire backend and frontend suites.
Compare failures against an unchanged baseline rather than attributing old failures
to this migration. The initial default pytest run hit Windows temporary-directory
permissions (33 setup errors), plus 16 test failures; rerun with a workspace-local
temporary directory to establish a usable baseline. Do not alter unrelated fixtures
or claim scripted tests measure live Qwen performance. Existing callers keep their
default conversation and legacy `text` field. No external services/models added.

Update KIP_ARCHITECTURE.md and run `.venv/Scripts/python.exe -m graphify update .`
after source changes. Record actual outcomes and remaining limits here.

## Verified outcome

- Unchanged HEAD exported into an isolated temporary directory: **459 backend tests
  passed**. The initial 16 failures came from local `.env` settings, not the baseline
  source. Temp-directory setup errors were eliminated by creating a workspace temp root.
- Final backend: **483 passed**, two existing dependency deprecation warnings.
  Command (PowerShell, test-process settings only; no `.env` edit):
  `$env:DIRECT_INITIAL_DELEGATION='false'; $env:IDLE_REFLECTION_ENABLED='true';
  .venv/Scripts/python.exe -m pytest backend/tests -q --basetemp=.test-tmp/release-suite
  -o cache_dir=.test-tmp/cache`. Create `.test-tmp` first; use a fresh child each run.
- Frontend: **13 passed**, including canonical text, legacy fallback and safe text
  rendering. `npm.cmd test` from frontend. Build passed with
  `npm.cmd run build -- --configLoader runner`; default bundled-config loading hit a
  Windows sandbox ancestor-directory denial. Existing large-bundle warning remains.
- Three existing tests were adjusted to inspect an active copy or standalone context
  instead of dispatching on a terminal run or racing a background planner. Assertions
  about recovery and grants remain; new tests explicitly reject terminal dispatch.
- Focused phases passed 120, 125 and 154 tests before broad validation. Final suite
  covers consent binding, contradictions, task evidence, multi-step scenarios, memory
  correction/recall, context, restart, cancellation, batches and streaming.
- Deterministic context fixture: an allergy correction, 25 unrelated 516-character
  assistant turns, a running follow-up and a rest-day reply, then “What should I eat?”.
  Baseline prompt: **4,411 characters, 10 messages, correction omitted**. Final prompt:
  **4,848 characters, 9 messages, correction retained**. Both retained recent running
  context. This is a retention improvement, not a token reduction or model-quality
  benchmark. The earlier 4,695-character intermediate result preceded the final
  correction-precedence instructions.
- Ordinary chat still makes one streamed call; context selection, summarization by
  exact excerpts and personality add zero calls. Generation cap defaults to 1,200
  tokens instead of 300; brevity is a prompt preference. Provider usage/TTFT metrics
  remain distinct from the conservative UTF-8 byte context estimate. Queue wait is
  recorded separately. No live inference, hardware throughput or cache benchmark ran.
- Graphify AST refresh completed: **1,827 nodes, 5,429 edges, 108 communities**; no
  extraction model calls. Two skill reference directories were unreadable; Graphify
  backed up the old graph and rebuilt from fresh extraction when reconciliation hit
  that permission error. Current code nodes were verified with a follow-up query.
  KIP_DEPENDENCIES.json was regenerated from resolved first-party import/call edges.
  Documentation semantic extraction was not rerun. `git diff --check` passed.

## Migration semantics and limits

SQLite gains an additive `conversation_turns` table/index. Existing memories/state
are retained. Default `CHAT_SESSION_ID=home` resumes the last 200 transcript turns;
the complete transcript remains stored. Explicit ChatSession IDs isolate transcript
rows, **not** account permissions, recalled shared memories or task/world ownership.
The legacy HTTP/WebSocket surface still serves one household conversation.

Task records (latest 50) and restrictions use existing atomic state transactions.
Interrupted running records become failed with an explicit uncertain-outcome message;
no task/offer/queued grant is automatically replayed. Reset clears the task journal
with the world; restrictions remain binding. Restarting does not restore world state.
Do not deploy multiple app workers against one physical robot with this migration.

Context selection uses full recent turns plus relevance/recency-ranked older exact
excerpts, prioritizing correction/commitment language. It is not an abstractive summary
or a guarantee of recalling every old commitment. It operates on the latest 200 turns
plus existing memory retrieval. Whole memory facts either fit or are omitted; oversized
mandatory context returns a clear error without inference. A local tokenizer callback
can replace the conservative byte estimate; provider tokenization is not integrated.

The screen receives canonical content and preserves whitespace safely as text. Code,
links and tables survive; a full rich Markdown renderer is still deferred. Voice tables
use labeled rows and code-only output points to the screen. Ordinary streamed deltas
remain screen-oriented, not a production TTS stream.

## Next migrations, in dependency order

1. Session ownership: `main.py`, `chat/session.py`, `events/bus.py`, `tasks/models.py`,
   `tasks/manager.py`, memory retrieval and frontend WebSocket/API helpers. Add scoped
   conversation/run IDs, filtered events/history/memory, session-local queues and one
   shared physical lease. Migrate legacy data to `home`. Acceptance: concurrent clients
   cannot observe or cancel each other's work; one robot action executor remains.
2. Recoverable digital execution: extract a world-independent dispatch interface from
   `capabilities/tools.py`/manager, persist typed step states and replay policies. Recover
   only authorized idempotent reads after restrictions are restored. Acceptance: crash
   tests at every dispatch/result boundary, no duplicate effects or false completion.
3. Routing and response synthesis: extend the existing chat proposal schema for bounded
   CHAT/THINK/ACT decisions; ambiguity uses the existing call rather than a new classifier
   on every message. Add post-tool synthesis only when useful. Acceptance: held-out
   conversational scenarios plus measured per-interaction calls, tokens and latency.
4. Memory ownership/deletion and commitment lifecycle before broader autonomy. Add
   provenance-aware deletion across memories, summaries and transcripts; scoped recall;
   initiative cooldowns/quiet schedules and suppression. Preserve deterministic persona.
5. Serving experiments: request-local metrics, prompt-tokenizer integration, caching and
   warm-up measurements, then optional fast/deep local model policies. The existing
   factory already supports separate local chat/task clients; no hosted API is required.
