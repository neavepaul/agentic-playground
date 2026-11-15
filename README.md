# agentic_friend — Kip, a companion you can talk to

Kip is a robot friend in a chat window. He remembers you, has a persistent personality,
and can look things up on the web, read pages and run small Python programs to answer you.
Behind the conversation, three software agents (Coordinator, Explorer and Critic) run those
lookups and check answers against their sources. Kip is built for a **Raspberry Pi 4 (8 GB)**: a
small local model (Qwen3.5 2B on Ollama) answers when Groq can't, Groq answers first while it has
quota, and F1, cricket and weather come straight from data feeds with no model at all.

## Start

Prerequisites: Python **3.12+**, Node.js **20.19+ or 22.12+** with npm, and
[Ollama](https://ollama.com/download). Install dependencies once while online.

Run these commands from the project directory unless a `cd` is shown. The examples are for
Raspberry Pi OS (Linux); on Windows use `.venv\Scripts\python.exe` and `Copy-Item`.

**The model** (once; keep `ollama serve` running, which the installer sets up as a service):

```bash
ollama pull qwen3.5:2b-q4_K_M
```

**Backend**

```bash
cd backend
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env          # add GROQ_API_KEY and CRICKETDATA_API_KEY if you have them
.venv/bin/python -m app.check_model
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Do not overwrite an existing `.env` when updating.

**Frontend**

```bash
cd frontend
npm install -g pnpm
pnpm install
pnpm dev
```

Open **[http://127.0.0.1:5173](http://127.0.0.1:5173)**. The development server proxies
`/api` and `/ws` to FastAPI on port 8000. Keep both processes running. Everything is
served from installed packages, with no CDN or external fonts.

## Use it

Open the page and talk. Kip answers in his own voice, remembers what you tell him about
yourself (see **About you** in the side panel), and looks things up when you ask about live
information such as scores, weather, prices or news. A lookup shows its sources under the
answer, and **Activity** lists what each lookup did. **Kip** shows his current mood, traits
and how well you know each other.

Only one lookup runs at a time; a second one waits for the first. Saying "stop" cancels it.

### Talking to Kip

The **Talk to Kip** panel is an ongoing conversation. Each message costs one streamed
model call with a short prompt: Kip's voice and mood, up to six recalled memories, the
last eight turns and a one-line status. There are no maps or tool schemas in this prompt.
When you ask for something that needs a tool ("search for…", "use Python to…"), Kip agrees and
starts an errand through the task system, ending its reply with a hidden `###` action line. A
second errand waits until the first finishes. Kip narrates errand results in the chat using
templates, without extra model calls.

The status line in every chat prompt carries the real local date, time and timezone, and
"what time is it?" is answered straight from the device clock with no model call. "What time?"
right after a question about a match still asks about the match. Kip has no body and no
surroundings: the prompt tells him never to describe any.

**Reply speed.** With `GROQ_API_KEY` set, conversation runs on Groq while it has quota
(`GROQ_CHAT_MODELS`, Qwen first so chat leaves the gpt-oss search quota alone), streamed, with
the first words in about a second. When every chat model is rate-limited for only a few seconds
(up to `GROQ_MAX_WAIT_SECONDS`, default 20), the reply waits for one rather than going local.
A model that sends back nothing gets one more try; one that tries to call a tool (gpt-oss reaches
for its own Python when asked to "use Python") is skipped for that reply. When no chat model can
answer, the reply comes from the local chat model and the log says why. A request that names a
tool ("use Python to…", "read https://…") gets no chat reply at all: like a live lookup, the
errand starts at once and its answer is the reply, and if the chat model fails, what the person
plainly asked for still starts. Set
`CHAT_HOSTED=false` to keep conversation local. The `chat_reply` log line names the provider and
model, and how the prompt's size splits between system, history and the latest message.

Without Groq, chat and errands share the one local model: while an errand runs, a reply goes
before its next model call.

The backend log reports `first_token_seconds` and `total_seconds` for every reply, and
hovering over a reply in the UI shows the same timing.

### Understanding a message before any model sees it

`backend/app/chat/router.py` reads each message locally, in about 0.1 ms (or 5–8 ms once the
embedding model is loaded), and says what it is about: an intent such as
`information.sports`, `conversation.emotional` or `action.request`, plus the domain, the
people or teams named, what is being asked (a qualifying position, a forecast, a price)
and when. In sport, "where did he finish" asks for a position, not a place. A short
follow-up leans on the last information question ("What about Max?", "And yesterday?"), a
pronoun with nobody to point at gets "Who you mean, question?", and "I meant what position"
is recognised as a correction.

The classifier is TF-IDF plus logistic regression, trained in numpy from
`backend/app/chat/router_data/intents.tsv` at startup. Once fastembed's bge-small model is
loaded (memory recall already loads it), its embeddings are averaged in, which held up
better on wording the examples never used.

`INTENT_ROUTER` sets how much it does:

- `on` (default): a confident live-information question (sport, weather, markets, news)
  goes straight to the one-call web lookup with a search built from what it means, with no
  chat-model reply first. Everything else (chat, feelings, memory, complex requests, anything
  uncertain) takes the usual path. Stop, permissions, offers and requested tools always come first.
- `shadow`: the router reads every message and records what it would have done next to
  what the usual path did, and changes nothing.
- `off`: no router.

Offline, from `backend`:

```bash
.venv/bin/python -m app.chat.router_train eval              # accuracy against evaluations/intent_eval.jsonl
.venv/bin/python -m app.chat.router_train eval --embeddings --compare
.venv/bin/python -m app.chat.router_train shadow            # router vs usual path, from kip.db
.venv/bin/python -m app.chat.router_train review            # corrections people made
.venv/bin/python -m app.chat.router_train accept 3          # use correction 3 as a training example
```

Corrections are stored in `kip.db` as unreviewed, deduplicated, with what was predicted,
what was meant and whether the corrected lookup was answered. Only accepted ones join the
training data, the next time Kip starts. Nothing is retrained during a conversation.

### Kip's memory: one SQLite file

`backend/memory/kip.db` holds everything Kip keeps between runs. Its `memories` table has
every kind of memory as a row: chat
exchanges, facts you tell Kip, task episodes and answers to past questions. Recall combines keyword search (FTS5) with meaning search (`sqlite-vec`
plus a small local `fastembed` model, `BAAI/bge-small-en-v1.5`, about 70 MB, downloaded to
`backend/memory/models/` on first use), then ranks by recency and importance.
Rewordings of a memory reinforce it instead of being stored twice. If the embedding
model cannot load, recall falls back to keyword search. Set `EMBEDDINGS_ENABLED=false` to
skip it entirely.

Chat replies and new errands (as `relevant_memories`, marked advisory) read from this store.
Browse or search it under **About you**, or with `GET /api/memories?q=...`.

Its `state` table holds structured documents: recent errands, Kip's self-model (which tools
have failed), his persona, the restrictions you set ("don't use Python") and the name you
gave him, so he still knows who he is talking to after a restart.

`backend/memory/persistent.json` and `beliefs.json` are the stores from before this. The first
time the backend starts with a `kip.db` that has no state, it imports them, copies them to
`backend/memory/backups/<time>-import/`, and never reads or writes them again. To make Kip
forget everything, delete `backend/memory/kip.db` (and move the old JSON files away, or they
are imported again).

## How the code fits together

```text
backend/app/
  chat/         the conversation: prompt, intent router, consent, claim checks, narration
  agents/       Coordinator/Explorer/Critic prompts, typed decisions, role-specific model calls
  tasks/        task context, permissions and the bounded asyncio agent loop
  capabilities/ web search, webpage reading, restricted Python, answer grounding
  memory/       kip.db: searchable memories and structured state
  persona/      deterministic personality: mood, traits, relationships (no model calls)
  llm/          small LLM protocol, JSON validation/retry, Ollama and Groq adapters
  events/       bounded event history and WebSocket subscriber queues
  clock.py      the one clock, in TIMEZONE (IST by default) whatever the device says
  main.py       FastAPI lifecycle, endpoints, WebSocket stream
  check_model.py  live structured-output smoke check
frontend/src/   the chat interface
```

`TaskManager._loop()` is the orchestration. A plain lookup of a live fact skips planning and,
with Groq, is answered in one browser-search call whose answer is checked against what it read.
Otherwise Coordinator turns the request into required answer outcomes, Explorer picks digital
tool calls (singly or as an independent batch) until it can cite successful evidence in
`finish_answer`, and Coordinator replans or fails when progress stops. Critic reviews plans on
request and any completion the deterministic checks cannot settle. Budgets on cycles, tool
calls, model calls and time bound every task.

### Model interface

The local model runs on Ollama's `/api/chat`, with schema-constrained output (`format`), low
temperature, bounded output and the model kept loaded. Groq, when its key is set, goes first.
Pydantic validates the result. Invalid JSON or a schema mismatch gets one retry,
then a safe task failure. HTTP failures and timeouts also become task failures.
Optional model reasoning fields are ignored. Invalid raw output, model prompts
and hidden reasoning are never stored in task history, events or application logs.
Only brief explicitly requested public summaries and validated tool results are shown.

### Realtime consistency

The WebSocket begins with an authoritative snapshot (`{sequence, task}` plus recent events)
after subscribing, closing the HTTP-to-WebSocket race. Events have monotonically increasing
sequence numbers; the UI ignores events already included in a snapshot. Slow subscribers
receive a resync snapshot instead of silently losing events.

## Configuration

Copy `backend/.env.example` to `backend/.env`. Environment variables override it.

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | unset | Optional hosted Groq: answers first while it has quota (chat, planning, web search); unset keeps everything on the Pi |
| `CRICKETDATA_API_KEY` | unset | Optional CricketData.org key for cricket (100 requests a day free); unset, ESPN's scoreboard answers cricket |
| `FOOTBALLDATA_API_KEY` | unset | Optional football-data.org key for football (10 requests a minute free); unset, ESPN's league scoreboards answer football |
| `GROWW_API_KEY` / `GROWW_API_SECRET` | unset | Optional Groww Trade API TOTP key and its TOTP secret, for portfolio questions; read-only (token and holdings only). Unset: no portfolio answers |
| `GROQ_MODELS` | `openai/gpt-oss-120b,qwen/qwen3.8-27b,openai/gpt-oss-20b` | Preference order. Free-tier limits are per model, so a model whose quota is spent is skipped until its reset |
| `GROQ_MAX_WAIT_SECONDS` | `20` | When every Groq model is rate-limited and one frees within this long, a call waits for it (once) instead of falling back to the local model, which can take minutes |
| `GROQ_PLAN_CALLS` / `GROQ_RESEARCH_CALLS` | `2` / `4` | Hard caps on Groq calls per goal plan and per web-research task; past a cap, or on any Groq failure, the local model continues |
| `CHAT_HOSTED` / `GROQ_CHAT_MODELS` | `true` / `qwen/qwen3.8-27b,openai/gpt-oss-20b` | Conversation on Groq while it has quota, streamed, falling back to the local chat model |
| `GROQ_SEARCH_REASONING` | `low` | Effort for Groq's browser search. Higher opens more pages, and every step re-reads each page read so far (one page can be ~50k tokens) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `OLLAMA_MODEL` | `qwen3.5:2b-q4_K_M` | The one local model (4-bit; the plain `qwen3.5:2b` tag is 8-bit, about half the speed on a Pi) |
| `OLLAMA_KEEP_ALIVE` | `-1` | How long Ollama keeps the model loaded (`-1` always, `30m`, or empty for Ollama's 5 minutes); reloading from microSD takes tens of seconds |
| `LOCAL_CHAT_ERRANDS` | `false` | A reply the local model writes keeps its memories, but its task and offer are dropped: errands start only from your own words. `true` lets it start errands. Groq-written replies are unaffected |
| `CHAT_HISTORY_TOKENS` | `250` | Earlier conversation each reply carries; less is faster on a Pi. A long message itself is never refused for it |
| `LLM_TIMEOUT_SECONDS` | `600` | Per model HTTP request; a small model on a Pi can take minutes over a long call |
| `TASK_TIMEOUT_SECONDS` | `1800` | Entire task, including retries and reviews; allows CPU-only inference |
| `TEMPERATURE` | `0.1` | Conservative structured generation |
| `CONTEXT_TOKENS` | `4096` | Ollama context size; smaller needs less memory |
| `MAX_COORDINATOR_CYCLES` | `20` | Maximum planning iterations |
| `MAX_EXPLORER_ACTIONS` | `40` | Hard safety ceiling per delegation, not the normal exit |
| `MAX_TOOL_CALLS` | `80` | Includes failed calls |
| `MAX_CRITIC_REVIEWS` | `8` | Bounds plan/completion reviews and recovery |
| `LOG_LEVEL` | `INFO` | Use `DEBUG` for application diagnostics |
| `TIMEZONE` | `Asia/Kolkata` | Every time Kip says or reasons with (the clock, match and race times, the status line), whatever the device's clock is set to; an IANA name |
| `CHAT_MAX_TOKENS` | `300` | Upper bound on one chat reply |
| `CHAT_TEMPERATURE` | `0.6` | Conversational variety (tasks keep `TEMPERATURE`) |
| `INTENT_ROUTER` | `on` | Local intent router: `on` sends confident live questions straight to the lookup, `shadow` only records what it would do, `off` |
| `MEMORY_DB_FILE` | `backend/memory/kip.db` | Kip's durable store: memories and structured state |
| `MEMORY_FILE` / `GRAPH_FILE` | `backend/memory/persistent.json` / `beliefs.json` | Legacy JSON stores, read once to import |
| `EMBEDDINGS_ENABLED` | `true` | Meaning search with fastembed; false means keyword-only |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | fastembed model name |
| `PERSONA_ENABLED` / `PERSONA_NAME` | `true` / `Kip` | Personality layer |
| `CHAT_DEFAULT_SPEAKER` | unset | Who is talking when nobody has said; unset, Kip learns it from "my name is …" |
| `PERSONA_MAKER` | unset | Who built Kip, as he should know it (e.g. `Neave`); unset, he says he does not know, never denying a claim or naming someone else |

The application uses in-process memory; use **one Uvicorn worker**. Bind it to
loopback as shown. There is intentionally no authentication or remote deployment setup.

### Speed on a Raspberry Pi 4 (8 GB)

Estimates from the Pi 4's memory bandwidth, not measurements: the 2B model reads a prompt at roughly
6–10 tokens a second and writes about 3.

- **Instant** (no model): the clock, router answers, F1, cricket, football, NBA, weather, air quality,
  markets and news from their feeds (below), live lookups with Groq, and explicit tool requests like "use Python to…" or "read
  https://…", whose errand starts at once. A named page is read with no planning call. Times are
  always in `TIMEZONE` (IST by default) and temperatures in °C, whichever model or feed answers.
- **Tens of seconds**: anything the local model writes. The system prompt is a few hundred tokens
  and the same every reply, so Ollama reuses it; each reply adds the mood, two memories, the
  status, recent history (`CHAT_HISTORY_TOKENS`) and the message.

A 2B model is a weak judge of what to do: tested without Groq, its replies proposed nonsense
errands, and one became an offer that blocked lookups for ten minutes. So it is kept to talking and
remembering (`LOCAL_CHAT_ERRANDS=false`); errands start from what you said. A request that only asks
for Python costs one narrow model call that writes just the code.

With `GROQ_API_KEY` set, Groq answers while it has quota, from the same prompt, and the Pi's model
takes over when Groq cannot. Measure your own numbers with `python -m app.bench` from `backend`
while Kip runs: it times chat, memory, a live lookup, Python and a page read. It talks to Kip like
the browser, so start that backend with `MEMORY_DB_FILE` pointing at a copy of `kip.db` if the test
conversation should not be kept. Keep `kip.db` on a USB 3 SSD if you can: it is written constantly,
which wears microSD cards.

## API

| Endpoint | Result |
|---|---|
| `GET /api/health` | Backend health plus provider/model availability in `llm` |
| `GET /api/state` | `{sequence, task}`: the event sequence and the active or latest task |
| `GET /api/chat` / `POST /api/chat` | Conversation history / `{ "message": "..." }` → 202 |
| `GET /api/memories` | `?q=&kind=&limit=`: search or list Kip's memories |
| `GET /api/persona` | Mood, traits, relationships and recent moments |
| `POST /api/tasks` | `{ "goal": "..." }` → 202 with task state |
| `GET /api/tasks/{id}` | Task state, observations and reviews |
| `POST /api/tasks/{id}/cancel` | Cancel active task; terminal tasks unchanged |
| `WS /ws` | Initial snapshot, then chat/agent/tool/task/persona events |

Interactive API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).
Task history is bounded to the latest 50 tasks. Event history retains 300 events.
Neither survives a process restart. The UI keeps at most 200 activity entries.

## Tests and evaluation

From `backend`, with its virtual environment installed:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m app.check_model
```

`pytest` uses scripted/mock model responses and does **not** require a running model server.
The smoke check intentionally calls your local model.

From `frontend`:

```bash
npm test
npm run build
```

The JavaScript tests cover the pure UI helpers. The production build checks module
resolution and bundling. See `VALIDATION.md` for
the actual validation performed during implementation.

## Troubleshooting and V1 limits

- **Model missing/offline:** start Ollama (`ollama serve`), check `ollama list` shows
  `OLLAMA_MODEL`, and run `python -m app.check_model`. Restart the backend after changing `.env`.
- **Slow replies:** a Pi's model takes tens of seconds for anything it writes. Set `GROQ_API_KEY`,
  lower `CHAT_HISTORY_TOKENS`, or raise the request/task timeout if calls time out.
- **Truncated model output:** raise `CONTEXT_TOKENS` or shorten the context. The adapter reports
  truncation as a task failure; it never accepts a partial JSON response as a completed decision.
- **Disconnected UI:** it retries automatically and fetches a fresh snapshot.
- **Port conflict:** stop the existing process on 8000 or 5173. If changing ports,
  update both Vite proxy targets and the backend launch command.
- Model decisions can be inefficient or wrong. Invalid actions are rejected,
  answers are checked against their cited evidence, and hard budgets stop stalled
  tasks. These checks cannot guarantee that a model will answer every question.

## Web research and small Python programs

Every digital request goes through the model, including an explicit URL plus a
simple calculation. The former direct-command parser and its setting have been
removed. The model (Groq, else the local one) chooses the tools, their arguments, and the final answer.

The initial goal plan can request up to four independent tools with fully known
arguments. Later decisions can use `batch_capabilities` to group independent
calls too. These public reads/restricted Python calls execute concurrently, with
the complete batch checked against the remaining tool budget before starting.
Each actual result retains its own evidence record. Final answers cannot be batched. A search must finish before the model selects a result
URL; programs using page data must wait for that data. Cancellation preserves
completed evidence and cancels unfinished calls.

Model prompts use smaller page excerpts (with continuation offsets), shorter
search snippets, and short citation labels. Full tool observations and original
evidence IDs remain in the audit history; aliases are resolved before validation.
Schema annotations are omitted from digital prompts while the original schema
still validates outputs. Metrics now include the initial GoalPlan call. These
changes reduce inference work and round trips; model/hardware speed still limits
response time, and general research is not guaranteed to be instantaneous.

Things you can ask:

- “Search the internet for Raspberry Pi 5 power requirements. Read the official source and explain them.”
- “Read https://www.raspberrypi.com/documentation/ and summarize what it covers.”
- “Use Python to calculate the average of 12, 18, 21 and 29.”

Install the updated dependencies in the backend's existing environment with
`python -m pip install -r requirements.txt`, then restart the backend and refresh
the frontend. No search API key is required. `DIGITAL_TOOLS_ENABLED=true` is the
default; set it to false to disable them.

**Data feeds.** Questions the router recognises on these subjects are answered from a feed before
any search, with one request (two or three for some), exact facts, an answer in fixed words and no
model call. Each is recorded like a web search, so "don't use the internet" still stops it, and
anything a feed does not cover goes to the usual web lookup. Before asking, Kip says which source he is checking
("Let me check Open-Meteo for that.").

| Subject | Source | Key | Covers |
|---|---|---|---|
| F1 | [Jolpica](https://api.jolpi.ca/ergast/f1) (the Ergast successor) | none | results, qualifying, sprints, standings, the next race |
| Cricket | [CricketData.org](https://cricketdata.org/) `cricScore`, then ESPN's scoreboard | `CRICKETDATA_API_KEY` (100 requests a day free) | results, live scores and fixtures for a week or more either side; ESPN (the last few days) when there is no key, the day's 100 are used or a match is missing |
| Football | [football-data.org](https://www.football-data.org/), then ESPN's league scoreboards | `FOOTBALLDATA_API_KEY` (10 requests a minute free) | twenty days of matches (ten either side) across the key's competitions; ESPN (this round of the Premier League, La Liga, Serie A, Bundesliga, Ligue 1, Champions League and ISL) otherwise |
| NBA | ESPN's scoreboard | none | yesterday's, today's and tomorrow's games |
| Weather | [Open-Meteo](https://open-meteo.com/) | none | now, today, tomorrow, yesterday or this weekend at a named place, in °C |
| Air quality | Open-Meteo air quality | none | AQI now (US scale, which differs from India's NAQI), PM2.5 and PM10 at a named place |
| Markets | [CoinGecko](https://www.coingecko.com/) (crypto), Yahoo Finance (Nifty, Sensex, Bank Nifty, Nasdaq, Dow, S&P 500, and any share named, found by Yahoo's search), [Frankfurter](https://www.frankfurter.app/) (exchange rates) | none | current prices; exchange rates are the European Central Bank's, published once per working day |
| Your portfolio | [Groww Trade API](https://groww.in/trade-api/docs) holdings, priced from Yahoo Finance | `GROWW_API_KEY` + `GROWW_API_SECRET` (a TOTP key) | "how is my portfolio doing": value, gain on what you paid, the day's move, best and worst. Read-only: Kip calls only the token and holdings endpoints, never orders, funds or positions |
| News | Google News RSS (India) | none | the top three headlines, or on a topic they name ("news about the elections") |

Keyed feeds keep one response for a minute, so follow-ups cost none of a free quota. Team names must
match exactly ("India" is not "India Under-19s"). ESPN and Yahoo Finance are unofficial and can change
without notice. Portfolio answers stay on the device: shown to you, but left out of every later
prompt (so never sent to Groq), memory and task record. Not covered: tennis, MotoGP, gold and silver, past prices,
tomorrow's air, UV, practice times, pit stops and broadcasters.

The agent chooses search queries, page URLs and short programs as needed. Search
uses DDGS, trying DuckDuckGo, Yahoo, Brave and Mojeek in turn until one answers (Yahoo up to
three times, in fresh sessions, since it sometimes serves a page layout DDGS cannot read);
queries are sent to those external search services,
and reading a page contacts its website. Model inference stays local. Search may
be throttled; errors are reported rather than replaced with invented results.
The webpage reader supports public HTML/plain text, validates every redirect,
pins connections to checked public addresses, rejects private/local networks,
and limits download size and reading time. It does not execute JavaScript, sign
into accounts, or read PDFs. Long pages can be read in successive text windows.

Python runs in Monty 1.3.0, a restricted Python interpreter in a separate worker,
with a fresh session, a 3-second execution limit, 64 MB interpreter memory limit,
bounded output, and no exposed host functions, filesystem mounts, network,
shell or credentials. It supports small calculations and data
transformations, not arbitrary CPython packages or installing dependencies.
The tool's input dictionary is available as `data`; print a result or end with
an expression. Failed programs provide errors for correction and do not count
as successful execution. This is bounded interpreter execution, not an OS-wide
security guarantee against runtime vulnerabilities.

Research and calculation results appear in the conversation with source links;
tool activity and failures appear in Activity. Final answers must reference
successful task-local tool evidence and the tools required by the interpreted
goal. This checks provenance/execution, not the factual correctness of every
sentence. External page content is treated as untrusted data.
