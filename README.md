# agentic_friend: Kip, a companion you can talk to

Kip is a robot friend in a chat window. He remembers you, has a persistent personality, and looks
things up for you: scores, weather, prices, news, your portfolio, any web page, or a small Python
calculation. He is built to run on a **Raspberry Pi 4 (8 GB)**:

- **Data feeds** answer sport, weather, markets, news and your Groww portfolio with no model call.
- **Groq** answers chat and web lookups first, while it has free quota.
- **Qwen3.5 2B on Ollama** is the one local model, used when Groq can't answer.

How it works inside, including his character, is in [ARCHITECTURE.md](ARCHITECTURE.md).

## Start

Prerequisites: Python **3.12+**, Node.js **20.19+ or 22.12+**, and [Ollama](https://ollama.com/download).
Install dependencies once while online. Commands are for Raspberry Pi OS; on Windows use
`.venv\Scripts\python.exe` and `Copy-Item`.

**The model** (once; the Ollama installer keeps `ollama serve` running as a service):

```bash
ollama pull qwen3.5:2b-q4_K_M
```

**Backend**

```bash
cd backend
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env          # then add whichever keys you have (see Configuration)
.venv/bin/python -m app.check_model
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Don't overwrite an existing `.env` when updating.

**Frontend**

```bash
cd frontend
npm install -g pnpm
pnpm install
pnpm dev
```

Open **[http://127.0.0.1:5173](http://127.0.0.1:5173)**. The dev server proxies `/api` and `/ws` to
the backend on port 8000; keep both running. Nothing loads from a CDN.

## Use it

Talk. Kip answers in his own voice and remembers what you tell him about yourself (**About you** in
the side panel). A lookup shows its sources under the answer, and **Activity** lists what it did.
The **Kip** panel shows his mood, traits and how well you know each other.

Things to ask:

- "who won the last f1 race?", "india score", "lakers score", "how did arsenal do?"
- "weather in bengaluru tomorrow", "aqi in delhi"
- "bitcoin price", "where is the nifty", "reliance share price", "dollar to rupee"
- "how is my portfolio doing?"
- "top headlines", "news about the elections"
- "Read https://www.raspberrypi.com/documentation/ and summarise it."
- "Use Python to calculate the average of 12, 18, 21 and 29."

Before a feed answers, Kip says where he is looking ("Let me check Open-Meteo for that."). One lookup
runs at a time; another waits its turn. "Stop" cancels it. "Don't use the internet" or "don't use
Python" stop those tools until you say otherwise.

### Data feeds

Recognised questions on these subjects are answered from a feed before any search: one request (two
or three for some), exact facts in fixed words, no model call. Anything a feed doesn't cover goes to
the web lookup.

| Subject | Source | Key | Covers |
|---|---|---|---|
| F1 | [Jolpica](https://api.jolpi.ca/ergast/f1) (the Ergast successor) | none | results, qualifying, sprints, standings, the next race |
| Cricket | [CricketData.org](https://cricketdata.org/), then ESPN | `CRICKETDATA_API_KEY` (100 requests a day free) | results, live scores and fixtures a week or more either side; ESPN (the last few days) without a key or when a match is missing |
| Football | [football-data.org](https://www.football-data.org/), then ESPN | `FOOTBALLDATA_API_KEY` (10 requests a minute free) | ten days either side across the key's competitions; ESPN covers this round of the top five leagues, the Champions League and the ISL |
| NBA | ESPN | none | yesterday's, today's and tomorrow's games for a named team |
| Weather | [Open-Meteo](https://open-meteo.com/) | none | now, today, tomorrow, yesterday or this weekend at a named place |
| Air quality | Open-Meteo | none | AQI now (US scale, not India's NAQI), PM2.5 and PM10 |
| Markets | [CoinGecko](https://www.coingecko.com/), Yahoo Finance, [Frankfurter](https://www.frankfurter.app/) | none | crypto; Nifty, Sensex, Bank Nifty, Nasdaq, Dow, S&P 500 and any named share; ECB exchange rates (once per working day) |
| Your portfolio | [Groww Trade API](https://groww.in/trade-api/docs) holdings, priced from Yahoo Finance | `GROWW_API_KEY` + `GROWW_API_SECRET` (a TOTP key) | value, gain on what you paid, the day's move, best and worst |
| News | Google News RSS (India) | none | the top three headlines, or on a topic you name |

Teams and players must be named, and names match exactly ("India" is not "India Under-19s"). ESPN
and Yahoo Finance are unofficial and can change without notice. Not covered: tennis, MotoGP, gold
and silver, past prices, tomorrow's air, UV, practice times, pit stops and broadcasters.

**Your portfolio stays on the device.** Kip calls only Groww's token and holdings endpoints, never
orders, funds or positions (a test enforces this). The answer is shown to you but left out of every
later prompt (so it never reaches Groq), every memory, every task record and the log.

## Configuration

Copy `backend/.env.example` to `backend/.env`; environment variables override it. Keys live only in
`.env`, which git ignores.

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | unset | Groq answers first while it has quota (chat, planning, web search). Unset keeps everything on the Pi |
| `CRICKETDATA_API_KEY` / `FOOTBALLDATA_API_KEY` | unset | Cricket and football feeds; unset, ESPN answers |
| `GROWW_API_KEY` / `GROWW_API_SECRET` | unset | Groww TOTP key and secret for portfolio questions (read-only) |
| `GROQ_MODELS` | `openai/gpt-oss-120b,qwen/qwen3.8-27b,openai/gpt-oss-20b` | Preference order for planning and search; a model whose quota is spent is skipped until its reset |
| `CHAT_HOSTED` / `GROQ_CHAT_MODELS` | `true` / `qwen/qwen3.8-27b,openai/gpt-oss-20b` | Conversation on Groq, streamed, falling back to the local model |
| `GROQ_MAX_WAIT_SECONDS` | `20` | When every Groq model is rate-limited and one frees within this, wait for it instead of going local |
| `GROQ_PLAN_CALLS` / `GROQ_RESEARCH_CALLS` | `2` / `4` | Caps on Groq calls per plan and per research task; past them the local model continues |
| `GROQ_SEARCH_REASONING` | `low` | Effort for Groq's browser search; higher reads more pages and costs far more tokens |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `OLLAMA_MODEL` | `qwen3.5:2b-q4_K_M` | The local model (4-bit; the plain tag is 8-bit and about half the speed on a Pi) |
| `OLLAMA_KEEP_ALIVE` | `-1` | Keep the model loaded (`-1` always); reloading from microSD takes tens of seconds |
| `LOCAL_CHAT_ERRANDS` | `false` | Whether a reply the local model writes may start an errand. Off: errands start only from your own words |
| `CHAT_HISTORY_TOKENS` | `250` | Earlier conversation carried by each reply; less is faster on a Pi |
| `CHAT_MAX_TOKENS` / `CHAT_TEMPERATURE` | `300` / `0.6` | Length and variety of a chat reply |
| `CONTEXT_TOKENS` / `TEMPERATURE` | `4096` / `0.1` | Ollama context size; temperature for structured task calls |
| `LLM_TIMEOUT_SECONDS` / `TASK_TIMEOUT_SECONDS` | `600` / `1800` | Per model request; per whole task |
| `MAX_COORDINATOR_CYCLES` / `MAX_EXPLORER_ACTIONS` / `MAX_TOOL_CALLS` / `MAX_CRITIC_REVIEWS` | `20` / `40` / `80` / `8` | Hard budgets on a task |
| `INTENT_ROUTER` | `on` | `on` sends confident live questions straight to a feed or lookup; `shadow` only records what it would do; `off` |
| `DIGITAL_TOOLS_ENABLED` | `true` | Web search, page reading and Python |
| `TIMEZONE` | `Asia/Kolkata` | Every time Kip says or reasons with, whatever the device clock is set to |
| `MEMORY_DB_FILE` | `backend/memory/kip.db` | Kip's memories and state |
| `EMBEDDINGS_ENABLED` / `EMBEDDING_MODEL` | `true` / `BAAI/bge-small-en-v1.5` | Meaning search in memory recall; off means keyword only |
| `PERSONA_ENABLED` / `PERSONA_NAME` | `true` / `Kip` | Personality layer |
| `PERSONA_MAKER` | unset | Who built Kip, as he should know it; unset, he says he doesn't know |
| `CHAT_DEFAULT_SPEAKER` | unset | Who is talking when nobody has said; otherwise learned from "my name is …" |
| `LOG_LEVEL` | `INFO` | `DEBUG` for diagnostics |
| `ALLOWED_HOSTS` | `127.0.0.1,localhost` | Host names the server answers to; any other is refused |
| `ALLOWED_ORIGINS` | `http://127.0.0.1:5173,http://localhost:5173,http://127.0.0.1:8000,http://localhost:8000` | Pages that may open the WebSocket or send changes; a page on any other site is refused |

Run **one Uvicorn worker**, bound to loopback as shown. **There is no login**: anyone who can reach
the port can read the conversation and memories and start errands. On loopback that is only you, and
the Host and Origin checks above stop other websites in your browser from reaching it. Don't expose
it on a network (or add its address to `ALLOWED_HOSTS`) until authentication has been added.

## Speed on a Raspberry Pi 4

Estimated from the Pi 4's memory bandwidth, not measured: the 2B model reads about 6–10 tokens a
second and writes about 3.

- **Instant**: the clock, feed answers, Groq web lookups, and named tool requests ("use Python to…",
  "read https://…"), which start at once.
- **Tens of seconds**: anything the local model writes.

Measure real numbers with `python -m app.bench` from `backend` while Kip runs (point
`MEMORY_DB_FILE` at a copy of `kip.db` if the test conversation shouldn't be kept). Keep `kip.db` on a
USB 3 SSD if you can: it is written constantly, which wears microSD cards.

## API

| Endpoint | Result |
|---|---|
| `GET /api/health` | Backend health and model availability |
| `GET /api/state` | The event sequence and the active or latest task |
| `GET /api/chat` / `POST /api/chat` | Conversation history / `{"message": "..."}` → 202 |
| `GET /api/memories` | `?q=&kind=&limit=`: search or list memories |
| `GET /api/persona` | Mood, traits, relationships and recent moments |
| `POST /api/tasks` | `{"goal": "..."}` → 202 with task state |
| `GET /api/tasks/{id}` / `POST /api/tasks/{id}/cancel` | Task state / cancel it |
| `WS /ws` | A snapshot, then chat, task, tool and persona events |

Interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## Tests

```bash
cd backend && .venv/bin/python -m pytest -q      # scripted models and offline feeds; no server needed
cd frontend && npm test && npm run build
```

`python -m app.check_model` calls your real local model. Router accuracy and corrections:

```bash
.venv/bin/python -m app.chat.router_train eval
.venv/bin/python -m app.chat.router_train review
.venv/bin/python -m app.chat.router_train accept 3
```

## Troubleshooting

- **Model missing:** check `ollama list` shows `OLLAMA_MODEL`, run `python -m app.check_model`, and
  restart the backend after changing `.env`.
- **Slow replies:** set `GROQ_API_KEY`, lower `CHAT_HISTORY_TOKENS`, or raise the timeouts.
- **Truncated model output:** raise `CONTEXT_TOKENS`. A partial reply is a task failure, never a decision.
- **Disconnected UI:** it reconnects and fetches a fresh snapshot.
- **Port conflict:** free 8000 or 5173, or change both the Vite proxy and the backend command.
- **Forget everything:** delete `backend/memory/kip.db`, and move `persistent.json` and
  `beliefs.json` away, or they are imported again.
