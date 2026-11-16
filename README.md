# agentic_friend: Kip, a companion you can talk to

Kip is a robot friend in a chat window. He remembers you, has a persistent personality, and looks
things up for you: scores, weather, prices, news, your portfolio, any web page, or a small Python
calculation. He is built to run on a **Raspberry Pi 4 (8 GB)**:

- **Data feeds** answer sport, weather, markets, news and your Groww portfolio with no model call.
- **Groq** answers chat and web lookups first, while it has free quota.
- **Qwen3.5 2B on Ollama** is the one local model, used when Groq can't answer.

How it works inside, including his character, is in [ARCHITECTURE.md](ARCHITECTURE.md).

## Run it on the Raspberry Pi

On a Pi 4 with Raspberry Pi OS 13 (Trixie, 64-bit), install [Ollama](https://ollama.com/download) once:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Then clone this repository and run the installer as your normal user:

```bash
git clone https://github.com/neavepaul/agentic-playground.git
```

```bash
bash agentic-playground/deploy/install.sh
```

The installer:
1. Installs whatever is missing from `python3-venv`, `nodejs` and `npm`.
2. Pulls the model.
3. Sets up the backend's Python environment and creates `backend/.env` from the example.
4. Builds the page and checks the model.
5. Installs `kip` as a systemd service that starts on boot and restarts if it crashes.

After that, add your keys to `backend/.env` and run `sudo systemctl restart kip`. To update, run
`git pull` and the installer again; it keeps `.env` and `kip.db`.

One service serves everything on port 8000, to the Pi itself and to phones and laptops on the same
Wi-Fi. Open **http://127.0.0.1:8000** in the Pi's browser, or **http://kip-pi.local:8000** (your
Pi's hostname, or its IP address) on a phone; the installer prints both.

**People.** Everyone picks their name and types a 4-digit PIN, or makes a profile. The first profile
made is the owner's: make yours first. Each person has their own chats and their own memories; one
Kip knows them all, but never tells one person what another said. Five wrong PINs lock a profile for
a while, doubling each time.

**Settings** (the gear beside your name). The owner decides, per person, who may ask about the Groww
portfolio and who may change their own PIN; the owner always may do both. Anyone allowed changes
their PIN there (the current one first), which signs out their other devices. To make a profile for
someone who picks their own PIN the first time they sign in, run `.venv/bin/python -m app.people add
NAME` in `backend` (`list` shows everyone).

**Chats.** Like a chat app: Kip opens on a new chat, your chats are listed newest first, and
deleting one removes its messages (not what Kip learned from it). The chat from before profiles
becomes the owner's "Earlier chats".

**On the Pi's own screen.** Run this once on the desktop version of Raspberry Pi OS:

```bash
bash agentic-playground/deploy/desktop.sh
```

It adds a Kip icon to the desktop and to the menu (Internet > Kip), and opens Kip in a
maximized window after the desktop logs in (once the service answers). Minimize or close it like
any window; the icon brings it back. Options:

- `--kiosk`: full screen at login instead, with no browser bars or window buttons, for a Pi that
  only ever shows Kip. Alt+F4 closes it.
- `--no-autostart`: don't open Kip at login; the icon still does.

**Bringing Kip's memories.** Everything Kip remembers is in `backend/memory/kip.db`. Stop Kip on
both machines, copy it across, then start Kip on the Pi:

```bash
scp backend/memory/kip.db pi@raspberrypi.local:agentic-playground/backend/memory/kip.db
```

```bash
sudo systemctl restart kip
```

Logs: `journalctl -u kip -f`. Keep `kip.db` on a USB 3 SSD if you can (set `MEMORY_DB_FILE`):
it is written constantly, which wears microSD cards.

## Develop

Prerequisites: Python **3.12+**, Node.js **20.19+ or 22.12+**, and [Ollama](https://ollama.com/download).
Install dependencies once while online. Commands are for Linux; on Windows use
`.venv\Scripts\python.exe` and `Copy-Item`.

**The model** (once; the Ollama installer keeps `ollama serve` running as a service):

```bash
ollama pull qwen3.5:2b-q4_K_M
```

**Backend**

```bash
cd backend
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip   # the bundled pip has known advisories
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
- "nba scores today", "nba games tonight", "premier league results", "isl fixtures"
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
| Football | [football-data.org](https://www.football-data.org/), then ESPN | `FOOTBALLDATA_API_KEY` (10 requests a minute free) | a named team's match, ten days either side across the key's competitions; ESPN covers this round of the top five leagues, the Champions League and the ISL. With a league named and no team ("premier league results"), the latest or next round's games |
| NBA | ESPN | none | a named team's game, or with no team named, the day's games across the league ("nba scores today", "nba games tonight") |
| Weather | [Open-Meteo](https://open-meteo.com/) | none | now, today, tomorrow, yesterday or this weekend at a named place |
| Air quality | Open-Meteo | none | AQI now (US scale, not India's NAQI), PM2.5 and PM10 |
| Markets | [CoinGecko](https://www.coingecko.com/), Yahoo Finance, [Frankfurter](https://www.frankfurter.app/) | none | crypto; Nifty, Sensex, Bank Nifty, Nasdaq, Dow, S&P 500 and any named share; ECB exchange rates (once per working day) |
| Your portfolio | [Groww Trade API](https://groww.in/trade-api/docs) holdings, priced from Yahoo Finance | `GROWW_API_KEY` + `GROWW_API_SECRET` (a TOTP key) | value, gain on what you paid, the day's move, best and worst |
| News | Google News RSS (India) | none | the top three headlines, or on a topic you name |

Cricket needs a team named; football needs a team or a league; names match exactly ("India" is not
"India Under-19s"). Football's free key doesn't cover the ISL, so ESPN answers it. ESPN
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
| `GROQ_CHAT_WAIT_SECONDS` | `30` | A chat reply waits this long in all for Groq (again, if limited again) before the local model answers; the local first word takes minutes |
| `GROQ_CHAT_RESERVE_TOKENS` | `3000` | Per-minute tokens an errand call leaves free on the models chat also uses, so a long lookup cannot push the next reply to the local model; `0` turns it off |
| `GROQ_PLAN_CALLS` / `GROQ_RESEARCH_CALLS` | `2` / `4` | Caps on Groq calls per plan and per research task; past them the local model continues |
| `GROQ_SEARCH_REASONING` | `low` | Effort for Groq's browser search; higher reads more pages and costs far more tokens |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `OLLAMA_MODEL` | `qwen3.5:2b-q4_K_M` | The local model (4-bit; the plain tag is 8-bit and about half the speed on a Pi) |
| `OLLAMA_KEEP_ALIVE` | `-1` | Keep the model loaded (`-1` always); reloading from microSD takes tens of seconds |
| `LOCAL_CHAT_ERRANDS` | `false` | Whether a reply the local model writes may start an errand. Off: errands start only from your own words |
| `CHAT_HISTORY_TOKENS` | `300` | Earlier conversation carried by each reply; less is faster on a Pi |
| `CHAT_COMPACTION` | `true` | When a chat outgrows the history budget, fold its older turns into a short note (one background model call after a reply). The transcript is kept whole; `false` just drops older turns from the prompt |
| `CHAT_CHECKPOINT_TOKENS` | `200` | Largest size of that note |
| `CHAT_HOSTED_CONTEXT_TOKENS` | `8192` | The window a chat prompt may use when Groq answers (the local model keeps `CONTEXT_TOKENS`; a reply that falls back to it is rebuilt at that size) |
| `CHAT_HOSTED_HISTORY_TOKENS` | `1600` | Earlier conversation Groq reads each reply. More remembers more but spends Groq quota faster; set equal to `CHAT_HISTORY_TOKENS` for the same prompt as the local model |
| `CHAT_EXTRACTION` | `true` | After a reply that could hold a lasting fact or correction, decide what to remember in its own short model call (the chat model only talks). A failed call still keeps a plain "remember that…". `false`: nothing new is learned from chat |
| `CHAT_MAX_TOKENS` / `CHAT_TEMPERATURE` | `300` / `0.6` | Length and variety of a chat reply |
| `CONTEXT_TOKENS` / `TEMPERATURE` | `4096` / `0.1` | Ollama context size; temperature for structured task calls |
| `LLM_TIMEOUT_SECONDS` / `TASK_TIMEOUT_SECONDS` | `600` / `1800` | Per model request; per whole task |
| `MAX_COORDINATOR_CYCLES` / `MAX_EXPLORER_ACTIONS` / `MAX_TOOL_CALLS` / `MAX_CRITIC_REVIEWS` | `20` / `40` / `80` / `8` | Hard budgets on a task |
| `INTENT_ROUTER` | `on` | `on` sends confident live questions straight to a feed or lookup; `shadow` only records what it would do; `off` |
| `DIGITAL_TOOLS_ENABLED` | `true` | Web search, page reading and Python |
| `TIMEZONE` | `Asia/Kolkata` | Every time Kip says or reasons with, whatever the device clock is set to |
| `MEMORY_DB_FILE` | `backend/memory/kip.db` | Kip's memories and state |
| `FRONTEND_DIST` | `frontend/dist` | The built page the backend serves at `/` when it exists (the installer builds it); in development Vite serves it instead |
| `EMBEDDINGS_ENABLED` / `EMBEDDING_MODEL` | `true` / `BAAI/bge-small-en-v1.5` | Meaning search in memory recall; off means keyword only |
| `PERSONA_ENABLED` / `PERSONA_NAME` | `true` / `Kip` | Personality layer |
| `PERSONA_MAKER` | unset | Who built Kip, as he should know it; unset, he says he doesn't know |
| `CHAT_DEFAULT_SPEAKER` | unset | Who is talking when nobody has said; otherwise learned from "my name is …" |
| `LOG_LEVEL` | `INFO` | `DEBUG` for diagnostics |
| `ALLOWED_HOSTS` | `127.0.0.1,localhost` | Extra host names the server answers to, besides the machine's own name (`kip-pi`, `kip-pi.local`) and private IP addresses; any other is refused |
| `ALLOWED_ORIGINS` | `http://127.0.0.1:5173,http://localhost:5173,http://127.0.0.1:8000,http://localhost:8000` | Extra pages that may open the WebSocket or send changes, besides the page Kip served (the dev server) |

Run **one Uvicorn worker**. On the Pi it listens on the home network, and everyone signs in with a
PIN. A PIN is a lock for a household, not for the internet: never forward Kip's port from your
router. The connection is plain HTTP, so someone on the same Wi-Fi who captures traffic could read
chats; keep the Wi-Fi itself private.

## Speed on a Raspberry Pi 4

Estimated from the Pi 4's memory bandwidth, not measured: the 2B model reads about 6–10 tokens a
second and writes about 3.

- **Instant**: the clock, feed answers, Groq web lookups, and named tool requests ("use Python to…",
  "read https://…"), which start at once.
- **Tens of seconds**: anything the local model writes.

Measure real numbers on the Pi with `.venv/bin/python -m app.bench` from `backend` while Kip runs.
The benchmark talks to Kip like the browser does, as the owner and in a new chat of its own; point
`MEMORY_DB_FILE` at a copy of `kip.db` for the run if it shouldn't be kept.

## API

| Endpoint | Result |
|---|---|
| `GET /api/health` | Backend health and model availability |
| `GET /api/profiles` | Profile names for the picker, and who this device is signed in as |
| `POST /api/profiles` / `POST /api/signin` / `POST /api/signout` | Make a profile `{"name", "pin"}` / sign in `{"profile_id", "pin"}` / sign out |
| `GET /api/me` | The signed-in profile (the owner also gets everyone else's) |
| `POST /api/profiles/{id}/access` | Owner only: `{"portfolio": true, "pin_change": true}` (either or both) |
| `POST /api/profiles/{id}/pin` | A profile made without a PIN: `{"pin"}` chooses it and signs in |
| `POST /api/me/pin` | `{"current", "pin"}`: change your own PIN, when allowed |
| `GET /api/conversations` / `DELETE /api/conversations/{id}` | Your chats, newest first / delete one |
| `GET /api/state` | The event sequence and your active or latest task |
| `GET /api/chat` / `POST /api/chat` | `?conversation_id=`: a chat's history / `{"message", "conversation_id"}` → 202 (a new id starts a chat) |
| `GET /api/memories` | `?q=&kind=&limit=`: search or list memories |
| `GET /api/persona` | Mood, traits, relationships and recent moments |
| `POST /api/tasks` | `{"goal": "..."}` → 202 with task state |
| `GET /api/tasks/{id}` / `POST /api/tasks/{id}/cancel` | Task state / cancel it |
| `WS /ws` | A snapshot, then your chat, task and tool events, and Kip's mood |

Everything but `/api/health`, `/api/profiles` and signing in needs a signed-in device (a cookie).
A program on the Pi itself may act as the owner by sending `X-Kip-Local: 1` (the benchmark does).

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
- **Kip won't start on the Pi:** `journalctl -u kip -n 50` shows why; `sudo systemctl restart kip`
  after fixing it.
- **Forget everything:** stop the backend and delete `backend/memory/kip.db`.
