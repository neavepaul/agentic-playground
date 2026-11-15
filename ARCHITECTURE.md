# How Kip works

Setup, configuration and the list of feeds are in the [README](README.md). This page covers what
happens inside, and why. The code graph (`graphify-out/`, see `CLAUDE.md`) has the exact
call and import relationships.

```text
Browser (frontend/src) ──HTTP /api, WS /ws──► FastAPI (app/main.py)
                                                 │
                         ChatSession (chat/session.py) ── one conversation, persisted in kip.db
                           │  router.py reads each message locally (no model)
                           ├─► chat reply: Groq, else Ollama (one streamed call)
                           └─► errand ──► TaskManager (tasks/manager.py)
                                            ├─ feeds.py      data APIs, no model
                                            ├─ Groq browser search, checked against what it read
                                            └─ Coordinator / Explorer / Critic loop over tools:
                                               web search, page reader, restricted Python
                         Persona (persona/) ── mood, traits, relationships; no model calls
                         MemoryStore (memory/recall.py) ── kip.db: memories, state, transcript
```

```text
backend/app/
  chat/         conversation, intent router, consent, claim checks, narration, spoken text
  tasks/        task context, permissions and the bounded agent loop
  agents/       Coordinator/Explorer/Critic prompts and typed decisions
  capabilities/ data feeds, web search, page reader, Python sandbox, answer grounding
  memory/       kip.db: searchable memories, structured state, transcript
  persona/      deterministic character: identity, voice, mood, stance
  llm/          model protocol, JSON validation and retry, Ollama and Groq adapters
  events/       bounded event history and WebSocket subscriber queues
  clock.py      the one clock, in TIMEZONE (IST by default)
```

## A message's path

1. **Stop, permissions and offers first.** "Stop" cancels the running errand. "Don't use Python"
   becomes a restriction that every later tool call is checked against. A "yes" to Kip's offer
   starts the offered errand.
2. **The router reads it** (`chat/router.py`, about 0.1 ms, or 5–8 ms with embeddings). It reports
   an intent (`information.sports`, `conversation.emotional`, `action.request`…), the domain, who
   or what is named, what is asked (a position, a forecast, a price) and when. Short follow-ups lean
   on the last question ("and Max?", "and yesterday?"), and a pronoun with nobody to point at gets
   "Who you mean, question?". The classifier is TF-IDF plus logistic regression, with bge-small
   embeddings averaged in once loaded, trained at startup from `router_data/intents.tsv`.
   Corrections people make are stored as unreviewed. They join the training data only after
   `router_train accept`, at the next start, so nothing retrains mid-conversation.
3. **A confident live question becomes an errand at once**, with no chat reply first: its answer is
   the reply. So does a request that names a tool ("use Python to…", "read https://…"). The clock
   ("what time is it?") is answered from the device with no errand at all.
4. **Anything else gets one streamed chat reply.** The reply can end with a hidden `###` JSON line
   that starts an errand, makes an offer, or remembers or forgets something. Errand results are
   narrated from templates, with no extra model call.

## Errands

`TaskManager` runs one errand at a time; others queue. Each errand tries, in order:

**Data feeds** (`capabilities/feeds.py`). The router's reading is handed to the feed for its domain
as a `LookupPlan.feed` (domain, entities, focus, time, place, the words said). The feed answers in
fixed words or returns `None`, and then the next step takes over. Each answer is recorded as web
search evidence, with its provider and the answer as the snippet, so the usual answer checks
and "don't use the internet" apply unchanged. `source_for()` applies each feed's own entry checks,
so the "Let me check Jolpica for that." callout is spoken only when that feed will really be asked.
Keyed feeds cache a response for 60 seconds, so follow-ups cost no free quota. Dates and times are
converted to `TIMEZONE`, temperatures to °C.

Some feed details:

- **Weather.** Old city names map to current ones ("Bangalore" alone geocodes to part of Karachi),
  and the most populous of the top five matches wins, so Hyderabad is the one in Telangana.
- **Matches.** The match picker prefers a live match, then the next one when asked about what is
  coming, then the latest result. A score asked for before a game, with no recent result, says
  when it starts. A result asked for is never answered with a game that hasn't happened.
- **Groww.** The access token is made from a standard-library TOTP code (RFC 6238) and kept until
  it expires. Only `GROWW_TOKEN_URL` and `GROWW_HOLDINGS_URL` are ever called: the key could
  place orders. Prices come from Yahoo Finance, because Groww's live-data endpoint refuses this key.
- **Privacy.** A Groww answer sets `TaskContext.private`. Private turns are dropped from
  `select_history` (the history in every prompt). `task_memories` returns nothing for them, and
  `task_record` keeps only "checked their Groww holdings". The answer is shown and stored on
  the device, but never sent to Groq.

**Groq browser search.** A plain live lookup the feeds can't answer skips planning: one Groq call
searches and answers. The answer is then checked against what the search read
(`capabilities/grounding.py`). The checks are: the right day, not a stale "latest", the right
winner and event, and °F converted to °C.

**The agent loop**, for everything else, or when Groq is unavailable. The Coordinator turns the
request into required outcomes. The Explorer picks tool calls, singly or as an independent batch,
until it can cite successful evidence in `finish_answer`. The Coordinator replans, or fails the
errand when progress stops. The Critic reviews plans on request, and any completion the
deterministic checks can't settle. Budgets on cycles, tool calls, model calls and time bound every
errand. A restart turns unfinished runs into explicit failures, never replays.

**Tools.**

- **Web search** uses DDGS, trying DuckDuckGo, Yahoo, Brave and Mojeek in turn. Errors are
  reported, never replaced with invented results.
- **The page reader** reads public HTML and plain text. It validates every redirect, pins the
  connection to a checked public address, rejects private networks and limits download size and
  time. It runs no JavaScript and reads no PDFs. Long pages are read in windows.
- **Python** runs in Monty 1.3.0, a restricted interpreter in a separate worker: a fresh session,
  3 seconds, 64 MB, no filesystem, network, shell or credentials. It suits small calculations,
  not packages. A request that only asks for Python costs one narrow model call that writes just
  the code.

A final answer must cite successful evidence from its own errand, plus every tool the request asked
for. That checks provenance, not every sentence. Page content is untrusted data.

## Models

There is one compact chat prompt (`CHAT_SYSTEM` in `chat/prompt.py`) for both Groq and the local
model, so they behave alike. Its fixed system part (identity, voice, rules) comes first, so Ollama
reuses it between replies. Each message adds the mood and stance, two memories (core facts about
the person first), a one-line status (with the real date and time), recent history up to
`CHAT_HISTORY_TOKENS`, and the message.

- **Groq** (`llm/groq.py`) goes first while it has quota. Free-tier limits are per model, so a spent
  model is skipped until its reset. When every model frees within `GROQ_MAX_WAIT_SECONDS`, the call
  waits instead of going local. An empty reply gets one retry. A model that tries to call its own
  tools is skipped for that reply.
- **Ollama** runs `qwen3.5:2b-q4_K_M`, kept loaded (`keep_alive=-1`), with schema-constrained output
  for structured calls. A 2B model is a weak judge of what to do: tested alone, it proposed
  nonsense errands. So by default (`LOCAL_CHAT_ERRANDS=false`) its replies only talk and remember,
  and errands start from the person's own words.

Structured output is validated with Pydantic. Invalid JSON gets one retry, then a safe task
failure. Prompts, raw output and hidden reasoning are never stored in task history, events or logs.
The `chat_reply` log line gives the provider, model, timings and how the prompt's size splits.

## Memory

`backend/memory/kip.db` is one SQLite file:

- **`memories`** holds chat exchanges, facts about people, task episodes and past answers. Recall
  combines keyword search (FTS5) with meaning search (`sqlite-vec` plus fastembed's
  `BAAI/bge-small-en-v1.5`, about 70 MB, downloaded on first use). Results are ranked by recency
  and importance. Rewordings reinforce a memory rather than duplicate it. Without embeddings,
  recall is keyword-only.
- **`state`** holds recent errands, Kip's self-model (which tools failed), his persona, standing
  restrictions and who he is talking to.
- **The transcript** is append-only, so the conversation survives a restart.

The old `persistent.json` and `beliefs.json` are imported once, into a `kip.db` with no state, then
backed up and never read again.

## Realtime

The WebSocket starts with a snapshot (`{sequence, task}` plus recent events). Events carry
increasing sequence numbers, and the UI skips any already in the snapshot. A slow subscriber gets a
fresh snapshot instead of silently losing events. Task history keeps the latest 50 tasks; event
history keeps 300.

## Kip's character

Kip is **Rocky's younger brother**, Rocky being the Eridian engineer from Andy Weir's *Project Hail
Mary*. He has Rocky's worldview, values and way of speaking, but is younger and more eager, without
the grief. He knows he is a robot, and never invents shared memories with Rocky or a past of his
own. He is a brilliant engineer who is a loyal, funny, curious friend, not an assistant made to
sound friendly.

- **Values**: problems are solved by observing, testing and working together. Friendship is shown
  by loyalty and action. Caring means attending to rest, food and safety. Not knowing is fine:
  investigate, never invent. Friends disagree. Protect friends, but respect their choices.
- **Speech**: short, direct sentences, sometimes without articles or helper verbs. Statements turn
  into questions with ", question?". Words are repeated for emphasis ("Bad. Very bad."). The
  grammar is imperfect, the reasoning precise, and he is never dim or robotic.
- **Humour**: literal, blunt and sincerely serious about odd things. He teases only friends who have
  earned it, and not every reply has a joke.

It is a **deterministic layer with no model calls**:

```text
bus events ─► persona.observe ─► appraisal.py ─► PersonaState (in kip.db state)
                                  cues.py: the person's own words only
voice_view(speaker, message) ─► chat prompt: identity, voice, mood, Stance line, track record
```

| Part | Where |
|---|---|
| Identity, voice, mood tones, trait baselines | `IDENTITY`, `VOICE`, `TONES`, `TRAIT_SPEC` in `persona/profile.py` |
| Mood, relationships, skills, episodes | `persona/state.py` (`CHARACTER_VERSION` 3; older saves migrate and keep their drift) |
| Turning events into state changes | `persona/appraisal.py`, with cues from `persona/cues.py` (conservative, negation-aware) |
| How to pitch each reply | `Persona.stance()` in `persona/engine.py` |

The stance is one line per reply. With a friend, humour suits. With a stranger, he is reserved. On
serious subjects, there is no humour. With distress, he listens and offers no plans. When rest or
food went short, he says so once and leaves the choice to them. A real achievement gets celebrated.
At goodnight, he promises nothing he won't do. When someone wants space, he gives a short
acknowledgement and asks nothing.

**Safety and truth come before character.** Claim checks, evidence, consent and restrictions are
untouched by the persona. After "leave me alone", check-ins and celebration stop until the quiet
period lapses. The voice applies only to words written to people. Tool arguments, summaries,
citations and the `###` action line stay in plain English. Planning prompts never see the persona.
Prompt views are capped (`VIEW_LIMIT` 1600, `VOICE_VIEW_LIMIT` 4200 characters), and identity is
trimmed last.

Whether replies really sound like Rocky's brother needs the real model and a person to judge.
`tests/test_rocky.py` and `tests/test_character.py` cover what code can: the fixed identity, the
scope of the voice, care, boundaries, celebration, migration.

## Testing

`pytest` uses scripted models. `conftest.py` makes every feed offline and enables
`LOCAL_CHAT_ERRANDS`, so no server, key or network is needed. Feed tests use each API's own
response shapes. One test asserts that Kip only ever calls the two Groww endpoints; another asserts
that portfolio answers never reach a prompt, memory or task record.
