"""An ongoing conversation with Kip. One streamed model call per message; errands run
in the background through the existing TaskManager, and progress is narrated without
any model calls.

The chat model only *proposes* errands and memories. Code decides what may start
(see consent.py), checks Kip's claims about its own actions against tool evidence, and
never runs a half-written or invalid action.
"""
import asyncio
import logging
import re
import time
from collections import deque
from datetime import datetime, timezone
from uuid import uuid4

from app.llm.base import ModelError
from app.memory.recall import prompt_memories
from app.tasks.manager import TERMINAL, TaskBusy
from app.tasks.permissions import READ_ONLY, TOOL_PLAIN, Grant, digital_scope, plain
from .consent import Offer, classify_reply, is_question, is_request, task_identity  # noqa: F401 (re-exported)
from .evidence import capabilities, denied_capabilities, task_record, unkept_promise, unsupported_execution
from .intent import Intent, given_name, introduced_as, read_intent
from .router import (THRESHOLD as ROUTER_THRESHOLD, LIVE, SPORTS, Frame, Route, clock_question, default_router,
                     fast_path, frame_of, read_slots)
from app.clock import date_str, spoken_time
from app.agents.schemas import GoalCondition, GoalPlan, InitialCapability, LookupPlan
from app.capabilities.feeds import FEEDS, is_portfolio
from .prompt import ChatAction, TrailerSplitter, build_messages
from .spoken import spoken
from .context import ContextBudgetError, estimate_tokens, message_tokens
from . import checkpoint as checkpoints, extract as extraction

_log = logging.getLogger("agentic_friend.chat")

HISTORY = 200
# Kip stating a total or comparison of his own: "over by 1k", "total 17k", "9k plus 3k equals 12k".
_SUM_IN_HEAD = re.compile(r"\d[\d,.]*\s*k?\s*(?:\+|plus|minus|-|times|x|\*)\s*\d|\b(?:equals?|totals?|over by|under by|short by|"
                          r"leaves?|left over)\b[^.]{0,25}\d")
SPEAKER_STATE = "chat-speaker"
# Past exchanges are recalled for what the person said. Kip's own old lines are left out:
# shown them, the model repeats them word for word.
_OWN_OLD_LINE = re.compile(r"\s*\|\s*I replied:.*$", re.S)
# A reading, price or score: a number with a unit or a currency.
_FIGURE = re.compile(r"\d[\d,.]*\s*(?:°|degrees?|celsius|fahrenheit|%|percent|runs?|wickets?|overs?|points?|goals?|"
                     r"rupees?|dollars?|pounds?|euros?|crore|lakh)|[₹$£€]\s*\d|\b\d{1,3}\s*(?:/|for)\s*\d{1,2}\b")
LOOKUP_PREFIX = "Look up online: "
_TOPIC_STOP = {"about", "again", "could", "does", "from", "have", "just", "know", "last", "like", "much", "next",
               "now", "okay", "please", "really", "right", "should", "still", "tell", "than", "that", "their",
               "them", "then", "there", "they", "this", "what", "when", "where", "which", "while", "will",
               "with", "would", "your", "today", "tonight", "tomorrow", "yesterday", "think", "going"}
_HOW_IT_WENT = {"match result", "finishing position", "score", "match status", "qualifying position", "standings"}
# How long a lookup's evidence answers follow-ups without searching again.
EVIDENCE_SECONDS = 600.0
_NEWER = re.compile(r"\b(now|still|yet|latest|update|current(ly)?|live|so far|anymore)\b")
_DISPUTE = re.compile(r"\b(you'?re wrong|you are wrong|that'?s wrong|thats wrong|not true|not right|incorrect|"
                      r"you got (it|that) wrong|wrong|actually (it'?s|it was|they)|it'?s actually)\b")
# Phrases Kip reaches for when he has nothing to add; fine once in a while, never as a habit.
_HABITS = ("good good", "amaze", "fist my bump", "i here", "hall quiet", "i like that", "you good", "i watch")
_HALF_YES = re.compile(r"^\W*(yes|yeah|yep|sure|ok|okay)\b")  # "Yes? What do you mean?" is not a yes
FIRST_PERSON = re.compile(r"\b(i|i'm|i've|me|my|mine|myself|the user)\b")
# Facts that matter in every conversation: who someone is, what they do, who they are to Kip.
_CORE_FACT = re.compile(r"\b(name|named|called|works?|job|engineer|developer|student|lives?|birthday|born|age|"
                        r"married|wife|husband|partner|girlfriend|boyfriend|son|daughter|mother|father|mom|dad|"
                        r"brother|sister|best friend|built|made|maker|creator|programmed|allergic|allergy)\b", re.I)


class ChatSession:
    def __init__(self, client, bus, manager, settings, store=None, persona=None,
                 shares_task_model: bool = True, session_id: str | None = None,
                 speaker: str | None = None, owner: bool = True, portfolio: bool = True,
                 household=None) -> None:
        self.session_id = session_id or str(uuid4())
        # A signed-in person's conversation: who is talking is fixed, and it sees only its own
        # errands and memories. Without one (tests, tools), Kip learns names as he always has.
        self.person, self.owner, self.portfolio = speaker, owner, portfolio
        # Who uses this Kip: {"owner": name, "people": [names]}, read fresh for each message.
        self.household = household
        if speaker:
            from .hub import OwnTasks
            manager = OwnTasks(manager, self.session_id, _speaker_slug(speaker), portfolio, owner)
        self.client, self.bus, self.manager = client, bus, manager
        self.settings, self.store, self.persona = settings, store, persona
        self.shares_task_model = shares_task_model
        self.name = persona.name if persona else settings.persona_name
        self.turns: deque[dict] = deque(maxlen=HISTORY)
        self.pending_goals: deque[str] = deque()
        self._lock = asyncio.Lock()
        self._jobs: set[asyncio.Task] = set()
        self.replying = False
        self._closed = False
        # Bumped by every reset: offers, queued errands and narration from before it
        # are never acted on.
        self.generation = 0
        if self.store:
            self.turns.extend(self.store.conversation_turns(self.session_id, HISTORY))
        # A note standing in for the older turns of a long chat (checkpoint.py), kept across restarts.
        self.checkpoint = (checkpoints.load(self.store.get_state(checkpoints.state_key(self.session_id)))
                           if self.store else None)
        self.compaction = "not_needed"   # not_needed | completed | failed: the last attempt
        self._checkpoint_epoch = 0       # bumped when the note is dropped, so a slower write never revives it
        self._compact_after = 0          # user turn before which a failed attempt is not repeated
        self._announced: set[tuple[str, ...]] = set()
        self.offer: Offer | None = None
        self.offer_log: deque[dict] = deque(maxlen=20)
        self._offer_seen: dict[str, str | None] = {}  # user turn id -> offer visible when sent
        self._started_by_turn: set[str] = set()
        self._remembered: list[str] = []
        self._queued_grants: dict[tuple[str, ...], Grant] = {}
        self._queued_plans: dict[tuple[str, ...], GoalPlan] = {}
        self._declined: set[tuple[str, ...]] = set()  # offers refused in this conversation: not offered again
        self._intents: dict[str, Intent] = {}         # user turn id -> what that message does
        self._work_epoch = 0
        self._turn_epochs: dict[str, int] = {}
        self._user_turns = 0
        self._offered: dict[tuple[str, ...], int] = {}  # offer identity -> user turn it was made on
        self._last_unasked_offer: tuple[int, Offer | None] = (-99, None)
        # Who Kip is talking with: a name they gave ("my name is Neave"), kept across restarts.
        self._speaker: str | None = None
        self._speaker_display: str | None = None
        if speaker:
            self._speaker, self._speaker_display = _speaker_slug(speaker), speaker
        elif self.store:
            saved = self.store.get_state(SPEAKER_STATE)
            if saved:
                self._speaker, self._speaker_display = _speaker_slug(saved), saved
        self.gate = None                               # shared-model priority (conversation first)
        # Intent router (router.py), per settings.intent_router. The app hands in one trained with
        # reviewed corrections and the embedding model; without one, the shipped TF-IDF router.
        self.router = None
        self._frame: Frame | None = None               # the last information request, for follow-ups
        self._frame_route: Route | None = None
        self._frame_answer = ""                        # what Kip's lookup said about that request
        self._frame_evidence = ""                      # what that lookup's source actually said
        self._frame_evidence_at = 0.0
        self._lookup_queries: dict[tuple[str, ...], str] = {}  # running or queued lookup -> what it searches
        self._routed = 0
        self._routes: dict[str, Route] = {}            # user turn id -> how the router read it
        self._correction_goals: dict[tuple[str, ...], int] = {}  # corrected lookup -> correction record

    @property
    def _hosted_chat(self) -> bool:
        """Whether chat can be served by Groq (with the local model as the fallback): then a reply is
        written against whichever window will answer it."""
        return bool(getattr(self.client, "prompt_alternatives", False))

    @property
    def _history_budget(self) -> int:
        """The history the checkpoint is sized against: Groq's, when it is the usual speaker. A local
        fallback then trims the oldest of the verbatim turns with the usual selection."""
        return (self.settings.chat_hosted_history_tokens if self._hosted_chat
                else self.settings.chat_history_tokens)

    def attach(self) -> None:
        self.bus.add_listener(self._narrate)

    # ── public API ──────────────────────────────────────────────────────────
    @property
    def busy(self) -> bool:
        return self.replying or any(not job.done() for job in self._jobs)

    @property
    def pending_offer(self) -> str | None:
        return self.offer.goal if self._offer_valid() else None

    def submit(self, text: str, speaker: str | None = None) -> dict:
        text = text.strip()
        if not text:
            raise ValueError("Say something first.")
        if self._closed:
            raise ValueError("Conversation is shutting down.")
        # "Don't use Python": binding from this moment on, for running and queued errands too.
        # Applied before anything else reads this message.
        before = set(self.manager.restrictions.describe())
        if self.manager.restrictions.update(text):
            _log.info("Restrictions now: %s", self.manager.restrictions.describe())
            self.manager.save_restrictions()
        after = set(self.manager.restrictions.describe())
        if self.person:
            speaker = self.person
        elif not speaker:
            self._learn_speaker(text)
        speaker_id = self._speaker_id(speaker)
        earlier = list(self.turns)
        turn = self._add_turn("user", text, speaker=speaker_id)
        self._user_turns += 1
        self._intents[turn["id"]] = read_intent(text, sorted(before - after), sorted(after - before))
        if self._intents[turn["id"]].cancel:
            # Cancellation must not wait behind a slow or unavailable chat model.
            self._cancel_work()
        self._turn_epochs[turn["id"]] = self._work_epoch
        if self.gate is not None:
            self.gate.conversation_waiting(+1)  # no new errand model call starts until this reply is done
        # Consent is judged against the offer the person could actually see when they
        # sent this message; a message queued earlier can never approve a later offer.
        self._offer_seen[turn["id"]] = self.offer.id if self._offer_valid() else None
        self._spawn(self._respond(turn, earlier))
        return turn

    def history(self) -> list[dict]:
        return list(self.turns)

    def state(self) -> dict:
        return {"session_id": self.session_id,
                "queued_errands": list(self.pending_goals), "offered_errand": self.pending_offer,
                "offer": self.offer.public() if self.offer else None, "generation": self.generation}

    async def drain(self) -> None:
        while self._jobs:
            jobs = list(self._jobs)
            await asyncio.gather(*jobs, return_exceptions=True)
            self._jobs.difference_update(job for job in jobs if job.done())

    async def close(self) -> None:
        self._closed = True
        self.pending_goals.clear()
        self._queued_grants.clear()
        self._queued_plans.clear()
        self._expire_offer("expired")
        jobs = list(self._jobs)
        for job in jobs:
            job.cancel()
        await asyncio.gather(*jobs, return_exceptions=True)
        self._jobs.difference_update(jobs)

    async def reset_pending(self) -> None:
        """Discard replies, offers and queued work from before, then accept chat again."""
        await self.close()
        self.generation += 1
        self._announced.clear()
        self._offer_seen.clear()
        self._closed = False

    # ── replying ────────────────────────────────────────────────────────────
    def _spawn(self, coroutine) -> None:
        job = asyncio.create_task(coroutine)
        self._jobs.add(job)
        job.add_done_callback(self._jobs.discard)

    def _add_turn(self, role: str, text: str, **extra) -> dict:
        canonical = text
        if role != "user":
            text = spoken(text)  # Kip talks; code, markdown and links never reach the voice
        turn = {"id": str(uuid4()), "role": role, "text": text,
                "content": canonical, "screen_text": canonical, "spoken_text": text,
                "session_id": self.session_id,
                "at": datetime.now(timezone.utc).isoformat(), **extra}
        if self.store:
            self.store.append_turn(self.session_id, turn)
        self.turns.append(turn)
        speaker = "user" if role == "user" else self.name
        self.bus.emit("chat_message", speaker, turn=turn, summary=f"{speaker}: {text}")
        return turn

    def _learn_speaker(self, text: str) -> None:
        """"My name is Neave", or "it's Neave" from someone Kip already knows: who is talking now."""
        known = {key: {"name": key.title()} for key in (self.persona.state.relationships if self.persona else {})}
        name = given_name(text) or introduced_as(text, known)
        if not name:
            return
        display = name if name != name.casefold() else name.title()
        if _speaker_slug(display) == self._speaker:
            return
        self._speaker, self._speaker_display = _speaker_slug(display), display
        if self.store:
            self.store.set_state({SPEAKER_STATE: display})

    def _speaker_id(self, speaker: str | None) -> str | None:
        """Who is talking: named by the caller, the configured default, or a name they gave."""
        named = (speaker or self.settings.chat_default_speaker or "").strip()
        return _speaker_slug(named) if named else self._speaker

    def _speaker_name(self, speaker_id: str | None) -> str:
        if not speaker_id:
            return "unknown person"
        if speaker_id == self._speaker and self._speaker_display:
            return self._speaker_display
        return speaker_id.replace("_", " ").title()

    def _holder(self) -> str | None:
        """The owner's name, when someone else is talking: whose Groww portfolio it is."""
        if not self.person or self.owner or self.household is None:
            return None
        return self.household().get("owner")

    def _status(self, speaker_id: str | None = None) -> dict:
        status = {"now": date_str(), "talking_with": self._speaker_name(speaker_id)}
        if self.household is not None:
            # Kip knows who has a profile and whose the Groww portfolio is; never what the others told him.
            known = self.household()
            names = [n + (" (owner: the Groww portfolio is theirs)" if n == known.get("owner") else "")
                     for n in known.get("people", [])]
            if names:
                status["people_with_profiles"] = ", ".join(names) + ". You know their names; what each " \
                                                 "told you stays with them."
        active = self.manager.active()
        # "none", not null: models repeat null verbatim ("my current errand is null").
        status["current_errand"] = errand_progress(active) if active else "none"
        if self.pending_goals:
            status["queued_errands"] = list(self.pending_goals)
        if self.pending_offer:
            status["offered_errand_awaiting_yes"] = self.pending_offer
        finished = [c for c in self.manager.tasks.values() if c.status != "running"][-3:]
        if finished:
            # What each errand actually did, tried, was refused and left undone.
            status["recent_errands"] = [task_record(c) for c in finished]
        status["my_tools"] = capabilities(self.manager, self.settings)
        if self.manager.restrictions:
            status["asked_not_to"] = self.manager.restrictions.describe()
        return status

    def _freshness(self) -> str:
        """What Kip has leaned on in his last few replies, so this one says something new: a
        catchphrase is seasoning only while it is rare."""
        replies = [t["text"].casefold() for t in self.turns if t["role"] == "kip" and not t.get("source")][-3:]
        if not replies:
            return ""
        notes = []
        if sum(r.rstrip(" .!").endswith("question?") or r.rstrip().endswith("?") for r in replies) >= 2:
            notes.append("do not end with a question")
        used = [phrase for phrase in _HABITS if any(re.search(rf"\b{re.escape(phrase)}\b", r) for r in replies)]
        if used:
            notes.append("do not say " + ", ".join(f"'{p}'" for p in used))
        openers = [" ".join(r.split()[:2]) for r in replies]
        if len(openers) >= 2 and len(set(openers)) < len(openers):
            notes.append(f"do not open with '{max(set(openers), key=openers.count)}'")
        return ("Your last replies leaned on the same habits: this time " + "; ".join(notes)
                + ". Say something new instead.") if notes else ""

    def _guidance(self, intent: Intent, live: bool | None = None) -> str:
        """How this particular message must be handled, decided by code from its words."""
        restrictions, parts = self.manager.restrictions, []
        if intent.cancel:
            parts.append("They want the current work stopped: it is being stopped now. Confirm in a few words.")
        elif intent.changes_state_only:
            parts.append("This only changes a permission or turns something down. Acknowledge in a word or two "
                         "('Alright.' is a complete reply) and stop. Do not redo, resume or offer anything earlier.")
        needs = intent.needs_tools
        if needs:
            off = [t for t in needs if not self.settings.digital_tools_enabled or restrictions.forbids_tool(t)]
            usable = [t for t in needs if t not in off]
            names = lambda tools: " and ".join(TOOL_PLAIN.get(t, t) for t in tools)
            if not usable:
                why = "they ruled that out" if self.settings.digital_tools_enabled else "that is switched off"
                parts.append(f"They asked for something that needs you to {names(off)}, and {why}. Turn it down "
                             "with one blunt, good-natured line. Don't fake it, and don't do it some other way.")
            else:
                parts.append(f"This needs you to actually {names(usable)}; an errand starts for it. Say so in a few "
                             "words. Never write its result, output or page content yourself: the real one follows."
                             + (f" You won't {names(off)}: it was ruled out." if off else ""))
        if intent.calculates and self.settings.digital_tools_enabled and not restrictions.forbids_tool("run_python"):
            parts.append("This needs arithmetic over several numbers. Never add, compare or total them yourself and "
                         "never state a result: put a Python calculation in task, written out with every number it "
                         "uses (from this message and earlier ones), and say in a few words you're running it. "
                         "The real result follows.")
        if (intent.live_fact if live is None else live) and not intent.changes_state_only:
            parts.append("This is live information (weather, a price, a score, news): a lookup starts now. Say so "
                         "in a few words, in character. Never state a figure yourself; the real one follows.")
        if intent.greeting:
            parts.append("A greeting: a line or two in your own voice, from something real (the hour, the "
                         "day, what's happened). Asked how you are, go by your actual mood and errands, not "
                         "an invented day. Don't ask them back unless you want to know.")
        if intent.wants_offer:
            parts.append("They want an offer, not action: put it in offer and wait for their yes.")
        if intent.about_past:
            parts.append("They ask what you did: use only status.recent_errands. Say what was "
                         "done, tried, refused or not done.")
        if intent.small_talk:
            parts.append("Small talk: just talk; don't suggest errands.")
        return " ".join(parts)

    async def _recall(self, text: str, speaker: str | None = None) -> list[dict]:
        if not self.store:
            return []
        try:
            # Core facts about the person first, whatever was said; then durable knowledge
            # relevant to this message. Past chat snippets would otherwise crowd it out,
            # since they often repeat the very facts they mention.
            # A signed-in person's conversation recalls only what they told Kip (and, for the owner,
            # what Kip knew before there were profiles): never another person's facts or chats.
            who = {"viewer": self._speaker, "legacy": self.owner} if self.person else {}
            core = await asyncio.to_thread(lambda: self.store.core_facts(speaker, 4, **who))
            pinned = {m["id"] for m in core}
            memories = [m for m in await self.store.recall(text, kinds=["fact", "preference", "episode", "skill"], k=4,
                                                           **who)
                        if m["id"] not in pinned]
            chats = await self.store.recall(text, kinds=["conversation"], k=2, **who)
            return core + memories + [{**m, "text": _OWN_OLD_LINE.sub("", m["text"])} for m in chats]
        except Exception:
            _log.exception("Memory recall failed; continuing conversation without recalled memories")
            return []

    async def _respond(self, user_turn: dict, earlier: list[dict]) -> None:
        reply_id = str(uuid4())
        finished = False
        queued_at = time.monotonic()
        try:
            async with self._lock:  # FIFO: one reply at a time, in order
                queue_seconds = time.monotonic() - queued_at
                said = user_turn["text"]
                intent = self._intents.get(user_turn["id"]) or read_intent(said)
                route = await self._route(user_turn, intent, earlier)
                if route is not None and self.settings.intent_router == "on":
                    question = self._router_clarifies(user_turn, intent, route)
                    if question:
                        # "How did he do?" with nobody named: ask who, no model call.
                        finished = self._finish_turn(user_turn, reply_id, question, None,
                                                     {"chat_model_calls": 0, "intent_router": route.intent})
                        self._intents.pop(user_turn["id"], None)
                        return
                    if route.known and route.followup and not (intent.cancel or intent.changes_state_only):
                        # "Who is playing?" when the last lookup already named both sides.
                        finished = self._finish_turn(user_turn, reply_id, route.known, None,
                                                     {"chat_model_calls": 0, "intent_router": "known"})
                        self._intents.pop(user_turn["id"], None)
                        return
                    if self._asks_the_time(intent, route, said):
                        # The clock answers "what time is it?": no model call, never a guess.
                        finished = self._finish_turn(user_turn, reply_id, spoken_time(), None,
                                                     {"chat_model_calls": 0, "intent_router": "clock"})
                        self._intents.pop(user_turn["id"], None)
                        return
                    if self._from_last_lookup(user_turn, route):
                        route.from_evidence = True  # answered from what was just found: no new search
                        lookup = None
                    else:
                        lookup = self._routed_lookup(user_turn, intent, route)
                else:
                    lookup = self._direct_lookup(user_turn, intent, earlier)
                if lookup:
                    # A live fact needs no chat reply first: the lookup starts now, and its
                    # answer, written in Kip's voice, is the reply. One model step, not two.
                    goal, query = lookup
                    if self._lookup_pending(query):
                        # "when is it starting today?" again, while that lookup still runs.
                        finished = self._finish_turn(user_turn, reply_id, "Still checking that. Soon.", None,
                                                     {"chat_model_calls": 0, "intent_router": "same lookup"})
                        self._intents.pop(user_turn["id"], None)
                        return
                    self._lookup_queries[task_identity(goal)] = query
                    self._start_or_queue(goal, user_turn["id"], Grant.read_only(said), plan=_lookup_plan(query, route, said, self._holder()))
                    self._intents.pop(user_turn["id"], None)
                    finished = True
                    return
                if self._tool_request(user_turn, intent) and self._started_errand(user_turn):
                    # "Use Python to…", "read https://…": like a live lookup, the errand's answer is
                    # the reply. A chat reply first can take minutes on the local model.
                    self._intents.pop(user_turn["id"], None)
                    self._routes.pop(user_turn["id"], None)
                    finished = True
                    return
                self.replying = True
                waiting = self.shares_task_model and self.manager.active_id is not None
                self.bus.emit("chat_thinking", self.name, skip_history=True, session_id=self.session_id,
                              reply_id=reply_id, waiting_on_errand=waiting)
                recall_started = time.monotonic()
                memories = await self._recall(user_turn["text"], user_turn.get("speaker"))
                recall_seconds = time.monotonic() - recall_started
                voice = (self.persona.voice_view(speaker=user_turn.get("speaker"), message=user_turn["text"])
                         if self.persona else None)
                # Include replies to preceding messages that finished while this one
                # waited, without leaking later user messages into its context.
                known = {t["id"] for t in earlier}
                earlier += [t for t in self.turns if t.get("in_reply_to") in known and t["id"] not in known]
                anchors = {t["id"]: index for index, t in enumerate(earlier)}
                earlier.sort(key=lambda t: (anchors.get(t.get("in_reply_to"), anchors[t["id"]]),
                                           bool(t.get("in_reply_to")), anchors[t["id"]]))
                guidance = self._guidance(intent, live=self._is_live(user_turn["id"], intent))
                fresh = self._freshness()
                if fresh:
                    guidance = (guidance + " " + fresh).strip()
                checked = self._evidence_guidance(user_turn)
                if checked:
                    guidance = (guidance + " " + checked).strip()
                note = self.checkpoint
                status = self._status(user_turn.get("speaker"))
                recalled = prompt_memories(memories)
                since = checkpoints.after(earlier, note)

                def prompt(context_tokens: int, history_tokens: int) -> list[dict]:
                    return build_messages(self.name, voice, recalled, status, since, user_turn["text"],
                                          guidance=guidance, checkpoint=note.text if note else "",
                                          context_tokens=context_tokens - self.settings.chat_max_tokens,
                                          history_tokens=history_tokens)
                # Sized for the local model, whose window is the small one: this is also what a Groq
                # reply falls back to. When Groq serves chat, it gets its own, roomier prompt.
                try:
                    messages = local = prompt(self.settings.context_tokens, self.settings.chat_history_tokens)
                except ContextBudgetError:
                    if not self._hosted_chat:
                        raise
                    local = None  # too large for the local model; Groq may still take it
                    messages = None
                alternate = {}
                if self._hosted_chat:
                    try:
                        messages = prompt(self.settings.chat_hosted_context_tokens,
                                          self.settings.chat_hosted_history_tokens)
                    except ContextBudgetError:
                        if local is None:
                            raise
                    alternate = {"local_messages": local if local is not None else False}
                splitter = TrailerSplitter()
                try:
                    async for chunk in self.client.stream_text(messages, max_tokens=self.settings.chat_max_tokens,
                                                               temperature=self.settings.chat_temperature,
                                                               **alternate):
                        visible = splitter.feed(chunk)
                        if visible:
                            self.bus.emit("chat_delta", self.name, skip_history=True,
                                          session_id=self.session_id, reply_id=reply_id, text=visible)
                except ModelError as exc:
                    if not splitter.text.strip():
                        raise
                    # The model started answering, then failed: keep what was said,
                    # report the interruption honestly, and run no action.
                    _log.warning("Chat stream failed mid-reply: %s", exc)
                    remaining, parsed = splitter.finish()
                    finished = self._finish_turn(user_turn, reply_id, parsed.visible + " …", None,
                                                 {"interrupted": True}, error=str(exc))
                    return
                remaining, parsed = splitter.finish()
                if remaining:
                    self.bus.emit("chat_delta", self.name, skip_history=True, session_id=self.session_id,
                                  reply_id=reply_id, text=remaining)
                metrics = dict(getattr(self.client, "last_metrics", {}) or {})
                metrics["memory_recall_seconds"] = round(recall_seconds, 3)
                # The prompt the serving model actually read: Groq the roomier one, the local model its own.
                sent = messages if (not alternate or metrics.get("provider") == "groq" or local is None) else local
                metrics["input_chars"] = sum(len(m["content"]) for m in sent)
                metrics["estimated_prompt_tokens"] = message_tokens(sent)
                metrics["prompt_for"] = "hosted" if alternate and sent is not local else "local"
                metrics["token_estimator"] = "utf8_bytes_div3_estimate"  # a label, not a secret  # nosec B105
                metrics["queue_seconds"] = round(queue_seconds, 3)
                metrics["chat_model_calls"] = 1
                # Whether older turns are carried as a note, and how the last attempt to write one went.
                metrics["checkpoint_tokens"] = estimate_tokens(note.text) if note else 0
                metrics["checkpoint_turns"] = note.turns if note else 0
                metrics["compaction"] = self.compaction
                # Where the prompt's size goes: a long prompt is what makes a local model slow to start.
                metrics["prompt_chars"] = {"system": len(sent[0]["content"]),
                                           "history": sum(len(m["content"]) for m in sent[1:-1]),
                                           "latest": len(sent[-1]["content"])}
                action, action_error = parsed.action, parsed.error
                if (action and (action.task or action.offer) and not self.settings.local_chat_errands
                        and metrics.get("provider") != "groq"):
                    # A small local model's errands can be nonsense ("What is your favorite thing?"),
                    # and an offer blocks lookups while it waits for a yes. Its memories are kept.
                    _log.info("Local reply's errand ignored (LOCAL_CHAT_ERRANDS=false): task=%r offer=%r",
                              action.task, action.offer)
                    action = action.model_copy(update={"task": "", "offer": ""})
                text = parsed.visible
                if metrics.get("truncated"):
                    # A cut-off reply is shown as cut off; a half-written action never runs.
                    text = (text + " …").strip()
                    action, action_error = None, "The reply was cut off before its action line finished."
                if action_error:
                    _log.warning("Chat action not executed: %s", action_error)
                _log.info("chat_reply provider=%s model=%s first_token_seconds=%s total_seconds=%s memories=%d task=%s "
                          "recall_seconds=%.3f input_chars=%d prompt_chars=%s truncated=%s",
                          metrics.get("provider", "local"), metrics.get("model", "-"), metrics.get("first_token_seconds"),
                          metrics.get("total_seconds"), len(memories), bool(action and action.task), recall_seconds,
                          metrics["input_chars"], metrics["prompt_chars"], bool(metrics.get("truncated")))
                text = text or "Lost my words. Say again, question?"
                reply_turn = self._finish_turn(user_turn, reply_id, text, action, metrics, action_error=action_error)
                finished = True
                await self._act(action, user_turn, reply_turn, written=text)
        except ContextBudgetError as exc:
            self._finish_turn(user_turn, reply_id,
                              "Too much context for one answer. Too big. You split in smaller parts, "
                              "question?", None, {"chat_model_calls": 0}, error=str(exc))
            finished = True
        except ModelError as exc:
            # The reply failed, not the request: what the person's own words ask for (a tool run, a
            # live lookup, a yes to an offer) still starts, as it would with a reply that proposed nothing.
            started = self._started_errand(user_turn)
            self._finish_turn(user_turn, reply_id, "Cannot think now. My language model not answer. " + (
                "But your errand I start." if started else "You check it run, question?"), None, {}, error=str(exc))
            finished = True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _log.exception("Chat reply failed")
            self._finish_turn(user_turn, reply_id, "Something break in me while I answer. Bad. Very bad.",
                              None, {}, error=type(exc).__name__)
            finished = True
        finally:
            self.replying = False
            if finished and not self._closed and self.settings.chat_compaction and self.store:
                self._spawn(self._compact())
            self._offer_seen.pop(user_turn["id"], None)
            self._turn_epochs.pop(user_turn["id"], None)
            if self.gate is not None:
                self.gate.conversation_waiting(-1)
            if not finished:
                # Cancelled (reset, shutdown): release any "Thinking…" indicator.
                self.bus.emit("chat_reply_ended", self.name, skip_history=True, session_id=self.session_id,
                              reply_id=reply_id, reason="cancelled")

    def _tool_request(self, user_turn: dict, intent: Intent) -> bool:
        """Their own words plainly ask for a tool to run, with nothing that needs a word first:
        not an offer they asked for, a refusal, a cancellation, a permission change, or an answer
        to an offer Kip made."""
        if not intent.needs_tools or intent.cancel or intent.wants_offer or intent.declines:
            return False
        if intent.granted or intent.revoked:
            return False
        offer = self.offer if self._offer_valid() else None
        return not (offer and self._offer_seen.get(user_turn["id"]) == offer.id)

    def _started_errand(self, user_turn: dict) -> bool:
        """Start (or queue) what the person's own words ask for, with no reply from the model.
        True if an errand started or was queued."""
        before = (len(self.manager.tasks), len(self.pending_goals))
        self._decide_errands(None, user_turn)
        return (len(self.manager.tasks), len(self.pending_goals)) != before

    def _finish_turn(self, user_turn, reply_id, text, action, metrics, **extra) -> dict:
        return self._add_turn("kip", text, reply_id=reply_id, in_reply_to=user_turn["id"],
                              to=user_turn.get("speaker"),
                              action=action.model_dump() if action else None, metrics=metrics,
                              **{k: v for k, v in extra.items() if v})

    # ── acting on a reply ───────────────────────────────────────────────────
    async def _act(self, action: ChatAction | None, user_turn: dict, reply_turn: dict, written: str = "") -> None:
        said, reply = user_turn["text"], reply_turn["text"]
        intent = self._intents.get(user_turn["id"]) or read_intent(said)
        marker = self._work_marker()
        self._decide_errands(action, user_turn)
        if (intent.calculates and self._work_marker() == marker and not self.manager.active()
                and not self.pending_goals):
            self._start_calculation(user_turn)   # they asked for arithmetic and the model proposed no run
        started = (self._work_marker() != marker or bool(self.manager.active()) or bool(self.pending_goals))
        # Claims are judged on what the model wrote, before it was tidied for speech.
        self._check_claims(written or reply, intent, live=self._is_live(user_turn["id"], intent),
                           started=started, user_turn=user_turn)
        self._routes.pop(user_turn["id"], None)
        self._intents.pop(user_turn["id"], None)
        if self.store:
            # A reply that wrote its own remember/forget lines is honoured; otherwise what is worth
            # keeping is decided after it, in a separate call that never delays the reply.
            volunteered = bool(action and (action.remember or action.forget))
            await self._remember(action if volunteered else None, user_turn, reply)
            if (not volunteered and self.settings.chat_extraction and not self._closed
                    and (extraction.worth_reading(said) or extraction.explicit_request(said))):
                self._spawn(self._extract(user_turn, reply))

    async def _extract(self, user_turn: dict, reply: str) -> None:
        """Read one exchange for lasting facts and corrections, and file them like any other memory."""
        said, speaker = user_turn["text"], user_turn.get("speaker")
        known = [m for m in await self._recall(said, speaker) if m.get("kind") in ("fact", "preference")]
        who = self._speaker_name(speaker) if speaker else ""
        found = await extraction.extract(self.client, said, reply, who or "unknown person", known)
        if found is None:
            # The call failed or was unreadable. A plain "remember that..." is still kept, as asked.
            asked = extraction.explicit_request(said, who)
            found = extraction.Extraction(remember=[asked]) if asked else None
        if found is None or self._closed or not (found.remember or found.forget):
            return
        await self._remember(ChatAction(remember=found.remember, forget=found.forget), user_turn, reply,
                             record=False)
        try:  # the conversation's episode now includes what was just learned
            await self.store.set_keyed(f"chat-session:{self.session_id}", "episode", self._session_summary(),
                                       about=sorted({t for turn in self.turns for t in self._mentioned(turn["text"])}),
                                       importance=0.5, source="conversation", speaker=speaker)
        except Exception:
            _log.exception("Could not update the conversation summary")

    def _start_calculation(self, user_turn: dict) -> None:
        """Someone asked for arithmetic and the reply started nothing: run it in Python, with the numbers
        they gave earlier in the chat, rather than trust a sum done in the model's head."""
        if not self.settings.digital_tools_enabled or self.manager.restrictions.forbids_tool("run_python"):
            return
        earlier = [t["text"] for t in self.turns if t["role"] == "user" and t["id"] != user_turn["id"]
                   and not t.get("private")][-6:]
        goal = (f"Calculate with Python and answer plainly: {user_turn['text']} Numbers and facts from earlier in "
                f"this chat: " + " | ".join(_clip(text, 220) for text in earlier))
        _log.info("Arithmetic asked for and none proposed: starting a calculation")
        self._start_or_queue(goal, user_turn["id"], Grant.read_only(user_turn["text"]))

    def _work_marker(self) -> tuple:
        """What is running, queued or on offer: it differs after a message only if the message set work in motion."""
        return (len(self.manager.tasks), len(self.pending_goals),
                self.offer.id if self.offer and self._offer_valid() else None)

    def _offer_valid(self) -> bool:
        offer = self.offer
        return bool(offer and offer.status == "pending" and offer.generation == self.generation
                    and time.monotonic() - offer.created_at <= self.settings.chat_offer_seconds)

    def _expire_offer(self, status: str) -> None:
        if self.offer and self.offer.status == "pending":
            self.offer.status = status
            self.offer_log.append(self.offer.public())
            if status == "declined":
                self._declined.add(self.offer.identity)

    def _decide_errands(self, action: ChatAction | None, user_turn: dict) -> None:
        if self._closed:
            return
        if self._turn_epochs.get(user_turn["id"], self._work_epoch) != self._work_epoch:
            return  # A cancellation superseded this still-generating proposal.
        said = user_turn["text"]
        intent = self._intents.get(user_turn["id"]) or read_intent(said)
        answer = classify_reply(said)
        proposed = " ".join(action.task.split()) if action else ""
        offered = " ".join(action.offer.split()) if action else ""
        seen = self._offer_seen.get(user_turn["id"])
        asks = intent.request or (bool(intent.needs_tools) and not intent.question)
        question = intent.question
        if self.offer and not self._offer_valid():
            self._expire_offer("expired")
        offer = self.offer if self._offer_valid() else None
        replying = bool(offer and seen == offer.id)
        if intent.cancel:
            # Already applied at submission. A delayed reply cannot cancel newer work.
            return
        if replying:
            # This message is the person's answer to Kip's offer.
            if answer == "yes":
                offer.status, offer.authorized_by = "consumed", user_turn["id"]
                self.offer_log.append(offer.public())
                # Only the offered goal runs, whatever the model wrote this turn, and only
                # with the effects that were offered.
                self._start_or_queue(offer.goal, user_turn["id"], Grant(
                    effects=sorted(offer.effects) if offer.effects is not None else Grant().effects,
                    source="offer", authorized_text=offer.goal))
                return
            if answer == "no" or intent.declines:
                self._expire_offer("declined")
                if not asks:
                    return
                # "No, please find the keys instead": the offer is declined, and the new
                # explicit request below is still honoured (negated actions never are).
            elif intent.changes_state_only or (question and not proposed and not intent.needs_tools
                                                and not _HALF_YES.match(said.casefold())):
                return  # not an answer ("What would that involve?"): the offer simply stays open
            elif not (asks or question) or (question and _HALF_YES.match(said.casefold())):
                # Ambiguous: keep the offer open and ask, rather than guess.
                offer.status = "deferred"
                self.offer_log.append(offer.public())
                self.offer = Offer(goal=offer.goal, reply_turn_id=offer.reply_turn_id, generation=self.generation,
                                   effects=offer.effects)
                self._add_turn("kip", f"Want to be sure. I {_lower_first(offer.goal)}, question?",
                               source="clarification")
                return
            else:
                self._expire_offer("superseded")  # they asked for something else instead
        if intent.changes_state_only:
            # Permissions changed or something was turned down: nothing new to do, and no
            # earlier request is replayed, whatever the model wrote.
            return
        if self._pure_lookup(intent, said, live=self._is_live(user_turn["id"], intent)):
            # Weather, prices, scores, news: always looked up, never recited from memory. A pure
            # lookup's plan is known, so it starts straight away without a planning call.
            goal = next((g for g in (proposed, offered) if g and digital_scope(g)), f"{LOOKUP_PREFIX}{said}")
            route = self._routes.get(user_turn["id"])
            query = route.query if self.settings.intent_router == "on" and route and route.live else said
            self._start_or_queue(goal, user_turn["id"], Grant.read_only(said), plan=_lookup_plan(query, route, said, self._holder()))
            return
        needs = intent.needs_tools
        if needs:
            usable = [t for t in needs if self.settings.digital_tools_enabled
                      and not self.manager.restrictions.forbids_tool(t)]
            if not usable:
                _log.info("Not starting: %s ruled out or unavailable", needs)
                return  # the reply turns it down; nothing may fake the run
            # They asked for a tool to run: that is work, even if the model only described it.
            proposed = proposed or said
        if task_identity(proposed) in self._declined:
            proposed = ""  # never start, or offer again, an errand the person declined
        if proposed and not needs and self.manager.restrictions.mentions_only_forbidden_tools(proposed):
            # Every tool it names was ruled out ("...using Python" after "don't use Python"):
            # nothing to run; the reply answers within the limits (it is told about them).
            _log.info("Not starting %r: it relies only on tools the person ruled out", proposed)
            return
        if intent.wants_offer:
            self._make_offer(offered or proposed, user_turn, asked_for=True)
            return
        # "Read https://…": the plan is known, so no planning call (minutes on a small local model).
        plan = _read_plan(said) if needs == ["read_webpage"] and proposed == said else None
        if proposed and asks:
            self._start_or_queue(proposed, user_turn["id"], Grant.full("request", said), plan=plan)
            return
        if proposed and self.settings.chat_autonomy == "act" and answer != "no":
            # Opted-in configuration: proposals start at once (restrictions still bind them).
            self._start_or_queue(proposed, user_turn["id"], Grant.full("request", said))
            return
        if proposed and question and (needs or digital_scope(proposed)):
            # A question may start a lookup, but only ever with read-only effects.
            self._start_or_queue(proposed, user_turn["id"], Grant.read_only(said), plan=plan)
            return
        refusing = answer == "no" if replying else intent.declines
        suggestion = offered or proposed
        if suggestion and not refusing and digital_scope(suggestion) and task_identity(suggestion) not in self._declined:
            # A lookup or a calculation changes nothing: Kip may simply do it, unasked.
            # The read-only grant guarantees it, and a plan that turns out to need more
            # stops before acting and becomes an offer.
            self._start_or_queue(suggestion, user_turn["id"], Grant.read_only(said))
            return
        # Anything else is a suggestion: never started without a yes.
        if not refusing:
            self._make_offer(suggestion, user_turn, asked_for=False, small_talk=intent.small_talk)

    def _pure_lookup(self, intent: Intent, said: str, live: bool | None = None) -> bool:
        """Weather, prices, scores, news: a web lookup."""
        live = intent.live_fact if live is None else live
        return (live and not intent.needs_tools and self.settings.digital_tools_enabled
                and not self.manager.restrictions.forbids_tool("web_search") and not intent.wants_offer)

    def _direct_lookup(self, user_turn: dict, intent: Intent,
                       earlier: list[dict] = ()) -> tuple[str, str] | None:
        """(goal, search query) for a live-fact question or request that can go straight to
        the lookup, unreplied: including a follow-up on the lookup Kip just answered
        ("is there a fifth match in that series?"), which carries that topic along.

        Anything else keeps the normal reply: small talk that merely mentions cricket, an
        answer to Kip's open offer, what Kip did earlier, a cancellation or permission change."""
        if self._closed or intent.cancel or intent.changes_state_only or intent.about_past:
            return None
        if not (intent.question or intent.request):
            return None
        if self._turn_epochs.get(user_turn["id"], self._work_epoch) != self._work_epoch:
            return None
        seen = self._offer_seen.get(user_turn["id"])
        if seen is not None and self._offer_valid() and self.offer.id == seen:
            return None
        said = user_turn["text"]
        topic = None if intent.live_fact else self._lookup_followed_up(said, earlier)
        if not self._pure_lookup(intent, said, live=intent.live_fact or topic is not None):
            return None
        if topic:
            return _clip(f"{LOOKUP_PREFIX}{said} (following up on: {topic})", 600), f"{topic} {said}"
        return f"{LOOKUP_PREFIX}{said}", said

    # ── intent router ───────────────────────────────────────────────────────
    async def _route(self, user_turn: dict, intent: Intent, earlier: list[dict]) -> Route | None:
        """How the router reads this message, in order (replies run one at a time), with the last
        information request as context. Shadow mode only records it; nothing here acts."""
        mode = self.settings.intent_router
        if mode == "off":
            return None
        if self.router is None:
            self.router = default_router()
        said, turn = user_turn["text"], self._routed
        started = time.perf_counter()
        try:
            if self.router.embedding is not None:  # an embedding is a few ms of CPU: off the event loop
                route = await asyncio.to_thread(self.router.route, said, self._frame, turn)
            else:
                route = self.router.route(said, self._frame, turn)
        except Exception:
            _log.exception("Intent router failed; the usual path decides")
            return None
        seconds = time.perf_counter() - started
        self._routed += 1
        if route.correction and self._frame_route is not None:
            self._record_correction(user_turn, route)
        route.same_subject = bool(route.followup and self._frame is not None
                                  and set(route.entities) <= set(self._frame.entities))
        route.prior_focus = self._frame.focus if self._frame is not None else None
        frame = frame_of(route, said, turn)
        if frame:
            if not route.followup:
                self._frame_answer = ""  # a new request: the old answer is about something else
            self._frame, self._frame_route = frame, route
        self._routes[user_turn["id"]] = route
        router_fast = self._router_fast(user_turn, intent, route)
        usual_fast = self._direct_lookup(user_turn, intent, earlier) is not None
        _log.info("intent_router mode=%s intent=%s conf=%.2f live=%.2f source=%s fast=%s usual_fast=%s "
                  "clarify=%s query=%r ms=%.1f", mode, route.intent, route.confidence, route.live_confidence,
                  route.source, router_fast, usual_fast, bool(route.clarify), route.query, seconds * 1000)
        if self.store:
            try:
                self.store.log_intent_shadow(said, {**route.public(), "backend": self.router.backend,
                                                    "ms": round(seconds * 1000, 2)}, router_fast, usual_fast)
            except Exception:
                _log.exception("Could not record the router's shadow decision")
        return route

    def _router_fast(self, user_turn: dict, intent: Intent, route: Route) -> bool:
        """The router's fast path, behind every check the usual path makes first."""
        if self._closed or self._turn_epochs.get(user_turn["id"], self._work_epoch) != self._work_epoch:
            return False
        seen = self._offer_seen.get(user_turn["id"])
        if seen is not None and self._offer_valid() and self.offer.id == seen:
            return False  # an answer to Kip's offer is never a fresh lookup
        allowed = self.settings.digital_tools_enabled and not self.manager.restrictions.forbids_tool("web_search")
        # "What's the best performer in my portfolio?": the router may be unsure of the wording, but
        # their holdings are never answered from the chat model's memory of earlier lookups.
        own = route.intent == "information.finance" and is_portfolio(_feed_question(route, user_turn["text"],
                                                                                    self._holder()))
        return fast_path(route, intent, web_allowed=allowed, own_data=own)

    def _routed_lookup(self, user_turn: dict, intent: Intent, route: Route) -> tuple[str, str] | None:
        """(goal, search query) when the router sends this straight to the lookup. The goal keeps
        their words and adds what they meant; the search uses the meaning, not the wording."""
        if not self._router_fast(user_turn, intent, route):
            return None
        said = user_turn["text"]
        if route.correction:
            goal = f"{LOOKUP_PREFIX}{route.earlier} (they corrected: {said}; they mean: {route.meaning})"
        elif route.followup:
            answered = f"; Kip answered: {_clip(self._frame_answer, 200)}" if self._frame_answer else ""
            goal = f"{LOOKUP_PREFIX}{said} (following up on: {route.earlier}{answered}; they mean: {route.meaning})"
        else:
            goal = f"{LOOKUP_PREFIX}{said} (they mean: {route.meaning})"
        goal = _clip(goal, 600)
        correction = getattr(route, "correction_id", None)
        if correction:
            self._correction_goals[task_identity(goal)] = correction
        return goal, route.query

    def _router_clarifies(self, user_turn: dict, intent: Intent, route: Route) -> str | None:
        """A short question back when a live question points at nobody ("how did he do?")."""
        if not route.clarify or route.intent not in LIVE or route.live_confidence < ROUTER_THRESHOLD:
            return None
        if intent.cancel or intent.changes_state_only or intent.needs_tools or self._closed:
            return None
        seen = self._offer_seen.get(user_turn["id"])
        if seen is not None and self._offer_valid() and self.offer.id == seen:
            return None
        return route.clarify

    def _lookup_pending(self, query: str) -> bool:
        """A lookup searching for the same thing is already running or queued."""
        active = self.manager.active()
        if active and self._lookup_queries.get(task_identity(active["goal"])) == query:
            return True
        return any(self._lookup_queries.get(task_identity(goal)) == query for goal in self.pending_goals)

    def _lookup_evidence(self, task_id: str | None) -> str:
        """What a finished lookup's source actually said: Groq's quote and the page it came from,
        or the text of what the answer cited. Kept for follow-ups and disputes."""
        context = self.manager.tasks.get(task_id) if task_id else None
        if context is None:
            return ""
        searches = [a for a in context.action_history if a["tool"] == "web_search" and a["success"]]
        for entry in reversed(searches):
            obs = entry.get("observation") or {}
            if obs.get("quote"):
                titles = "; ".join(r.get("title", "") for r in obs.get("results", [])[:2])
                return _clip(f'"{obs["quote"]}" ({titles})', 700)
        cited = set((context.answer or {}).get("evidence_ids", []))
        parts = []
        for entry in context.action_history:
            if entry.get("evidence_id") in cited and entry["success"]:
                obs = entry.get("observation") or {}
                parts.append(str(obs.get("text") or "")[:400])
                parts += [f"{r.get('title', '')}: {r.get('snippet', '')}" for r in obs.get("results", [])[:4]]
        return _clip(" | ".join(p for p in parts if p.strip()), 900)

    def _evidence_fresh(self) -> bool:
        return bool(self._frame_evidence) and time.monotonic() - self._frame_evidence_at < EVIDENCE_SECONDS

    def _from_last_lookup(self, user_turn: dict, route: Route) -> bool:
        """"India lost?" right after the lookup that found the result: answered from that
        evidence, not searched again. Asking for something newer ("now", "latest") searches."""
        said = user_turn["text"].casefold()
        prior = getattr(route, "prior_focus", None)
        # What the last lookup found: the same aspect, or how the same match went ("india lost?"
        # after the score). "Which channel?" after the start time is something new: it searches.
        same_aspect = route.focus == prior or (route.focus in _HOW_IT_WENT and prior in _HOW_IT_WENT)
        return (self._evidence_fresh() and getattr(route, "same_subject", False) and not route.correction
                and same_aspect and len(said.split()) <= 7 and not _NEWER.search(said))

    def _disputes(self, route: Route | None, turn_id: str) -> bool:
        turn = next((t for t in reversed(self.turns) if t["id"] == turn_id), None)
        if turn is None or not self._evidence_fresh():
            return False
        said = turn["text"].casefold()
        statement = not (route and route.asks) and not said.rstrip().endswith("?")
        return bool(_DISPUTE.search(said) or (statement and re.search(r"\d", said)))

    def _evidence_guidance(self, user_turn: dict) -> str:
        """Kip's last lookup, side by side with what its source said, when the person asks
        about it again or disputes it: what the page says outranks what he said it says."""
        route = self._routes.get(user_turn["id"])
        evidence = f"Your last lookup's source said: {self._frame_evidence}. You told them: {self._frame_answer}"
        if self._disputes(route, user_turn["id"]):
            return ("They may be correcting you. " + evidence + ". Compare what they say with the source, not with "
                    "your own words: if the source does not back what you said, say plainly you got it wrong and "
                    "give what it does say; if it backs you, say what it says, kindly; if it cannot settle it, say "
                    "so and offer to look again. Never just agree, never just defend yourself.")
        if route is not None and getattr(route, "from_evidence", False):
            return ("They ask about what you just looked up: answer from that, no new lookup. " + evidence
                    + ". If it does not say, say so plainly and offer to check.")
        return ""

    def _learn_from_answer(self, text: str) -> None:
        """A lookup's answer joins the conversation topic: "New Zealand vs India, 12:30" names the
        opponent, so "who is playing?" next needs no search."""
        if self._frame is None:
            return
        self._frame_answer = text
        if self._frame.domain in SPORTS:
            slots = read_slots(text)
            found = sorted(((c, d) for c, d in slots.entities + slots.ambiguous if d in SPORTS),
                           key=lambda item: slots.order.get(item[0], 0))
            for name, _ in found:
                if name not in self._frame.entities:
                    self._frame.entities.append(name)

    def _asks_the_time(self, intent: Intent, route: Route, said: str) -> bool:
        """ "What time is it?" on its own, not "what time?" about the match just asked about."""
        return (not (route.followup or route.correction)
                and not (intent.cancel or intent.changes_state_only or self._closed) and clock_question(said))

    def _is_live(self, turn_id: str, intent: Intent) -> bool:
        """Live information? When it is on, the router decides for messages that ask something;
        a statement ("I wonder what the air is like") keeps the live-fact words, so the router
        never turns remarks into lookups nobody asked for."""
        route = self._routes.get(turn_id)
        asks = route is not None and (intent.question or intent.request or route.followup or route.correction
                                      or route.lookup_phrase or route.asks)
        if route is not None and (getattr(route, "from_evidence", False) or self._disputes(route, turn_id)):
            return False  # answered from the last lookup's evidence, never a fresh search
        if self.settings.intent_router == "on" and asks:
            return route.live
        return intent.live_fact

    def _record_correction(self, user_turn: dict, route: Route) -> None:
        """"I meant what position": the earlier reading was wrong. Kept, unreviewed, for offline
        review; never fed back into the router on its own."""
        if not self.store or self._frame is None:
            return
        previous = self._frame_route
        try:
            if previous is not None and getattr(previous, "correction_id", None):
                self.store.set_intent_outcome(previous.correction_id, "corrected again")
            route.correction_id = self.store.add_intent_correction(
                self._frame.said, {"frame": vars(self._frame)}, previous.public() if previous else {},
                route.public(), user_turn["text"])
        except Exception:
            _log.exception("Could not record an intent correction")

    def _lookup_followed_up(self, said: str, earlier: list[dict]) -> str | None:
        """The question Kip's last web lookup answered, if its answer came within the last three
        exchanges and this message shares its topic; otherwise None."""
        last = next((t for t in reversed(earlier[-6:]) if t.get("role") == "kip" and t.get("source") == "narration"),
                    None)
        if not last or last.get("outcome") != "completed":
            return None
        finished = [t for t in self.manager.tasks.values() if t.status != "running"]
        if not finished or finished[-1].status != "completed" or not finished[-1].goal.startswith(LOOKUP_PREFIX):
            return None
        asked = finished[-1].goal.removeprefix(LOOKUP_PREFIX)
        if not _topic_words(said) & _topic_words(asked + " " + last.get("text", "")):
            return None
        return _clip(asked, 200)

    def _make_offer(self, goal: str, user_turn: dict, asked_for: bool, small_talk: bool = False) -> None:
        identity = task_identity(goal)
        if not goal or identity in self._declined:
            return
        if not asked_for:
            # Unasked suggestions: once per errand, and in small talk never on top of one
            # that is still waiting unanswered from a few messages ago.
            recent, previous = self._last_unasked_offer
            ignored = previous is not None and previous.status in {"pending", "expired", "superseded", "deferred"}
            if identity in self._offered or (small_talk and ignored and self._user_turns - recent < 4):
                _log.info("Not repeating an unasked offer: %s", goal)
                return
        self._offered[identity] = self._user_turns
        self._expire_offer("superseded")
        self.offer = Offer(goal=goal, reply_turn_id=user_turn["id"], generation=self.generation)
        if not asked_for:
            self._last_unasked_offer = (self._user_turns, self.offer)
        _log.info("Holding errand as an offer awaiting a yes: %s", goal)

    def _cancel_work(self) -> None:
        """"Stop that": the current errand is cancelled and nothing queued or offered runs."""
        self._work_epoch += 1
        self._expire_offer("declined" if self.offer and self._offer_valid() else "expired")
        self.pending_goals.clear()
        self._queued_grants.clear()
        self._queued_plans.clear()
        active = self.manager.active()  # only this conversation's errand: never someone else's
        if active:
            goal = active["goal"]

            async def stop(task_id: str) -> None:
                await self.manager.cancel(task_id)
                self._add_turn("kip", f"Stopped: {goal}", source="narration")
            self._spawn(stop(active["id"]))

    def _check_claims(self, reply: str, intent: Intent | None = None, live: bool = False,
                      started: bool = True, user_turn: dict | None = None) -> None:
        """Kip must not claim runs or lookups his tool evidence does not support, nor promise one that
        was never started, nor do a sum in his head that should have been run."""
        notes = []
        if intent is not None and (intent.live_fact or live) and _FIGURE.search(reply.casefold()):
            # A live figure before any lookup came back is a guess wearing a number.
            notes.append("that figure was a guess; I'm looking up the real one now")
        running = (self.manager.active() or {}).get("requested_tools", [])
        for tool in unsupported_execution(reply, self.manager):
            what = {"run_python": "no Python has run for that", "read_webpage": "I haven't read that page",
                    "web_search": "I haven't searched for that"}[tool]
            notes.append(what + ("; it's running now, and the real result will follow" if tool in running else ""))
        denied = denied_capabilities(reply, self.manager, self.settings)
        if denied:
            notes.append("I can " + " and ".join(denied) + " — that's working")
        if intent is not None and intent.calculates and _SUM_IN_HEAD.search(reply.casefold()):
            notes.append("that sum was my own guess; the real one is running now" if started else
                         "I worked that sum out in my head and nothing ran, so check it; ask me to run it in Python")
        if notes:
            _log.warning("Unsupported claims in reply: %s", notes)
            self._add_turn("kip", "Scratch that: " + "; ".join(notes) + ".", source="correction", claims=notes)
        elif not started and user_turn is not None:
            self._hold_promise(reply, user_turn)

    def _hold_promise(self, reply: str, user_turn: dict) -> None:
        """"I check flight prices" with no errand behind it: say so, and hold the lookup for a yes
        instead of leaving the promise hanging or starting work nobody asked for."""
        promise = unkept_promise(reply)
        if not promise or self._closed or not self.settings.digital_tools_enabled:
            return
        clause, kind = promise
        if kind == "lookup" and not self.manager.restrictions.forbids_tool("web_search"):
            goal = f"{LOOKUP_PREFIX}{clause}. They said: {user_turn['text']}"
            if task_identity(goal) not in self._declined:
                self._make_offer(goal, user_turn, asked_for=False)
                if self.offer and self.offer.goal == goal:
                    self.offer.effects = sorted(READ_ONLY)  # a lookup only reads
                    self._add_turn("kip", f"Scratch that: I haven't started that. Say yes and I {clause}.",
                                   source="correction", claims=[f"promised: {clause}"])
                    return
        self._add_turn("kip", f"Scratch that: I haven't started that, so nothing is done. Ask me again and I {clause}.",
                       source="correction", claims=[f"promised: {clause}"])

    async def _remember(self, action: ChatAction | None, user_turn: dict, reply: str, record: bool = True) -> None:
        said, speaker = user_turn["text"], user_turn.get("speaker")
        about = self._mentioned(said + " " + reply)
        added = []
        for item in (action.remember if action else []):
            text = item.text.strip()
            if not text:
                continue
            context = item.context
            if context == "actual" and re.search(r"\b(pretend|imagine|hypothetical(ly)?|fictional|for this test|"
                                                 r"make[- ]believe|let's say)\b", said, re.I):
                context = "pretend"  # the person framed it as not real; keep that framing
            tags = self._mentioned(text)
            if speaker and speaker not in tags and FIRST_PERSON.search(text.casefold()):
                # "I like tea" belongs to whoever is speaking, never anyone else.
                tags.append(speaker)
            meta = {"context": context, "speaker": speaker or "unknown", "said_in": user_turn["id"]}
            try:
                added.append(await self.store.add("preference" if re.search(r"\b(prefer|like|love|favou?rite|"
                                                                            r"enjoy|hate|dislike)", text, re.I)
                                                  else "fact", text, about=tags or about,
                                                  # Who they are outranks what they had for lunch.
                                                  importance=0.9 if _CORE_FACT.search(text) else 0.7,
                                                  source="conversation", meta=meta))
                self._remembered.append(text if context == "actual" else f"{text} ({context})")
            except Exception:
                _log.exception("Could not save a conversational fact")
        # Corrections retire the wrong memory (kept as history), never the fix itself.
        for wrong in (action.forget if action else []):
            if wrong.strip():
                self._drop_checkpoint()  # the old note may repeat what was just corrected; it is rewritten
                try:
                    who = {"viewer": self._speaker, "legacy": self.owner} if self.person else {}
                    retired = await self.store.supersede(wrong, by=added[0] if added else None,
                                                         exclude=set(added), about=self._mentioned(wrong), **who)
                    if retired:
                        _log.info("Memory %s superseded by a correction", retired["id"])
                except Exception:
                    _log.exception("Could not retire a corrected memory")
        if not record:
            return
        try:
            await self.store.add("conversation", f"They said: {said} | I replied: {reply}", about=about,
                                 importance=0.3, source="conversation", meta={"speaker": speaker or "unknown"})
            await self.store.set_keyed(f"chat-session:{self.session_id}", "episode", self._session_summary(),
                                       about=sorted({t for turn in self.turns for t in self._mentioned(turn["text"])}),
                                       importance=0.5, source="conversation", speaker=speaker)
        except Exception:
            _log.exception("Could not save conversation")

    # ── checkpoint (older turns folded into a note) ─────────────────────────
    def _drop_checkpoint(self) -> None:
        """Forget the note. The turns it covered are folded again, correction included, on the next
        attempt; until then the usual history selection applies."""
        if self.checkpoint is None:
            return
        self.checkpoint = None
        self._checkpoint_epoch += 1
        if self.store:
            self.store.set_state({checkpoints.state_key(self.session_id): ""})

    async def _compact(self) -> None:
        """After a reply: fold older turns into the note once they outgrow the history budget.
        Background and best effort: a reply never waits for it, and a failure changes nothing."""
        if (self._user_turns < self._compact_after or self.replying or self._closed
                or (self.shares_task_model and self.manager.active_id is not None)):
            return  # a busy local model is left to the reply or errand that needs it; tried again later
        epoch, previous = self._checkpoint_epoch, self.checkpoint
        status, found = await checkpoints.compact(self.client, list(self.turns), previous,
                                                  self._history_budget,
                                                  self.settings.chat_checkpoint_tokens)
        if epoch != self._checkpoint_epoch or self._closed:
            return  # the note was dropped (a correction) or the chat reset while this was being written
        self.compaction = status
        if status == "failed":
            self._compact_after = self._user_turns + 3
        if found is not None:
            self.checkpoint = found
            if self.store:
                self.store.set_state({checkpoints.state_key(self.session_id): found.dump()})
            _log.info("chat_checkpoint covered_turns=%d tokens=%d", found.turns, estimate_tokens(found.text))

    def _session_summary(self) -> str:
        """A deterministic episode of this conversation, so it can be recalled after a restart."""
        turns = list(self.turns)
        start = turns[0]["at"][:16].replace("T", " ") if turns else ""
        topics = [_clip(t["text"], 60) for t in turns if t["role"] == "user"][-6:]
        # A private answer (their own accounts) is never summarised: this episode is recalled into prompts.
        errands = [f"{t['text'][:80]} ({t['outcome']})" for t in turns if t.get("outcome") and not t.get("private")][-4:]
        speakers = [t.get("speaker") for t in turns if t["role"] == "user" and t.get("speaker")]
        parts = [f"Conversation that started {start} UTC with {self._speaker_name(speakers[-1] if speakers else None)}."]
        if topics:
            parts.append("We talked about: " + "; ".join(topics) + ".")
        if errands:
            parts.append("Errands: " + "; ".join(errands) + ".")
        if self._remembered:
            parts.append("I learned: " + "; ".join(self._remembered[-5:]) + ".")
        if self.persona:
            parts.append(f"I felt {self.persona.mood_label()}.")
        return " ".join(parts)

    def _mentioned(self, text: str) -> list[str]:
        """Tag memories with the people Kip knows who are mentioned by name."""
        lowered = text.casefold()
        known = set(self.persona.state.relationships) if self.persona else set()
        if self._speaker:
            known.add(self._speaker)
        return sorted(key for key in known
                      if re.search(r"(?<!\w)" + re.escape(key.replace("_", " ")) + r"(?!\w)", lowered))

    # ── errands ─────────────────────────────────────────────────────────────
    def _already_planned(self, goal: str) -> bool:
        active = self.manager.active()
        planned = ([active["goal"]] if active else []) + list(self.pending_goals)
        identity = task_identity(goal)
        return any(task_identity(other) == identity for other in planned)

    def _start_or_queue(self, goal: str, origin_turn: str | None = None, grant: Grant | None = None,
                        plan: GoalPlan | None = None) -> None:
        if self._closed:
            return
        # Without a grant this is an explicit request (e.g. queued directly).
        grant = grant or Grant.full("request", goal)
        if origin_turn is not None:
            if origin_turn in self._started_by_turn:
                _log.info("One message starts at most one errand; ignoring %s", goal)
                return
            self._started_by_turn.add(origin_turn)
        if self._already_planned(goal):
            _log.info("Ignoring a repeated errand request: %s", goal)
            return
        try:
            # Kip's own reply already acknowledged this errand; skip "Start now".
            self._announced.add(task_identity(goal))
            self.manager.start(goal, grant, plan=plan)
        except TaskBusy:
            # It will start later, when a "Start now" is genuinely news again. It keeps the
            # grant it was given; restrictions are re-checked when its tools run.
            self._announced.discard(task_identity(goal))
            self.pending_goals.append(goal)
            self._queued_grants[task_identity(goal)] = grant
            if plan is not None:
                self._queued_plans[task_identity(goal)] = plan
            _log.info("Queued behind task %s: %s (queue now %d)", self.manager.active_id, goal, len(self.pending_goals))
            active = self.manager.tasks.get(self.manager.active_id)
            if active is None or active.status in TERMINAL:
                # Its end was already announced and it is only saving what it learned: no task
                # event is left to start this one, so wait for the slot here.
                self._spawn(self._start_next_when_free(self.generation))
            if goal.startswith(LOOKUP_PREFIX):
                self._add_turn("kip", "One errand first. Then I check this.", source="narration")
            else:
                self._add_turn("kip", f"First I finish current errand. Right after, I do: {goal}", source="narration")

    async def _start_next_when_free(self, generation: int) -> None:
        while self.manager.active_id is not None and not self._closed:
            await asyncio.sleep(0.05)
        if self.pending_goals and not self._closed and generation == self.generation:
            goal = self.pending_goals.popleft()
            self._start_or_queue(goal, grant=self._queued_grants.pop(task_identity(goal), None),
                                 plan=self._queued_plans.pop(task_identity(goal), None))

    def _narrate(self, event: dict) -> None:
        """Speak about errands in Kip's voice using templates, never a model call."""
        kind, data = event["type"], event.get("data", {})
        if self._closed:
            return
        task = data.get("task") or {}
        if self.person and not self._mine(data):
            # Someone else's errand. Its end may free the slot this conversation's queue waits for.
            if kind in {"task_cancelled", "task_completed", "task_failed"} and self.pending_goals:
                self._spawn(self._start_next_when_free(self.generation))
            return
        if kind == "task_cancelled":
            if self.pending_goals:
                self._spawn(self._start_next_when_free(self.generation))
            return
        if kind == "task_started":
            goal = task.get("goal", data.get("summary", ""))
            if task_identity(goal) in self._announced:
                self._announced.discard(task_identity(goal))
            else:
                self._add_turn("kip", f"Start now: {goal}", source="narration")
        elif kind == "feed_checking":
            self._add_turn("kip", f"Let me check {data.get('source', 'that')} for that.", source="narration")
        elif kind in {"task_completed", "task_failed"}:
            outcome = kind.removeprefix("task_")
            self._lookup_queries.pop(task_identity(task.get("goal", "")), None)
            correction = self._correction_goals.pop(task_identity(task.get("goal", "")), None)
            if correction and self.store:
                self.store.set_intent_outcome(correction, "answered" if outcome == "completed" else "failed")
            answer = task.get("answer") or {}
            private = bool(task.get("private"))  # their own accounts: shown, never fed to a later prompt
            if private:
                # Events carry a private task without its answer's words; the reply comes from the
                # task itself, in this process.
                context = getattr(self.manager, "tasks", {}).get(task.get("id"))
                answer = (context.answer if context is not None else None) or {}
            if answer.get("answer") and str(task.get("goal", "")).startswith(LOOKUP_PREFIX) and not private:
                self._learn_from_answer(answer["answer"])
                self._frame_evidence, self._frame_evidence_at = self._lookup_evidence(task.get("id")), time.monotonic()
            if answer.get("answer"):
                # The answer is already written in Kip's voice; no stock opener.
                text = answer["answer"]
            elif outcome == "completed":
                # Plain facts, no stock reaction: character belongs in replies, not templates.
                text = data.get("summary", "") or "Done."
            else:
                text = " ".join(part for part in ("Not finish. Bad.", data.get("summary", "")) if part)
            notes = list(task.get("answer_notes") or [])
            if task.get("blocked"):
                notes.append("Not done: " + "; ".join(task["blocked"]) + ".")
            text = " ".join([text, *notes])
            self._add_turn("kip", text, source="narration", sources=answer.get("sources", []), outcome=outcome,
                           private=private)
            if self.pending_goals:
                self._spawn(self._start_next_when_free(self.generation))
        elif kind == "task_needs_permission":
            self._needs_permission(data)


    def _mine(self, data: dict) -> bool:
        """The event is about an errand this conversation asked for."""
        task_id = (data.get("task") or {}).get("id") or data.get("task_id")
        return self.manager.owns(task_id)

    def _needs_permission(self, data: dict) -> None:
        """A task stopped before acting because it needed more than it was cleared for."""
        goal, needs = data.get("goal", ""), set(data.get("needs", []))
        if data.get("ruled_out"):
            self._add_turn("kip", f"I leave it. It mean {plain(needs)}, and you say no to that.",
                           source="narration")
            return
        if task_identity(goal) in self._declined:
            return  # they already said no to this once; don't ask again
        self._expire_offer("superseded")
        self.offer = Offer(goal=goal, reply_turn_id="", generation=self.generation, effects=frozenset(needs) | READ_ONLY)
        self._add_turn("kip", f"Cannot answer from here. It mean {plain(needs)}. I do it, question?",
                       source="narration")


def _feed_question(route: Route, words: str, holder: str | None = None) -> dict:
    """What a feed reads: the router's reading and their words. A follow-up ("what about Max?")
    or correction also carries the question it leans on, whose words say what is being asked.
    holder: someone else is asking about the owner's portfolio ("does Neave own HDFC?"); the
    finance reader reads it in the owner's words, and the answer names them."""
    leans = bool(route.followup or route.correction)
    earlier = route.earlier if leans else ""
    question = {"domain": route.domain, "entities": list(route.entities), "focus": route.focus, "time": route.time,
                "place": route.place, "said": words, "followup": leans, "earlier": earlier}
    if holder and route.domain == "finance":
        from app.capabilities.feeds import as_asked_by_holder
        question.update(said=as_asked_by_holder(words, holder), earlier=as_asked_by_holder(earlier, holder),
                        holder=holder)
    return question


def _lookup_plan(said: str, route: Route | None = None, words: str = "", holder: str | None = None) -> LookupPlan:
    """A plain web lookup of what they asked, searched in their own words. A subject with a feed
    (F1, cricket, weather) carries the router's reading and their words, so the feed can answer first."""
    query = re.sub(r"^\W*(?:hey |ok |so )?kip\W+", "", said.strip(), flags=re.I)[:200]
    feed = (_feed_question(route, words, holder) if route and route.domain in FEEDS else None)
    return LookupPlan(summary="Look it up.", conditions=[GoalCondition(kind="answer", required_tools=["web_search"])],
                      initial_actions=[InitialCapability(tool="web_search", arguments={"query": query})], feed=feed)


_PAGE_URL = re.compile(r"https?://[^\s,;)<>\"']+")


def _read_plan(said: str) -> GoalPlan | None:
    """Read the one page they named and answer from it. None unless exactly one http(s) URL."""
    urls = {url.rstrip(".!?:") for url in _PAGE_URL.findall(said)}
    if len(urls) != 1:
        return None
    return GoalPlan(summary="Read the page they named.",
                    conditions=[GoalCondition(kind="answer", required_tools=["read_webpage"])],
                    initial_actions=[InitialCapability(tool="read_webpage", arguments={"url": urls.pop()})])


# ── status and narration helpers (deterministic, no model calls) ───────────────
def _topic_words(text: str) -> set[str]:
    words = (w.split("'")[0] for w in re.findall(r"[a-z][a-z0-9]*(?:'[a-z]+)?", text.casefold().replace("’", "'")))
    return {w for w in words if len(w) >= 4 and w not in _TOPIC_STOP}


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:] if text else text


def _speaker_slug(name: str) -> str:
    return "_".join(name.casefold().split())


def errand_progress(task: dict) -> dict:
    """What the running errand has actually done so far, so chat never invents progress."""
    return {"goal": task.get("goal"), "outcomes_done": [c.get("kind") for c in task.get("conditions") or []
                                                        if c.get("satisfied")],
            "latest": task.get("summary", "")}
