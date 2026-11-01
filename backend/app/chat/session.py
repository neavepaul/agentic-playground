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
from app.tasks.manager import TaskBusy
from app.tasks.permissions import READ_ONLY, TOOL_PLAIN, Grant, digital_scope, plain
from .consent import Offer, classify_reply, is_question, is_request, task_identity  # noqa: F401 (re-exported)
from .evidence import (capabilities, contradicted_presence, denied_capabilities, last_seen, room_scans,
                       task_record, unsupported_execution)
from .intent import Intent, introduced_as, is_live_fact, read_intent
from .router import (THRESHOLD as ROUTER_THRESHOLD, LIVE, SPORTS, Frame, Route, clock_question, default_router,
                     fast_path, frame_of, read_slots)
from app.world.clock import spoken_time
from app.agents.schemas import GoalCondition, GoalPlan, InitialCapability
from .prompt import ChatAction, TrailerSplitter, build_messages
from .spoken import spoken
from .context import ContextBudgetError, message_tokens

_log = logging.getLogger("agentic_friend.chat")

HISTORY = 200
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
_HALF_YES = re.compile(r"^\W*(yes|yeah|yep|sure|ok|okay)\b")  # "Yes? What do you mean?" is not a yes
FIRST_PERSON = re.compile(r"\b(i|i'm|i've|me|my|mine|myself|the user)\b")


class ChatSession:
    def __init__(self, client, bus, manager, engine, settings, store=None, persona=None,
                 shares_task_model: bool = True, session_id: str | None = None) -> None:
        self.client, self.bus, self.manager, self.engine = client, bus, manager, engine
        self.settings, self.store, self.persona = settings, store, persona
        self.shares_task_model = shares_task_model
        self.name = persona.name if persona else settings.persona_name
        self.turns: deque[dict] = deque(maxlen=HISTORY)
        self.pending_goals: deque[str] = deque()
        self._lock = asyncio.Lock()
        self._jobs: set[asyncio.Task] = set()
        self.replying = False
        self.last_activity = 0.0
        self._closed = False
        # Bumped by every world reset: offers, queued errands and narration from an
        # older world are never acted on.
        self.generation = 0
        self.session_id = session_id or str(uuid4())
        if self.store:
            self.turns.extend(self.store.conversation_turns(self.session_id, HISTORY))
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
        self._introduced: str | None = None            # who said "it's Neave" when the picker was unset
        self.gate = None                               # shared-model priority (conversation first)
        # Intent router (router.py), per settings.intent_router. The app hands in one trained with
        # reviewed corrections and the embedding model; without one, the shipped TF-IDF router.
        self.router = None
        self._frame: Frame | None = None               # the last information request, for follow-ups
        self._frame_route: Route | None = None
        self._frame_answer = ""                        # what Kip's lookup said about that request
        self._lookup_queries: dict[tuple[str, ...], str] = {}  # running or queued lookup -> what it searches
        self._routed = 0
        self._routes: dict[str, Route] = {}            # user turn id -> how the router read it
        self._correction_goals: dict[tuple[str, ...], int] = {}  # corrected lookup -> correction record

    def attach(self) -> None:
        self.bus.add_listener(self._narrate)

    # ── public API ──────────────────────────────────────────────────────────
    @property
    def busy(self) -> bool:
        return self.replying or any(not job.done() for job in self._jobs)

    @property
    def pending_offer(self) -> str | None:
        return self.offer.goal if self._offer_valid() else None

    def quiet_seconds(self) -> float:
        return time.monotonic() - self.last_activity if self.last_activity else float("inf")

    def submit(self, text: str, speaker: str | None = None) -> dict:
        text = text.strip()
        if not text:
            raise ValueError("Say something first.")
        if self._closed:
            raise ValueError("Conversation is shutting down.")
        self.last_activity = time.monotonic()
        # "Don't use Python", "don't go into the kitchen": binding from this moment on, for
        # running and queued errands too. Applied before anything else reads this message.
        world = self.engine.snapshot()
        rooms = {key: room.get("name", key) for key, room in world.get("rooms", {}).items()}
        before = set(self.manager.restrictions.describe())
        if self.manager.restrictions.update(text, rooms):
            _log.info("Restrictions now: %s", self.manager.restrictions.describe())
            self.manager.save_restrictions()
        after = set(self.manager.restrictions.describe())
        speaker_id = self._speaker_id(speaker)
        if speaker_id is None and not speaker:
            # The picker says "someone": a known person introducing themselves is good evidence.
            self._introduced = introduced_as(text, world.get("people", {})) or self._introduced
            speaker_id = self._introduced
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
        """Discard replies, offers and queued work tied to the old world, then accept chat again."""
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

    def _speaker_id(self, speaker: str | None) -> str | None:
        """A known person id, or None when we don't know who is talking."""
        speaker = (speaker or self.settings.chat_default_speaker or "").strip().casefold()
        people = self.engine.snapshot().get("people", {})
        return speaker if speaker in people else None

    def _speaker_name(self, speaker_id: str | None) -> str:
        people = self.engine.snapshot().get("people", {})
        return people[speaker_id].get("name", speaker_id) if speaker_id in people else "unknown person"

    def _status(self, speaker_id: str | None = None) -> dict:
        status = self.engine.get_status()
        status["talking_with"] = self._speaker_name(speaker_id)
        # Live senses: the room Kip is standing in, as he sees and hears it now (nothing beyond it),
        # and who came, went or changed what they were doing since he got here.
        around = self.engine.surroundings()
        # Compact: every token here is read again on every reply. Full detail comes from look.
        held = {h["object"]: h["person"] for h in around["held_objects"]}
        view = {"room": around["room_name"], "description": around.get("description", ""),
                "people": [p["name"] + (f" ({p['activity']})" if p.get("activity") else "") for p in around["people"]],
                "things": [o["name"] for o in around["objects"]]
                          + [f"{obj} (held by {person})" for obj, person in held.items()],
                "fixtures": [f["name"] + (f": {f['state']}" if f.get("state") else "")
                             for f in around.get("features", [])]}
        status["around_me_now"] = {key: value for key, value in view.items() if value}
        noticed = []
        for event in reversed(self.bus.history):
            if event["type"] == "robot_moved":
                break
            if event["type"] == "presence_noticed":
                noticed.append(event["data"]["summary"])
                if len(noticed) == 3:
                    break
        if noticed:
            status["just_noticed"] = noticed[::-1]
        active = self.manager.active()
        # "none", not null: models repeat null verbatim ("my current errand is null").
        status["current_errand"] = errand_progress(active) if active else "none"
        if self.pending_goals:
            status["queued_errands"] = list(self.pending_goals)
        if self.pending_offer:
            status["offered_errand_awaiting_yes"] = self.pending_offer
        world = self.engine.snapshot()
        finished = [c for c in self.manager.tasks.values() if c.status != "running"][-3:]
        if finished:
            # What each errand actually did, tried, was refused and left undone.
            status["recent_errands"] = [task_record(c, world) for c in finished]
        people = last_seen(self.manager, world)
        if any(note != "unknown" for note in people.values()):
            status["last_seen"] = people  # fresh looks vs older records, never merged
        status["my_tools"] = capabilities(self.manager, self.settings)
        if self.manager.restrictions:
            status["asked_not_to"] = self.manager.restrictions.describe()
        return status

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
        if (intent.live_fact if live is None else live) and not intent.changes_state_only:
            parts.append("This is live information (weather, a price, a score, news): a lookup starts now. Say so "
                         "in a few words, in character. Never state a figure yourself; the real one follows.")
        if intent.greeting:
            parts.append("A greeting: a line or two in your own voice, from something real (the hour, where "
                         "you are, what's happened). Asked how you are, go by your actual mood and errands, not "
                         "an invented day. Don't ask them back unless you want to know.")
        if intent.wants_offer:
            parts.append("They want an offer, not action: put it in offer and wait for their yes.")
        if intent.about_past:
            parts.append("They ask what you did: use only status.recent_errands and status.last_seen. Say what was "
                         "done, tried, refused or not done, and how old each sighting is.")
        if intent.small_talk:
            parts.append("Small talk: no suggestions to do things around the house.")
        return " ".join(parts)

    async def _recall(self, text: str) -> list[dict]:
        if not self.store:
            return []
        try:
            # Durable knowledge first; past chat snippets would otherwise crowd it
            # out, since they often repeat the very facts they mention.
            memories = await self.store.recall(text, kinds=["fact", "preference", "episode", "skill"], k=4)
            chats = await self.store.recall(text, kinds=["conversation"], k=2)
            return memories + [{**m, "text": _OWN_OLD_LINE.sub("", m["text"])} for m in chats]
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
                        finished = self._finish_turn(user_turn, reply_id, spoken_time(self.engine.clock), None,
                                                     {"chat_model_calls": 0, "intent_router": "clock"})
                        self._intents.pop(user_turn["id"], None)
                        return
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
                    self._start_or_queue(goal, user_turn["id"], Grant.read_only(said), plan=_lookup_plan(query))
                    self._intents.pop(user_turn["id"], None)
                    finished = True
                    return
                self.replying = True
                waiting = self.shares_task_model and self.manager.active_id is not None
                self.bus.emit("chat_thinking", self.name, skip_history=True, reply_id=reply_id,
                              waiting_on_errand=waiting)
                recall_started = time.monotonic()
                memories = await self._recall(user_turn["text"])
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
                messages = build_messages(self.name, voice, prompt_memories(memories),
                                          self._status(user_turn.get("speaker")), earlier, user_turn["text"],
                                          guidance=guidance, context_tokens=min(
                                              self.settings.chat_context_tokens,
                                              self.settings.context_tokens - self.settings.chat_max_tokens))
                splitter = TrailerSplitter()
                try:
                    async for chunk in self.client.stream_text(messages, max_tokens=self.settings.chat_max_tokens,
                                                               temperature=self.settings.chat_temperature):
                        visible = splitter.feed(chunk)
                        if visible:
                            self.bus.emit("chat_delta", self.name, skip_history=True, reply_id=reply_id,
                                          text=visible)
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
                    self.bus.emit("chat_delta", self.name, skip_history=True, reply_id=reply_id, text=remaining)
                metrics = dict(getattr(self.client, "last_metrics", {}) or {})
                metrics["memory_recall_seconds"] = round(recall_seconds, 3)
                metrics["input_chars"] = sum(len(m["content"]) for m in messages)
                metrics["estimated_prompt_tokens"] = message_tokens(messages)
                metrics["token_estimator"] = "utf8_bytes_div3_estimate"
                metrics["queue_seconds"] = round(queue_seconds, 3)
                metrics["chat_model_calls"] = 1
                # Where the prompt's size goes: a long prompt is what makes a local model slow to start.
                metrics["prompt_chars"] = {"system": len(messages[0]["content"]),
                                           "history": sum(len(m["content"]) for m in messages[1:-1]),
                                           "latest": len(messages[-1]["content"])}
                action, action_error = parsed.action, parsed.error
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
            self._finish_turn(user_turn, reply_id, "Cannot think now. My language model not "
                                                   "answer. You check it run, question?", None, {},
                              error=str(exc))
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
            self.last_activity = time.monotonic()
            self._offer_seen.pop(user_turn["id"], None)
            self._turn_epochs.pop(user_turn["id"], None)
            if self.gate is not None:
                self.gate.conversation_waiting(-1)
            if not finished:
                # Cancelled (reset, shutdown): release any "Thinking…" indicator.
                self.bus.emit("chat_reply_ended", self.name, skip_history=True, reply_id=reply_id,
                              reason="cancelled")

    def _finish_turn(self, user_turn, reply_id, text, action, metrics, **extra) -> dict:
        return self._add_turn("kip", text, reply_id=reply_id, in_reply_to=user_turn["id"],
                              to=user_turn.get("speaker"),
                              action=action.model_dump() if action else None, metrics=metrics,
                              **{k: v for k, v in extra.items() if v})

    # ── acting on a reply ───────────────────────────────────────────────────
    async def _act(self, action: ChatAction | None, user_turn: dict, reply_turn: dict, written: str = "") -> None:
        said, reply = user_turn["text"], reply_turn["text"]
        intent = self._intents.get(user_turn["id"]) or read_intent(said)
        self._decide_errands(action, user_turn)
        # Claims are judged on what the model wrote, before it was tidied for speech.
        self._check_claims(written or reply, intent, live=self._is_live(user_turn["id"], intent))
        self._routes.pop(user_turn["id"], None)
        self._intents.pop(user_turn["id"], None)
        if self.store:
            await self._remember(action, user_turn, reply)

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
            self._start_or_queue(goal, user_turn["id"], Grant.read_only(said), plan=_lookup_plan(query))
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
        if proposed and asks:
            self._start_or_queue(proposed, user_turn["id"], Grant.full("request", said))
            return
        if proposed and self.settings.chat_autonomy == "act" and answer != "no":
            # Opted-in configuration: proposals start at once (restrictions still bind them).
            self._start_or_queue(proposed, user_turn["id"], Grant.full("request", said))
            return
        if proposed and question and (needs or digital_scope(proposed)):
            # A question may start a lookup, but only ever with read-only effects.
            self._start_or_queue(proposed, user_turn["id"], Grant.read_only(said))
            return
        refusing = answer == "no" if replying else intent.declines
        suggestion = offered or proposed
        if suggestion and not refusing and digital_scope(suggestion) and task_identity(suggestion) not in self._declined:
            # A lookup or a calculation changes nothing in the house: Kip may simply do it,
            # unasked. The read-only grant guarantees it, and a plan that turns out to need
            # more stops before acting and becomes an offer.
            self._start_or_queue(suggestion, user_turn["id"], Grant.read_only(said))
            return
        # Anything else is a suggestion: never physical without a yes.
        if not refusing:
            self._make_offer(suggestion, user_turn, asked_for=False, small_talk=intent.small_talk)

    def _pure_lookup(self, intent: Intent, said: str, live: bool | None = None) -> bool:
        """Weather, prices, scores, news, with nothing in the house involved: a web lookup."""
        live = intent.live_fact if live is None else live
        return (live and not intent.needs_tools and self.settings.digital_tools_enabled
                and not self.manager.restrictions.forbids_tool("web_search") and not intent.wants_offer
                and not self._involves_the_house(said))

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
        return fast_path(route, intent, web_allowed=allowed,
                         involves_house=self._involves_the_house(user_turn["text"]))

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
        return (self.engine.clock is not None and not (route.followup or route.correction)
                and not (intent.cancel or intent.changes_state_only or self._closed) and clock_question(said))

    def _is_live(self, turn_id: str, intent: Intent) -> bool:
        """Live information? When it is on, the router decides for messages that ask something;
        a statement ("I wonder what the air is like") keeps the live-fact words, so the router
        never turns remarks into lookups nobody asked for."""
        route = self._routes.get(turn_id)
        asks = route is not None and (intent.question or intent.request or route.followup or route.correction
                                      or route.lookup_phrase or route.asks)
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

    def _involves_the_house(self, text: str) -> bool:
        """Mentions someone or somewhere in the house ("...and tell Paul"): not a pure lookup.

        Chat before the lookup question ("Let Moira enjoy her book. When's the next race?")
        counts only if it asks for something; from the question on, any mention counts."""
        world = self.engine.snapshot()
        sentences = re.split(r"(?<=[.!?;])\s+", text.strip())
        start = next((i for i, sentence in enumerate(sentences) if is_live_fact(sentence)), 0)
        lowered = " ".join(sentence for i, sentence in enumerate(sentences)
                           if i >= start or is_request(sentence) or is_question(sentence)).casefold().replace("_", " ")
        names = [str(v.get("name", k)).casefold() for k, v in {**world.get("people", {}),
                                                                 **world.get("rooms", {})}.items()]
        names += [k.replace("_", " ") for k in {**world.get("people", {}), **world.get("rooms", {})}]
        return any(re.search(r"\b" + re.escape(name) + r"\b", lowered) for name in names if name)

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
        if self.manager.active_id:
            goal = self.manager.active()["goal"]

            async def stop(task_id: str) -> None:
                await self.manager.cancel(task_id)
                self._add_turn("kip", f"Stopped: {goal}", source="narration")
            self._spawn(stop(self.manager.active_id))

    def _check_claims(self, reply: str, intent: Intent | None = None, live: bool = False) -> None:
        """Kip must not claim actions, runs or sightings his tool evidence does not support."""
        world = self.engine.snapshot()
        found, notes = unsupported_claims(reply, evidence_ledger(self.manager), world), []
        if intent is not None and (intent.live_fact or live) and _FIGURE.search(reply.casefold()):
            # A live figure before any lookup came back is a guess wearing a number.
            notes.append("that figure was a guess; I'm looking up the real one now")
        if found:
            notes.append("I haven't actually done that; there's no record of it")
        running = (self.manager.active() or {}).get("requested_tools", [])
        for tool in unsupported_execution(reply, self.manager):
            what = {"run_python": "no Python has run for that", "read_webpage": "I haven't read that page",
                    "web_search": "I haven't searched for that"}[tool]
            notes.append(what + ("; it's running now, and the real result will follow" if tool in running else ""))
        denied = denied_capabilities(reply, self.manager, self.settings)
        if denied:
            notes.append("I can " + " and ".join(denied) + " — that's working")
        scans, names = room_scans(self.manager), {k: v.get("name", k) for k, v in world.get("people", {}).items()}
        for person, room in dict.fromkeys(contradicted_presence(reply, self.manager, world)):
            at = scans[room].get("at")
            notes.append(f"when I looked in the {room.replace('_', ' ')}{' at ' + at if at else ''}, "
                         f"{names.get(person, person)} wasn't there; anything older is only a record")
        if notes:
            _log.warning("Unsupported claims in reply: %s", notes)
            self._add_turn("kip", "Scratch that: " + "; ".join(notes) + ".", source="correction", claims=notes)

    async def _remember(self, action: ChatAction | None, user_turn: dict, reply: str) -> None:
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
            people = set(self.engine.snapshot().get("people", {}))
            if not people & set(tags) and FIRST_PERSON.search(text.casefold()):
                # "I like tea" belongs to whoever is speaking, never an arbitrary resident.
                if speaker:
                    tags.append(speaker)
            meta = {"context": context, "speaker": speaker or "unknown", "said_in": user_turn["id"]}
            try:
                added.append(await self.store.add("preference" if re.search(r"\b(prefer|like|love|favou?rite|"
                                                                            r"enjoy|hate|dislike)", text, re.I)
                                                  else "fact", text, about=tags or about, importance=0.7,
                                                  source="conversation", meta=meta))
                self._remembered.append(text if context == "actual" else f"{text} ({context})")
            except Exception:
                _log.exception("Could not save a conversational fact")
        # Corrections retire the wrong memory (kept as history), never the fix itself.
        for wrong in (action.forget if action else []):
            if wrong.strip():
                try:
                    retired = await self.store.supersede(wrong, by=added[0] if added else None,
                                                         exclude=set(added), about=self._mentioned(wrong))
                    if retired:
                        _log.info("Memory %s superseded by a correction", retired["id"])
                except Exception:
                    _log.exception("Could not retire a corrected memory")
        try:
            await self.store.add("conversation", f"They said: {said} | I replied: {reply}", about=about,
                                 importance=0.3, source="conversation", meta={"speaker": speaker or "unknown"})
            await self.store.set_keyed(f"chat-session:{self.session_id}", "episode", self._session_summary(),
                                       about=sorted({t for turn in self.turns for t in self._mentioned(turn["text"])}),
                                       importance=0.5, source="conversation")
        except Exception:
            _log.exception("Could not save conversation")

    def _session_summary(self) -> str:
        """A deterministic episode of this conversation, so it can be recalled after a restart."""
        turns = list(self.turns)
        start = turns[0]["at"][:16].replace("T", " ") if turns else ""
        topics = [_clip(t["text"], 60) for t in turns if t["role"] == "user"][-6:]
        errands = [f"{t['text'][:80]} ({t['outcome']})" for t in turns if t.get("outcome")][-4:]
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
        """Tag memories with people and objects mentioned by name or ID."""
        world = self.engine.snapshot()
        lowered = text.casefold()
        tags = []
        for key, item in {**world.get("people", {}), **world.get("objects", {})}.items():
            name = str(item.get("name", key)).casefold()
            if any(alias and re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", lowered)
                   for alias in (key.casefold(), name)):
                tags.append(key)
        return tags

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
        elif kind in {"task_completed", "task_failed"}:
            outcome = kind.removeprefix("task_")
            self._lookup_queries.pop(task_identity(task.get("goal", "")), None)
            correction = self._correction_goals.pop(task_identity(task.get("goal", "")), None)
            if correction and self.store:
                self.store.set_intent_outcome(correction, "answered" if outcome == "completed" else "failed")
            answer = task.get("answer") or {}
            if answer.get("answer") and str(task.get("goal", "")).startswith(LOOKUP_PREFIX):
                self._learn_from_answer(answer["answer"])
            if answer.get("answer"):
                # The answer is already written in Kip's voice; no stock opener.
                text = answer["answer"]
            else:
                # Plain facts, no stock reaction: character belongs in replies, not templates.
                found = describe_findings(task)
                if outcome == "completed":
                    text = found or data.get("summary", "") or "Done."
                else:
                    text = " ".join(part for part in ("Not finish. Bad.", found, data.get("summary", ""))
                                    if part)
            notes = list(task.get("answer_notes") or [])
            if task.get("blocked"):
                notes.append("Not done: " + "; ".join(task["blocked"]) + ".")
            text = " ".join([text, *notes])
            self._add_turn("kip", text, source="narration", sources=answer.get("sources", []), outcome=outcome)
            if self.pending_goals:
                self._spawn(self._start_next_when_free(self.generation))
        elif kind == "task_needs_permission":
            self._needs_permission(data)
        elif kind == "agent_intention":
            self._announced.add(task_identity(data.get("summary", "")))
            self._add_turn("kip", f"I take care of something: {data.get('summary', '')}. "
                                  f"{data.get('reason', '')}".strip(), source="narration")


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


def _lookup_plan(said: str) -> GoalPlan:
    """A plain web lookup of what they asked, searched in their own words."""
    query = re.sub(r"^\W*(?:hey |ok |so )?kip\W+", "", said.strip(), flags=re.I)[:200]
    return GoalPlan(summary="Look it up.", conditions=[GoalCondition(kind="answer", required_tools=["web_search"])],
                    initial_actions=[InitialCapability(tool="web_search", arguments={"query": query})])


# ── status and narration helpers (deterministic, no model calls) ───────────────
def _topic_words(text: str) -> set[str]:
    words = (w.split("'")[0] for w in re.findall(r"[a-z][a-z0-9]*(?:'[a-z]+)?", text.casefold().replace("’", "'")))
    return {w for w in words if len(w) >= 4 and w not in _TOPIC_STOP}


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:] if text else text


def _place(location) -> str | None:
    if isinstance(location, dict):
        kind, key = location.get("kind"), location.get("id")
        if kind == "room":
            return f"in the {key.replace('_', ' ')}"
        if kind == "person":
            return f"with {key.title()}"
        if kind == "robot":
            return "with me"
        return None
    return f"in the {location.replace('_', ' ')}" if location else None


def errand_progress(task: dict) -> dict:
    """What has actually been observed so far, so chat never invents progress."""
    memory = task.get("task_memory") or {}
    found = {item.get("name", key): _place(item.get("location"))
             for key, item in (memory.get("objects") or {}).items()}
    people = {item.get("name", key): _place(item.get("location"))
              for key, item in (memory.get("people") or {}).items()}
    return {"goal": task.get("goal"),
            "rooms_looked_in": [r.replace("_", " ") for r in (memory.get("observed_rooms") or {})],
            "objects_found": {k: v for k, v in found.items() if v},
            "people_seen": {k: v for k, v in people.items() if v},
            "outcomes_done": [o.get("kind") for o in memory.get("completed_outcomes", [])],
            "latest": task.get("summary", "")}


def recent_errands(manager, limit: int = 3) -> list[dict]:
    """The last finished errands and what they actually found or answered."""
    finished = [t for t in manager.tasks.values() if t.status != "running"][-limit:]
    result = []
    for context in finished:
        task = context.public()
        answer = (task.get("answer") or {}).get("answer", "")
        result.append({"goal": task["goal"], "outcome": task["status"],
                       "result": _clip(answer, 160) if answer else describe_findings(task) or task.get("summary", "")})
    return result


def known_people(persistent) -> dict:
    if persistent is None:
        return {}
    people = persistent.prompt().get("people", {})
    return {p.get("name", key): ("last seen " + _place(p["last_seen_room"]) if p.get("last_seen_room") else "")
            for key, p in people.items()}


def describe_findings(task: dict) -> str:
    """A plain-language account of what an errand actually observed, room by room."""
    memory = task.get("task_memory") or {}
    people, objects = memory.get("people") or {}, memory.get("objects") or {}
    rooms = [r for r in (memory.get("observed_rooms") or {})]
    shown_people, shown_objects, parts = set(), set(), []
    looked = []
    for room in rooms[:5]:
        here = [p.get("name") or key for key, p in people.items() if p.get("location") == room]
        things = [f"the {(o.get('name') or key).lower()}" for key, o in objects.items()
                  if (o.get("location") or {}) == {"kind": "room", "id": room}]
        shown_people |= {key for key, p in people.items() if p.get("location") == room}
        shown_objects |= {key for key, o in objects.items() if (o.get("location") or {}) == {"kind": "room", "id": room}}
        seen = here + things
        looked.append(f"the {room.replace('_', ' ')} ({', '.join(seen) if seen else 'empty'})")
    if looked:
        parts.append("I looked in " + (", ".join(looked[:-1]) + " and " + looked[-1] if len(looked) > 1 else looked[0]) + ".")
    progress = errand_progress(task)
    elsewhere = {name: where for key, name, where in
                 ((k, o.get("name", k), _place(o.get("location"))) for k, o in objects.items())
                 if where and key not in shown_objects}
    if elsewhere:
        parts.append("I found the " + ", and the ".join(f"{name.lower()} {where}"
                                         for name, where in list(elsewhere.items())[:3]) + ".")
    others = {name: where for name, where in progress["people_seen"].items()
              if name not in {people[k].get("name") or k for k in shown_people}}
    if others:
        parts.append("I saw " + ", ".join(f"{name} {where}" for name, where in list(others.items())[:3]) + ".")
    return " ".join(parts)


# ── evidence for Kip's own claims ────────────────────────────────────────────
def evidence_ledger(manager) -> dict[str, set[str]]:
    """Everything the robot has actually done or observed in this world, from tool results."""
    ledger = {"rooms": set(), "objects": set(), "people": set(), "gave": set(), "told": set()}
    for context in manager.tasks.values():
        ledger["rooms"] |= set(context.memory.rooms)
        ledger["objects"] |= set(context.memory.objects)
        ledger["people"] |= {key for key, person in context.memory.people.items() if person.location}
        for entry in context.action_history:
            if not entry.get("success"):
                continue
            observation = entry.get("observation") or {}
            if entry["tool"] == "give":
                ledger["gave"] |= {observation.get("object", ""), observation.get("person", "")}
            elif entry["tool"] == "talk_to":
                ledger["told"].add(observation.get("person", ""))
    return ledger


_CLAIM = re.compile(r"\bi(?:'ve| have| just| already|\s)*\s*(checked|looked|searched|found|delivered|brought|gave|"
                    r"told|visited)\b(.*)")
_HEDGE = re.compile(r"\b(n't|not|never|haven't|couldn't|didn't|will|'ll|going to|can|could|would|should|"
                    r"if|when|before|after|yet)\b")


def unsupported_claims(reply: str, ledger: dict[str, set[str]], world: dict) -> list[str]:
    """First-person past-tense claims about physical actions with no matching tool evidence."""
    rooms = {key: {key.casefold(), room.get("name", key).casefold()} for key, room in world.get("rooms", {}).items()}
    objects = {key: {key.casefold(), obj.get("name", key).casefold()} for key, obj in world.get("objects", {}).items()}
    people = {key: {key.casefold(), person.get("name", key).casefold()} for key, person in world.get("people", {}).items()}
    claims = []
    for sentence in re.split(r"(?<=[.!?])\s+", reply.casefold().replace("’", "'")):
        match = _CLAIM.search(sentence)
        if not match or _HEDGE.search(sentence[:match.start()] + " " + match.group(2)):
            continue
        verb, rest = match.group(1), match.group(2)
        def named(table):
            return {key for key, names in table.items() if any(re.search(r"\b" + re.escape(n) + r"\b", rest) for n in names)}
        mentioned_rooms, mentioned_objects, mentioned_people = named(rooms), named(objects), named(people)
        if verb in {"checked", "looked", "searched", "visited"}:
            evidence = (mentioned_rooms & ledger["rooms"]) | (mentioned_objects & ledger["objects"]) | \
                       (mentioned_people & ledger["people"])
        elif verb == "found":
            evidence = (mentioned_objects & ledger["objects"]) | (mentioned_people & ledger["people"])
        elif verb == "told":
            evidence = mentioned_people & ledger["told"]
        else:  # delivered / brought / gave
            evidence = (mentioned_objects | mentioned_people) & ledger["gave"]
        mentioned = mentioned_rooms | mentioned_objects | mentioned_people
        unknown_thing = re.search(r"\b(the|his|her|their|your|my)\s+(\w+)", rest)
        if evidence:
            continue
        if mentioned or unknown_thing:
            claims.append(sentence.strip())
    return claims
