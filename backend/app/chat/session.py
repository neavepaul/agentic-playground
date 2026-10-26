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
from app.tasks.permissions import READ_ONLY, Grant, digital_scope, plain
from .consent import Offer, classify_reply, is_question, is_request, task_identity
from .prompt import ChatAction, TrailerSplitter, build_messages

_log = logging.getLogger("agentic_friend.chat")

HISTORY = 50
FIRST_PERSON = re.compile(r"\b(i|i'm|i've|me|my|mine|myself|the user)\b")


class ChatSession:
    def __init__(self, client, bus, manager, engine, settings, store=None, persona=None,
                 shares_task_model: bool = True) -> None:
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
        self.session_id = str(uuid4())
        self._announced: set[tuple[str, ...]] = set()
        self.offer: Offer | None = None
        self.offer_log: deque[dict] = deque(maxlen=20)
        self._offer_seen: dict[str, str | None] = {}  # user turn id -> offer visible when sent
        self._started_by_turn: set[str] = set()
        self._remembered: list[str] = []
        self._queued_grants: dict[tuple[str, ...], Grant] = {}
        self._declined: set[tuple[str, ...]] = set()  # offers refused in this conversation: not offered again

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
        rooms = {key: room.get("name", key) for key, room in self.engine.snapshot().get("rooms", {}).items()}
        if self.manager.restrictions.update(text, rooms):
            _log.info("Restrictions now: %s", self.manager.restrictions.describe())
        earlier = list(self.turns)
        turn = self._add_turn("user", text, speaker=self._speaker_id(speaker))
        # Consent is judged against the offer the person could actually see when they
        # sent this message; a message queued earlier can never approve a later offer.
        self._offer_seen[turn["id"]] = self.offer.id if self._offer_valid() else None
        self._spawn(self._respond(turn, earlier))
        return turn

    def history(self) -> list[dict]:
        return list(self.turns)

    def state(self) -> dict:
        return {"queued_errands": list(self.pending_goals), "offered_errand": self.pending_offer,
                "offer": self.offer.public() if self.offer else None, "generation": self.generation}

    async def drain(self) -> None:
        while self._jobs:
            jobs = list(self._jobs)
            await asyncio.gather(*jobs, return_exceptions=True)
            self._jobs.difference_update(job for job in jobs if job.done())

    async def close(self) -> None:
        self._closed = True
        self.pending_goals.clear()
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
        turn = {"id": str(uuid4()), "role": role, "text": text,
                "at": datetime.now(timezone.utc).isoformat(), **extra}
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
        active = self.manager.active()
        # "none", not null: models repeat null verbatim ("my current errand is null").
        status["current_errand"] = errand_progress(active) if active else "none"
        if self.pending_goals:
            status["queued_errands"] = list(self.pending_goals)
        if self.pending_offer:
            status["offered_errand_awaiting_yes"] = self.pending_offer
        recent = recent_errands(self.manager)
        if recent:
            status["recent_errands"] = recent
        people = known_people(self.manager.persistent)
        if people:
            status["people_i_know"] = people
        if self.manager.restrictions:
            status["asked_not_to"] = self.manager.restrictions.describe()
        return status

    async def _recall(self, text: str) -> list[dict]:
        if not self.store:
            return []
        try:
            # Durable knowledge first; past chat snippets would otherwise crowd it
            # out, since they often repeat the very facts they mention.
            memories = await self.store.recall(text, kinds=["fact", "preference", "episode", "skill"], k=4)
            memories += await self.store.recall(text, kinds=["conversation"], k=2)
            return memories
        except Exception:
            _log.exception("Memory recall failed; continuing conversation without recalled memories")
            return []

    async def _respond(self, user_turn: dict, earlier: list[dict]) -> None:
        reply_id = str(uuid4())
        finished = False
        try:
            async with self._lock:  # FIFO: one reply at a time, in order
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
                messages = build_messages(self.name, voice, prompt_memories(memories),
                                          self._status(user_turn.get("speaker")), earlier, user_turn["text"])
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
                action, action_error = parsed.action, parsed.error
                text = parsed.visible
                if metrics.get("truncated"):
                    # A cut-off reply is shown as cut off; a half-written action never runs.
                    text = (text + " …").strip()
                    action, action_error = None, "The reply was cut off before its action line finished."
                if action_error:
                    _log.warning("Chat action not executed: %s", action_error)
                _log.info("chat_reply first_token_seconds=%s total_seconds=%s memories=%d task=%s recall_seconds=%.3f "
                          "input_chars=%d truncated=%s", metrics.get("first_token_seconds"),
                          metrics.get("total_seconds"), len(memories), bool(action and action.task), recall_seconds,
                          metrics["input_chars"], bool(metrics.get("truncated")))
                text = text or "Sorry, I lost my words for a moment. Could you say that again?"
                reply_turn = self._finish_turn(user_turn, reply_id, text, action, metrics, action_error=action_error)
                finished = True
                await self._act(action, user_turn, reply_turn)
        except ModelError as exc:
            self._finish_turn(user_turn, reply_id, "I can't think clearly right now: my language model isn't "
                                                   "answering. Please check that it's running.", None, {},
                              error=str(exc))
            finished = True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _log.exception("Chat reply failed")
            self._finish_turn(user_turn, reply_id, "Sorry, something went wrong on my side while answering.",
                              None, {}, error=type(exc).__name__)
            finished = True
        finally:
            self.replying = False
            self.last_activity = time.monotonic()
            self._offer_seen.pop(user_turn["id"], None)
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
    async def _act(self, action: ChatAction | None, user_turn: dict, reply_turn: dict) -> None:
        said, reply = user_turn["text"], reply_turn["text"]
        self._decide_errands(action, user_turn)
        self._check_claims(reply)
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
        said = user_turn["text"]
        answer = classify_reply(said)
        proposed = " ".join(action.task.split()) if action else ""
        offered = " ".join(action.offer.split()) if action else ""
        seen = self._offer_seen.get(user_turn["id"])
        asks = is_request(said)  # their own words ask for an action
        question = not asks and is_question(said)
        if self.offer and not self._offer_valid():
            self._expire_offer("expired")
        offer = self.offer if self._offer_valid() else None
        if offer and seen == offer.id:
            # This message is the person's answer to Kip's offer.
            if answer == "yes":
                offer.status, offer.authorized_by = "consumed", user_turn["id"]
                self.offer_log.append(offer.public())
                # Only the offered goal runs, whatever the model wrote this turn, and only
                # with the effects that were offered.
                self._start_or_queue(offer.goal, user_turn["id"], Grant(
                    effects=sorted(offer.effects) if offer.effects else Grant().effects,
                    source="offer", authorized_text=offer.goal))
                return
            if answer == "no":
                self._expire_offer("declined")
                if not asks:
                    return
                # "No, please find the keys instead": the offer is declined, and the new
                # explicit request below is still honoured (negated actions never are).
            elif not (asks or (question and proposed)):
                # Ambiguous: keep the offer open and ask, rather than guess.
                offer.status = "deferred"
                self.offer_log.append(offer.public())
                self.offer = Offer(goal=offer.goal, reply_turn_id=offer.reply_turn_id, generation=self.generation)
                self._add_turn("kip", f"Just to be sure: would you like me to {_lower_first(offer.goal)}?",
                               source="clarification")
                return
            else:
                self._expire_offer("superseded")  # they asked for something else instead
        # is_request() already rejects negated actions ("Don't deliver…"), so refusal words
        # elsewhere in the sentence ("…dinner is NOT ready", "…instead") don't block a request.
        if task_identity(proposed) in self._declined:
            proposed = ""  # never start, or offer again, an errand the person declined
        if proposed and self.manager.restrictions.mentions_only_forbidden_tools(proposed):
            # Every tool it names was ruled out ("...using Python" after "don't use Python"):
            # nothing to run; the reply answers within the limits (it is told about them).
            _log.info("Not starting %r: it relies only on tools the person ruled out", proposed)
            return
        if proposed and asks:
            self._start_or_queue(proposed, user_turn["id"], Grant.full("request", said))
            return
        if proposed and self.settings.chat_autonomy == "act" and answer != "no":
            # Opted-in configuration: proposals start at once (restrictions still bind them).
            self._start_or_queue(proposed, user_turn["id"], Grant.full("request", said))
            return
        if proposed and question:
            if digital_scope(proposed):
                # A question may start a lookup, but only ever with read-only effects.
                self._start_or_queue(proposed, user_turn["id"], Grant.read_only(said))
                return
            # Anything else a question leads to is a suggestion: never physical without a yes.
        suggestion = offered or proposed
        if task_identity(suggestion) in self._declined:
            suggestion = ""
        if suggestion and answer != "no":
            # Kip may suggest; nothing runs until the person says yes to this offer.
            self._expire_offer("superseded")
            self.offer = Offer(goal=suggestion, reply_turn_id=user_turn["id"], generation=self.generation)
            _log.info("Holding errand as an offer awaiting a yes: %s", suggestion)

    def _check_claims(self, reply: str) -> None:
        """Kip must not claim physical actions it has no tool evidence for."""
        world = self.engine.snapshot()
        unsupported = unsupported_claims(reply, evidence_ledger(self.manager), world)
        if unsupported:
            _log.warning("Unsupported action claim in reply: %s", unsupported)
            self._add_turn("kip", "Sorry, I should correct myself: I don't have any record of actually doing "
                                  "that yet.", source="correction", claims=unsupported)

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

    def _start_or_queue(self, goal: str, origin_turn: str | None = None, grant: Grant | None = None) -> None:
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
            # Kip's own reply already acknowledged this errand; skip "On it".
            self._announced.add(task_identity(goal))
            self.manager.start(goal, grant)
        except TaskBusy:
            # It will start later, when an "On it" is genuinely news again. It keeps the
            # grant it was given; restrictions are re-checked when its tools run.
            self._announced.discard(task_identity(goal))
            self.pending_goals.append(goal)
            self._queued_grants[task_identity(goal)] = grant
            self._add_turn("kip", f"I'll do that right after my current errand: {goal}", source="narration")

    async def _start_next_when_free(self, generation: int) -> None:
        while self.manager.active_id is not None and not self._closed:
            await asyncio.sleep(0.05)
        if self.pending_goals and not self._closed and generation == self.generation:
            goal = self.pending_goals.popleft()
            self._start_or_queue(goal, grant=self._queued_grants.pop(task_identity(goal), None))

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
                self._add_turn("kip", f"On it: {goal}", source="narration")
        elif kind in {"task_completed", "task_failed"}:
            outcome = kind.removeprefix("task_")
            answer = task.get("answer") or {}
            if answer.get("answer"):
                # The answer is already written in Kip's voice; no stock opener.
                text = answer["answer"]
            else:
                # Plain facts, no stock reaction: character belongs in replies, not templates.
                found = describe_findings(task)
                if outcome == "completed":
                    text = found or data.get("summary", "") or "Done."
                else:
                    text = " ".join(part for part in ("I couldn't finish that.", found, data.get("summary", ""))
                                    if part)
            self._add_turn("kip", text, source="narration", sources=answer.get("sources", []), outcome=outcome)
            if self.pending_goals:
                self._spawn(self._start_next_when_free(self.generation))
        elif kind == "task_needs_permission":
            self._needs_permission(data)
        elif kind == "agent_intention":
            self._announced.add(task_identity(data.get("summary", "")))
            self._add_turn("kip", f"I'm going to take care of something: {data.get('summary', '')}. "
                                  f"{data.get('reason', '')}".strip(), source="narration")


    def _needs_permission(self, data: dict) -> None:
        """A task stopped before acting because it needed more than it was cleared for."""
        goal, needs = data.get("goal", ""), set(data.get("needs", []))
        if data.get("ruled_out"):
            self._add_turn("kip", f"I've left that alone: it would mean {plain(needs)}, and you asked me not to.",
                           source="narration")
            return
        if task_identity(goal) in self._declined:
            return  # they already said no to this once; don't ask again
        self._expire_offer("superseded")
        self.offer = Offer(goal=goal, reply_turn_id="", generation=self.generation, effects=frozenset(needs) | READ_ONLY)
        self._add_turn("kip", f"I can't answer that from here: it would mean {plain(needs)}. Want me to?",
                       source="narration")


# ── status and narration helpers (deterministic, no model calls) ───────────────
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
    """A plain-language account of what an errand actually observed."""
    progress = errand_progress(task)
    parts = []
    if progress["rooms_looked_in"]:
        rooms = progress["rooms_looked_in"][:4]
        parts.append("I looked in the " + (", the ".join(rooms[:-1]) + " and the " + rooms[-1]
                                         if len(rooms) > 1 else rooms[0]) + ".")
    if progress["objects_found"]:
        parts.append("I found the " + ", and the ".join(f"{name.lower()} {where}"
                                          for name, where in list(progress["objects_found"].items())[:3]) + ".")
    if progress["people_seen"]:
        parts.append("I saw " + ", ".join(f"{name} {where}"
                                         for name, where in list(progress["people_seen"].items())[:3]) + ".")
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
