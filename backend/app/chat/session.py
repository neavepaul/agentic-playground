"""An ongoing conversation with Kip. One streamed model call per message; errands run
in the background through the existing TaskManager, and progress is narrated without
any model calls."""
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
from .prompt import ChatAction, TrailerSplitter, build_messages, parse_action

_log = logging.getLogger("agentic_friend.chat")

HISTORY = 50


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
        self._announced: set[str] = set()
        # An errand Kip offered but nobody has agreed to yet. Lasts one turn.
        self.pending_offer: str | None = None

    def attach(self) -> None:
        self.bus.add_listener(self._narrate)

    # ── public API ──────────────────────────────────────────────────────────
    @property
    def busy(self) -> bool:
        return self.replying or any(not job.done() for job in self._jobs)

    def quiet_seconds(self) -> float:
        return time.monotonic() - self.last_activity if self.last_activity else float("inf")

    def submit(self, text: str) -> dict:
        text = text.strip()
        if not text:
            raise ValueError("Say something first.")
        if self._closed:
            raise ValueError("Conversation is shutting down.")
        self.last_activity = time.monotonic()
        earlier = list(self.turns)
        turn = self._add_turn("user", text)
        self._spawn(self._respond(turn, earlier))
        return turn

    def history(self) -> list[dict]:
        return list(self.turns)

    async def drain(self) -> None:
        while self._jobs:
            jobs = list(self._jobs)
            await asyncio.gather(*jobs, return_exceptions=True)
            self._jobs.difference_update(job for job in jobs if job.done())

    async def close(self) -> None:
        self._closed = True
        self.pending_goals.clear()
        jobs = list(self._jobs)
        for job in jobs:
            job.cancel()
        await asyncio.gather(*jobs, return_exceptions=True)
        self._jobs.difference_update(jobs)

    async def reset_pending(self) -> None:
        """Discard replies and queued work tied to the old world, then accept chat again."""
        await self.close()
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

    def _status(self) -> dict:
        status = self.engine.get_status()
        active = self.manager.active()
        # "none", not null: models repeat null verbatim ("my current errand is null").
        status["current_errand"] = errand_progress(active) if active else "none"
        if self.pending_goals:
            status["queued_errands"] = list(self.pending_goals)
        if self.pending_offer:
            status["offered_errand_awaiting_yes"] = self.pending_offer
        people = known_people(self.manager.persistent)
        if people:
            status["people_i_know"] = people
        return status

    async def _respond(self, user_turn: dict, earlier: list[dict]) -> None:
        async with self._lock:  # FIFO: one reply at a time, in order
            self.replying = True
            reply_id = str(uuid4())
            waiting = self.shares_task_model and self.manager.active_id is not None
            self.bus.emit("chat_thinking", self.name, skip_history=True, reply_id=reply_id,
                          waiting_on_errand=waiting)
            try:
                memories = []
                recall_started = time.monotonic()
                if self.store:
                    # Durable knowledge first; past chat snippets would otherwise crowd it
                    # out, since they often repeat the very facts they mention.
                    try:
                        memories = await self.store.recall(
                            user_turn["text"], kinds=["fact", "preference", "episode", "skill"], k=4)
                        memories += await self.store.recall(user_turn["text"], kinds=["conversation"], k=2)
                    except Exception:
                        _log.exception("Memory recall failed; continuing conversation without recalled memories")
                recall_seconds = time.monotonic() - recall_started
                voice = self.persona.voice_view() if self.persona else None
                # Include replies to preceding messages that finished while this one
                # waited, without leaking later user messages into its context.
                known = {t["id"] for t in earlier}
                earlier += [t for t in self.turns if t.get("in_reply_to") in known and t["id"] not in known]
                anchors = {t["id"]: index for index, t in enumerate(earlier)}
                earlier.sort(key=lambda t: (anchors.get(t.get("in_reply_to"), anchors[t["id"]]),
                                           bool(t.get("in_reply_to")), anchors[t["id"]]))
                messages = build_messages(self.name, voice, prompt_memories(memories), self._status(),
                                          earlier, user_turn["text"])
                splitter = TrailerSplitter()
                async for chunk in self.client.stream_text(messages, max_tokens=self.settings.chat_max_tokens,
                                                           temperature=self.settings.chat_temperature):
                    visible = splitter.feed(chunk)
                    if visible:
                        self.bus.emit("chat_delta", self.name, skip_history=True, reply_id=reply_id, text=visible)
                remaining, reply, trailer = splitter.finish()
                if remaining:
                    self.bus.emit("chat_delta", self.name, skip_history=True, reply_id=reply_id, text=remaining)
                action = parse_action(trailer)
                metrics = dict(getattr(self.client, "last_metrics", {}) or {})
                metrics["memory_recall_seconds"] = round(recall_seconds, 3)
                metrics["input_chars"] = sum(len(m["content"]) for m in messages)
                _log.info("chat_reply first_token_seconds=%s total_seconds=%s memories=%d task=%s recall_seconds=%.3f input_chars=%d",
                          metrics.get("first_token_seconds"), metrics.get("total_seconds"),
                          len(memories), bool(action.task), recall_seconds, metrics["input_chars"])
                reply = reply or "Sorry, I lost my words for a moment. Could you say that again?"
                self._add_turn("kip", reply, reply_id=reply_id, in_reply_to=user_turn["id"],
                               action=action.model_dump(), metrics=metrics)
                await self._act(action, user_turn["text"], reply)
            except ModelError as exc:
                self._add_turn("kip", "I can't think clearly right now: my language model isn't answering. "
                               "Please check that it's running.", reply_id=reply_id,
                               in_reply_to=user_turn["id"], error=str(exc))
            finally:
                self.replying = False
                self.last_activity = time.monotonic()

    async def _act(self, action: ChatAction, said: str, reply: str) -> None:
        goal = " ".join(action.task.split())
        offer, self.pending_offer = self.pending_offer, None
        agreed = bool(offer) and is_affirmative(said)
        if agreed and not goal:
            goal = offer  # "Yes please" is enough, even if the model forgot the task line
        if goal and not self._closed:
            authorized = self.settings.chat_autonomy == "act" or agreed or is_request(said)
            if self._already_planned(goal):
                _log.info("Ignoring a repeated errand request: %s", goal)
            elif authorized:
                self._start_or_queue(goal)
            else:
                # The model proposed an errand nobody asked for: Kip's words may offer it,
                # but code decides. It waits for a yes on the next turn.
                self.pending_offer = goal
                _log.info("Holding unrequested errand as an offer: %s", goal)
        if self.store:
            about = self._mentioned(said + " " + reply)
            added = []
            for fact in action.remember:
                if fact.strip():
                    try:
                        added.append(await self.store.add("fact", fact, about=self._mentioned(fact) or about,
                                                          importance=0.7, source="conversation"))
                    except Exception:
                        _log.exception("Could not save a conversational fact")
            # Corrections retire the wrong memory (kept as history), never the fix itself.
            for wrong in action.forget:
                if wrong.strip():
                    try:
                        retired = await self.store.supersede(wrong, by=added[0] if added else None,
                                                             exclude=set(added))
                        if retired:
                            _log.info("Memory %s superseded by a correction", retired["id"])
                    except Exception:
                        _log.exception("Could not retire a corrected memory")
            try:
                await self.store.add("conversation", f"They said: {said} | I replied: {reply}", about=about,
                                     importance=0.3, source="conversation")
            except Exception:
                _log.exception("Could not save conversation")

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
        planned = [active["goal"]] if active else []
        return _same_goal(goal, planned + list(self.pending_goals))

    def _start_or_queue(self, goal: str) -> None:
        if self._closed:
            return
        try:
            # Kip's own reply already acknowledged this errand; skip "On it".
            self._announced.add(_goal_key(goal))
            self.manager.start(goal)
        except TaskBusy:
            # It will start later, when an "On it" is genuinely news again.
            self._announced.discard(_goal_key(goal))
            self.pending_goals.append(goal)
            self._add_turn("kip", f"I'll do that right after my current errand: {goal}", source="narration")

    async def _start_next_when_free(self) -> None:
        while self.manager.active_id is not None and not self._closed:
            await asyncio.sleep(0.05)
        if self.pending_goals and not self._closed:
            self._start_or_queue(self.pending_goals.popleft())

    def _narrate(self, event: dict) -> None:
        """Speak about errands in Kip's voice using templates, never a model call."""
        kind, data = event["type"], event.get("data", {})
        if self._closed:
            return
        task = data.get("task") or {}
        if kind == "task_cancelled":
            if self.pending_goals:
                self._spawn(self._start_next_when_free())
            return
        if kind == "task_started":
            goal = task.get("goal", data.get("summary", ""))
            if _goal_key(goal) in self._announced:
                self._announced.discard(_goal_key(goal))
            else:
                self._add_turn("kip", f"On it: {goal}", source="narration")
        elif kind in {"task_completed", "task_failed"}:
            outcome = kind.removeprefix("task_")
            answer = task.get("answer") or {}
            if answer.get("answer"):
                # The answer is already written in Kip's voice; no stock opener.
                text = answer["answer"]
            else:
                found = describe_findings(task)
                reaction = self.persona.reaction(outcome) if self.persona else None
                if outcome == "completed" and found:
                    text = found
                else:
                    detail = found or data.get("summary", "")
                    text = " ".join(part for part in (reaction, detail) if part)
            self._add_turn("kip", text, source="narration",
                           sources=answer.get("sources", []), outcome=outcome)
            if self.pending_goals:
                self._spawn(self._start_next_when_free())
        elif kind == "agent_intention":
            self._announced.add(_goal_key(data.get("summary", "")))
            self._add_turn("kip", f"I'm going to take care of something: {data.get('summary', '')}. "
                                  f"{data.get('reason', '')}".strip(), source="narration")


# ── status and narration helpers (deterministic, no model calls) ───────────────
def _goal_key(goal: str) -> str:
    return " ".join(re.findall(r"\w+", goal.casefold()))


def _same_goal(goal: str, others: list[str]) -> bool:
    """Same errand, allowing small rewordings ("Find the keys" vs "find keys.")."""
    words = set(_goal_key(goal).split()) - {"the", "a", "an", "my", "please"}
    for other in others:
        theirs = set(_goal_key(other).split()) - {"the", "a", "an", "my", "please"}
        if words and theirs and len(words & theirs) / len(words | theirs) >= 0.75:
            return True
    return False


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


# ── consent (deterministic) ──────────────────────────────────────────────────
_POLITE_REQUEST = re.compile(
    r"\b(please|could you|can you|would you|will you|would you mind|i need you to|i want you to|"
    r"i'd like you to|go and|help me)\b")
_IMPERATIVE = re.compile(
    r"(?:^|[.!?;]\s+|,\s*|\band\s+|\bthen\s+)(?:also\s+|just\s+|now\s+|quickly\s+)?(find|bring|fetch|get|grab|take|tell|let|check|look|search|"
    r"calculate|compute|use|run|read|deliver|give|go|ask|remind|put|place|carry|work out|figure out|"
    r"notify|inform|show|count)\b")
_AFFIRMATIVE = re.compile(
    r"^\W*(yes|yeah|yep|yup|sure|ok|okay|please|please do|go ahead|do it|go for it|sounds good|"
    r"that would be (great|lovely|nice)|absolutely|of course|definitely|why not|alright|all right)\b")


def is_request(text: str) -> bool:
    """Does the person's message itself ask Kip to do something?"""
    lowered = " ".join(text.casefold().split())
    return bool(_POLITE_REQUEST.search(lowered) or _IMPERATIVE.search(lowered))


def is_affirmative(text: str) -> bool:
    return bool(_AFFIRMATIVE.search(" ".join(text.casefold().split())))
