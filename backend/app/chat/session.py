"""An ongoing conversation with Kip. One streamed model call per message; errands run
in the background through the existing TaskManager, and progress is narrated without
any model calls."""
import asyncio
import logging
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

    def attach(self) -> None:
        self.bus.add_listener(self._narrate)

    # ── public API ──────────────────────────────────────────────────────────
    @property
    def busy(self) -> bool:
        return self.replying or any(not job.done() for job in self._jobs)

    def quiet_seconds(self) -> float:
        return time.monotonic() - self.last_activity if self.last_activity else float("inf")

    def submit(self, text: str) -> dict:
        text = " ".join(text.split())
        if not text:
            raise ValueError("Say something first.")
        self.last_activity = time.monotonic()
        turn = self._add_turn("user", text)
        self._spawn(self._respond(turn))
        return turn

    def history(self) -> list[dict]:
        return list(self.turns)

    async def drain(self) -> None:
        while self._jobs:
            await asyncio.gather(*list(self._jobs), return_exceptions=True)

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
        status["current_errand"] = active["goal"] if active else None
        if self.pending_goals:
            status["queued_errands"] = list(self.pending_goals)
        return status

    async def _respond(self, user_turn: dict) -> None:
        async with self._lock:  # FIFO: one reply at a time, in order
            self.replying = True
            reply_id = str(uuid4())
            waiting = self.shares_task_model and self.manager.active_id is not None
            self.bus.emit("chat_thinking", self.name, skip_history=True, reply_id=reply_id,
                          waiting_on_errand=waiting)
            try:
                memories = []
                if self.store:
                    # Durable knowledge first; past chat snippets would otherwise crowd it
                    # out, since they often repeat the very facts they mention.
                    memories = await self.store.recall(
                        user_turn["text"], kinds=["fact", "preference", "episode", "skill"], k=4)
                    memories += await self.store.recall(user_turn["text"], kinds=["conversation"], k=2)
                voice = self.persona.voice_view() if self.persona else None
                earlier = [t for t in self.turns if t["id"] != user_turn["id"]]
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
                _log.info("chat_reply first_token_seconds=%s total_seconds=%s memories=%d task=%s",
                          metrics.get("first_token_seconds"), metrics.get("total_seconds"),
                          len(memories), bool(action.task))
                reply = reply or "Sorry, I lost my words for a moment. Could you say that again?"
                self._add_turn("kip", reply, reply_id=reply_id, action=action.model_dump(), metrics=metrics)
                await self._act(action, user_turn["text"], reply)
            except ModelError as exc:
                self._add_turn("kip", "I can't think clearly right now: my language model isn't answering. "
                               "Please check that it's running.", reply_id=reply_id, error=str(exc))
            finally:
                self.replying = False
                self.last_activity = time.monotonic()

    async def _act(self, action: ChatAction, said: str, reply: str) -> None:
        if self.store:
            about = self._mentioned(said + " " + reply)
            for fact in action.remember:
                if fact.strip():
                    await self.store.add("fact", fact, about=self._mentioned(fact) or about,
                                         importance=0.7, source="conversation")
            await self.store.add("conversation", f"They said: {said} | I replied: {reply}", about=about,
                                 importance=0.3, source="conversation")
        goal = " ".join(action.task.split())
        if goal:
            self._start_or_queue(goal)

    def _mentioned(self, text: str) -> list[str]:
        """Tag memories with people and objects mentioned by name or ID."""
        world = self.engine.snapshot()
        lowered = text.casefold()
        tags = []
        for key, item in {**world.get("people", {}), **world.get("objects", {})}.items():
            name = str(item.get("name", key)).casefold()
            if key.casefold() in lowered or name in lowered:
                tags.append(key)
        return tags

    # ── errands ─────────────────────────────────────────────────────────────
    def _start_or_queue(self, goal: str) -> None:
        try:
            self.manager.start(goal)
        except TaskBusy:
            self.pending_goals.append(goal)
            self._add_turn("kip", f"I'll do that right after my current errand: {goal}", source="narration")

    async def _start_next_when_free(self) -> None:
        while self.manager.active_id is not None:
            await asyncio.sleep(0.05)
        if self.pending_goals:
            self._start_or_queue(self.pending_goals.popleft())

    def _narrate(self, event: dict) -> None:
        """Speak about errands in Kip's voice using templates, never a model call."""
        kind, data = event["type"], event.get("data", {})
        task = data.get("task") or {}
        if kind == "task_started":
            self._add_turn("kip", f"On it: {task.get('goal', data.get('summary', ''))}", source="narration")
        elif kind in {"task_completed", "task_failed"}:
            reaction = self.persona.reaction(kind.removeprefix("task_")) if self.persona else None
            parts = [reaction] if reaction else []
            answer = task.get("answer") or {}
            if answer.get("answer"):
                parts.append(answer["answer"])
            elif data.get("summary"):
                parts.append(data["summary"])
            self._add_turn("kip", " ".join(parts), source="narration",
                           sources=answer.get("sources", []), outcome=kind.removeprefix("task_"))
            if self.pending_goals:
                self._spawn(self._start_next_when_free())
        elif kind == "agent_intention":
            self._add_turn("kip", f"I'm going to take care of something: {data.get('summary', '')}. "
                                  f"{data.get('reason', '')}".strip(), source="narration")
