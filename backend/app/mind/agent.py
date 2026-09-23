import asyncio
import logging
import time

from app.config import Settings
from app.events.bus import EventBus
from app.llm.base import LLMClient, structured
from app.memory.graph import BeliefGraph
from app.tasks.manager import TaskBusy, TaskManager
from app.world.engine import WorldEngine
from .models import GraphConsolidation, IntentionOrIdle
from .prompts import CONSOLIDATION, INTENTION_GENERATOR

_log = logging.getLogger("agentic_friend.mind")


class AgentMind:
    """Persistent autonomous loop that generates goals when the robot is idle.

    On each tick, if no user task is active:
    - If new events have arrived: consolidate them into the belief graph first.
    - If enough time has passed or new events arrived: ask the LLM for an intention.
    - If the intention's priority clears the threshold: submit it as a new task.
    """

    def __init__(self, client: LLMClient, manager: TaskManager, engine: WorldEngine,
                 bus: EventBus, settings: Settings,
                 graph: BeliefGraph | None = None) -> None:
        self._client = client
        self._manager = manager
        self._engine = engine
        self._bus = bus
        self._settings = settings
        self._graph = graph if graph is not None else BeliefGraph()
        self._task: asyncio.Task | None = None
        self._seen_events: int = 0
        self._last_reflection: float = 0.0

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop(), name="agent-mind")

    async def close(self) -> None:
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    async def _loop(self) -> None:
        while True:
            try:
                if not self._manager.active_id:
                    await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                _log.exception("AgentMind tick failed")
            await asyncio.sleep(self._settings.idle_tick_seconds)

    async def _tick(self) -> None:
        now = time.monotonic()
        history = list(self._bus.history)
        new_events = history[self._seen_events:]
        self._seen_events = len(history)

        since_last = now - self._last_reflection

        if new_events:
            await self._consolidate(new_events)

        if not new_events and since_last < self._settings.idle_reflection_interval:
            return

        self._last_reflection = now
        robot_status = self._engine.get_status()

        context = {
            "robot_status": robot_status,
            "beliefs": self._graph.prompt(),
            "drive_helpfulness": self._settings.drive_helpfulness,
            "recent_events": [
                {"type": e["type"], "summary": e.get("data", {}).get("summary", "")}
                for e in new_events[-8:]
                if e["type"] not in {"task_updated", "world_reset"}
            ],
            "idle_seconds": int(since_last),
        }

        result = await structured(self._client, IntentionOrIdle, INTENTION_GENERATOR, context)

        if result.intention is None:
            _log.debug("AgentMind: no intention, staying idle")
            return

        intention = result.intention
        if intention.priority < self._settings.intention_threshold:
            _log.debug("AgentMind: intention priority=%.2f below threshold %.2f, staying idle",
                       intention.priority, self._settings.intention_threshold)
            return

        _log.info("AgentMind: pursuing goal=%r priority=%.2f reason=%r",
                  intention.goal, intention.priority, intention.reason)
        self._bus.emit("agent_intention", summary=intention.goal,
                       reason=intention.reason, priority=intention.priority)
        try:
            self._manager.start(intention.goal)
        except TaskBusy:
            _log.warning("AgentMind: manager busy, deferring intention")

    async def _consolidate(self, new_events: list[dict]) -> None:
        """Ask the LLM to distil recent events into belief graph updates."""
        task_outcomes = [
            {"goal": e["data"].get("goal", ""), "outcome": e["data"].get("status", ""),
             "summary": e["data"].get("summary", "")}
            for e in new_events
            if e["type"] == "task_completed"
        ]
        relevant = [
            {"type": e["type"], "summary": e.get("data", {}).get("summary", ""),
             "observation": e.get("data", {}).get("observation", "")}
            for e in new_events
            if e["type"] not in {"task_updated", "world_reset", "agent_intention"}
        ]
        if not relevant and not task_outcomes:
            return

        context = {
            "recent_events": relevant[-12:],
            "current_beliefs": self._graph.prompt(),
            "drive_helpfulness": self._settings.drive_helpfulness,
            "task_outcomes": task_outcomes,
        }

        try:
            diff = await structured(self._client, GraphConsolidation, CONSOLIDATION, context)
        except Exception:
            _log.exception("AgentMind consolidation failed")
            return

        for edge in diff.upsert_edges:
            self._graph.upsert_edge(edge.subject, edge.relation, edge.target,
                                    edge.confidence, edge.reason)
            _log.debug("belief upsert: %s -[%s]-> %s (%.2f)",
                       edge.subject, edge.relation, edge.target, edge.confidence)
        for edge in diff.remove_edges:
            self._graph.remove_edge(edge.subject, edge.relation, edge.target)
            _log.debug("belief remove: %s -[%s]-> %s",
                       edge.subject, edge.relation, edge.target)

        if diff.upsert_edges or diff.remove_edges:
            self._graph.save()
