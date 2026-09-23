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

_CONSOLIDATION = """You are the memory system of a household robot.
Review your recent experiences and decide what to remember long-term.

recent_events: what just happened — observations, conversations, task outcomes.
current_beliefs: your existing belief graph (edges with confidence scores).
drive_helpfulness: how strongly you prioritise tracking people's needs (0=low, 1=high).
task_outcomes: completed tasks and whether they succeeded or failed.

For each belief to add or update, return an EdgeUpsert with:
  subject  — entity id (person or object, lowercase, no articles)
  relation — one of: located_in | needs | has | recurring_need
  target   — room or object or person id (lowercase, no articles)
  confidence:
    0.9  directly observed moments ago
    0.7  observed recently, likely still true
    0.5  inferred or somewhat stale
    0.3  secondhand or old
  reason   — one short sentence: what you observed

For beliefs to remove: only when a direct observation explicitly contradicts an existing edge.

Rules:
- Return empty lists when nothing meaningful changed. Conservative updates are better.
- A high drive_helpfulness means pay extra attention to needs and recurring_need edges.
- A failed task should lower (not remove) confidence on beliefs you acted on.
- A successful delivery is evidence of a recurring_need — record it.
- Use entity ids from the world, not display names.
- Beliefs show both stored_confidence and effective_confidence (time-decayed floor).
  Act on effective_confidence. If it is below 0.3 and you have no new evidence,
  you may lower the stored confidence to reflect genuine uncertainty — or leave it
  for the next time you can verify it in person.
"""

_INTENTION_GENERATOR = """You are the autonomous mind of a household robot.
You are idle — no user has given you a task. Decide whether anything is worth doing now.

Output JSON matching the schema. Return intention=null to remain idle.
Only return an intention when you have genuine grounding in memory or recent events:
a person who may need help, a stale belief worth verifying, or something unfinished.
Do not invent busywork. Null is the correct choice most of the time.

robot_status: your current location and what you are carrying.
beliefs: your long-term belief graph — where people and objects were last seen, recurring needs,
         with stored_confidence and effective_confidence (time-decayed).
drive_helpfulness: a 0-1 scalar; higher means more motivated to help people proactively.
recent_events: what just happened that may be relevant.
idle_seconds: how long since your last task completed.

Priority guide (be conservative):
  0.9  someone explicitly needs something you remember and can address
  0.7  a likely unfinished commitment or a clear recurring need
  0.5  a low-cost belief check that could prevent a future failure
  <0.5 not worth interrupting your rest

goal should be a natural-language description a person would understand.
reason should explain the specific memory or event that motivated this.
conditions is optional — leave empty and the Coordinator will derive them from goal.
If you set conditions, use lowercase entity IDs from beliefs without articles.
"""

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
        # Track by monotonic sequence number — safe when the deque wraps at maxlen.
        self._last_seen_sequence: int = 0
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
        new_events = [e for e in history if e["sequence"] > self._last_seen_sequence]
        if new_events:
            self._last_seen_sequence = new_events[-1]["sequence"]

        since_last = now - self._last_reflection

        # Snapshot before consolidation so the LLM sees the pre-update state as context.
        beliefs_snapshot = self._graph.prompt()
        if new_events:
            await self._consolidate(new_events, beliefs_snapshot)

        if not new_events and since_last < self._settings.idle_reflection_interval:
            return

        self._last_reflection = now
        robot_status = self._engine.get_status()

        context = {
            "robot_status": robot_status,
            "beliefs": self._graph.prompt(),  # post-consolidation state
            "drive_helpfulness": self._settings.drive_helpfulness,
            "recent_events": [
                {"type": e["type"], "summary": e.get("data", {}).get("summary", "")}
                for e in new_events[-8:]
                if e["type"] not in {"task_updated", "world_reset"}
            ],
            "idle_seconds": int(since_last),
        }

        result = await structured(self._client, IntentionOrIdle, _INTENTION_GENERATOR, context)

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

    async def _consolidate(self, new_events: list[dict], beliefs_snapshot: dict) -> None:
        """Ask the LLM to distil recent events into belief graph updates."""
        task_outcomes = [
            {
                "goal": e["data"]["task"].get("goal", ""),
                "outcome": e["data"]["task"].get("status", ""),
                "summary": e["data"].get("summary", ""),
            }
            for e in new_events
            if e["type"] == "task_completed"
        ]
        # Exclude task_completed — already captured in task_outcomes above.
        relevant = [
            {"type": e["type"], "summary": e.get("data", {}).get("summary", ""),
             "observation": e.get("data", {}).get("observation", "")}
            for e in new_events
            if e["type"] not in {"task_updated", "world_reset", "agent_intention", "task_completed"}
        ]
        if not relevant and not task_outcomes:
            return

        context = {
            "recent_events": relevant[-12:],
            "current_beliefs": beliefs_snapshot,
            "drive_helpfulness": self._settings.drive_helpfulness,
            "task_outcomes": task_outcomes,
        }

        try:
            diff = await structured(self._client, GraphConsolidation, _CONSOLIDATION, context)
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
