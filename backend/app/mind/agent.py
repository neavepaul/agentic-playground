import asyncio
import logging
import time

from app.config import Settings
from app.events.bus import EventBus
from app.llm.base import LLMClient, structured
from app.memory.graph import BeliefGraph
from app.memory.store import PersistentMemory
from app.tasks.manager import TaskBusy, TaskManager
from app.world.engine import WorldEngine
from .models import GraphConsolidation, IntentionOrIdle

_CONSOLIDATION = """You are the memory system of a household robot.
Review your recent experiences and decide what to remember long-term.

recent_events: what just happened — observations, conversations, task outcomes.
current_beliefs: your existing belief graph (edges with confidence scores).
drive_helpfulness: how strongly you prioritise tracking people's needs (0=low, 1=high).
task_outcomes: completed tasks and whether they succeeded or failed.
current_time: simulated wall-clock time (HH:MM). Use it when reasoning about
  whether a located_in belief is plausible — people follow routines.

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
current_time: simulated wall-clock time (HH:MM). Use it to judge whether a
  person belief is plausible for the time of day before acting on it.

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

_DISTILLATION = """You are the long-term memory system of a household robot.
Review episodic history — patterns across many past tasks — and distil them
into durable semantic beliefs. This is a batch pass over time, not a reaction
to a single event. Look for recurring patterns strong enough to become stable,
high-confidence beliefs.

episode_history: recent task records — goals, outcomes, summaries.
people_tallies: for each person: how many times they were observed in each room,
  their last-seen room, known needs, and delivery_counts (object → n deliveries).
objects_history: where objects were last found.
current_beliefs: existing belief graph (do not duplicate high-confidence beliefs).
drive_helpfulness: 0–1 scalar — higher means prioritise needs-based beliefs.

Output GraphConsolidation with upsert_edges and remove_edges.

When to add a recurring_need edge:
  delivery_counts gives the exact number of confirmed deliveries.
  2 deliveries → confidence 0.65  |  3–4 → 0.8  |  5+ → 0.9
  reason must name the count: "Delivered medicine to paul 3 times."

When to add a located_in edge:
  Only when a person has a strong tally concentration — at least 3 total
  observations and ≥60% in one room.
  confidence: 60% → 0.5  |  75% → 0.65  |  90%+ → 0.8
  Do not emit a location belief based on a single observation.

Rules:
  - Do not re-emit edges already stored at confidence > 0.7 unless you have
    stronger evidence that raises them further.
  - Return empty lists when history is too thin to support a pattern.
  - Prefer fewer, stronger beliefs over many weak ones.
  - Use lowercase entity IDs from the data, never display names.
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
                 graph: BeliefGraph | None = None,
                 persistent: PersistentMemory | None = None) -> None:
        self._client = client
        self._manager = manager
        self._engine = engine
        self._bus = bus
        self._settings = settings
        self._graph = graph if graph is not None else BeliefGraph()
        self._persistent = persistent
        self._task: asyncio.Task | None = None
        # Track by monotonic sequence number — safe when the deque wraps at maxlen.
        self._last_seen_sequence: int = 0
        self._last_reflection: float = 0.0
        self._last_distilled_task_count: int = 0
        self._last_world_push: float = 0.0

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop(), name="agent-mind")

    async def close(self) -> None:
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    async def _loop(self) -> None:
        while True:
            try:
                now = time.monotonic()
                if now - self._last_world_push >= 10.0:
                    self._bus.emit("world_tick", world=self._engine.snapshot(), skip_history=True)
                    self._last_world_push = now
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

        await self._maybe_distill()

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
            "current_time": robot_status.get("time"),
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
        _TASK_OUTCOME_TYPES = {"task_completed", "task_failed", "task_cancelled"}
        task_outcomes = [
            {
                "goal": e["data"]["task"].get("goal", ""),
                "outcome": e["data"]["task"].get("status", ""),
                "summary": e["data"].get("summary", ""),
            }
            for e in new_events
            if e["type"] in _TASK_OUTCOME_TYPES
        ]
        # Exclude task outcomes (handled above), bookkeeping events, and self-referential
        # events that would cause consolidation to trigger itself.
        _EXCLUDE = {"task_updated", "world_reset", "agent_intention", "world_updated",
                    "world_tick", "beliefs_updated", "status_observed", "map_observed"}
        _EXCLUDE |= _TASK_OUTCOME_TYPES
        relevant = [
            {"type": e["type"], "summary": e.get("data", {}).get("summary", ""),
             "observation": e.get("data", {}).get("observation", "")}
            for e in new_events
            if e["type"] not in _EXCLUDE
        ]
        if not relevant and not task_outcomes:
            return

        context = {
            "recent_events": relevant[-12:],
            "current_beliefs": beliefs_snapshot,
            "drive_helpfulness": self._settings.drive_helpfulness,
            "task_outcomes": task_outcomes,
            "current_time": self._engine.get_status().get("time"),
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
            self._bus.emit("beliefs_updated", summary="Belief graph updated.")

    async def _maybe_distill(self) -> None:
        if self._persistent is None:
            return
        task_count = len(self._persistent.recent_tasks)
        new_since_last = task_count - self._last_distilled_task_count
        if new_since_last < self._settings.distillation_task_interval:
            return
        self._last_distilled_task_count = task_count
        await self._distill()

    async def _distill(self) -> None:
        """Batch distillation: promote recurring patterns from episodic history into semantic beliefs."""
        p = self._persistent
        people_tallies = {}
        for pid, person in p.people.items():
            total = sum(person.location_tally.values())
            people_tallies[pid] = {
                "name": person.name,
                "location_tally": person.location_tally,
                "total_observations": total,
                "last_seen_room": person.last_seen_room,
                "known_needs": person.known_needs,
                "delivery_counts": person.delivery_counts,
            }
        context = {
            "episode_history": [
                {"goal": t.goal, "outcome": t.outcome, "summary": t.summary}
                for t in p.recent_tasks[-20:]
            ],
            "people_tallies": people_tallies,
            "objects_history": {
                oid: {"last_seen_room": o.last_seen_room}
                for oid, o in p.objects.items()
            },
            "current_beliefs": self._graph.prompt(),
            "drive_helpfulness": self._settings.drive_helpfulness,
        }
        try:
            diff = await structured(self._client, GraphConsolidation, _DISTILLATION, context)
        except Exception:
            _log.exception("AgentMind distillation failed")
            return

        for edge in diff.upsert_edges:
            self._graph.upsert_edge(edge.subject, edge.relation, edge.target,
                                    edge.confidence, edge.reason)
            _log.info("distillation upsert: %s -[%s]-> %s (%.2f)",
                      edge.subject, edge.relation, edge.target, edge.confidence)
        for edge in diff.remove_edges:
            self._graph.remove_edge(edge.subject, edge.relation, edge.target)
            _log.info("distillation remove: %s -[%s]-> %s",
                      edge.subject, edge.relation, edge.target)

        if diff.upsert_edges or diff.remove_edges:
            self._graph.save()
            self._bus.emit("beliefs_updated", summary="Belief graph distilled.")
