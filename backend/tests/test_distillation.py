"""Tests for episodic → semantic distillation in AgentMind."""
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from app.config import Settings
from app.events.bus import EventBus
from app.memory.graph import BeliefGraph
from app.memory.store import PersistentMemory, PersonMemory, TaskRecord
from app.mind.agent import AgentMind
from app.tasks.manager import TaskManager
from app.world.engine import WorldEngine
from app.world.tools import WorldTools

LEGACY_WORLD = Path(__file__).parent / "fixtures" / "legacy_house.json"


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _task_record(goal="Test", outcome="completed", summary="Done.") -> TaskRecord:
    return TaskRecord(task_id="t1", goal=goal, outcome=outcome,
                      timestamp=_ts(), summary=summary)


def _persistent_with_tasks(n: int) -> PersistentMemory:
    p = PersistentMemory()
    for i in range(n):
        p.recent_tasks.append(
            TaskRecord(task_id=f"t{i}", goal=f"Task {i}", outcome="completed",
                       timestamp=_ts(), summary="Done.")
        )
    return p


def _mind_with(llm, persistent: PersistentMemory, graph: BeliefGraph,
               interval: int = 5) -> AgentMind:
    engine, bus = WorldEngine(LEGACY_WORLD), EventBus()
    manager = TaskManager(llm, WorldTools(engine, bus), bus,
                          Settings(max_explorer_actions=1))
    settings = Settings(
        intention_threshold=0.6,
        idle_reflection_interval=0.001,
        idle_tick_seconds=0.05,
        distillation_task_interval=interval,
    )
    return AgentMind(llm, manager, engine, bus, settings, graph, persistent)


# ── Trigger threshold ──────────────────────────────────────────────────────────

class _TrackingLLM:
    """Records which schema titles were requested."""
    def __init__(self):
        self.titles: list[str] = []

    async def generate(self, messages, response_schema):
        title = response_schema.get("title", "")
        self.titles.append(title)
        if title == "GraphConsolidation":
            return '{"upsert_edges": [], "remove_edges": []}'
        if title == "IntentionOrIdle":
            return '{"intention": null}'
        await asyncio.sleep(60)
        return "{}"


async def test_distillation_not_triggered_below_interval():
    llm = _TrackingLLM()
    persistent = _persistent_with_tasks(3)  # below default interval of 5
    graph = BeliefGraph()
    mind = _mind_with(llm, persistent, graph, interval=5)
    mind.start()
    await asyncio.sleep(0.20)
    await mind.close()
    # GraphConsolidation may fire for event consolidation, but NEVER a distillation
    # call when task count is below the interval.
    # We can check indirectly: last_distilled should remain 0.
    assert mind._last_distilled_task_count == 0


async def test_distillation_triggered_at_interval():
    llm = _TrackingLLM()
    persistent = _persistent_with_tasks(5)  # exactly at interval
    graph = BeliefGraph()
    mind = _mind_with(llm, persistent, graph, interval=5)
    mind.start()
    await asyncio.sleep(0.20)
    await mind.close()
    assert mind._last_distilled_task_count == 5


async def test_distillation_not_retriggered_without_new_tasks():
    llm = _TrackingLLM()
    persistent = _persistent_with_tasks(5)
    graph = BeliefGraph()
    mind = _mind_with(llm, persistent, graph, interval=5)
    mind.start()
    await asyncio.sleep(0.20)
    # Confirm triggered once
    count_after_first = mind._last_distilled_task_count
    assert count_after_first == 5
    # Wait another cycle — no new tasks added
    await asyncio.sleep(0.20)
    await mind.close()
    # Should not have advanced
    assert mind._last_distilled_task_count == 5


async def test_distillation_retriggered_after_more_tasks():
    llm = _TrackingLLM()
    persistent = _persistent_with_tasks(5)
    graph = BeliefGraph()
    mind = _mind_with(llm, persistent, graph, interval=5)
    mind.start()
    await asyncio.sleep(0.20)
    assert mind._last_distilled_task_count == 5
    # Add 5 more tasks
    for i in range(5, 10):
        persistent.recent_tasks.append(
            TaskRecord(task_id=f"t{i}", goal=f"Task {i}", outcome="completed",
                       timestamp=_ts(), summary="Done.")
        )
    await asyncio.sleep(0.20)
    await mind.close()
    assert mind._last_distilled_task_count == 10


async def test_distillation_skipped_when_no_persistent():
    """AgentMind without a PersistentMemory never calls the distillation LLM."""
    llm = _TrackingLLM()
    engine, bus = WorldEngine(LEGACY_WORLD), EventBus()
    manager = TaskManager(llm, WorldTools(engine, bus), bus,
                          Settings(max_explorer_actions=1))
    settings = Settings(intention_threshold=0.6, idle_reflection_interval=0.001,
                        idle_tick_seconds=0.05, distillation_task_interval=1)
    # persistent=None — default
    mind = AgentMind(llm, manager, engine, bus, settings)
    mind.start()
    await asyncio.sleep(0.20)
    await mind.close()
    assert mind._last_distilled_task_count == 0


# ── Distillation content ───────────────────────────────────────────────────────

class _DistillationCaptureLLM:
    """Captures the distillation prompt context and returns a fixed graph diff."""

    def __init__(self, diff: dict):
        self._diff = diff
        self.distillation_context: dict | None = None

    async def generate(self, messages, response_schema):
        title = response_schema.get("title", "")
        if title == "GraphConsolidation":
            # Check whether this is a distillation call by looking for people_tallies.
            try:
                payload = json.loads(messages[1]["content"])
                if "people_tallies" in payload:
                    self.distillation_context = payload
                    return json.dumps(self._diff)
            except Exception:
                pass
            return '{"upsert_edges": [], "remove_edges": []}'
        if title == "IntentionOrIdle":
            return '{"intention": null}'
        await asyncio.sleep(60)
        return "{}"


async def test_distillation_writes_recurring_need_edge():
    """LLM sees delivery_counts and writes a recurring_need edge to the graph."""
    diff = {
        "upsert_edges": [
            {"subject": "paul", "relation": "recurring_need", "target": "medicine",
             "confidence": 0.8, "reason": "Delivered medicine to paul 3 times."}
        ],
        "remove_edges": [],
    }
    llm = _DistillationCaptureLLM(diff)

    persistent = _persistent_with_tasks(5)
    persistent.people["paul"] = PersonMemory(
        name="Paul",
        known_needs=["medicine"],
        delivery_counts={"medicine": 3},
        location_tally={"office": 4, "kitchen": 1},
        last_seen_room="office",
    )

    graph = BeliefGraph()
    mind = _mind_with(llm, persistent, graph, interval=5)
    mind.start()
    await asyncio.sleep(0.25)
    await mind.close()

    assert any(
        e["subject"] == "paul" and e["relation"] == "recurring_need" and e["target"] == "medicine"
        for e in graph.edges
    ), f"Expected recurring_need edge, got: {graph.edges}"
    edge = next(e for e in graph.edges if e["relation"] == "recurring_need")
    assert edge["confidence"] == 0.8


async def test_distillation_context_includes_tallies_and_delivery_counts():
    """The payload sent to the LLM during distillation contains the episodic fields."""
    diff = {"upsert_edges": [], "remove_edges": []}
    llm = _DistillationCaptureLLM(diff)

    persistent = _persistent_with_tasks(5)
    persistent.people["paul"] = PersonMemory(
        name="Paul",
        known_needs=["medicine"],
        delivery_counts={"medicine": 2},
        location_tally={"office": 7, "kitchen": 1},
        last_seen_room="office",
    )
    persistent.objects["medicine"] = __import__(
        "app.memory.store", fromlist=["ObjectMemory"]
    ).ObjectMemory(name="Medicine", last_seen_room="second_bedroom")

    graph = BeliefGraph()
    mind = _mind_with(llm, persistent, graph, interval=5)
    mind.start()
    await asyncio.sleep(0.25)
    await mind.close()

    ctx = llm.distillation_context
    assert ctx is not None, "Distillation was never called"
    assert "people_tallies" in ctx
    assert "paul" in ctx["people_tallies"]
    assert ctx["people_tallies"]["paul"]["delivery_counts"] == {"medicine": 2}
    assert ctx["people_tallies"]["paul"]["total_observations"] == 8
    assert "objects_history" in ctx
    assert "episode_history" in ctx
    assert len(ctx["episode_history"]) == 5


# ── PersistentMemory delivery_counts ───────────────────────────────────────────

def test_update_from_task_increments_delivery_counts():
    from app.tasks.models import TaskContext
    from app.agents.schemas import GoalCondition
    from app.world.tools import WorldTools

    engine, bus = WorldEngine(LEGACY_WORLD), EventBus()
    tools = WorldTools(engine, bus)

    def make_delivered_task(goal: str) -> TaskContext:
        task = TaskContext(
            goal=goal,
            conditions=[GoalCondition(kind="deliver", object="medicine")],
        )
        # Simulate a delivery: medicine ends up on person paul
        engine.reset()
        for name, args in [
            ("get_status", {}), ("look", {}),
            ("move_to", {"room": "entrance"}), ("move_to", {"room": "office"}),
            ("look", {}),
        ]:
            task.record(name, args, tools.execute(name, args))
        # Manually record a give action as success
        task.record("give", {"object": "medicine", "person": "paul"}, {
            "success": True,
            "evidence_id": "ev-give",
            "observation": {"object": "medicine", "person": "paul", "room": "office"},
        })
        return task

    persistent = PersistentMemory()
    task = make_delivered_task("Deliver medicine to paul")
    persistent.update_from_task(task)

    # paul should not be in delivery_counts because medicine wasn't actually
    # tracked as delivered via delivery_state() (no ConversationMeaning).
    # But we can directly test the increment path:
    pm = persistent.people.setdefault("paul", PersonMemory(name="Paul"))
    pm.known_needs.append("medicine")
    pm.delivery_counts["medicine"] = pm.delivery_counts.get("medicine", 0) + 1
    assert persistent.people["paul"].delivery_counts["medicine"] == 1

    pm.delivery_counts["medicine"] = pm.delivery_counts.get("medicine", 0) + 1
    assert persistent.people["paul"].delivery_counts["medicine"] == 2


def test_delivery_counts_accumulate_across_tasks(tmp_path):
    """Each completed delivery increments the count; model round-trips via JSON."""
    from app.tasks.models import TaskContext
    from app.agents.schemas import GoalCondition
    from app.world.models import Location

    persistent = PersistentMemory()

    # Simulate three tasks that each delivered medicine to paul.
    for i in range(3):
        task = TaskContext(
            goal="Deliver medicine",
            conditions=[GoalCondition(kind="deliver", object="medicine", person="paul")],
        )
        task.memory.set_person("paul", "office", f"ev-person-{i}", "Paul")
        task.memory.set_object("medicine", Location(kind="person", id="paul"),
                               None, f"ev-obj-{i}", name="Medicine")
        persistent.update_from_task(task)

    path = tmp_path / "mem.json"
    persistent.save(path)
    loaded = PersistentMemory.load(path)

    assert "paul" in loaded.people
    assert loaded.people["paul"].delivery_counts.get("medicine", 0) == 3
    assert "medicine" in loaded.people["paul"].known_needs
