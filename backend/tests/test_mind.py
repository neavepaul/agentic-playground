"""Tests for the autonomous idle loop (AgentMind)."""
import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.config import Settings
from app.events.bus import EventBus
from app.memory.graph import BeliefGraph, _effective
from app.mind.agent import AgentMind
from app.mind.models import GraphConsolidation, IntentionOrIdle, Intention
from app.tasks.manager import TaskBusy, TaskManager
from app.world.engine import WorldEngine
from app.world.tools import WorldTools
from tests.fakes import ScriptedLLM

LEGACY_WORLD = Path(__file__).parent / "fixtures" / "legacy_house.json"


def _mind(llm, *, threshold=0.6, reflection_interval=0.001, tick_seconds=0.05):
    """Build an AgentMind backed by LEGACY_WORLD with fast tick settings for tests."""
    engine, bus = WorldEngine(LEGACY_WORLD), EventBus()
    manager = TaskManager(llm, WorldTools(engine, bus), bus,
                          Settings(max_explorer_actions=1))
    settings = Settings(
        intention_threshold=threshold,
        idle_reflection_interval=reflection_interval,
        idle_tick_seconds=tick_seconds,
    )
    return AgentMind(llm, manager, engine, bus, settings), manager, bus


class _IntentionLLM:
    """Returns a fixed IntentionOrIdle JSON for the first call, then stalls."""
    def __init__(self, intention: dict | None):
        self._response = json.dumps({"intention": intention})
        self._calls = 0

    async def generate(self, messages, response_schema):
        self._calls += 1
        title = response_schema.get("title", "")
        if title == "IntentionOrIdle":
            return self._response
        if title == "GraphConsolidation":
            return '{"upsert_edges": [], "remove_edges": []}'
        # Stall any subsequent calls (task execution) so the test can assert before they run.
        await asyncio.sleep(60)
        return "{}"

    @property
    def calls(self):
        return self._calls


def test_belief_confidence_decays_over_time():
    """effective_confidence falls as time passes; stored_confidence is unchanged."""
    fresh = datetime.now(timezone.utc).isoformat()
    stale = (datetime.now(timezone.utc) - timedelta(hours=8)).isoformat()

    # Fresh observation: effective should be close to stored.
    assert _effective(0.9, "located_in", fresh) > 0.85

    # 8 hours old with 4h half-life → two half-lives → 0.9 × 0.25 ≈ 0.225
    effective_stale = _effective(0.9, "located_in", stale)
    assert effective_stale < 0.25

    # recurring_need has a 7-day half-life — 8 hours barely moves it.
    assert _effective(0.9, "recurring_need", stale) > 0.85


def test_belief_graph_prompt_shows_both_confidences():
    graph = BeliefGraph()
    stale_ts = (datetime.now(timezone.utc) - timedelta(hours=8)).isoformat()
    graph.edges.append({
        "subject": "paul", "relation": "located_in", "target": "office",
        "confidence": 0.9, "last_observed": stale_ts, "observation_count": 1,
    })
    beliefs = graph.prompt()["beliefs"]
    assert len(beliefs) == 1
    b = beliefs[0]
    assert b["stored_confidence"] == 0.9
    assert b["effective_confidence"] < 0.25  # two half-lives elapsed


async def test_mind_stays_idle_when_null_intention():
    llm = _IntentionLLM(None)
    mind, manager, _ = _mind(llm, reflection_interval=0.001)
    mind.start()
    await asyncio.sleep(0.12)
    await mind.close()
    assert manager.active_id is None


async def test_mind_stays_idle_when_priority_below_threshold():
    low_priority = {"goal": "Wander around", "reason": "bored", "priority": 0.3, "conditions": []}
    llm = _IntentionLLM(low_priority)
    mind, manager, _ = _mind(llm, threshold=0.6, reflection_interval=0.001)
    mind.start()
    await asyncio.sleep(0.12)
    await mind.close()
    assert manager.active_id is None


async def test_mind_starts_task_when_priority_clears_threshold():
    high_priority = {"goal": "Check if paul needs anything",
                     "reason": "Paul often needs help", "priority": 0.75, "conditions": []}
    llm = _IntentionLLM(high_priority)
    mind, manager, _ = _mind(llm, threshold=0.6, reflection_interval=0.001)
    mind.start()
    # Give the tick time to fire and call manager.start.
    await asyncio.sleep(0.15)
    await mind.close()
    assert manager.active_id is not None


async def test_mind_does_not_tick_when_task_running():
    """While a task is active, the mind skips the tick entirely."""
    high_priority = {"goal": "Do something", "reason": "just because", "priority": 0.9, "conditions": []}
    llm = _IntentionLLM(high_priority)
    mind, manager, _ = _mind(llm, reflection_interval=0.001)
    # Manually mark a task as active so the mind sees it.
    from app.tasks.models import TaskContext
    fake_task = TaskContext(goal="Existing task")
    manager.tasks[fake_task.id] = fake_task
    manager.active_id = fake_task.id
    mind.start()
    await asyncio.sleep(0.15)
    await mind.close()
    # LLM should never have been called for intention generation.
    assert llm.calls == 0


async def test_mind_emits_agent_intention_event():
    """A pursued intention is broadcast on the EventBus."""
    high_priority = {"goal": "Deliver medicine to paul",
                     "reason": "Paul's recurring need", "priority": 0.8, "conditions": []}
    llm = _IntentionLLM(high_priority)
    mind, manager, bus = _mind(llm, threshold=0.6, reflection_interval=0.001)
    mind.start()
    await asyncio.sleep(0.15)
    await mind.close()
    events = [e for e in bus.history if e["type"] == "agent_intention"]
    assert len(events) >= 1
    assert events[0]["data"]["summary"] == "Deliver medicine to paul"


async def test_mind_consolidates_events_into_belief_graph():
    """When new events arrive, the mind consolidates them into its belief graph."""

    class _ConsolidatingLLM:
        """Returns a graph diff for consolidation, stays idle on intention."""
        async def generate(self, messages, response_schema):
            title = response_schema.get("title", "")
            if title == "GraphConsolidation":
                return json.dumps({"upsert_edges": [
                    {"subject": "paul", "relation": "located_in", "target": "office",
                     "confidence": 0.8, "reason": "Saw paul in office."}
                ], "remove_edges": []})
            if title == "IntentionOrIdle":
                return '{"intention": null}'
            await asyncio.sleep(60)
            return "{}"

    graph = BeliefGraph()
    engine, bus = WorldEngine(LEGACY_WORLD), EventBus()
    manager = TaskManager(_ConsolidatingLLM(), WorldTools(engine, bus), bus,
                          Settings(max_explorer_actions=1))
    settings = Settings(intention_threshold=0.6, idle_reflection_interval=0.001,
                        idle_tick_seconds=0.05)
    mind = AgentMind(_ConsolidatingLLM(), manager, engine, bus, settings, graph)

    # Emit a real event so the consolidation branch fires.
    bus.emit("look_result", summary="Saw paul in the office.")

    mind.start()
    await asyncio.sleep(0.15)
    await mind.close()

    beliefs = graph.edges
    assert any(
        e["subject"] == "paul" and e["relation"] == "located_in" and e["target"] == "office"
        for e in beliefs
    ), f"Expected paul→located_in→office in graph, got: {beliefs}"


async def test_mind_handles_task_busy_gracefully():
    """If manager is already busy when the mind tries to start a task, no exception surfaces."""
    high_priority = {"goal": "Do something", "reason": "always", "priority": 0.9, "conditions": []}
    llm = _IntentionLLM(high_priority)
    mind, manager, _ = _mind(llm, reflection_interval=0.001)
    # Start a real task first so TaskBusy is raised on the second attempt.
    from app.tasks.models import TaskContext
    fake_task = TaskContext(goal="Blocking task")
    fake_task.status = "running"
    manager.tasks[fake_task.id] = fake_task
    manager.active_id = fake_task.id
    # Unblock after one tick so the mind tries to start on the second tick.
    async def unblock():
        await asyncio.sleep(0.08)
        manager.active_id = None
    asyncio.create_task(unblock())
    mind.start()
    await asyncio.sleep(0.20)
    await mind.close()
    # No exception should have propagated (mind continues running).
    assert mind._task is None or mind._task.cancelled()
