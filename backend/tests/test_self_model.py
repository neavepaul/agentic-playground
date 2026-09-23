"""Tests for the robot self-model — performance history extracted from task execution."""
import json
from pathlib import Path

import pytest

from app.agents.schemas import GoalCondition
from app.config import Settings
from app.events.bus import EventBus
from app.memory.store import PersistentMemory, PersonMemory
from app.tasks.manager import TaskManager
from app.tasks.models import TaskContext
from app.world.engine import WorldEngine
from app.world.models import Location
from app.world.tools import WorldTools

LEGACY_WORLD = Path(__file__).parent / "fixtures" / "legacy_house.json"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _tools():
    engine, bus = WorldEngine(LEGACY_WORLD), EventBus()
    return WorldTools(engine, bus), engine


def _failed_task(tools, actions: list[tuple]) -> TaskContext:
    task = TaskContext(goal="Test task")
    task.status = "failed"
    for name, args in actions:
        result = tools.execute(name, args)
        task.record(name, args, result)
    return task


def _completed_task(tools, actions: list[tuple]) -> TaskContext:
    task = TaskContext(goal="Test task")
    task.status = "completed"
    for name, args in actions:
        result = tools.execute(name, args)
        task.record(name, args, result)
    return task


# ── Room visit tracking ─────────────────────────────────────────────────────────

def test_self_model_counts_successful_move_to_rooms():
    tools, _ = _tools()
    task = _failed_task(tools, [
        ("get_status", {}),
        ("move_to", {"room": "kitchen"}),
        ("move_to", {"room": "hall"}),
        ("move_to", {"room": "kitchen"}),
    ])
    p = PersistentMemory()
    p.update_from_task(task)
    assert p.self_model.room_visit_counts["kitchen"] == 2
    assert p.self_model.room_visit_counts["hall"] == 1


def test_self_model_room_visits_accumulate_across_tasks():
    p = PersistentMemory()
    for _ in range(3):
        fresh_tools, _ = _tools()  # fresh engine each task — robot starts in hall
        task = _failed_task(fresh_tools, [
            ("get_status", {}),
            ("move_to", {"room": "kitchen"}),
        ])
        p.update_from_task(task)
    assert p.self_model.room_visit_counts["kitchen"] == 3


def test_self_model_failed_move_not_counted():
    tools, _ = _tools()
    task = TaskContext(goal="Test")
    task.status = "failed"
    # Try to move to a non-adjacent room — will fail
    result = tools.execute("move_to", {"room": "office"})  # not connected to hall
    task.record("move_to", {"room": "office"}, result)
    p = PersistentMemory()
    p.update_from_task(task)
    assert "office" not in p.self_model.room_visit_counts


# ── Tool failure tracking ───────────────────────────────────────────────────────

def test_self_model_tracks_tool_failures():
    tools, _ = _tools()
    task = TaskContext(goal="Test")
    task.status = "failed"
    # move_to non-adjacent room twice
    for _ in range(2):
        result = tools.execute("move_to", {"room": "office"})
        task.record("move_to", {"room": "office"}, result)
    p = PersistentMemory()
    p.update_from_task(task)
    assert p.self_model.tool_failure_counts.get("move_to", 0) == 2


def test_self_model_get_status_not_counted_as_failure():
    """get_status and get_map are bookkeeping calls, not meaningful failures."""
    tools, _ = _tools()
    task = TaskContext(goal="Test")
    task.status = "failed"
    # Manually inject a fake get_status failure
    task.action_history.append({
        "tool": "get_status", "arguments": {}, "success": False,
        "evidence_id": "ev1", "observation": {"error": "some error"},
    })
    p = PersistentMemory()
    p.update_from_task(task)
    assert "get_status" not in p.self_model.tool_failure_counts


# ── Entity search failure tracking ─────────────────────────────────────────────

def test_self_model_records_entity_search_failure_for_failed_task():
    tools, engine = _tools()
    # Navigate around but don't find the laptop (it's somewhere we didn't look)
    task = TaskContext(
        goal="Find laptop",
        conditions=[GoalCondition(kind="deliver", object="laptop")],
    )
    task.status = "failed"
    task.record("get_status", {}, tools.execute("get_status", {}))
    task.record("move_to", {"room": "kitchen"}, tools.execute("move_to", {"room": "kitchen"}))
    task.record("look", {}, tools.execute("look", {}))
    # laptop not found in kitchen — last_observed_location is None

    p = PersistentMemory()
    p.update_from_task(task)
    assert p.self_model.entity_search_failures.get("laptop", 0) == 1


def test_self_model_no_entity_failure_for_completed_task():
    tools, engine = _tools()
    task = TaskContext(
        goal="Deliver medicine",
        conditions=[GoalCondition(kind="deliver", object="medicine", person="paul")],
    )
    task.status = "completed"
    # Simulate medicine found and delivered to paul
    task.memory.set_object("medicine", Location(kind="person", id="paul"),
                           None, "ev-obj", name="Medicine")
    task.memory.set_person("paul", "office", "ev-paul", "Paul")

    p = PersistentMemory()
    p.update_from_task(task)
    # Completed task — no entity search failures recorded
    assert p.self_model.entity_search_failures.get("medicine", 0) == 0
    assert p.self_model.entity_search_failures.get("paul", 0) == 0


def test_self_model_records_missing_recipient_failure():
    tools, _ = _tools()
    task = TaskContext(
        goal="Deliver medicine to paul",
        conditions=[GoalCondition(kind="deliver", object="medicine", person="paul")],
    )
    task.status = "failed"
    # Object found but recipient never located
    task.memory.set_object("medicine", Location(kind="room", id="second_bedroom"),
                           "second_bedroom", "ev-obj", name="Medicine")
    # paul never added to task memory → recipient_last_seen will be None

    p = PersistentMemory()
    p.update_from_task(task)
    assert p.self_model.entity_search_failures.get("paul", 0) == 1


# ── self_model_prompt ───────────────────────────────────────────────────────────

def test_self_model_prompt_empty_with_no_tasks():
    p = PersistentMemory()
    assert p.self_model_prompt() == {}


def test_self_model_prompt_structure():
    tools, _ = _tools()
    task = _failed_task(tools, [
        ("get_status", {}),
        ("move_to", {"room": "kitchen"}),
        ("move_to", {"room": "hall"}),
    ])
    p = PersistentMemory()
    p.update_from_task(task)

    prompt = p.self_model_prompt()
    assert "task_summary" in prompt
    assert prompt["task_summary"]["total"] == 1
    assert prompt["task_summary"]["failed"] == 1
    assert "rooms_visited_by_frequency" in prompt
    assert "kitchen" in prompt["rooms_visited_by_frequency"]
    assert "tool_failures" in prompt
    assert "entity_search_failures" in prompt


def test_self_model_prompt_rooms_ordered_by_frequency():
    tools, _ = _tools()
    p = PersistentMemory()
    p.self_model.room_visit_counts = {"office": 5, "kitchen": 2, "entrance": 8}
    p.recent_tasks.append(__import__(
        "app.memory.store", fromlist=["TaskRecord"]
    ).TaskRecord(task_id="t1", goal="x", outcome="completed",
                 timestamp="2024-01-01T00:00:00+00:00", summary="done"))
    prompt = p.self_model_prompt()
    rooms = prompt["rooms_visited_by_frequency"]
    assert rooms[0] == "entrance"
    assert rooms[1] == "office"
    assert rooms[2] == "kitchen"


def test_self_model_prompt_caps_top_5_failures():
    p = PersistentMemory()
    for i in range(8):
        p.self_model.tool_failure_counts[f"tool_{i}"] = i
    p.recent_tasks.append(__import__(
        "app.memory.store", fromlist=["TaskRecord"]
    ).TaskRecord(task_id="t1", goal="x", outcome="failed",
                 timestamp="2024-01-01T00:00:00+00:00", summary="failed"))
    prompt = p.self_model_prompt()
    assert len(prompt["tool_failures"]) <= 5


# ── Persistence ─────────────────────────────────────────────────────────────────

def test_self_model_persists_across_save_load(tmp_path):
    tools, _ = _tools()
    task = _failed_task(tools, [
        ("get_status", {}),
        ("move_to", {"room": "kitchen"}),
    ])
    p = PersistentMemory()
    p.update_from_task(task)

    path = tmp_path / "mem.json"
    p.save(path)
    loaded = PersistentMemory.load(path)

    assert loaded.self_model.room_visit_counts == p.self_model.room_visit_counts
    assert loaded.self_model.tool_failure_counts == p.self_model.tool_failure_counts


# ── Integration: injected into compact() ───────────────────────────────────────

async def test_self_model_injected_into_compact_when_persistent_wired():
    from tests.fakes import WaitingLLM

    engine_main, bus_main = WorldEngine(LEGACY_WORLD), EventBus()
    persistent = PersistentMemory()
    persistent.self_model.room_visit_counts["study"] = 4
    persistent.self_model.entity_search_failures["alice"] = 3
    from app.memory.store import TaskRecord
    persistent.recent_tasks.append(
        TaskRecord(task_id="t0", goal="g", outcome="failed",
                   timestamp="2024-01-01T00:00:00+00:00", summary="failed")
    )

    manager = TaskManager(
        WaitingLLM(), WorldTools(engine_main, bus_main), bus_main,
        Settings(), persistent
    )
    task = manager.start("Find alice")
    payload = task.compact()

    assert "self_model" in payload
    sm = payload["self_model"]
    assert sm["task_summary"]["failed"] == 1
    assert "study" in sm["rooms_visited_by_frequency"]
    assert sm["entity_search_failures"].get("alice") == 3

    await manager.close()
