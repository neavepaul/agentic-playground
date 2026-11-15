"""Tests for Kip's self-model — performance history extracted from task execution."""
from app.config import Settings
from app.events.bus import EventBus
from app.memory.store import PersistentMemory, TaskRecord
from app.tasks.manager import TaskManager
from app.tasks.models import TaskContext


# ── Helpers ────────────────────────────────────────────────────────────────────

def _task(status: str, actions: list[tuple]) -> TaskContext:
    """A finished task whose tool calls had the given (tool, result) outcomes."""
    task = TaskContext(goal="Test task")
    task.status = status
    for index, (name, result) in enumerate(actions):
        task.record(name, {}, {"evidence_id": f"ev{index}", **result})
    return task


OK = {"success": True, "observation": {}}
FAILED = {"success": False, "error": "offline"}
REFUSED = {"success": False, "error": "Not permitted: you asked me not to run Python.", "refused": True}


def _record(outcome: str) -> TaskRecord:
    return TaskRecord(task_id="t1", goal="x", outcome=outcome, timestamp="2024-01-01T00:00:00+00:00",
                      summary=outcome)


# ── Tool failure tracking ───────────────────────────────────────────────────────

def test_self_model_tracks_tool_failures():
    task = _task("failed", [("read_webpage", FAILED), ("read_webpage", FAILED), ("web_search", OK)])
    p = PersistentMemory()
    p.update_from_task(task)
    assert p.self_model.tool_failure_counts == {"read_webpage": 2}


def test_self_model_failures_accumulate_across_tasks():
    p = PersistentMemory()
    for _ in range(3):
        p.update_from_task(_task("completed", [("run_python", FAILED), ("run_python", OK)]))
    assert p.self_model.tool_failure_counts["run_python"] == 3
    assert p.task_total == 3


def test_self_model_refusals_are_not_counted_as_failures():
    """A call the person ruled out was never attempted: it says nothing about the tool."""
    p = PersistentMemory()
    p.update_from_task(_task("failed", [("run_python", REFUSED)]))
    assert "run_python" not in p.self_model.tool_failure_counts


def test_every_finished_task_is_recorded_whatever_the_outcome():
    p = PersistentMemory()
    for outcome in ("completed", "failed", "cancelled"):
        p.update_from_task(_task(outcome, []))
    assert [t.outcome for t in p.recent_tasks] == ["completed", "failed", "cancelled"]


# ── self_model_prompt ───────────────────────────────────────────────────────────

def test_self_model_prompt_empty_with_no_tasks():
    p = PersistentMemory()
    assert p.self_model_prompt() == {}


def test_self_model_prompt_structure():
    p = PersistentMemory()
    p.update_from_task(_task("failed", [("web_search", OK), ("read_webpage", FAILED)]))

    prompt = p.self_model_prompt()
    assert "task_summary" in prompt
    assert prompt["task_summary"]["total"] == 1
    assert prompt["task_summary"]["failed"] == 1
    assert prompt["tool_failures"] == {"read_webpage": 1}


def test_self_model_prompt_orders_failures_by_frequency():
    p = PersistentMemory()
    p.self_model.tool_failure_counts = {"web_search": 2, "run_python": 5, "read_webpage": 8}
    p.recent_tasks.append(_record("completed"))
    assert list(p.self_model_prompt()["tool_failures"]) == ["read_webpage", "run_python", "web_search"]


def test_self_model_prompt_caps_top_5_failures():
    p = PersistentMemory()
    for i in range(8):
        p.self_model.tool_failure_counts[f"tool_{i}"] = i
    p.recent_tasks.append(_record("failed"))
    prompt = p.self_model_prompt()
    assert len(prompt["tool_failures"]) <= 5


# ── Persistence ─────────────────────────────────────────────────────────────────

def test_self_model_persists_across_save_load(tmp_path):
    p = PersistentMemory()
    p.update_from_task(_task("failed", [("read_webpage", FAILED)]))

    path = tmp_path / "mem.json"
    p.save(path)
    loaded = PersistentMemory.load(path)

    assert loaded.self_model.tool_failure_counts == p.self_model.tool_failure_counts == {"read_webpage": 1}
    assert [t.goal for t in loaded.recent_tasks] == ["Test task"]


# ── Integration: injected into compact() ───────────────────────────────────────

async def test_self_model_injected_into_compact_when_persistent_wired():
    from tests.fakes import WaitingLLM

    persistent = PersistentMemory()
    persistent.self_model.tool_failure_counts["read_webpage"] = 3
    persistent.recent_tasks.append(_record("failed"))

    manager = TaskManager(WaitingLLM(), EventBus(), Settings(), persistent)
    task = manager.start("Look up the AQI")
    payload = task.compact()

    assert "self_model" in payload
    sm = payload["self_model"]
    assert sm["task_summary"]["failed"] == 1
    assert sm["tool_failures"] == {"read_webpage": 3}
    assert payload["long_term_memory"]["recent_tasks"] == [{"goal": "x", "outcome": "failed", "summary": "failed"}]

    await manager.close()
