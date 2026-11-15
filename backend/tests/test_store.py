"""One durable store: Kip's structured state lives in kip.db next to his searchable memories.

Covers the state table, saving and reloading Kip's persistent memory, and protection of
unreadable data.
"""
import pytest

from app.memory.durable import UnreadableStoreError
from app.memory.recall import MemoryStore
from app.memory.store import STATE_KEY, PersistentMemory
from app.tasks.models import TaskContext


# ── state table ────────────────────────────────────────────────────────────────
def test_state_keys_are_written_together_and_survive_reopening(tmp_path):
    store = MemoryStore(tmp_path / "kip.db")
    store.set_state({"a": "1", "b": "2"})
    store.set_state({"a": "3"})
    store.close()
    reopened = MemoryStore(tmp_path / "kip.db")
    assert (reopened.get_state("a"), reopened.get_state("b"), reopened.get_state("c")) == ("3", "2", None)
    reopened.close()


def test_persistent_memory_saves_to_and_loads_from_the_store(tmp_path):
    store = MemoryStore(tmp_path / "kip.db")
    memory = PersistentMemory.load(store)
    task = TaskContext(goal="Look up the AQI", status="completed", summary="Answered.")
    task.record("web_search", {"query": "AQI"}, {"success": False, "error": "offline"})
    memory.update_from_task(task)
    assert memory.save() is True
    store.close()
    again = PersistentMemory.load(MemoryStore(tmp_path / "kip.db"))
    assert again.self_model.tool_failure_counts == {"web_search": 1}
    assert [(t.goal, t.outcome) for t in again.recent_tasks] == [("Look up the AQI", "completed")]
    assert again.task_total == 1


def test_a_memory_with_nowhere_to_go_is_not_saved():
    assert PersistentMemory().save() is False


def test_unreadable_state_is_reported_and_never_overwritten():
    store = MemoryStore(":memory:")
    store.set_state({STATE_KEY: "{ not json"})
    memory = PersistentMemory.load(store)
    assert memory.load_error
    with pytest.raises(UnreadableStoreError):
        memory.save()
    assert store.get_state(STATE_KEY) == "{ not json"
