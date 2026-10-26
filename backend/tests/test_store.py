"""One durable store: Kip's structured state lives in kip.db next to his searchable memories.

Covers the state table, the one-time import of the old JSON files (with backups and
without touching the originals), protection of unreadable data, and beliefs computed
from the tallies instead of a model-maintained graph.
"""
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.agents.schemas import GoalCondition
from app.config import Settings
from app.memory.beliefs import derive_beliefs, effective
from app.memory.durable import UnreadableStoreError
from app.memory.recall import MemoryStore
from app.memory.store import IMPORT_KEY, STATE_KEY, PersistentMemory, PersonMemory, ObjectMemory, open_persistent
from app.main import create_app
from app.tasks.models import TaskContext
from app.world.models import Location
from tests.fakes import WaitingLLM

LEGACY = {"version": 1,
          "people": {"paul": {"name": "Paul", "location_tally": {"office": 2, "kitchen": 1},
                              "last_seen_room": "office", "known_needs": ["laptop"],
                              "delivery_counts": {"laptop": 2}}},
          "objects": {"laptop": {"name": "Laptop", "last_seen_room": "office"}},
          "persona": {"traits": {"empathy": 0.9, "loyalty": 0.8, "curiosity": 0.703, "enthusiasm": 0.71,
                                 "caution": 0.51},
                      "relationships": {"moira": {"familiarity": 0.02, "warmth": 0.51, "interactions": 1}},
                      "episodes": [{"summary": "Could not finish: Find the keys", "feeling": "concerned",
                                    "at": 9.0}],
                      "age_seconds": 10.0}}
GRAPH = {"nodes": {}, "edges": [{"subject": "paul", "relation": "located_in", "target": "office",
                                 "confidence": 0.65, "last_observed": "2026-10-24T05:33:38+00:00"}]}


def _legacy_files(folder: Path) -> tuple[Path, Path]:
    memory, graph = folder / "persistent.json", folder / "beliefs.json"
    memory.write_text(json.dumps(LEGACY), encoding="utf-8")
    graph.write_text(json.dumps(GRAPH), encoding="utf-8")
    return memory, graph


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    memory.people["neave"] = PersonMemory(name="Neave", last_seen_room="kitchen")
    assert memory.save() is True
    store.close()
    again = PersistentMemory.load(MemoryStore(tmp_path / "kip.db"))
    assert again.people["neave"].last_seen_room == "kitchen"


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


# ── one-time import ────────────────────────────────────────────────────────────
def test_json_files_are_imported_once_backed_up_and_left_untouched(tmp_path):
    memory_file, graph_file = _legacy_files(tmp_path)
    before = _digest(memory_file), _digest(graph_file)
    store = MemoryStore(tmp_path / "kip.db")
    memory = open_persistent(store, memory_file, graph_file, backups_dir=tmp_path / "backups")
    # Everything came across, including the persona (migrated) and its history.
    assert memory.people["paul"].delivery_counts == {"laptop": 2}
    assert memory.persona.relationships["moira"].warmth == 0.51
    assert memory.persona.legacy_traits["enthusiasm"] == 0.71
    assert memory.persona.episodes[0].summary == "Could not finish: Find the keys"
    # Originals untouched, verified copies in a timestamped backup folder.
    assert (_digest(memory_file), _digest(graph_file)) == before
    record = json.loads(store.get_state(IMPORT_KEY))
    backup = Path(record["backup"])
    assert _digest(backup / "persistent.json") == before[0] and _digest(backup / "beliefs.json") == before[1]
    assert record["files"]["persistent.json"]["sha256"] == before[0]
    assert json.loads(store.get_state("imported:beliefs.json")) == GRAPH  # kept as history
    # From now on kip.db is the source of truth: later edits to the JSON are ignored.
    memory.people["paul"].last_seen_room = "garden"
    memory.save()
    memory_file.write_text(json.dumps({"version": 1}), encoding="utf-8")
    again = open_persistent(store, memory_file, graph_file, backups_dir=tmp_path / "backups")
    assert again.people["paul"].last_seen_room == "garden"
    assert len(list((tmp_path / "backups").iterdir())) == 1
    assert _digest(graph_file) == before[1]


def test_an_unreadable_json_file_is_not_imported_and_saving_stays_off(tmp_path):
    memory_file = tmp_path / "persistent.json"
    memory_file.write_text("{ broken", encoding="utf-8")
    store = MemoryStore(":memory:")
    memory = open_persistent(store, memory_file, backups_dir=tmp_path / "backups")
    assert memory.load_error and store.get_state(STATE_KEY) is None
    with pytest.raises(UnreadableStoreError):
        memory.save()
    assert memory_file.read_text(encoding="utf-8") == "{ broken"


def test_a_fresh_install_without_json_starts_empty_in_the_store(tmp_path):
    store = MemoryStore(":memory:")
    memory = open_persistent(store, tmp_path / "missing.json", backups_dir=tmp_path / "backups")
    assert memory.people == {} and memory.save() is True
    assert store.get_state(IMPORT_KEY) is None and not (tmp_path / "backups").exists()


def test_the_app_imports_then_runs_from_kip_db_across_restarts(tmp_path):
    memory_file, graph_file = _legacy_files(tmp_path)
    before = _digest(memory_file)
    settings = Settings(memory_file=memory_file, graph_file=graph_file, idle_reflection_enabled=False)
    db = tmp_path / "kip.db"
    with TestClient(create_app(client=WaitingLLM(), settings=settings, memory_store=MemoryStore(db))) as client:
        beliefs = client.get("/api/beliefs").json()["beliefs"]
        assert any(b["subject"] == "paul" and b["relation"] == "recurring_need" for b in beliefs)
        client.post("/api/chat", json={"message": "haha", "speaker": "neave"})
    with TestClient(create_app(client=WaitingLLM(), settings=settings, memory_store=MemoryStore(db))) as client:
        persona = client.get("/api/persona").json()
    assert persona["relationships"]["neave"]["chats"] == 1
    assert "moira" in persona["relationships"]
    assert _digest(memory_file) == before


# ── computed beliefs ──────────────────────────────────────────────────────────
def _beliefs(memory, **filters):
    return [b for b in derive_beliefs(memory)["beliefs"]
            if all(b[key] == value for key, value in filters.items())]


@pytest.mark.parametrize("count, confidence", [(2, 0.65), (3, 0.8), (4, 0.8), (5, 0.9)])
def test_repeated_deliveries_become_a_recurring_need(count, confidence):
    memory = PersistentMemory(people={"paul": PersonMemory(name="Paul", known_needs=["laptop"],
                                                           delivery_counts={"laptop": count})})
    [belief] = _beliefs(memory, relation="recurring_need")
    assert belief["stored_confidence"] == belief["effective_confidence"] == confidence
    assert belief["reason"] == f"Delivered laptop to paul {count} times."
    assert not _beliefs(memory, relation="needs")


def test_a_single_delivery_is_only_a_tentative_need():
    memory = PersistentMemory(people={"paul": PersonMemory(name="Paul", known_needs=["tea"],
                                                           delivery_counts={"tea": 1})})
    [belief] = _beliefs(memory, relation="needs")
    assert belief["target"] == "tea" and belief["effective_confidence"] <= 0.5


def test_a_habit_needs_enough_sightings_mostly_in_one_room():
    def person(tally):
        return PersistentMemory(people={"minu": PersonMemory(name="Minu", location_tally=tally)})
    assert not _beliefs(person({"bedroom": 2}), relation="located_in")              # too few
    assert not _beliefs(person({"bedroom": 2, "kitchen": 2}), relation="located_in")  # no clear room
    [habit] = _beliefs(person({"bedroom": 3}), relation="located_in")
    assert habit["target"] == "bedroom" and habit["stored_confidence"] == 0.8
    [weaker] = _beliefs(person({"office": 2, "kitchen": 1}), relation="located_in")
    assert 0.5 < weaker["stored_confidence"] < 0.65 and weaker["reason"] == "Seen in office 2 of 3 times."


def test_sightings_fade_with_time_and_undated_ones_count_as_old():
    now = datetime.now(timezone.utc)
    eight_hours_ago = (now - timedelta(hours=8)).isoformat()
    assert effective(0.7, "located_in", now.isoformat(), now) == 0.7
    assert effective(0.7, "located_in", eight_hours_ago, now) == pytest.approx(0.175, abs=0.001)
    assert effective(0.7, "located_in", None) == 0.3
    memory = PersistentMemory(objects={"keys": ObjectMemory(name="Keys", last_seen_room="hall",
                                                            last_seen_at=eight_hours_ago)})
    [belief] = _beliefs(memory, subject="keys")
    assert belief["stored_confidence"] == 0.7 and belief["effective_confidence"] < 0.2


def test_finished_tasks_feed_beliefs_with_observation_times():
    memory = PersistentMemory()
    for index in range(3):
        task = TaskContext(goal="Deliver medicine",
                           conditions=[GoalCondition(kind="deliver", object="medicine", person="paul")])
        task.memory.set_person("paul", "office", f"ev-person-{index}", "Paul")
        task.memory.set_object("medicine", Location(kind="person", id="paul"), None, f"ev-obj-{index}",
                               name="Medicine")
        memory.update_from_task(task)
    assert memory.people["paul"].delivery_counts == {"medicine": 3}
    assert memory.people["paul"].last_seen_at
    [habit] = _beliefs(memory, subject="paul", relation="located_in")
    assert habit["target"] == "office" and habit["effective_confidence"] == 0.8
    assert _beliefs(memory, relation="recurring_need")[0]["stored_confidence"] == 0.8
