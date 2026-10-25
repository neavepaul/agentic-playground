"""The searchable memory store: hybrid recall, merging, filters and task memories."""
import asyncio
import math
from pathlib import Path

import pytest

from app.agents.schemas import GoalCondition
from app.config import Settings
from app.events.bus import EventBus
from app.memory.recall import MemoryStore, task_memories
from app.tasks.manager import TaskManager
from app.tasks.models import TaskContext
from app.world.engine import WorldEngine
from app.world.tools import WorldTools
from tests.fakes import WaitingLLM

LEGACY_WORLD = Path(__file__).parent / "fixtures" / "legacy_house.json"

# Words that mean the same thing share a dimension, so this fake behaves like a
# real embedding model on these sentences without downloading one.
CONCEPTS = [{"tired", "exhausted", "sleepy", "weary"}, {"tea", "drink", "beverage"},
            {"charger", "cable"}, {"neave"}, {"paul"}, {"office"}, {"likes", "enjoy", "prefers"},
            {"keys"}]


class FakeEmbedder:
    dim = len(CONCEPTS) + 1

    def embed_sync(self, texts):
        vectors = []
        for text in texts:
            words = {w.strip(".,?!'").casefold() for w in text.split()}
            vector = [1.0 if words & concept else 0.0 for concept in CONCEPTS] + [0.1]
            norm = math.sqrt(sum(x * x for x in vector))
            vectors.append([x / norm for x in vector])
        return vectors

    async def embed(self, texts):
        return self.embed_sync(texts)


@pytest.fixture
def store():
    memory = MemoryStore(":memory:", FakeEmbedder())
    yield memory
    memory.close()


@pytest.mark.asyncio
async def test_meaning_matches_without_shared_words(store):
    await store.add("episode", "Neave looked exhausted after work", ["neave"])
    await store.add("fact", "The keys hang by the door", ["keys"])
    found = await store.recall("is he sleepy", k=1)
    assert found[0]["text"] == "Neave looked exhausted after work"


@pytest.mark.asyncio
async def test_keyword_search_works_without_embeddings():
    store = MemoryStore(":memory:")
    await store.add("fact", "Neave keeps his charger in the office", ["neave", "charger"])
    await store.add("preference", "Paul likes green tea", ["paul"])
    assert not store.vectors
    assert (await store.recall("where is the charger", k=1))[0]["kind"] == "fact"
    assert (await store.recall("Paul", k=1))[0]["kind"] == "preference"


@pytest.mark.asyncio
async def test_repeats_and_near_duplicates_reinforce_instead_of_copying(store):
    first = await store.add("fact", "Neave keeps his charger in the office", importance=0.5)
    again = await store.add("fact", "Neave keeps his charger in the office", importance=0.5)
    reworded = await store.add("fact", "In the office, Neave keeps his charger.", importance=0.5)
    assert first == again == reworded and store.count() == 1
    assert (await store.recall("charger"))[0]["importance"] > 0.5


@pytest.mark.asyncio
async def test_importance_and_recency_break_ties(store):
    old = await store.add("fact", "Paul likes tea in the morning", importance=0.3)
    await store.add("fact", "Paul likes tea after dinner", importance=0.9)
    store._db.execute("UPDATE memories SET created_at = '2020-01-01T00:00:00+00:00' WHERE id = ?", (old,))
    found = await store.recall("what does paul enjoy", k=2)
    assert found[0]["text"] == "Paul likes tea after dinner"


@pytest.mark.asyncio
async def test_kind_and_about_filters(store):
    await store.add("skill", "For delivering tea to Paul: kitchen first", ["paul"])
    await store.add("fact", "Paul likes tea", ["paul"])
    await store.add("fact", "Neave likes tea", ["neave"])
    assert [m["kind"] for m in await store.recall("tea", kinds=["skill"])] == ["skill"]
    assert {m["text"] for m in await store.recall("tea", about=["neave"])} == {"Neave likes tea"}


@pytest.mark.asyncio
async def test_keyword_index_follows_edits(store):
    key = await store.add("fact", "The keys are on the shelf")
    store._db.execute("UPDATE memories SET text = 'The remote is on the sofa' WHERE id = ?", (key,))
    store._db.commit()
    assert not [m for m in await store.recall("shelf") if m["text"].endswith("shelf")]
    assert (await store.recall("sofa"))[0]["id"] == key


@pytest.mark.asyncio
async def test_database_file_is_created_only_when_used(tmp_path):
    path = tmp_path / "kip.db"
    store = MemoryStore(path)
    assert not path.exists()
    await store.add("fact", "Hello")
    assert path.exists() and store.recent(5)[0]["text"] == "Hello"
    store.close()


def test_finished_delivery_forms_episode_and_skill_memories():
    context = TaskContext(goal="Bring the charger to Neave")
    context.conditions = [GoalCondition(kind="deliver", object="charger", person="neave")]
    context.status, context.summary = "completed", "Delivered."
    for tool, observation in [("move_to", {"room": "office"}), ("pick_up", {"object": "charger", "room": "office"}),
                              ("move_to", {"room": "bedroom"}), ("give", {"object": "charger", "person": "neave", "room": "bedroom"})]:
        context.action_history.append({"tool": tool, "success": True, "observation": observation, "arguments": {}})
    memories = {m["kind"]: m for m in task_memories(context, "cheerful")}
    assert "I felt cheerful" in memories["episode"]["text"]
    skill = memories["skill"]["text"]
    assert "found the charger in the office" in skill and "neave was in the bedroom" in skill
    assert "office → bedroom" in skill and memories["skill"]["about"] == ["neave", "charger"]


@pytest.mark.asyncio
async def test_next_task_recalls_relevant_skills(store):
    await store.add("skill", "For 'Bring the charger to Neave': found the charger in the office.", ["charger"])
    engine, bus = WorldEngine(LEGACY_WORLD), EventBus()
    manager = TaskManager(WaitingLLM(), WorldTools(engine, bus), bus, Settings(), memory_store=store)
    context = manager.start("Neave needs the charger again")
    await asyncio.sleep(0.05)
    items = context.long_term_memory["relevant_memories"]["items"]
    assert "found the charger in the office" in items[0]["text"]
    await manager.cancel(context.id)
