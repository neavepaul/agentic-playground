"""The searchable memory store: hybrid recall, merging, filters and task memories."""
import asyncio
import math

import pytest

from app.agents.schemas import GoalCondition
from app.config import Settings
from app.events.bus import EventBus
from app.memory.recall import MemoryStore, task_memories
from app.tasks.manager import TaskManager
from app.tasks.models import TaskContext
from tests.fakes import WaitingLLM

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


def test_finished_lookup_forms_episode_and_answer_memories():
    context = TaskContext(goal="Look up the AQI in Mumbai")
    context.conditions = [GoalCondition(kind="answer", required_tools=["web_search"])]
    context.status, context.summary = "completed", "Answered."
    context.record("web_search", {"query": "AQI Mumbai"}, {"success": True, "evidence_id": "s1", "observation": {
        "query": "AQI Mumbai", "results": [{"url": "https://aqi.example/mumbai", "title": "AQI"}]}})
    context.record("finish_answer", {}, {"success": True, "evidence_id": "a1", "observation": {
        "answer": "The AQI is 150.", "sources": [{"url": "https://aqi.example/mumbai"}]}})
    memories = {m["kind"]: m for m in task_memories(context, "cheerful")}
    assert set(memories) == {"episode", "observation"}
    assert memories["episode"]["text"] == "Finished: Look up the AQI in Mumbai. Answered. I felt cheerful."
    assert memories["observation"]["text"] == "Q: Look up the AQI in Mumbai → A: The AQI is 150."
    assert memories["observation"]["meta"] == {"sources": ["https://aqi.example/mumbai"]}
    assert memories["episode"]["about"] == memories["observation"]["about"] == []


def test_a_failed_task_forms_only_an_episode():
    context = TaskContext(goal="Look up the AQI", status="failed", summary="No source.")
    [memory] = task_memories(context)
    assert memory["kind"] == "episode" and memory["text"] == "Could not finish: Look up the AQI. No source."
    assert memory["importance"] == 0.7


@pytest.mark.asyncio
async def test_old_tool_answers_are_archived_as_observations(tmp_path):
    path = tmp_path / "observations.db"
    original = MemoryStore(path)
    await original.add("fact", "The score was 145.", source="task")
    await original.add("fact", "Neave likes tea.", source="conversation")
    original.close()
    reopened = MemoryStore(path)
    try:
        assert [m["text"] for m in await reopened.recall("score")] == ["Neave likes tea."]
        assert [m["text"] for m in await reopened.recall("score", kinds=["observation"])] == ["The score was 145."]
    finally:
        reopened.close()


@pytest.mark.asyncio
async def test_correction_replaces_preference_after_reopen(tmp_path):
    path = tmp_path / "correction.db"
    store = MemoryStore(path)
    old = await store.add("preference", "Neave prefers coffee.", meta={"speaker": "neave"})
    new = await store.add("preference", "Neave prefers tea.", meta={"speaker": "neave"})
    await store.add("preference", "Neave prefers coffee.", meta={"speaker": "moira"})
    retired = await store.supersede("Neave prefers coffee.", by=new, viewer="neave")
    assert retired["id"] == old
    store.close()
    reopened = MemoryStore(path)
    try:
        assert [m["id"] for m in await reopened.recall("prefers", viewer="neave")] == [new]
        assert [m["text"] for m in await reopened.recall("prefers", viewer="moira")] == ["Neave prefers coffee."]
    finally:
        reopened.close()


@pytest.mark.asyncio
async def test_next_task_recalls_relevant_memories(store):
    await store.add("fact", "Q: Where does Neave keep the charger cable? → A: In the office.", [])
    manager = TaskManager(WaitingLLM(), EventBus(), Settings(), memory_store=store)
    context = manager.start("Neave needs the charger again")
    await asyncio.sleep(0.05)
    items = context.long_term_memory["relevant_memories"]["items"]
    assert "In the office" in items[0]["text"]
    await manager.cancel(context.id)


@pytest.mark.asyncio
async def test_superseded_memories_stay_as_history_but_leave_recall(store):
    wrong = await store.add("fact", "Neave is allergic to peanuts", ["neave"])
    fix = await store.add("fact", "Paul is allergic to peanuts", ["paul"])
    retired = await store.supersede("Neave is allergic to peanuts", by=fix, exclude={fix})
    assert retired["id"] == wrong
    assert [m["id"] for m in await store.recall("peanuts allergy")] == [fix]
    assert [m["id"] for m in await store.recall("peanuts allergy", include_superseded=True)] == [fix, wrong] or \
        {m["id"] for m in await store.recall("peanuts allergy", include_superseded=True)} == {fix, wrong}
    assert {m["id"]: m["superseded_by"] for m in store.recent(5)} == {fix: None, wrong: fix}
    # Re-stating a superseded fact stores it afresh rather than reviving the old row silently.
    again = await store.add("fact", "Neave is allergic to peanuts", ["neave"])
    assert again not in {wrong, fix}


@pytest.mark.asyncio
async def test_existing_databases_gain_the_superseded_column(tmp_path):
    import sqlite3
    path = tmp_path / "old.db"
    legacy = sqlite3.connect(path)
    legacy.execute("CREATE TABLE memories (id INTEGER PRIMARY KEY, kind TEXT NOT NULL, text TEXT NOT NULL, "
                   "about TEXT NOT NULL DEFAULT '[]', importance REAL NOT NULL DEFAULT 0.5, source TEXT NOT NULL "
                   "DEFAULT '', created_at TEXT NOT NULL, last_recalled TEXT, recall_count INTEGER NOT NULL "
                   "DEFAULT 0, meta TEXT NOT NULL DEFAULT '{}')")
    legacy.execute("INSERT INTO memories(kind, text, created_at) VALUES ('fact', 'Old fact', '2026-10-25T00:00:00+00:00')")
    legacy.commit(); legacy.close()
    store = MemoryStore(path)
    assert store.recent(1)[0]["superseded_by"] is None
    store.close()


@pytest.mark.asyncio
async def test_backfill_indexes_memories_saved_without_embeddings(tmp_path):
    path = tmp_path / "kip.db"
    plain = MemoryStore(path)
    await plain.add("episode", "Neave looked exhausted after work", ["neave"])
    plain.close()
    store = MemoryStore(path, FakeEmbedder())
    assert (await store.recall("is he sleepy", k=1))[0]["score"] < 0.45  # no meaning match yet
    assert await store.backfill() == 1
    assert await store.backfill() == 0
    assert (await store.recall("is he sleepy", k=1))[0]["score"] >= 0.45
    store.close()


@pytest.mark.asyncio
async def test_two_people_saving_the_same_words_keep_separate_memories(store):
    text = "Likes tea in the morning"
    a = await store.add("preference", text, [], meta={"speaker": "neave"})
    b = await store.add("preference", text, [], meta={"speaker": "paul"})
    assert a != b
    assert [m["id"] for m in await store.recall("tea", viewer="neave")] == [a]
    assert [m["id"] for m in await store.recall("tea", viewer="paul")] == [b]
    # The same person saying it again still reinforces, not copies.
    assert await store.add("preference", text, [], meta={"speaker": "paul"}) == b


@pytest.mark.asyncio
async def test_a_reworded_repeat_merges_only_within_one_person(store):
    a = await store.add("preference", "Neave likes tea", ["neave"], meta={"speaker": "neave"})
    assert await store.add("preference", "Neave likes tea!", ["neave"], meta={"speaker": "paul"}) != a


@pytest.mark.asyncio
async def test_a_correction_never_retires_another_persons_memory(store):
    theirs = await store.add("fact", "Neave is allergic to peanuts", ["neave"], meta={"speaker": "paul"})
    assert await store.supersede("Neave is allergic to peanuts", viewer="neave") is None
    assert [m["id"] for m in await store.recall("peanuts", viewer="paul")] == [theirs]
    mine = await store.add("fact", "Neave is allergic to peanuts", ["neave"], meta={"speaker": "neave"})
    assert (await store.supersede("Neave is allergic to peanuts", viewer="neave"))["id"] == mine
    assert [m["id"] for m in await store.recall("peanuts", viewer="paul")] == [theirs]


@pytest.mark.asyncio
async def test_a_task_recalls_only_its_own_persons_memories(store):
    await store.add("fact", "Q: Where is the charger cable? → A: In the office.", [], meta={"speaker": "paul"})
    manager = TaskManager(WaitingLLM(), EventBus(), Settings(), memory_store=store)
    context = manager.start("Where is the charger again", speaker="neave")
    await asyncio.sleep(0.05)
    assert "relevant_memories" not in context.long_term_memory
    await manager.cancel(context.id)
    mine = manager.start("Where is the charger again", speaker="paul")
    await asyncio.sleep(0.05)
    assert "In the office" in mine.long_term_memory["relevant_memories"]["items"][0]["text"]
    await manager.cancel(mine.id)

@pytest.mark.asyncio
async def test_changeable_facts_age_without_aging_preferences(store):
    from app.chat.prompt import build_messages
    from app.memory.recall import prompt_memories

    work = await store.add("fact", "Neave works at Acme.", source="conversation")
    tea = await store.add("preference", "Neave prefers tea.", source="conversation")
    with store._lock, store._db:
        store._db.execute("UPDATE memories SET created_at = ?", ("2020-01-01T00:00:00+00:00",))
    memories = {m["id"]: m for m in store.recent()}
    assert memories[work]["confirmed_on"] == "2020-01-01"
    assert memories[work]["needs_confirmation"]
    assert "needs_confirmation" not in memories[tea]
    prompt = build_messages("Kip", None, prompt_memories(list(memories.values())), {}, [], "Where do I work?")
    assert "may have changed" in prompt[-1]["content"]
    assert "ask whether it is still true" in prompt[-1]["content"]

    # Recalling the fact does not confirm it. Hearing it again does.
    await store.recall("Neave works at Acme")
    assert next(m for m in store.recent() if m["id"] == work)["needs_confirmation"]
    assert await store.add("fact", "Neave works at Acme.", source="conversation") == work
    assert not next(m for m in store.recent() if m["id"] == work)["needs_confirmation"]
