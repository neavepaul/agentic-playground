from fastapi.testclient import TestClient

from app.chat.checkpoint import Checkpoint, state_key
from app.config import Settings
from app.main import create_app
from app.memory.recall import MemoryStore
from tests.test_chat import StreamingModel
from tests.test_people import Device, two_people


def test_memory_controls_are_private_and_invalidate_old_chat_context(tmp_path):
    path = tmp_path / "memory.db"
    settings = Settings(intent_router="off", chat_compaction=False)
    store = MemoryStore(path)
    model = StreamingModel(["Okay."])
    app = create_app(client=model, settings=settings, memory_store=store)
    marker = "emerald lighthouse"
    old_at = "2025-01-01T00:00:00+00:00"

    async def seed(neave_id, asha_id):
        people = app.state.people
        neave = people.get(neave_id)
        asha = people.get(asha_id)
        cid = people.new_conversation(neave, "Past chat")
        store.append_turn(cid, {"id": "old", "role": "user", "text": marker, "at": old_at})
        store.set_state({state_key(cid): Checkpoint(marker, old_at, 1).dump()})
        await store.set_keyed(f"chat-session:{cid}", "episode", marker, speaker="neave")
        await store.add("conversation", marker, meta={"speaker": "neave"})
        old = await store.add("preference", "I prefer coffee.", meta={"speaker": "neave"})
        forgotten = await store.add("fact", marker, meta={"speaker": "neave"})
        other = await store.add("fact", "Asha's fact.", meta={"speaker": "asha"})
        legacy = await store.add("fact", "Unattributed fact.")
        app.state.hub.get(cid, neave)  # cache the checkpoint before changing memory
        return cid, old, forgotten, other, legacy

    with TestClient(app) as client:
        neave, asha, neave_profile, asha_profile = two_people(client)
        cid, old, forgotten, other, legacy = client.portal.call(seed, neave_profile["id"], asha_profile["id"])
        assert asha.patch(f"/api/memories/{old}", json={"text": "Wrong profile"}).status_code == 404
        assert asha.delete(f"/api/memories/{forgotten}").status_code == 404
        assert neave.delete(f"/api/memories/{other}").status_code == 404
        assert neave.delete(f"/api/memories/{legacy}").status_code == 404
        assert neave.patch(f"/api/memories/{old}", json={"text": "I prefer tea."}).status_code == 204
        assert neave.delete(f"/api/memories/{forgotten}").status_code == 204
        assert client.portal.call(lambda: store.set_keyed(
            f"chat-session:{cid}", "episode", marker, speaker="neave", expected_cutoff="")) == 0
        changes = neave.get("/api/memories").json()
        assert "I prefer tea." in [m["text"] for m in changes if m["editable"]]
        assert all(marker not in m["text"] and "coffee" not in m["text"] for m in changes)
        assert [m["id"] for m in asha.get("/api/memories").json()] == [other]
        assert neave.post("/api/chat", json={"message": "What do you know about me?", "conversation_id": cid}).status_code == 202
        client.portal.call(app.state.hub.peek(cid).drain)
        assert marker not in str(model.prompts)
        assert "I prefer tea." in str(model.prompts)
        assert store.get_state(state_key(cid)) in (None, "")
        assert marker in str(neave.get("/api/chat", params={"conversation_id": cid}).json()["turns"])
        cookies = neave.jar

    store = MemoryStore(path)
    model = StreamingModel(["Okay."])
    app = create_app(client=model, settings=settings, memory_store=store)
    with TestClient(app) as client:
        neave = Device(client, cookies)
        assert neave.post("/api/chat", json={"message": "Continue that thought.", "conversation_id": cid}).status_code == 202
        client.portal.call(app.state.hub.peek(cid).drain)
        assert marker not in str(model.prompts)
        assert all(marker not in m["text"] for m in neave.get("/api/memories").json())
