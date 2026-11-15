from fastapi.testclient import TestClient

from app.main import create_app
from tests.fakes import WaitingLLM


def test_api_lifecycle_and_websocket():
    app = create_app(client=WaitingLLM())
    with TestClient(app) as client:
        assert client.get("/api/health").json()["status"] == "ok"
        assert client.get("/api/state").json() == {"sequence": 0, "task": None}
        assert client.get("/api/world").status_code == 404
        assert client.post("/api/world/reset").status_code in (404, 405)
        assert client.get("/api/beliefs").status_code == 404
        assert client.post("/api/tasks", json={"goal": "  "}).status_code == 422
        with client.websocket_connect("/ws") as socket:
            snapshot = socket.receive_json()
            assert snapshot["type"] == "snapshot"
            assert set(snapshot["data"]) == {"sequence", "task"}
            response = client.post("/api/tasks", json={"goal": "Find charger."})
            assert response.status_code == 202
            id = response.json()["id"]
            assert "action_history" not in response.json()
            assert socket.receive_json()["type"] == "task_started"
            assert client.post("/api/tasks", json={"goal": "Other task."}).status_code == 409
            assert client.get(f"/api/tasks/{id}").json()["status"] == "running"
            state = client.get("/api/state").json()
            assert set(state) == {"sequence", "task"}
            assert state["sequence"] > 0 and state["task"]["id"] == id
            assert state["task"]["status"] == "running"
            assert client.post(f"/api/tasks/{id}/cancel").json()["status"] == "cancelled"
            assert client.post(f"/api/tasks/{id}/cancel").json()["status"] == "cancelled"
        assert app.state.manager.active_id is None
        latest = client.get("/api/state").json()["task"]
        assert latest["id"] == id and latest["status"] == "cancelled"
        assert client.get("/api/tasks/missing").status_code == 404
        with client.websocket_connect("/ws") as socket:
            data = socket.receive_json()["data"]
            assert set(data) == {"sequence", "task"} and data["task"]["id"] == id


def test_event_queue_overflow_requests_resync():
    from app.events.bus import EventBus
    bus = EventBus()
    queue = bus.subscribe()
    for _ in range(129):
        bus.emit("test")
    assert queue.get_nowait()["type"] == "resync_required"
    bus.unsubscribe(queue)
    assert not bus.subscribers


def test_only_kips_own_pages_and_host_names_get_in():
    """No login, so the boundary is the browser's: another site's page may neither read the event
    stream nor send changes, and another host name (DNS rebinding) is refused outright."""
    import pytest
    from starlette.websockets import WebSocketDisconnect
    with TestClient(create_app(client=WaitingLLM())) as client:
        with pytest.raises(WebSocketDisconnect) as refused:
            with client.websocket_connect("/ws", headers={"origin": "https://untrusted.example"}) as socket:
                socket.receive_json()
        assert refused.value.code == 1008
        with client.websocket_connect("/ws", headers={"origin": "http://127.0.0.1:5173"}) as socket:
            assert socket.receive_json()["type"] == "snapshot"
        foreign = {"origin": "https://untrusted.example"}
        assert client.post("/api/chat", json={"message": "hi"}, headers=foreign).status_code == 403
        assert client.post("/api/tasks", json={"goal": "Find charger."}, headers=foreign).status_code == 403
        assert client.post("/api/chat", json={"message": "hi"}, headers={"origin": "http://localhost:5173"}).status_code == 202
        assert client.get("/api/health", headers={"host": "kip.attacker.example"}).status_code == 400


def test_a_private_event_reaches_live_pages_whole_but_replays_without_its_words():
    from app.events.bus import EventBus
    bus = EventBus()
    queue = bus.subscribe()
    turn = {"id": "t1", "role": "kip", "text": "Worth 46,905.", "content": "Worth 46,905.", "private": True}
    bus.emit("chat_message", "Kip", turn=turn, summary="Kip: Worth 46,905.")
    assert queue.get_nowait()["data"]["turn"]["text"] == "Worth 46,905."  # the live reply shows
    stored = bus.history[-1]["data"]
    assert "46,905" not in str(stored) and stored["turn"]["id"] == "t1" and stored["turn"]["private"]
    assert turn["text"] == "Worth 46,905."  # redaction copies; the live event is untouched


def test_a_task_from_the_api_is_its_public_view():
    """No audit trail (page text, tool output), and a private answer's words hidden."""
    from app.tasks.models import TaskContext
    app = create_app(client=WaitingLLM())
    with TestClient(app) as client:
        task = TaskContext(goal="how is my portfolio doing?", status="completed", private=True,
                           answer={"answer": "Worth 46,905.", "sources": [{"title": "Groww holdings", "url": "https://groww.in/"}]},
                           action_history=[{"tool": "web_search", "success": True, "observation": {"quote": "Worth 46,905."}}])
        app.state.manager.tasks[task.id] = task
        shown = client.get(f"/api/tasks/{task.id}").json()
        assert "46,905" not in str(shown) and "action_history" not in shown
        assert shown["answer"]["sources"][0]["url"] == "https://groww.in/"


def test_the_task_journal_keeps_tool_output_short():
    """Saved several times per tool call: a page's text is not rewritten in full each time."""
    import json
    from app.events.bus import EventBus
    from app.config import Settings
    from app.memory.recall import MemoryStore
    from app.tasks.manager import JOURNAL_TEXT, TaskManager
    from app.tasks.models import TaskContext
    store = MemoryStore(":memory:")
    manager = TaskManager(WaitingLLM(), EventBus(), Settings(), memory_store=store)
    task = TaskContext(goal="read a long page", status="completed",
                       action_history=[{"tool": "read_webpage", "success": True, "observation": {"text": "x" * 200_000}}])
    manager.tasks[task.id] = task
    manager.checkpoint()
    saved = json.loads(store.get_state("task-journal"))[-1]
    assert len(saved["action_history"][0]["observation"]["text"]) <= JOURNAL_TEXT + 2
    assert len(task.action_history[0]["observation"]["text"]) == 200_000  # the live task is untouched
    assert TaskContext.model_validate(saved).status == "completed"  # and it still loads after a restart
