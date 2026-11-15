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
