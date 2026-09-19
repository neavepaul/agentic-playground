from fastapi.testclient import TestClient

from app.main import create_app
from tests.fakes import WaitingLLM


def test_api_lifecycle_and_websocket():
    app = create_app(client=WaitingLLM())
    with TestClient(app) as client:
        assert client.get("/api/health").json()["status"] == "ok"
        initial = client.get("/api/world").json()["world"]
        assert initial["robot"]["room"] == "hall"
        assert client.post("/api/tasks", json={"goal": "  "}).status_code == 422
        with client.websocket_connect("/ws") as socket:
            assert socket.receive_json()["type"] == "snapshot"
            response = client.post("/api/tasks", json={"goal": "Find charger."})
            assert response.status_code == 202
            id = response.json()["id"]
            assert socket.receive_json()["type"] == "task_started"
            assert client.post("/api/tasks", json={"goal": "Other task."}).status_code == 409
            assert client.get(f"/api/tasks/{id}").json()["status"] == "running"
            assert client.post(f"/api/tasks/{id}/cancel").json()["status"] == "cancelled"
            assert client.post(f"/api/tasks/{id}/cancel").json()["status"] == "cancelled"
        client.post("/api/tasks", json={"goal": "Try again."})
        assert client.post("/api/world/reset").json()["world"] == initial
        assert app.state.manager.active_id is None
        assert client.get(f"/api/tasks/{id}").status_code == 404
        with client.websocket_connect("/ws") as socket:
            assert socket.receive_json()["data"]["world"] == initial


def test_event_queue_overflow_requests_resync():
    from app.events.bus import EventBus
    bus = EventBus()
    queue = bus.subscribe()
    for _ in range(129):
        bus.emit("test")
    assert queue.get_nowait()["type"] == "resync_required"
    bus.unsubscribe(queue)
    assert not bus.subscribers
