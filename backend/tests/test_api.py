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
