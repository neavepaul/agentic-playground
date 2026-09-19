import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings, get_settings
from app.events.bus import EventBus
from app.llm.ollama import OllamaClient
from app.tasks.manager import TaskBusy, TaskManager
from app.world.engine import WorldEngine
from app.world.tools import WorldTools


class GoalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    goal: str = Field(min_length=1, max_length=1000)


def create_app(client=None, settings: Settings | None = None) -> FastAPI:
    config = settings or get_settings()
    logging.basicConfig(level=getattr(logging, config.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Never enable HTTP wire-body logging, even with application DEBUG.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    bus, engine = EventBus(), WorldEngine()
    llm = client or OllamaClient(config)
    manager = TaskManager(llm, WorldTools(engine, bus), bus, config)
    mutation_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(app):
        yield
        await manager.close()
        if client is None:
            await llm.close()

    app = FastAPI(title="agentic_friend", lifespan=lifespan)
    app.state.manager, app.state.engine, app.state.bus = manager, engine, bus

    def snapshot() -> dict:
        latest = next(reversed(manager.tasks.values()), None) if manager.tasks else None
        return {"world": engine.snapshot(), "sequence": bus.sequence,
                "task": manager.active() or (latest.public() if latest else None)}

    @app.get("/api/health")
    async def health():
        model = await llm.health() if isinstance(llm, OllamaClient) else {"mode": "test"}
        return {"status": "ok", "ollama": model}

    @app.get("/api/world")
    async def world():
        return snapshot()

    @app.post("/api/world/reset")
    async def reset():
        async with mutation_lock:
            await manager.close()
            engine.reset()
            # A new world starts a new session; old tasks should not appear active in UI.
            manager.tasks.clear()
            bus.history.clear()
            bus.emit("world_reset", world=engine.snapshot(), summary="World restored to its initial state.")
            return snapshot()

    @app.post("/api/tasks", status_code=202)
    async def create_task(request: GoalRequest):
        async with mutation_lock:
            try:
                return manager.start(request.goal).public()
            except TaskBusy as exc:
                raise HTTPException(409, str(exc)) from None

    @app.get("/api/tasks/{task_id}")
    async def task(task_id: str):
        if task_id not in manager.tasks:
            raise HTTPException(404, "Task not found (tasks are held in memory until reset/restart).")
        return manager.tasks[task_id].model_dump()

    @app.post("/api/tasks/{task_id}/cancel")
    async def cancel(task_id: str):
        async with mutation_lock:
            if task_id not in manager.tasks:
                raise HTTPException(404, "Task not found.")
            return (await manager.cancel(task_id)).public()

    @app.websocket("/ws")
    async def websocket(socket: WebSocket):
        await socket.accept()
        queue = bus.subscribe()
        # Subscribe before taking a snapshot: no startup race can lose an action.
        await socket.send_json({"type": "snapshot", "data": snapshot(), "events": list(bus.history)})

        async def send():
            while True:
                event = await queue.get()
                if event["type"] == "resync_required":
                    await socket.send_json({"type": "snapshot", "data": snapshot(), "events": list(bus.history)})
                else:
                    await socket.send_json(event)

        async def receive():
            while True:
                await socket.receive_text()

        sender, receiver = asyncio.create_task(send()), asyncio.create_task(receive())
        try:
            done, _ = await asyncio.wait([sender, receiver], return_when=asyncio.FIRST_COMPLETED)
            for future in done:
                future.result()
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            sender.cancel()
            receiver.cancel()
            await asyncio.gather(sender, receiver, return_exceptions=True)
            bus.unsubscribe(queue)

    return app


app = create_app()
