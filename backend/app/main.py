import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings, get_settings
from app.events.bus import EventBus
from app.chat import ChatSession
from app.llm.factory import create_chat_client, create_client
from app.llm.priority import ConversationFirst
from app.memory.recall import Embedder, MemoryStore
from app.memory.beliefs import derive_beliefs
from app.memory.store import open_persistent
from app.mind.agent import AgentMind
from app.persona import Persona
from app.tasks.manager import TaskBusy, TaskManager
from app.world.clock import WorldClock
from app.world.engine import WorldEngine
from app.world.tools import WorldTools


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    message: str = Field(min_length=1, max_length=1000)
    # Who is talking (a person id from the world), so "I"/"my" facts are filed under them.
    speaker: str | None = Field(default=None, max_length=40)


class GoalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    goal: str = Field(min_length=1, max_length=1000)


def create_app(client=None, settings: Settings | None = None, memory_store: MemoryStore | None = None) -> FastAPI:
    config = settings or get_settings()
    logging.basicConfig(level=getattr(logging, config.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Never enable HTTP wire-body logging, even with application DEBUG.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    clock = WorldClock(speed=config.clock_speed, start_hour=config.clock_start_hour,
                       mode=config.simulation_mode, action_seconds=config.action_seconds)
    bus, engine = EventBus(), WorldEngine(config.world_file, clock=clock)
    llm = client if client is not None else create_client(config)
    # Injected test clients get a throwaway in-memory store and no embedding model,
    # unless the test hands in its own store (e.g. a file, to test restarts).
    live = client is None
    embedder = (Embedder(config.embedding_model, config.embedding_cache_dir)
                if live and config.embeddings_enabled else None)
    store = memory_store or MemoryStore(config.memory_db_file if live else ":memory:", embedder)
    # kip.db holds everything durable; the old JSON files are imported once, then left alone.
    persistent = open_persistent(store, config.memory_file, config.graph_file,
                                 backups_dir=config.memory_file.parent / "backups")
    persona = None
    if config.persona_enabled:
        persona = Persona(persistent.persona, config.persona_name, now=clock.simulated_seconds,
                          helpfulness_baseline=config.drive_helpfulness)
        persona.attach(bus)
    chat_llm = create_chat_client(config, llm) if live else llm
    shared = chat_llm is llm
    # One local model for both: a person waiting for a reply goes before the next errand call.
    task_llm = ConversationFirst(llm) if shared else llm
    manager = TaskManager(task_llm, WorldTools(engine, bus), bus, config, persistent, persona=persona,
                          memory_store=store)
    chat = ChatSession(chat_llm, bus, manager, engine, config, store=store, persona=persona,
                       shares_task_model=shared)
    chat.gate = task_llm if shared else None
    chat.attach()
    mind = AgentMind(task_llm, manager, engine, bus, config, persistent, persona=persona,
                     memory_store=store, chat=chat)
    mutation_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(app):
        if embedder:
            # Load the embedding model in the background so the first reply doesn't wait,
            # then index any memories saved while meaning search was unavailable.
            async def prepare_memory():
                await asyncio.to_thread(embedder.embed_sync, ["warm up"])
                await store.backfill()
            app.state.memory_ready = asyncio.create_task(prepare_memory())
        mind.start()
        yield
        memory_ready = getattr(app.state, "memory_ready", None)
        if memory_ready and not memory_ready.done():
            memory_ready.cancel()  # never let a backfill touch the store after shutdown
            await asyncio.gather(memory_ready, return_exceptions=True)
        await chat.close()
        await mind.close()
        await manager.close()
        if persona and persona.dirty:
            # Who Kip is must survive a restart, including what changed since the last save.
            try:
                persistent.save()
            except Exception:
                logging.getLogger("agentic_friend").exception("Could not save persona state at shutdown")
        if client is None:
            await llm.close()
            if chat_llm is not llm:
                await chat_llm.close()
        store.close()

    app = FastAPI(title="agentic_friend", lifespan=lifespan)
    app.state.manager, app.state.engine, app.state.bus = manager, engine, bus

    def snapshot() -> dict:
        latest = next(reversed(manager.tasks.values()), None) if manager.tasks else None
        return {"world": engine.snapshot(), "sequence": bus.sequence,
                "task": manager.active() or (latest.public() if latest else None)}

    @app.get("/api/beliefs")
    async def get_beliefs():
        return derive_beliefs(persistent)

    @app.post("/api/chat", status_code=202)
    async def send_chat(request: ChatRequest):
        try:
            return chat.submit(request.message, request.speaker)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None

    @app.get("/api/chat")
    async def chat_history():
        return {"turns": chat.history(), "busy": chat.busy, "name": chat.name, **chat.state()}

    @app.get("/api/memories")
    async def memories(q: str = "", kind: str = "", limit: int = 20):
        limit = max(1, min(limit, 100))
        kinds = [kind] if kind else None
        if q.strip():
            return await store.recall(q, kinds=kinds, k=limit, touch=False)
        return await asyncio.to_thread(store.recent, limit, kind or None)

    @app.get("/api/persona")
    async def get_persona():
        return persona.public() if persona else {"enabled": False}

    @app.get("/api/health")
    async def health():
        model = await llm.health() if client is None else {"mode": "test"}
        memory = {"meaning_search": store.vectors and bool(embedder and embedder.available),
                  "keyword_search": True, "persistent_error": persistent.load_error or None}
        return {"status": "ok" if not persistent.load_error else "degraded",
                "llm": {"provider": config.llm_provider, **model}, "memory": memory}

    @app.get("/api/world")
    async def world():
        return snapshot()

    @app.post("/api/world/reset")
    async def reset():
        async with mutation_lock:
            # Reset also discards pending replies/errands from the previous world.
            await chat.reset_pending()
            await manager.close()
            try:
                engine.reset()
            except (ValueError, OSError) as exc:
                raise HTTPException(422, "Cannot reload world definition: " + str(exc)) from None
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

        # Receive in the handler itself: a disconnect then ends the handler in one step,
        # instead of waking a separate receiver task first (which raced with shutdown).
        sender = asyncio.create_task(send())
        try:
            while not sender.done():
                await socket.receive_text()
            sender.result()  # surface a send failure
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)
            bus.unsubscribe(queue)

    return app


app = create_app()
