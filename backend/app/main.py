import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings, get_settings
from app.events.bus import EventBus
from app.chat import ChatSession
from app.llm.factory import create_client, create_hosted_client
from app.llm.priority import ConversationFirst
from app.memory.recall import Embedder, MemoryStore
from app.memory.store import PersistentMemory
from app.persona import Persona
from app.tasks.manager import TaskBusy, TaskManager

_log = logging.getLogger("agentic_friend")


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    message: str = Field(min_length=1, max_length=1000)
    # Who is talking (a name), so "I"/"my" facts are filed under them.
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
    bus = EventBus()
    llm = client if client is not None else create_client(config)
    # Injected test clients get a throwaway in-memory store and no embedding model,
    # unless the test hands in its own store (e.g. a file, to test restarts).
    live = client is None
    embedder = (Embedder(config.embedding_model, config.embedding_cache_dir)
                if live and config.embeddings_enabled else None)
    store = memory_store or MemoryStore(config.memory_db_file if live else ":memory:", embedder)
    # kip.db holds everything durable: memories, persona, recent errands and state.
    persistent = PersistentMemory.load(store)
    persona = None
    if config.persona_enabled:
        persona = Persona(persistent.persona, config.persona_name, now=time.time, maker=config.persona_maker)
        persona.attach(bus)
    # One local model for both: a person waiting for a reply goes before the next errand call.
    task_llm = ConversationFirst(llm)
    # Optional Groq for goal planning and web research; injected test clients never get it.
    hosted = create_hosted_client(config) if live else None
    chat_llm = llm
    if hosted is not None and config.chat_hosted:
        # Conversation on Groq while it has quota (about a second to the first word, against
        # minutes for a long prompt on a local CPU model); the local chat model otherwise.
        from app.llm.groq import HostedChat
        chat_llm = HostedChat(hosted, llm, [m.strip() for m in config.groq_chat_models.split(",") if m.strip()])
    manager = TaskManager(task_llm, bus, config, persistent, persona=persona, memory_store=store, hosted=hosted)
    chat = ChatSession(chat_llm, bus, manager, config, store=store, persona=persona,
                       session_id=config.chat_session_id)
    chat.gate = task_llm
    chat.attach()
    if config.intent_router != "off":
        # Reviewed corrections join the shipped examples; the embedding model joins once loaded.
        from app.chat.router import IntentRouter
        chat.router = IntentRouter(store.accepted_intent_examples(), embedder=embedder)
    mutation_lock = asyncio.Lock()

    def save_persona() -> None:
        if persona and persona.dirty:
            persona.dirty = False
            try:
                persistent.save()
            except Exception:
                persona.dirty = True
                _log.exception("Could not save persona state")

    async def keep_persona_saved() -> None:
        # Conversations change Kip: who he is must survive a crash, not only a clean shutdown.
        while True:
            await asyncio.sleep(5)
            save_persona()

    @asynccontextmanager
    async def lifespan(app):
        if embedder:
            # Load the embedding model in the background so the first reply doesn't wait,
            # then index any memories saved while meaning search was unavailable.
            async def prepare_memory():
                await asyncio.to_thread(embedder.embed_sync, ["warm up"])
                if chat.router is not None:
                    # A second or two in a worker thread; until then the router uses TF-IDF alone.
                    await asyncio.to_thread(chat.router.add_embeddings)
                await store.backfill()
            app.state.memory_ready = asyncio.create_task(prepare_memory())
        saver = asyncio.create_task(keep_persona_saved(), name="persona-saver")
        yield
        saver.cancel()
        await asyncio.gather(saver, return_exceptions=True)
        memory_ready = getattr(app.state, "memory_ready", None)
        if memory_ready and not memory_ready.done():
            memory_ready.cancel()  # never let a backfill touch the store after shutdown
            await asyncio.gather(memory_ready, return_exceptions=True)
        await chat.close()
        await manager.close()
        save_persona()  # including what changed since the last save
        if client is None:
            await llm.close()  # the hosted chat wrapper owns nothing of its own
        if hosted is not None:
            await hosted.close()
        store.close()

    app = FastAPI(title="agentic_friend", lifespan=lifespan)
    app.state.manager, app.state.bus = manager, bus
    # No login: Kip trusts whoever reaches him, so only his own pages and host names get in. The
    # Host check (HTTP and WebSocket) stops another site pointing its own name at this machine.
    origins = {o.strip().rstrip("/") for o in config.allowed_origins.split(",") if o.strip()}
    app.add_middleware(TrustedHostMiddleware,
                       allowed_hosts=[h.strip() for h in config.allowed_hosts.split(",") if h.strip()])

    def trusted_origin(origin: str | None) -> bool:
        return origin is None or origin.rstrip("/") in origins

    @app.middleware("http")
    async def same_site_changes(request: Request, call_next):
        # A page on another site can make the browser send a POST here (a chat message, a new task,
        # a cancel). Browsers name that page in Origin, so a foreign one is refused.
        if request.method not in ("GET", "HEAD", "OPTIONS") and not trusted_origin(request.headers.get("origin")):
            return JSONResponse({"detail": "Requests from other sites are not accepted."}, status_code=403)
        return await call_next(request)

    def snapshot() -> dict:
        latest = next(reversed(manager.tasks.values()), None) if manager.tasks else None
        return {"sequence": bus.sequence, "task": manager.active() or (latest.public() if latest else None)}

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
        result = {"status": "ok" if not persistent.load_error else "degraded",
                  "llm": {"provider": "ollama", **model}, "memory": memory}
        if chat_llm is not llm:
            # Conversation runs on Groq while it has quota; the local model only stands in.
            result["chat"] = {"provider": "groq", "model": chat_llm.models[0] if chat_llm.models else ""}
        return result

    @app.get("/api/state")
    async def state():
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
            raise HTTPException(404, "Task not found (only the latest 50 tasks are retained).")
        # The public view: no audit trail (page text, tool output), and a private answer's words hidden.
        return manager.tasks[task_id].public()

    @app.post("/api/tasks/{task_id}/cancel")
    async def cancel(task_id: str):
        async with mutation_lock:
            if task_id not in manager.tasks:
                raise HTTPException(404, "Task not found.")
            return (await manager.cancel(task_id)).public()

    @app.websocket("/ws")
    async def websocket(socket: WebSocket):
        # A browser always says which page opened the socket: only Kip's own may read the stream.
        # Clients that are not browsers (the benchmark) send no Origin and are already on this machine.
        if not trusted_origin(socket.headers.get("origin")):
            await socket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
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
