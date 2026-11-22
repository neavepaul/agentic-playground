import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from app.access import OwnHostsOnly, origin_allowed, own_names
from app.config import Settings, get_settings
from app.events.bus import EventBus
from app.chat import ChatSession
from app.chat.hub import ConversationHub
from app.chat.reminders import reminder_message
from app.chat.session import HISTORY
from app.clock import now as clock_now
from app.llm.factory import create_client, create_hosted_client
from app.llm.priority import ConversationFirst
from app.memory.recall import Embedder, MemoryStore
from app.memory.store import PersistentMemory
from app.people import COOKIE, SESSION_SECONDS, People, Profile, SignInError
from app.persona import Persona
from app.phone_push import PhonePush, Subscription
from app.scheduler import Scheduler
from app.tasks.manager import TaskBusy, TaskManager

_log = logging.getLogger("agentic_friend")


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    message: str = Field(min_length=1, max_length=1000)
    # The conversation to continue. A new id (the page makes one per new chat) or none starts one.
    conversation_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")


class NewProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str = Field(min_length=1, max_length=30)
    pin: str = Field(min_length=4, max_length=4)


class SignIn(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    profile_id: str = Field(min_length=1, max_length=64)
    pin: str = Field(min_length=1, max_length=8)


class FirstPin(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    pin: str = Field(min_length=4, max_length=4)


class Access(BaseModel):
    """What the owner allows someone; an omitted field is left as it is."""
    model_config = ConfigDict(extra="forbid")
    portfolio: bool | None = None
    pin_change: bool | None = None


class NewPin(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    current: str = Field(min_length=1, max_length=8)
    pin: str = Field(min_length=4, max_length=4)


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
    # Profiles, signed-in devices and each person's conversations. The chat from before profiles
    # (config.chat_session_id) becomes the first profile's, the owner's.
    people = People(store, legacy_session=config.chat_session_id)
    phone_push = PhonePush(store)
    router = None
    if config.intent_router != "off":
        # Reviewed corrections join the shipped examples; the embedding model joins once loaded.
        from app.chat.router import IntentRouter
        router = IntentRouter(store.accepted_intent_examples(), embedder=embedder)
    kip_name = persona.name if persona else config.persona_name

    def household() -> dict:
        """Who has a profile, and whose the Groww portfolio is: for Kip's status line."""
        everyone = people.profiles()
        return {"owner": next((p.name for p in everyone if p.owner), None), "people": [p.name for p in everyone]}

    async def deliver_scheduled(job, late: float) -> None:
        """A reminder, timer or scheduled check is due: it arrives in the conversation it was set in (or the
        person's latest one, if that was deleted), whatever Kip is doing. It takes no lock and no errand slot."""
        profile = people.get(job.owner)
        if profile is None:
            _log.warning("A %s was set by someone who no longer has a profile; dropped.", job.kind)
            return
        conversation = job.session_id
        if owner_of(conversation) != profile.id:
            recent = people.conversations(profile, 1)
            conversation = recent[0]["id"] if recent else people.new_conversation(profile, "Reminders")
            owners[conversation] = profile.id
        session = hub.get(conversation, profile)
        text = reminder_message(job, late)
        if job.kind == "lookup":
            session.run_scheduled_lookup(job.text, text)
        else:
            session.deliver(text)
        people.touch(conversation)
        await hub.trim()

    scheduler = Scheduler(store, clock_now, deliver_scheduled)

    def open_conversation(conversation_id: str, profile: Profile) -> ChatSession:
        session = ChatSession(chat_llm, bus, manager, config, store=store, persona=persona,
                              session_id=conversation_id, speaker=profile.name, owner=profile.owner,
                              portfolio=profile.portfolio, household=household, scheduler=scheduler,
                              owner_key=profile.id)
        session.gate = task_llm
        session.router = router
        session.attach()
        return session

    hub = ConversationHub(open_conversation, bus)
    owners: dict[str, str | None] = {}  # conversation id -> profile id (never changes once made)
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
        # Reopen durable queues and deliver journaled outcomes missed before a crash.
        with store._lock:
            rows = store._db.execute('SELECT id, profile_id FROM conversations').fetchall()
        for row in rows:
            queued = store.get_state(f"queue:{row['id']}")
            tasks = [t for t in manager.tasks.values() if t.session_id == row['id']]
            if queued not in (None, '', '[]') or tasks:
                profile = people.get(row['profile_id'])
                if profile is None:
                    continue
                session = hub.get(row['id'], profile)
                delivered = {t.get('task_id') for t in store.conversation_turns(row['id'])}
                for task in tasks:
                    if task.status in {'completed', 'failed', 'cancelled'} and task.id not in delivered:
                        if task.status == 'cancelled':
                            session._add_turn('kip', task.summary, source='narration', task_id=task.id)
                        else:
                            session._narrate({'type': f'task_{task.status}',
                                              'data': {'task': task.public(), 'summary': task.summary}})
        if embedder:
            # Load the embedding model in the background so the first reply doesn't wait,
            # then index any memories saved while meaning search was unavailable.
            async def prepare_memory():
                await asyncio.to_thread(embedder.embed_sync, ["warm up"])
                if router is not None:
                    # A second or two in a worker thread; until then the router uses TF-IDF alone.
                    await asyncio.to_thread(router.add_embeddings)
                await store.backfill()
            app.state.memory_ready = asyncio.create_task(prepare_memory())
        # The feed readers train now (half a second on the Pi), not on the first F1 or markets question.
        from app.capabilities.feeds import warm_readers
        app.state.readers_ready = asyncio.create_task(asyncio.to_thread(warm_readers))
        saver = asyncio.create_task(keep_persona_saved(), name="persona-saver")
        ticker = asyncio.create_task(scheduler.run(), name="scheduler")
        yield
        saver.cancel()
        ticker.cancel()
        await asyncio.gather(saver, ticker, return_exceptions=True)
        await scheduler.close()
        memory_ready = getattr(app.state, "memory_ready", None)
        if memory_ready and not memory_ready.done():
            memory_ready.cancel()  # never let a backfill touch the store after shutdown
            await asyncio.gather(memory_ready, return_exceptions=True)
        await hub.close()
        await manager.close()
        await phone_push.close()
        save_persona()  # including what changed since the last save
        if client is None:
            await llm.close()  # the hosted chat wrapper owns nothing of its own
        if hosted is not None:
            await hosted.close()
        store.close()

    app = FastAPI(title="agentic_friend", lifespan=lifespan)
    app.state.manager, app.state.bus, app.state.people, app.state.hub = manager, bus, people, hub
    app.state.scheduler = scheduler
    app.state.persona = persona
    # Everyone signs in to a profile (a 4-digit PIN). Only Kip's own pages and host names get in: the
    # Host check (HTTP and WebSocket) stops another site pointing its own name at this machine, and
    # a page may act only when Kip served it (access.py).
    origins = {o.strip().rstrip("/") for o in config.allowed_origins.split(",") if o.strip()}
    app.add_middleware(OwnHostsOnly, names=own_names(config.allowed_hosts))

    def trusted_origin(origin: str | None, host: str | None) -> bool:
        return origin_allowed(origin, host, origins)

    @app.middleware("http")
    async def same_site_changes(request: Request, call_next):
        # A page on another site can make the browser send a POST here (a chat message, a new task,
        # a cancel). Browsers name that page in Origin, so a foreign one is refused.
        changes = request.method not in ("GET", "HEAD", "OPTIONS")
        if changes and not trusted_origin(request.headers.get("origin"), request.headers.get("host")):
            return JSONResponse({"detail": "Requests from other sites are not accepted."}, status_code=403)
        return await call_next(request)

    # ── who is asking ───────────────────────────────────────────────────────
    def local_tool(client, headers) -> bool:
        """A program on the Pi itself (the benchmark), which acts as the owner. A browser page can
        never qualify: it always sends Origin, and cannot add this header to another site's request."""
        return (client is not None and client.host in ("127.0.0.1", "::1")
                and headers.get("x-kip-local") == "1" and "origin" not in headers)

    def signed_in(cookies, client, headers) -> Profile | None:
        profile = people.signed_in(cookies.get(COOKIE))
        if profile is None and local_tool(client, headers):
            profile = people.owner()
        return profile

    def who(request: Request) -> Profile:
        profile = signed_in(request.cookies, request.client, request.headers)
        if profile is None:
            raise HTTPException(401, "Sign in first.")
        return profile

    def owner_of(conversation_id: str | None) -> str | None:
        if conversation_id not in owners:
            owners[conversation_id] = people.conversation_owner(conversation_id)
        return owners[conversation_id]

    def notify_phone(event):
        if event["type"] != "chat_message" or not phone_push.public_key:
            return
        turn = event["data"]["turn"]
        if turn["role"] == "kip" and (turn.get("source") != "narration" or turn.get("task_id")):
            cid = turn["session_id"]
            phone_push.notify(owner_of(cid), cid)

    bus.add_listener(notify_phone)

    def task_owner(task_id: str | None, task: dict | None = None) -> tuple[bool, str | None]:
        """(is a task event, the profile id it belongs to). API tasks belong to the owner (None)."""
        context = manager.tasks.get(task_id) if task_id else None
        if context is None and not task:
            return False, None
        session_id = context.session_id if context is not None else (task or {}).get("session_id")
        return True, owner_of(session_id) if session_id else None

    def visible(event: dict, profile: Profile) -> bool:
        """Each person sees their own conversations and errands; Kip's mood is everyone's."""
        data = event.get("data") or {}
        session_id = data.get("session_id") or (data.get("turn") or {}).get("session_id")
        if session_id:
            return owner_of(session_id) == profile.id
        task = data.get("task") or {}
        is_task, belongs = task_owner(task.get("id") or data.get("task_id"), task)
        if is_task:
            return belongs == profile.id if belongs else profile.owner
        return True

    def task_visible(context, profile: Profile) -> bool:
        return owner_of(context.session_id) == profile.id if context.session_id else profile.owner

    def snapshot(profile: Profile) -> dict:
        mine = [task for task in manager.tasks.values() if task_visible(task, profile)]
        active = manager.tasks.get(manager.active_id) if manager.active_id else None
        if active is not None and not task_visible(active, profile):
            active = None
        latest = active or (mine[-1] if mine else None)
        return {"sequence": bus.sequence, "task": latest.public() if latest else None}

    def signed_in_response(response: Response, request: Request, profile: Profile) -> dict:
        phone_push.revoke_session(request.cookies.get(COOKIE))
        token = people.sign_in(profile)
        response.set_cookie(COOKIE, token, max_age=SESSION_SECONDS, httponly=True, samesite="strict",
                            secure=request.url.scheme == "https", path="/")
        return profile.public()

    # ── profiles and signing in ─────────────────────────────────────────────
    @app.get("/api/profiles")
    async def profiles(request: Request):
        me = signed_in(request.cookies, request.client, request.headers)
        return {"profiles": [{"id": p.id, "name": p.name, "needs_pin": p.needs_pin} for p in people.profiles()],
                "me": me.public() if me else None, "name": kip_name}

    @app.post("/api/profiles", status_code=201)
    async def create_profile(body: NewProfile, request: Request, response: Response):
        try:
            profile = await asyncio.to_thread(people.create, body.name, body.pin)
        except SignInError as exc:
            raise HTTPException(409, str(exc)) from None
        return signed_in_response(response, request, profile)

    @app.post("/api/signin")
    async def sign_in(body: SignIn, request: Request, response: Response):
        try:
            profile = await asyncio.to_thread(people.verify, body.profile_id, body.pin)
        except SignInError as exc:
            raise HTTPException(403, str(exc)) from None
        return signed_in_response(response, request, profile)

    @app.post("/api/profiles/{profile_id}/pin")
    async def choose_pin(profile_id: str, body: FirstPin, request: Request, response: Response):
        """A profile made without a PIN: its person chooses one, and is signed in."""
        try:
            profile = await asyncio.to_thread(people.choose_pin, profile_id, body.pin)
        except SignInError as exc:
            raise HTTPException(409, str(exc)) from None
        return signed_in_response(response, request, profile)

    @app.post("/api/signout", status_code=204)
    async def sign_out(request: Request, response: Response):
        phone_push.revoke_session(request.cookies.get(COOKIE))
        people.sign_out(request.cookies.get(COOKIE))
        response.delete_cookie(COOKIE, path="/")

    @app.get("/api/phone-push")
    async def push_settings(profile: Profile = Depends(who)):
        return {"public_key": phone_push.public_key}

    @app.post("/api/phone-push", status_code=204)
    async def push_subscribe(body: Subscription, request: Request, profile: Profile = Depends(who)):
        token = request.cookies.get(COOKIE)
        if not phone_push.public_key or not token or not body.valid():
            raise HTTPException(400, "Phone notifications are unavailable for this subscription.")
        phone_push.subscribe(profile.id, token, body)

    @app.delete("/api/phone-push", status_code=204)
    async def push_disable(request: Request, profile: Profile = Depends(who)):
        phone_push.revoke_session(request.cookies.get(COOKIE))

    @app.post("/api/chat-view/{tab}", status_code=204)
    async def chat_view(tab: str, request: Request, conversation: str = "", profile: Profile = Depends(who)):
        token = request.cookies.get(COOKIE)
        if not phone_push.public_key or not token:
            return
        if conversation:
            own_conversation(conversation, profile)
        phone_push.view(token, tab[:64], profile.id, conversation)

    @app.get("/api/me")
    async def me(profile: Profile = Depends(who)):
        result = profile.public()
        if profile.owner:
            result["people"] = [p.public() for p in people.profiles() if not p.owner]
        return result

    @app.post("/api/me/pin")
    async def change_pin(body: NewPin, request: Request, response: Response, profile: Profile = Depends(who)):
        """Change your own PIN (when the owner allows it). Other devices are signed out; this one stays."""
        if not profile.pin_change:
            raise HTTPException(403, "The owner hasn't allowed this profile to change its PIN.")
        try:
            changed = await asyncio.to_thread(people.change_pin, profile.id, body.current, body.pin)
        except SignInError as exc:
            raise HTTPException(403, str(exc)) from None
        return signed_in_response(response, request, changed)

    @app.post("/api/profiles/{profile_id}/access")
    async def set_access(profile_id: str, body: Access, profile: Profile = Depends(who)):
        if not profile.owner:
            raise HTTPException(403, "Only the owner can change what people may do.")
        try:
            changed = people.set_access(profile_id, portfolio=body.portfolio, pin_change=body.pin_change)
        except SignInError as exc:
            raise HTTPException(404, str(exc)) from None
        for conversation in people.conversations(changed):
            session = hub.peek(conversation["id"])
            if session is not None:
                session.portfolio = session.manager.portfolio = changed.portfolio
        return changed.public()

    # ── conversations ───────────────────────────────────────────────────────
    def own_conversation(conversation_id: str, profile: Profile) -> str:
        if owner_of(conversation_id) != profile.id:
            raise HTTPException(404, "Conversation not found.")
        return conversation_id

    @app.get("/api/conversations")
    async def conversations(profile: Profile = Depends(who), q: str = ""):
        query = q.strip()[:200]
        matches = await asyncio.to_thread(people.search_conversations, profile, query) if query else people.conversations(profile)
        return {"conversations": matches}

    @app.delete("/api/conversations/{conversation_id}", status_code=204)
    async def delete_conversation(conversation_id: str, profile: Profile = Depends(who)):
        own_conversation(conversation_id, profile)
        await hub.drop(conversation_id)
        store.set_state({f'queue:{conversation_id}': '[]'})
        people.delete_conversation(conversation_id)
        owners.pop(conversation_id, None)

    @app.post("/api/chat", status_code=202)
    async def send_chat(request: ChatRequest, profile: Profile = Depends(who)):
        conversation_id = request.conversation_id
        if conversation_id and owner_of(conversation_id) is not None:
            own_conversation(conversation_id, profile)  # someone else's id is "not found", never theirs
        else:
            conversation_id = people.new_conversation(profile, request.message, conversation_id)
            owners[conversation_id] = profile.id
        session = hub.get(conversation_id, profile)
        try:
            turn = session.submit(request.message)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None
        people.touch(conversation_id)
        await hub.trim()
        return turn

    @app.get("/api/chat")
    async def chat_history(conversation_id: str = "", profile: Profile = Depends(who)):
        if not conversation_id:
            return {"turns": [], "busy": False, "name": kip_name, "session_id": None}
        own_conversation(conversation_id, profile)
        session = hub.peek(conversation_id)
        if session is not None:
            return {"turns": session.history(), "busy": session.busy, "name": kip_name, **session.state()}
        turns = await asyncio.to_thread(store.conversation_turns, conversation_id, HISTORY)
        return {"turns": turns, "busy": False, "name": kip_name, "session_id": conversation_id}

    @app.get("/api/memories")
    async def memories(q: str = "", kind: str = "", limit: int = 20, profile: Profile = Depends(who)):
        limit = max(1, min(limit, 100))
        kinds = [kind] if kind else None
        whose = {"viewer": profile.speaker, "legacy": profile.owner}
        if q.strip():
            return await store.recall(q, kinds=kinds, k=limit, touch=False, **whose)
        return await asyncio.to_thread(lambda: store.recent(limit, kind or None, include_superseded=False, **whose))

    class MemoryEdit(BaseModel):
        text: str = Field(max_length=1200)

    async def change_memory(memory_id: int, profile: Profile, replacement: str | None):
        changed = await asyncio.to_thread(store.manage_memory, memory_id, profile.speaker,
                                          profile.id, profile.owner, replacement)
        if not changed:
            raise HTTPException(404, "Editable memory not found.")

    @app.patch("/api/memories/{memory_id}", status_code=204)
    async def edit_memory(memory_id: int, body: MemoryEdit, profile: Profile = Depends(who)):
        replacement = body.text.strip()
        if not replacement:
            raise HTTPException(422, "Enter the corrected memory.")
        await change_memory(memory_id, profile, replacement)

    @app.delete("/api/memories/{memory_id}", status_code=204)
    async def forget_memory(memory_id: int, profile: Profile = Depends(who)):
        await change_memory(memory_id, profile, None)

    @app.get("/api/persona")
    async def get_persona(profile: Profile = Depends(who)):
        return persona.public(profile.speaker, legacy=profile.owner) if persona else {"enabled": False}

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
    async def state(profile: Profile = Depends(who)):
        return snapshot(profile)

    @app.post("/api/tasks", status_code=202)
    async def create_task(request: GoalRequest, profile: Profile = Depends(who)):
        if not profile.owner:
            raise HTTPException(403, "Only the owner can start errands directly.")
        async with mutation_lock:
            try:
                return manager.start(request.goal, speaker=profile.speaker, owner=True).public()
            except TaskBusy as exc:
                raise HTTPException(409, str(exc)) from None

    @app.get("/api/tasks/{task_id}")
    async def task(task_id: str, profile: Profile = Depends(who)):
        if task_id not in manager.tasks or not task_visible(manager.tasks[task_id], profile):
            raise HTTPException(404, "Task not found (only the latest 50 tasks are retained).")
        # The public view: no audit trail (page text, tool output), and a private answer's words hidden.
        return manager.tasks[task_id].public()

    @app.post("/api/tasks/{task_id}/cancel")
    async def cancel(task_id: str, profile: Profile = Depends(who)):
        async with mutation_lock:
            if task_id not in manager.tasks or not task_visible(manager.tasks[task_id], profile):
                raise HTTPException(404, "Task not found.")
            return (await manager.cancel(task_id)).public()

    @app.websocket("/ws")
    async def websocket(socket: WebSocket):
        # A browser always says which page opened the socket: only Kip's own may read the stream.
        # Clients that are not browsers (the benchmark) send no Origin and are already on this machine.
        profile = signed_in(socket.cookies, socket.client, socket.headers)
        if not trusted_origin(socket.headers.get("origin"), socket.headers.get("host")) or profile is None:
            await socket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        await socket.accept()
        queue = bus.subscribe()

        def replay() -> dict:
            # Subscribed before this snapshot is taken: no startup race can lose an action.
            return {"type": "snapshot", "data": snapshot(profile),
                    "events": [event for event in bus.history if visible(event, profile)]}
        await socket.send_json(replay())

        async def send():
            while True:
                event = await queue.get()
                if event["type"] == "resync_required":
                    await socket.send_json(replay())
                elif event["type"] == "persona_updated" and persona is not None:
                    # Kip's mood is everyone's; his relationship and moments with a person are theirs.
                    mine = persona.public(profile.speaker, legacy=profile.owner)
                    await socket.send_json({**event, "data": {**event["data"], "persona": mine}})
                elif visible(event, profile):
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

    # The built page (pnpm build), when there is one: on the Pi one service serves Kip whole, with no
    # Vite dev server. Mounted last, so /api and /ws always win.
    if (config.frontend_dist / "index.html").is_file():
        app.mount("/", StaticFiles(directory=config.frontend_dist, html=True), name="frontend")
    return app


app = create_app()
