"""Conversation with Kip: streaming, hidden action trailers, errands and memory."""
import asyncio
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app.chat import ChatSession
from app.chat.prompt import TrailerSplitter, build_messages, parse_action
from app.config import Settings
from app.events.bus import EventBus
from app.llm.base import ModelError
from app.llm.colibri import ColibriClient
from app.llm.ollama import OllamaClient
from app.memory.recall import MemoryStore
from app.mind.agent import AgentMind
from app.persona import Persona
from app.tasks.manager import TaskManager
from app.world.engine import WorldEngine
from app.world.tools import WorldTools
from tests.fakes import WaitingLLM

LEGACY_WORLD = Path(__file__).parent / "fixtures" / "legacy_house.json"


class StreamingModel(WaitingLLM):
    """Streams scripted replies in small chunks; task calls (generate) wait forever."""

    def __init__(self, replies):
        self.replies = iter(replies)
        self.prompts = []
        self.last_metrics = {}

    async def stream_text(self, messages, max_tokens=300, temperature=None):
        self.prompts.append(messages)
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        for start in range(0, len(reply), 3):
            yield reply[start:start + 3]
        self.last_metrics = {"first_token_seconds": 0.01, "total_seconds": 0.02}


def make_session(replies, store=None):
    engine, bus = WorldEngine(LEGACY_WORLD), EventBus()
    model = StreamingModel(replies)
    manager = TaskManager(model, WorldTools(engine, bus), bus, Settings())
    persona = Persona()
    session = ChatSession(model, bus, manager, engine, Settings(), store=store or MemoryStore(":memory:"),
                          persona=persona)
    session.attach()
    events = []
    bus.add_listener(events.append)
    return session, model, manager, events


def reply(text, task="", remember=()):
    return f"{text}\n###\n" + json.dumps({"task": task, "remember": list(remember)})


@pytest.mark.asyncio
async def test_reply_streams_without_leaking_the_trailer():
    session, _, _, events = make_session([reply("Hello! I'm doing well, thank you. Item #3 is fine.")])
    session.submit("Hi Kip")
    await session.drain()
    deltas = [e["data"]["text"] for e in events if e["type"] == "chat_delta"]
    assert "".join(deltas) == "Hello! I'm doing well, thank you. Item #3 is fine."
    assert not any("#" * 3 in d or "{" in d for d in deltas)
    kip = session.history()[-1]
    assert kip["role"] == "kip" and kip["text"].startswith("Hello!") and kip["metrics"]["total_seconds"]
    assert [e["type"] for e in events if e["type"].startswith("chat_")][:2] == ["chat_message", "chat_thinking"]


@pytest.mark.parametrize("chunks", [["Hi", "\n#", "##\n{\"task\": \"x\"}"], ["Hi\n###", "\n{}"], ["###\n{}"]])
def test_splitter_handles_markers_split_across_chunks(chunks):
    splitter = TrailerSplitter()
    visible = "".join(splitter.feed(chunk) for chunk in chunks)
    remaining, full, trailer = splitter.finish()
    assert "#" not in visible + remaining
    assert full == ("" if chunks[0].startswith("###") else "Hi")
    assert parse_action(trailer).task in {"", "x"}


@pytest.mark.asyncio
async def test_task_trailer_starts_an_errand_and_queues_the_next():
    session, _, manager, _ = make_session([reply("Sure, I'll look for the keys.", task="Find the missing keys."),
                                           reply("I'll do that next.", task="Find Neave and tell him dinner is ready.")])
    session.submit("Could you find the keys?")
    await session.drain()
    assert manager.active()["goal"] == "Find the missing keys."
    assert any(t["text"] == "On it: Find the missing keys." for t in session.history())
    session.submit("Also tell Neave dinner is ready")
    await session.drain()
    assert list(session.pending_goals) == ["Find Neave and tell him dinner is ready."]
    assert "right after my current errand" in session.history()[-1]["text"]
    await manager.cancel(manager.active_id)
    await session.close()
    await manager.close()


@pytest.mark.asyncio
async def test_broken_or_missing_trailer_only_replies():
    session, _, manager, _ = make_session(["Just chatting, no trailer here.", "Hmm\n###\n{not json"])
    session.submit("Hello")
    session.submit("Again")
    await session.drain()
    assert manager.active_id is None
    assert [t["text"] for t in session.history() if t["role"] == "kip"] == ["Just chatting, no trailer here.", "Hmm"]


@pytest.mark.asyncio
async def test_remembered_facts_are_recalled_in_later_turns():
    store = MemoryStore(":memory:")
    session, model, _, _ = make_session([reply("Noted!", remember=["Neave prefers green tea after 6pm"]),
                                         reply("Green tea, after six.")], store)
    session.submit("Remember that Neave prefers green tea after 6pm")
    await session.drain()
    facts = store.recent(10, "fact")
    assert facts[0]["text"] == "Neave prefers green tea after 6pm" and facts[0]["about"] == ["neave"]
    assert store.recent(10, "conversation")
    session.submit("What tea does Neave like?")
    await session.drain()
    system = model.prompts[-1][0]["content"]
    assert "Neave prefers green tea after 6pm" in system
    # Earlier turns keep their trailers so the model keeps the format.
    assert any(m["role"] == "assistant" and "###" in m["content"] for m in model.prompts[-1])


@pytest.mark.asyncio
async def test_errand_outcomes_are_narrated_without_model_calls():
    session, model, manager, _ = make_session([])
    manager.bus.emit("task_completed", task={"goal": "Look it up", "answer": {
        "answer": "It is 42.", "sources": [{"title": "Calc", "url": "https://example.org"}]}}, summary="Done.")
    narration = session.history()[-1]
    assert narration["source"] == "narration" and "It is 42." in narration["text"]
    assert narration["sources"][0]["url"] == "https://example.org"
    assert model.prompts == []


@pytest.mark.asyncio
async def test_model_errors_become_a_friendly_reply():
    session, _, _, _ = make_session([ModelError("offline")])
    session.submit("Hello?")
    await session.drain()
    assert "language model isn't answering" in session.history()[-1]["text"]
    assert not session.busy


def test_chat_prompt_stays_small():
    memories = [{"kind": "fact", "text": "x" * 1000}] * 6
    turns = [{"role": "user" if i % 2 else "kip", "text": "y" * 1000} for i in range(30)]
    messages = build_messages("Kip", Persona().voice_view(), memories, {"room": "hall", "inventory": []},
                              turns, "z" * 2000)
    assert len(messages[0]["content"]) < 3500
    assert len(messages) == 1 + 8 + 1
    assert sum(len(m["content"]) for m in messages) < 9000
    assert messages[-1]["content"] == "z" * 2000


@pytest.mark.asyncio
async def test_current_message_keeps_trailing_constraints_and_code_formatting():
    session, model, _, _ = make_session([reply("Understood.")])
    text = "A" * 450 + "\nPlease preserve this constraint.\nfor i in range(2):\n    print(i)"
    session.submit(text)
    await session.drain()
    assert model.prompts[0][-1]["content"] == text
    assert session.history()[0]["text"] == text


@pytest.mark.asyncio
async def test_rapid_messages_keep_each_question_and_reply_in_order():
    session, model, _, _ = make_session([reply("First reply"), reply("Second reply"), reply("Third reply")])
    session.submit("First message")
    session.submit("Second message")
    session.submit("Third message")
    await session.drain()
    assert [m["content"].split("\n###")[0] for m in model.prompts[0][1:]] == ["First message"]
    assert [m["content"].split("\n###")[0] for m in model.prompts[1][1:]] == [
        "First message", "First reply", "Second message"]
    assert [m["content"].split("\n###")[0] for m in model.prompts[2][1:]] == [
        "First message", "First reply", "Second message", "Second reply", "Third message"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["recall", "add"])
async def test_memory_failure_does_not_block_reply_or_promised_errand(failure):
    store = MemoryStore(":memory:")
    async def broken(*args, **kwargs):
        raise OSError("database unavailable")
    setattr(store, failure, broken)
    session, model, manager, _ = make_session([
        reply("I'll find the keys.", task="Find keys.", remember=["Neave likes tea"]), reply("Still here.")], store)
    session.submit("Remember I like tea, and find my keys.")
    await session.drain()
    assert manager.active()["goal"] == "Find keys."
    assert any(t["text"] == "I'll find the keys." for t in session.history())
    session.submit("Are you still there?")
    await session.drain()
    assert session.history()[-1]["text"] == "Still here."
    assert len(model.prompts) == 2 and not session.busy
    await session.close()
    await manager.close()


@pytest.mark.asyncio
async def test_cancelled_errand_starts_the_next_queued_errand():
    session, _, manager, _ = make_session([
        reply("On it.", task="Find keys."), reply("Next.", task="Tell Neave dinner is ready.")])
    session.submit("Find keys")
    await session.drain()
    first = manager.active_id
    session.submit("Then tell Neave dinner is ready")
    await session.drain()
    await manager.cancel(first)
    await asyncio.wait_for(session.drain(), 1)
    assert manager.active()["goal"] == "Tell Neave dinner is ready."
    assert manager.active_id != first
    assert not session.pending_goals
    await session.close()
    await manager.close()


@pytest.mark.asyncio
async def test_close_cancels_generation_and_pending_messages_without_starting_work():
    session, model, manager, _ = make_session([])
    entered, stopped = asyncio.Event(), asyncio.Event()
    async def blocked(*args, **kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()
        yield "never"
    model.stream_text = blocked
    session.submit("Hello")
    session.submit("Follow up")
    session.pending_goals.append("Find keys.")
    await entered.wait()
    await asyncio.wait_for(session.close(), 1)
    assert stopped.is_set() and not session.busy and not session.pending_goals
    manager.bus.emit("task_completed", task={}, summary="Done")
    assert manager.active_id is None
    with pytest.raises(ValueError, match="shutting down"):
        session.submit("After shutdown")


@pytest.mark.asyncio
async def test_reset_discards_old_pending_work_and_accepts_new_chat():
    session, _, manager, _ = make_session([reply("Hello again.")])
    session.pending_goals.append("Old world errand")
    await session.reset_pending()
    session.submit("New world")
    await session.drain()
    assert session.history()[-1]["text"] == "Hello again."
    assert not session.pending_goals and manager.active_id is None


def test_memory_tags_match_names_and_ids_without_substring_collisions():
    session, _, _, _ = make_session([])
    session.engine = WorldEngine()
    assert "minu" not in session._mentioned("Calculate the minutes; minimum effort.")
    assert "paul" not in session._mentioned("Pauline likes tea.")
    assert set(session._mentioned("MINU, could you ask Neave's friend Paul?")) == {"neave", "paul", "minu"}


@pytest.mark.asyncio
async def test_idle_mind_waits_while_a_conversation_is_active():
    session, model, manager, _ = make_session([reply("Hi!")])
    mind = AgentMind(model, manager, manager.tools._engine, manager.bus, Settings(), chat=session)
    assert not mind._in_conversation()
    session.submit("Hello")
    assert mind._in_conversation()
    await session.drain()
    assert mind._in_conversation()  # still within the quiet period


@pytest.mark.asyncio
async def test_colibri_streams_server_sent_events():
    body = ("data: " + json.dumps({"choices": [{"delta": {"content": "Hel"}}]}) + "\n\n"
            "data: " + json.dumps({"choices": [{"delta": {"content": "lo"}}]}) + "\n\ndata: [DONE]\n\n")
    seen = {}
    def respond(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
    client = ColibriClient(Settings())
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(respond), base_url="http://colibri/")
    assert "".join([chunk async for chunk in client.stream_text([{"role": "user", "content": "hi"}])]) == "Hello"
    assert seen["stream"] is True and seen["enable_thinking"] is False
    assert "first_token_seconds" in client.last_metrics
    await client.close()


@pytest.mark.asyncio
async def test_colibri_records_stream_usage_without_changing_the_reply():
    events = [{"choices": [{"delta": {"content": "Hello"}}]},
              {"choices": [], "usage": {"prompt_tokens": 120, "completion_tokens": 7}}]
    body = "".join("data: " + json.dumps(event) + "\n\n" for event in events) + "data: [DONE]\n\n"
    def respond(request):
        assert json.loads(request.content)["stream_options"] == {"include_usage": True}
        return httpx.Response(200, text=body)
    client = ColibriClient(Settings())
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(respond), base_url="http://colibri/")
    assert "".join([chunk async for chunk in client.stream_text([])]) == "Hello"
    assert client.last_metrics["prompt_tokens"] == 120
    assert client.last_metrics["completion_tokens"] == 7
    await client.close()


@pytest.mark.asyncio
async def test_colibri_truncated_reply_is_reported_as_an_error():
    body = 'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n'
    client = ColibriClient(Settings())
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, text=body)), base_url="http://colibri/")
    with pytest.raises(ModelError, match="reply limit"):
        async for _ in client.stream_text([]):
            pass
    await client.close()


@pytest.mark.asyncio
async def test_ollama_streams_ndjson():
    lines = [{"message": {"content": "Hi "}}, {"message": {"content": "there"}}, {"done": True}]
    body = "\n".join(json.dumps(line) for line in lines)
    seen = {}
    def respond(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, text=body)
    client = OllamaClient(Settings())
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(respond), base_url="http://ollama")
    assert "".join([chunk async for chunk in client.stream_text([{"role": "user", "content": "hi"}])]) == "Hi there"
    assert seen["stream"] is True and "format" not in seen
    await client.close()


def test_chat_and_memories_api():
    from app.main import create_app
    with TestClient(create_app(client=StreamingModel([reply("Hello from Kip!")]))) as client:
        assert client.post("/api/chat", json={"message": "  "}).status_code == 422
        turn = client.post("/api/chat", json={"message": "Hi Kip"})
        assert turn.status_code == 202 and turn.json()["role"] == "user"
        for _ in range(100):
            history = client.get("/api/chat").json()
            if len(history["turns"]) >= 2:
                break
            import time
            time.sleep(0.02)
        assert history["turns"][-1]["text"] == "Hello from Kip!"
        assert client.get("/api/memories", params={"q": "Kip"}).json()[0]["kind"] == "conversation"
        assert client.get("/api/memories").json()
