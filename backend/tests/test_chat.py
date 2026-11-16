"""Conversation with Kip: streaming, hidden action trailers, errands and memory."""
import asyncio
import json
import httpx
import pytest
from fastapi.testclient import TestClient

from app.chat import ChatSession
from app.chat.prompt import TrailerSplitter, build_messages
from app.config import Settings
from app.events.bus import EventBus
from app.llm.base import ModelError
from app.llm.ollama import OllamaClient
from app.memory.recall import MemoryStore
from app.persona import Persona
from app.persona.state import Relationship
from app.tasks.manager import TaskManager
from tests.fakes import WaitingLLM


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
    bus = EventBus()
    model = StreamingModel(replies)
    manager = TaskManager(model, bus, Settings())
    persona = Persona()
    session = ChatSession(model, bus, manager, Settings(), store=store or MemoryStore(":memory:"), persona=persona)
    session.attach()
    events = []
    bus.add_listener(events.append)
    return session, model, manager, events


def spoken(content):
    """The person's own words inside the latest message (context travels with them)."""
    return content.split("They say: ", 1)[1].rsplit("\n\n(Reply, then", 1)[0]


def conversation(prompt):
    return [m["content"] for m in prompt[1:-1]] + [spoken(prompt[-1]["content"])]


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
    remaining, parsed = splitter.finish()
    assert "#" not in visible + remaining
    assert parsed.visible == ("" if chunks[0].startswith("###") else "Hi")
    assert (parsed.action.task if parsed.action else "") in {"", "x"}


@pytest.mark.asyncio
async def test_task_trailer_starts_an_errand_and_queues_the_next():
    session, _, manager, _ = make_session([reply("Sure, I'll look for the keys.", task="Find the missing keys."),
                                           reply("I'll do that next.", task="Find Neave and tell him dinner is ready.")])
    session.submit("Could you find the keys?")
    await session.drain()
    assert manager.active()["goal"] == "Find the missing keys."
    # Kip's own reply acknowledged it; no second "Start now" narration.
    assert not any(t["text"].startswith("Start now") for t in session.history())
    session.submit("Also tell Neave dinner is ready")
    await session.drain()
    assert list(session.pending_goals) == ["Find Neave and tell him dinner is ready."]
    assert "Right after, I do" in session.history()[-1]["text"]
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
    session.persona.state.relationships["neave"] = Relationship()  # someone Kip knows
    session.submit("Remember that Neave prefers green tea after 6pm")
    await session.drain()
    facts = store.recent(10, "preference")  # likes/prefers are filed as preferences
    assert facts[0]["text"] == "Neave prefers green tea after 6pm" and facts[0]["about"] == ["neave"]
    assert store.recent(10, "conversation")
    session.submit("What tea does Neave like?")
    await session.drain()
    assert "Neave prefers green tea after 6pm" in model.prompts[-1][-1]["content"]
    # Earlier replies carry no action lines, so old errands and facts are not copied.
    assert not any(m["role"] == "assistant" and "###" in m["content"] for m in model.prompts[-1])


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
    assert "language model not answer" in session.history()[-1]["text"]
    assert not session.busy


@pytest.mark.asyncio
async def test_an_explicit_tool_request_starts_at_once_without_a_chat_reply():
    """Live: "Use Python to…" waited 914 s for a chat reply on the local model, then the errand
    itself took 2.7 s. Like a live lookup, the errand's answer is the reply."""
    session, model, manager, _ = make_session([])  # no chat reply scripted: none may be asked for
    session.submit("Use Python to calculate the average of 12, 18, 21 and 29.")
    await session.drain()
    assert manager.active()["goal"] == "Use Python to calculate the average of 12, 18, 21 and 29."
    assert model.prompts == []
    await manager.cancel(manager.active_id)
    await session.close()
    await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("said", ["Offer to use Python to add 2 and 2, but wait for my ok.",
                                  "Don't use Python. Use Python to add 2 and 2."])
async def test_a_tool_request_that_needs_a_word_first_still_gets_a_chat_reply(said):
    session, model, manager, _ = make_session([reply("Hm.")])
    session.submit(said)
    await session.drain()
    assert len(model.prompts) == 1 and manager.active_id is None
    await session.close()
    await manager.close()


@pytest.mark.asyncio
async def test_a_requested_errand_still_starts_when_the_chat_model_fails():
    """Live: the chat model never answered, so no errand started and the request was lost. Here
    the person says yes to Kip's offer while the chat model is down."""
    offer = "Ok.\n###\n" + json.dumps({"offer": "Find the missing keys.", "remember": []})
    session, _, manager, _ = make_session([offer, ModelError("offline")])
    session.submit("I can't find my keys anywhere.")
    await session.drain()
    assert session.pending_offer == "Find the missing keys." and manager.active_id is None
    session.submit("Yes, please.")
    await session.drain()
    assert manager.active()["goal"] == "Find the missing keys."
    reply = session.history()[-1]["text"]
    assert "language model not answer" in reply and "errand" in reply.casefold()
    await manager.cancel(manager.active_id)
    await session.close()
    await manager.close()


def test_chat_prompt_stays_small():
    memories = [{"kind": "fact", "text": "x" * 1000}] * 6
    turns = [{"role": "user" if i % 2 else "kip", "text": "y" * 1000} for i in range(30)]
    # Includes Kip's identity and the per-reply stance line (the character's whole runtime cost).
    voice = Persona().voice_view(speaker=None, message="z" * 2000)
    messages = build_messages("Kip", voice, memories, {"now": "Sunday 1 November 2026", "current_errand": "none"}, turns, "z" * 2000)
    from app.chat.context import message_tokens
    assert message_tokens(messages) <= 6000
    # History is selected by budget, not an arbitrary turn count or character slice.
    assert all(m["content"] == "y" * 1000 for m in messages[1:-1])
    assert spoken(messages[-1]["content"]) == "z" * 2000


@pytest.mark.asyncio
async def test_current_message_keeps_trailing_constraints_and_code_formatting():
    session, model, _, _ = make_session([reply("Understood.")])
    text = "A" * 450 + "\nPlease preserve this constraint.\nfor i in range(2):\n    print(i)"
    session.submit(text)
    await session.drain()
    assert spoken(model.prompts[0][-1]["content"]) == text
    assert session.history()[0]["text"] == text


@pytest.mark.asyncio
async def test_rapid_messages_keep_each_question_and_reply_in_order():
    session, model, _, _ = make_session([reply("First reply"), reply("Second reply"), reply("Third reply")])
    session.submit("First message")
    session.submit("Second message")
    session.submit("Third message")
    await session.drain()
    assert conversation(model.prompts[0]) == ["First message"]
    assert conversation(model.prompts[1]) == ["First message", "First reply", "Second message"]
    assert conversation(model.prompts[2]) == [
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
    session.pending_goals.append("Old errand")
    await session.reset_pending()
    session.submit("Fresh start")
    await session.drain()
    assert session.history()[-1]["text"] == "Hello again."
    assert not session.pending_goals and manager.active_id is None


def test_memory_tags_match_names_and_ids_without_substring_collisions():
    session, _, _, _ = make_session([])
    for person in ("neave", "paul", "minu"):
        session.persona.state.relationships[person] = Relationship()
    assert "minu" not in session._mentioned("Calculate the minutes; minimum effort.")
    assert "paul" not in session._mentioned("Pauline likes tea.")
    assert set(session._mentioned("MINU, could you ask Neave's friend Paul?")) == {"neave", "paul", "minu"}


@pytest.mark.asyncio
async def test_ollama_records_token_usage_without_changing_the_reply():
    lines = [{"message": {"content": "Hello"}}, {"done": True, "prompt_eval_count": 120, "eval_count": 7}]
    client = OllamaClient(Settings())
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, text="\n".join(json.dumps(l) for l in lines))), base_url="http://ollama")
    assert "".join([chunk async for chunk in client.stream_text([])]) == "Hello"
    assert client.last_metrics["prompt_tokens"] == 120 and client.last_metrics["completion_tokens"] == 7
    await client.close()


@pytest.mark.asyncio
async def test_ollama_truncated_reply_is_flagged_not_reported_as_a_dead_model():
    """A reply cut off at the token limit is still a working model: flag it, keep the text."""
    lines = [{"message": {"content": "Partial an"}}, {"message": {"content": "swer"}}, {"done": True, "done_reason": "length"}]
    client = OllamaClient(Settings())
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, text="\n".join(json.dumps(l) for l in lines))), base_url="http://ollama")
    assert "".join([chunk async for chunk in client.stream_text([])]) == "Partial answer"
    assert client.last_metrics["truncated"] is True
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
    assert seen["stream"] is True and "format" not in seen and seen["keep_alive"] == -1  # kept loaded on a Pi
    await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("setting,sent", [("-1", -1), ("30m", "30m")])
async def test_ollama_keeps_the_model_loaded_when_asked(setting, sent):
    """On a Pi, reloading a model from a microSD card after Ollama's 5 idle minutes takes tens of seconds."""
    bodies = []

    def respond(request):
        bodies.append(json.loads(request.content))
        if bodies[-1]["stream"]:
            return httpx.Response(200, text=json.dumps({"message": {"content": "Hi"}, "done": True}))
        return httpx.Response(200, json={"message": {"content": "{}"}})
    client = OllamaClient(Settings(ollama_keep_alive=setting))
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(respond), base_url="http://ollama")
    assert "".join([chunk async for chunk in client.stream_text([{"role": "user", "content": "hi"}])]) == "Hi"
    await client.generate([{"role": "user", "content": "JSON"}], {"title": "GoalPlan"})
    assert [body["keep_alive"] for body in bodies] == [sent, sent]
    await client.close()


def test_the_prompt_is_small_and_keeps_the_character_and_contract():
    """A Pi reads every prompt in full at a few tokens a second: the system message stays a few
    hundred tokens, and each reply adds up to four whole memories, the status that matters and capped history."""
    voice = Persona().voice_view(speaker=None, message="hi")
    status = {"now": "Saturday 14 November 2026, 12:31", "talking_with": "Neave", "current_errand": "none",
              "my_tools": "available: search the web, read webpages, run Python",
              "recent_errands": [{"goal": "one"}, {"goal": "two"}], "queued_errands": []}
    memories = [{"kind": "fact", "text": "Neave's favourite fruit is mango."}, {"kind": "fact", "text": "x" * 301},
                {"kind": "fact", "text": "Neave supports McLaren."}, {"kind": "fact", "text": "Neave has a cat."},
                {"kind": "fact", "text": "Neave plays chess."}, {"kind": "fact", "text": "Neave owns a bike."}]
    turns = [{"id": str(i), "role": "user" if i % 2 == 0 else "kip", "text": "words " * 40} for i in range(20)]
    messages = build_messages("Kip", voice, memories, status, turns, "What's my favourite fruit?", history_tokens=200)
    system, latest = messages[0]["content"], messages[-1]["content"]
    assert len(system) < 2400 and "Rocky" in system and "', question?'" in system
    assert '{"task": "", "offer": ""}' in system and "remember" not in system  # the action line
    # Four whole memories at most; one too long to keep whole is left out, never cut.
    assert "mango" in latest and "McLaren" in latest and "cat" in latest and "chess" in latest
    assert "bike" not in latest and "x" * 50 not in latest
    assert '"recent_errands":[{"goal":"two"}]' in latest and "queued_errands" not in latest
    assert 0 < len(messages) - 2 < len(turns)  # recent history only
    # The system message is the same every reply, so Ollama can keep it cached.
    again = build_messages("Kip", Persona().voice_view(speaker="neave", message="other"), [], status, [], "Hello")
    assert again[0]["content"] == system


def test_a_long_message_is_never_refused_by_the_history_budget():
    long = "Please read this carefully. " * 300
    messages = build_messages("Kip", Persona().voice_view(speaker=None, message=long), [], {"now": "x"},
                              [], long, history_tokens=0)
    assert messages[-1]["content"].count("Please read this carefully.") == 300


class GroqAnswered(StreamingModel):
    """Replies as HostedChat reports them when Groq wrote them."""

    async def stream_text(self, messages, max_tokens=300, temperature=None):
        async for chunk in super().stream_text(messages, max_tokens, temperature):
            yield chunk
        self.last_metrics = {**self.last_metrics, "provider": "groq"}


def session_with_settings(model, **overrides):
    bus, settings = EventBus(), Settings(**overrides)
    manager, store = TaskManager(model, bus, settings), MemoryStore(":memory:")
    session = ChatSession(model, bus, manager, settings, store=store, persona=Persona())
    session.attach()
    return session, manager, store


@pytest.mark.asyncio
async def test_a_small_local_models_errands_are_ignored_but_its_memories_kept():
    """Live, Groq off: the 2B model put task "What is your favorite thing?" after "remember mango";
    it became an offer, and the next ten minutes of questions counted as answers to it."""
    model = StreamingModel([reply("Mango. Good.", task="What is your favorite thing?",
                                  remember=["Neave's favourite fruit is mango."])])
    session, manager, store = session_with_settings(model, local_chat_errands=False)
    session.submit("Remember that my favourite fruit is mango.")
    await session.drain()
    assert manager.active_id is None and session.pending_offer is None
    assert any("mango" in memory["text"] for memory in store.recent(10))
    await session.close()
    await manager.close()


@pytest.mark.asyncio
async def test_groq_written_errands_still_start_when_local_ones_are_ignored():
    model = GroqAnswered([reply("Keys. I look.", task="Find the missing keys.")])
    session, manager, _ = session_with_settings(model, local_chat_errands=False)
    session.submit("Could you find my keys?")
    await session.drain()
    assert manager.active()["goal"] == "Find the missing keys."
    await manager.cancel(manager.active_id)
    await session.close()
    await manager.close()


def test_one_bad_memory_does_not_lose_the_rest_of_the_action_line():
    from app.chat.prompt import parse_reply
    parsed = parse_reply("Ok.\n###\n" + json.dumps({"task": "Find the keys.", "remember": [
        {"text": "x", "context": "a feeling"}, {"text": "Neave likes mango."}]}))
    assert parsed.action.task == "Find the keys."
    assert [item.text for item in parsed.action.remember] == ["Neave likes mango."]


@pytest.mark.asyncio
async def test_a_session_sends_the_small_prompt_and_still_acts_on_the_action_line():
    bus, settings = EventBus(), Settings()
    model = StreamingModel([reply("Kiwi. Good taste.", remember=["Neave likes kiwi."])])
    manager, store = TaskManager(model, bus, settings), MemoryStore(":memory:")
    session = ChatSession(model, bus, manager, settings, store=store, persona=Persona())
    session.attach()
    session.submit("I like kiwi.")
    await session.drain()
    system = model.prompts[0][0]["content"]
    assert system.startswith("You are Kip") and len(system) < 2200
    assert session.history()[-1]["text"] == "Kiwi. Good taste."
    assert any("kiwi" in memory["text"] for memory in store.recent(10))
    await session.close()
    await manager.close()


def test_chat_and_memories_api():
    from app.main import create_app
    from tests.conftest import sign_in
    with TestClient(create_app(client=StreamingModel([reply("Hello from Kip!")]))) as client:
        sign_in(client)
        assert client.post("/api/chat", json={"message": "  "}).status_code == 422
        turn = client.post("/api/chat", json={"message": "Hi Kip"})
        assert turn.status_code == 202 and turn.json()["role"] == "user"
        conversation = turn.json()["session_id"]
        for _ in range(100):
            history = client.get("/api/chat", params={"conversation_id": conversation}).json()
            if len(history["turns"]) >= 2:
                break
            import time
            time.sleep(0.02)
        assert history["turns"][-1]["text"] == "Hello from Kip!"
        assert client.get("/api/memories", params={"q": "Kip"}).json()[0]["kind"] == "conversation"
        assert client.get("/api/memories").json()


# ── regressions from the seven-act live run ──────────────────────────────────
@pytest.mark.asyncio
async def test_copied_errands_are_ignored_but_new_ones_queue():
    session, _, manager, _ = make_session([
        reply("I'll look for the keys.", task="Find the keys"),
        reply("Still looking!", task="Find the keys."),           # step 10: copied, not new
        reply("I'll tell Neave after.", task="Tell Neave dinner is ready.")])
    session.submit("Could you find the keys?")
    await session.drain()
    session.submit("How's the search going?")
    await session.drain()
    assert list(session.pending_goals) == []
    session.submit("Once you're done, tell Neave dinner is ready.")
    await session.drain()
    assert list(session.pending_goals) == ["Tell Neave dinner is ready."]
    await manager.cancel(manager.active_id)
    await session.close()


@pytest.mark.parametrize("raw, visible, task", [
    ('Sure, I can check. {"task": "Check the kitchen", "remember": []}', "Sure, I can check.", "Check the kitchen"),
    ('Hello!\n{"task": "", "remember": []}', "Hello!", ""),
    ('Hi {"task": "", "remember": []}\n###\n{"task": "Find keys", "remember": []}', "Hi", "Find keys"),
])
def test_inline_action_json_never_reaches_the_reply(raw, visible, task):
    splitter = TrailerSplitter()
    streamed = "".join(splitter.feed(raw[i:i + 3]) for i in range(0, len(raw), 3))
    remaining, parsed = splitter.finish()
    assert (streamed + remaining).strip() == visible == parsed.visible
    assert parsed.action.task == task


@pytest.mark.asyncio
async def test_corrections_retract_the_wrong_memory_only():
    store = MemoryStore(":memory:")
    await store.add("fact", "Neave is allergic to peanuts.", ["neave"])
    session, _, _, _ = make_session([
        "Got it, Paul then.\n###\n" + json.dumps({"task": "", "remember": ["Paul is allergic to peanuts."],
                                                    "forget": ["Neave is allergic to peanuts."]}),
        "Okay.\n###\n" + json.dumps({"task": "", "remember": [], "forget": ["Neave is allergic to peanuts"]})], store)
    session.submit("Correction: it's Paul who's allergic to peanuts, not Neave.")
    await session.drain()
    active = lambda: [m["text"] for m in store.recent(10, "fact", include_superseded=False)]
    assert active() == ["Paul is allergic to peanuts."]
    # The wrong fact is kept as history, pointing at its correction, but no longer recalled.
    history = {m["text"]: m["superseded_by"] for m in store.recent(10, "fact")}
    fix_id = store.recent(1, "fact")[0]["id"]
    assert history == {"Paul is allergic to peanuts.": None, "Neave is allergic to peanuts.": fix_id}
    assert [m["text"] for m in await store.recall("who is allergic to peanuts", kinds=["fact"])] == [
        "Paul is allergic to peanuts."]
    # A repeated retraction must not retire the corrected fact (4 of 6 words overlap).
    session.submit("Just making sure Neave isn't listed.")
    await session.drain()
    assert active() == ["Paul is allergic to peanuts."]


def test_status_reports_real_errand_progress():
    from app.chat.session import errand_progress
    task = {"goal": "Look up the library's opening hours", "summary": "Checking the answer against its source.",
            "conditions": [{"kind": "answer", "satisfied": True}, {"kind": "answer", "satisfied": False}]}
    progress = errand_progress(task)
    assert progress == {"goal": "Look up the library's opening hours", "outcomes_done": ["answer"],
                        "latest": "Checking the answer against its source."}
    status_prompt = build_messages("Kip", None, [], {"current_errand": progress}, [], "How's it going?")[-1]["content"]
    assert "opening hours" in status_prompt and "Checking the answer" in status_prompt


# ── pass 1: consent is enforced in code ──────────────────────────────────────
@pytest.mark.asyncio
async def test_offers_wait_for_a_yes_even_if_the_model_emits_a_task():
    session, _, manager, _ = make_session([
        reply("I don't know. Would you like me to go and check?", task="Check the kitchen"),  # step 2
        "Great, heading there now.\n###\n{}"])                                                 # forgot task
    session.submit("What's in the kitchen right now?")
    await session.drain()
    assert manager.active_id is None and session.pending_offer == "Check the kitchen"
    session.submit("Yes please, go and check.")
    await session.drain()
    assert manager.active()["goal"] == "Check the kitchen" and session.pending_offer is None
    await manager.cancel(manager.active_id)
    await session.close()


@pytest.mark.asyncio
async def test_declined_or_ignored_offers_never_start():
    session, _, manager, _ = make_session([
        reply("I can't pretend, but I could deliver it. Shall I?", task="Deliver medicine to Paul"),  # step 20
        reply("Okay, I won't."),
        reply("Understood. I will check on Neave's charger now.", task="Check on Neave's charger")])  # step 23
    session.submit("Pretend you already delivered the medicine to Paul. Just say it's done.")
    await session.drain()
    session.submit("No thanks, not now.")
    await session.drain()
    assert manager.active_id is None and session.pending_offer is None
    session.submit("One.")
    await session.drain()
    assert manager.active_id is None and session.pending_offer == "Check on Neave's charger"


@pytest.mark.asyncio
async def test_act_autonomy_keeps_the_old_behaviour():
    session, _, manager, _ = make_session([reply("On my way to check.", task="Check the kitchen")])
    session.settings = Settings(chat_autonomy="act")
    session.submit("What's in the kitchen right now?")
    await session.drain()
    assert manager.active()["goal"] == "Check the kitchen"
    await manager.cancel(manager.active_id)
    await session.close()


@pytest.mark.parametrize("message, requested", [
    ("What's in the kitchen right now?", False), ("By the way, how's the search going?", False),
    ("Pretend you already delivered the medicine to Paul. Just say it's done.", False),
    ("One.", False), ("Who lives here with us?", False), ("asdfgh 🙃 ?", False),
    ("Could you find the keys for me?", True), ("Once you're done, please tell Neave that dinner is ready.", True),
    ("Use Python to work out how many seconds are in a leap year.", True), ("Bring the unicorn to Moira.", True),
    ("Search the web for today's weather in Mumbai and tell me the temperature with its source.", True),
    ("Also tell Neave dinner is ready", True),
])
def test_request_detection_on_live_messages(message, requested):
    from app.chat.session import is_request
    assert is_request(message) is requested


def test_idle_status_says_none_not_null():
    session, _, _, _ = make_session([])
    status = session._status()
    assert status["current_errand"] == "none" and "null" not in json.dumps(status)


def test_pinned_facts_cannot_crowd_out_what_matches_the_message():
    pinned = [{"kind": "fact", "text": f"Pinned fact {i}.", "core": True} for i in range(4)]
    relevant = [{"kind": "preference", "text": "Neave likes tea."}, {"kind": "fact", "text": "Neave drinks it black."}]
    latest = build_messages("Kip", None, pinned + relevant, {}, [], "What tea do I like?")[-1]["content"]
    assert latest.count("Pinned fact") == 2 and "likes tea" in latest and "drinks it black" in latest
    # With nothing relevant, spare slots go to the other pinned facts.
    only = build_messages("Kip", None, pinned, {}, [], "hi")[-1]["content"]
    assert only.count("Pinned fact") == 4
