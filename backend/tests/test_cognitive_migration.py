"""Behavioral regressions for the incremental cognitive architecture migration."""
import json
import asyncio

import httpx
import pytest

from app.chat.consent import task_identity
from app.chat.prompt import TrailerSplitter, parse_reply
from app.config import Settings
from app.llm.base import ModelError
from app.llm.colibri import ColibriClient
from app.chat.context import assemble, message_tokens, select_history
from app.memory.recall import MemoryStore
from tests.test_chat import make_session, reply, StreamingModel
from app.chat import ChatSession
from app.tasks.manager import TaskManager, LimitReached
from app.tasks.models import TaskMetrics
from app.agents.schemas import CriticReview
from app.llm.base import structured
from app.chat.spoken import spoken


@pytest.mark.parametrize("left,right", [
    ("Bring my keys to me", "Bring my keys to us"),
    ("Bring my keys", "Bring our keys"),
    ("Transfer 1.5 dollars", "Transfer 15 dollars"),
    ("Calculate -5 + 3", "Calculate 5 + 3"),
    ("Tell James yes", "Tell Jame yes"),
    ("Go to Paul", "Go for Paul"),
    ("Bring tea now", "Bring tea"),
    ("Tell nobody", "Tell nothing"),
])
def test_changed_intent_is_never_a_duplicate(left, right):
    assert task_identity(left) != task_identity(right)
    assert task_identity("Find the keys.") == task_identity("find keys")


@pytest.mark.parametrize("content", [
    "### Analysis\nThis is a heading, not an action.",
    'Example:\n```json\n{"task": "Find keys"}\n```',
    '```text\n###\n{"task": "Find keys"}\n```',
])
def test_rich_content_survives_chunked_control_parser(content):
    output = content + '\n###\n{"task": ""}'
    splitter = TrailerSplitter()
    streamed = "".join(splitter.feed(char) for char in output)
    tail, parsed = splitter.finish()
    assert streamed + tail == content
    assert parsed.visible == content
    assert parsed.action.task == ""
    assert parse_reply(content).action is None


@pytest.mark.asyncio
async def test_unconfirmed_stream_eof_is_an_error_even_with_complete_action_json():
    payload = 'Ready.\n###\n{"task": "Find keys"}'
    async def respond(request):
        event = {"choices": [{"delta": {"content": payload}}]}
        return httpx.Response(200, text="data: " + json.dumps(event) + "\n\n")
    client = ColibriClient(Settings())
    await client.http.aclose()
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(respond), base_url="http://local/")
    try:
        with pytest.raises(ModelError, match="before confirming"):
            async for _ in client.stream_text([]):
                pass
    finally:
        await client.close()


def test_history_budget_keeps_complete_correction_and_recent_followup():
    turns = [{"role": "user", "text": "Actually, I am allergic to peanuts. Never recommend them."}]
    turns += [{"role": "kip", "text": "Old unrelated detail. " * 80} for _ in range(15)]
    turns += [{"role": "user", "text": "My legs are cooked after yesterday's run."},
              {"role": "kip", "text": "A rest day sounds sensible."}]
    messages = select_history(turns, "What should I eat?", 600)
    assert message_tokens(messages) <= 600
    assert "allergic to peanuts" in json.dumps(messages)
    assert messages[-2]["content"] == turns[-2]["text"]


def test_context_preserves_long_code_and_unicode_or_omits_whole_turn():
    code = "```python\n" + "print('你好')\n" * 60 + "```"
    turns = [{"role": "kip", "text": code}]
    assert select_history(turns, "code", 5000)[0]["content"] == code
    assert select_history(turns, "unrelated", 100) == []
    with pytest.raises(ModelError, match="budget"):
        assemble("rules", "你好" * 200, [], "", 100)


@pytest.mark.asyncio
async def test_transcript_restart_and_id_isolation_without_replaying_actions(tmp_path):
    path = tmp_path / "conversation.db"
    store = MemoryStore(path)
    session, model, manager, events = make_session([reply("Rest sounds sensible.")], store)
    stable_id = session.session_id
    session.submit("My legs are cooked.")
    await session.drain()
    expected = session.history()
    await session.close()
    store.close()
    reopened = MemoryStore(path)
    restored = ChatSession(model, session.bus, manager, session.engine, Settings(), store=reopened,
                           session_id=stable_id)
    other = ChatSession(model, session.bus, manager, session.engine, Settings(), store=reopened,
                        session_id="another-conversation")
    assert restored.history() == expected
    assert other.history() == []
    assert restored.pending_offer is None and not restored.pending_goals and manager.active_id is None
    reopened.append_turn(stable_id, expected[0])
    assert len(reopened.conversation_turns(stable_id)) == 2
    await restored.close()
    await other.close()
    reopened.close()


@pytest.mark.asyncio
async def test_terminal_task_cannot_dispatch_or_increment_counters():
    session, _, manager, _ = make_session([])
    task = manager.start("Find keys")
    await manager.cancel(task.id)
    before = task.tool_count
    with pytest.raises(LimitReached, match="terminal"):
        manager.call_tool(task, "move_to", {"room": "kitchen"})
    with pytest.raises(LimitReached, match="terminal"):
        await manager.call_capability(task, "web_search", {"query": "weather"})
    assert task.tool_count == before
    await session.close()


@pytest.mark.asyncio
async def test_repair_attempts_cannot_exceed_task_model_budget():
    class Invalid:
        calls = 0
        async def generate(self, messages, response_schema):
            self.calls += 1
            return "invalid"
    model, metrics = Invalid(), TaskMetrics(llm_call_limit=1)
    with pytest.raises(ModelError, match="budget"):
        await structured(model, CriticReview, "Review", {}, metrics=metrics, role="critic")
    assert model.calls == metrics.llm_calls == metrics.critic_llm_calls == 1


@pytest.mark.asyncio
async def test_failed_model_attempts_are_counted():
    class Offline:
        async def generate(self, messages, response_schema):
            raise ModelError("offline")
    metrics = TaskMetrics()
    with pytest.raises(ModelError, match="offline"):
        await structured(Offline(), CriticReview, "Review", {}, metrics=metrics)
    assert metrics.llm_calls == 1


def test_spoken_tables_keep_labels_and_values_without_markdown():
    text = "| Day | Distance |\n|---|---:|\n| Monday | 5 km |\n| Tuesday | Rest |"
    voice = spoken(text)
    assert "Day: Monday; Distance: 5 km" in voice
    assert "Day: Tuesday; Distance: Rest" in voice
    assert "|" not in voice and "---" not in voice
    assert spoken("```python\nprint(42)\n```") == "The code is on screen."


@pytest.mark.asyncio
async def test_canonical_screen_and_voice_are_separate():
    text = "Try this:\n```python\nprint(42)\n```\n[Reference](https://example.com)"
    session, _, _, _ = make_session([reply(text)])
    session.submit("Show me the code.")
    await session.drain()
    turn = next(t for t in session.history() if t.get("in_reply_to"))
    assert turn["content"] == turn["screen_text"] == text
    assert "```" not in turn["spoken_text"]
    assert turn["text"] == turn["spoken_text"]
    assert turn["metrics"]["chat_model_calls"] == 1
    assert turn["metrics"]["estimated_prompt_tokens"] <= session.settings.chat_context_tokens
    await session.close()


@pytest.mark.asyncio
async def test_context_overflow_is_reported_without_claiming_model_outage():
    session, model, _, _ = make_session([])
    session.settings.chat_context_tokens = 1024
    session.submit("Hello")
    await session.drain()
    turn = session.history()[-1]
    assert "context" in turn["text"]
    assert "model isn't" not in turn["text"]
    assert turn["metrics"]["chat_model_calls"] == 0
    assert not model.prompts
    await session.close()


@pytest.mark.asyncio
async def test_cancel_does_not_wait_for_chat_model():
    session, model, manager, _ = make_session([])
    entered = asyncio.Event()
    async def blocked(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()
        yield "unreachable"
    model.stream_text = blocked
    task = manager.start("Find keys")
    session.submit("Cancel that")
    await asyncio.wait_for(entered.wait(), 1)
    await asyncio.sleep(0)
    assert task.status == "cancelled"
    await session.close()
    await manager.close()


@pytest.mark.asyncio
async def test_cancel_invalidates_older_reply_proposals():
    session, model, manager, _ = make_session([])
    entered, release = asyncio.Event(), asyncio.Event()
    async def delayed(*args, **kwargs):
        entered.set()
        await release.wait()
        yield reply("I will find them.", task="Find keys")
    model.stream_text = delayed
    session.submit("Find keys")
    await entered.wait()
    session.submit("Cancel that")
    release.set()
    await session.drain()
    assert not manager.tasks
    assert not session.pending_goals
    await session.close()


@pytest.mark.asyncio
async def test_restart_retains_task_evidence_and_restrictions_without_replay(tmp_path):
    session, model, _, _ = make_session([])
    store = MemoryStore(tmp_path / "tasks.db")
    manager = TaskManager(model, session.manager.tools, session.bus, Settings(), memory_store=store)
    task = manager.start("Find keys")
    manager.call_tool(task, "get_status", {})
    manager.restrictions.tools.add("run_python")
    manager.save_restrictions()
    # A crash between dispatch and recording its result has an uncertain outcome.
    manager._before_dispatch(task, "move_to", {"room": "kitchen"})
    journal = store.get_state("task-journal")
    await manager.close()
    store.set_state({"task-journal": journal})  # crash image, before graceful cancellation
    store.close()
    reopened = MemoryStore(tmp_path / "tasks.db")
    restored = TaskManager(model, session.manager.tools, session.bus, Settings(), memory_store=reopened)
    recovered = restored.tasks[task.id]
    assert recovered.status == "failed"
    assert "nothing was replayed" in recovered.summary
    assert recovered.action_history[0]["tool"] == "get_status"
    assert recovered.pending_action["tool"] == "move_to"
    assert restored.restrictions.forbids_tool("run_python")
    assert restored.runner is None and restored.active_id is None
    await session.close()
    reopened.close()
