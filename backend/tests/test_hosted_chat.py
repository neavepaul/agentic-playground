"""Conversation on Groq while it has quota, the local model otherwise; and the fixes from a
live run where a casual reply took 148 s, a live question missed the fast path, an old
answer was repeated as new, and a repeated question queued a second errand."""
import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from app.chat.context import _age_label
from app.chat.intent import read_intent
from app.chat.router import default_router, fast_path
from app.config import Settings
from app.llm.groq import HostedChat
from tests.test_conversation_truth import PlanningModel, done, ran, session_with  # noqa: F401
from tests.test_groq import groq, searched
from tests.test_permissions import settle


def sse(*pieces: str, finish: str = "stop") -> bytes:
    lines = [f"data: {json.dumps({'choices': [{'delta': {'content': p}, 'finish_reason': None}]})}" for p in pieces]
    lines.append(f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': finish}]})}")
    return ("\n\n".join(lines + ["data: [DONE]"]) + "\n\n").encode()


class Local:
    def __init__(self, text="Local words."):
        self.text, self.calls, self.last_metrics = text, 0, {}

    async def stream_text(self, messages, max_tokens=300, temperature=None):
        self.calls += 1
        yield self.text
        self.last_metrics = {"first_token_seconds": 90.0}


async def chat_with(respond, local=None, models="qwen/qwen3.8-27b,openai/gpt-oss-20b"):
    client = await groq(respond, models=models)
    return client, HostedChat(client, local or Local(), models.split(","))


async def said(chat, messages=None):
    return "".join([chunk async for chunk in chat.stream_text(messages or [{"role": "user", "content": "hi"}],
                                                              max_tokens=200)])


async def test_conversation_streams_from_groq_qwen_first_without_reasoning():
    bodies = []

    def respond(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, content=sse("Bad ", "day. ", "Very bad."))

    client, chat = await chat_with(respond)
    try:
        assert await said(chat) == "Bad day. Very bad."
        assert bodies[0]["model"] == "qwen/qwen3.8-27b" and bodies[0]["stream"] is True
        assert bodies[0]["reasoning_effort"] == "none"  # no visible thinking before the first word
        assert chat.last_metrics["provider"] == "groq" and chat.last_metrics["model"] == "qwen/qwen3.8-27b"
    finally:
        await client.close()


async def test_a_rate_limited_chat_model_hands_over_and_no_quota_means_the_local_model():
    seen = []

    def respond(request):
        model = json.loads(request.content)["model"]
        seen.append(model)
        if model.startswith("qwen/"):
            return httpx.Response(429, headers={"retry-after": "120"})
        return httpx.Response(200, content=sse("From ", "gpt-oss."))

    local = Local()
    client, chat = await chat_with(respond, local)
    try:
        assert await said(chat) == "From gpt-oss."
        assert seen == ["qwen/qwen3.8-27b", "openai/gpt-oss-20b"] and local.calls == 0
        client.quota["openai/gpt-oss-20b"].blocked_until = client.now() + 600
        assert await said(chat) == "Local words." and local.calls == 1  # qwen still benched: straight to local
        assert chat.last_metrics["provider"] == "local"
    finally:
        await client.close()


async def test_inline_reasoning_is_never_spoken_and_a_cut_reply_is_marked():
    client, chat = await chat_with(lambda r: httpx.Response(200, content=sse("<think>hm", "m</think>", "Hi.",
                                                                             finish="length")))
    try:
        assert await said(chat) == "Hi."
        assert chat.last_metrics["truncated"] is True
    finally:
        await client.close()


async def test_a_search_whose_output_could_not_be_parsed_gets_one_more_try():
    calls = []

    def respond(request):
        calls.append(json.loads(request.content)["model"])
        if len(calls) == 1:
            return httpx.Response(400, json={"error": {"code": "output_parse_failed", "message": "Parsing failed."}})
        return searched()

    client = await groq(respond, models="openai/gpt-oss-20b")
    try:
        found = await client.search([{"role": "user", "content": "q"}])
        assert calls == ["openai/gpt-oss-20b", "openai/gpt-oss-20b"] and found.answer.startswith("Lando")
    finally:
        await client.close()


def test_a_question_after_a_topic_switch_takes_the_fast_path():
    """Live: "ok leave f1. when is the india new zealand game starting'" scored live 0.93
    but the whole-message question reader saw no question, so it waited 155 s on chat."""
    text = "ok leave f1. when is the india new zealand game starting'"
    assert not read_intent(text).question and not read_intent(text).request
    route = default_router().route(text)
    assert route.asks and fast_path(route, read_intent(text), web_allowed=True, involves_house=False)


def test_an_old_lookup_answer_carries_its_age_in_the_chat_context():
    now = datetime.now(timezone.utc)
    fresh = {"source": "narration", "at": (now - timedelta(minutes=3)).isoformat()}
    old = {"source": "narration", "at": (now - timedelta(minutes=54)).isoformat()}
    assert _age_label(fresh, now) == "[update] "
    assert _age_label(old, now) == "[earlier update, 54 min ago, may be out of date] "
    assert "3 days ago" in _age_label({"at": (now - timedelta(days=3)).isoformat()}, now)


@pytest.mark.asyncio
async def test_asking_again_while_the_lookup_runs_does_not_queue_a_second_errand(ran):
    model = PlanningModel([])  # the lookup's research step never returns: it is still running
    session, manager = session_with(model, settings=Settings(direct_initial_delegation=True))
    session.submit("when is the india new zealand game starting")
    await settle(session, manager)
    assert manager.active_id is not None
    session.submit("when is the india versus new zealand game starting today")
    await settle(session, manager)
    assert session.history()[-1]["text"] == "Still checking that. Soon."
    assert len(manager.tasks) == 1 and not session.pending_goals and model.prompts == []
    await done(session, manager)


@pytest.mark.asyncio
async def test_a_lookup_queued_behind_an_errand_says_so_without_reading_out_its_goal(ran):
    model = PlanningModel([])
    session, manager = session_with(model, settings=Settings(direct_initial_delegation=True))
    manager.start("Find the keys.")  # an errand already under way
    await settle(session, manager)
    session.submit("whats the weather in delhi")
    await settle(session, manager)
    assert session.history()[-1]["text"] == "One errand first. Then I check this."
    assert len(session.pending_goals) == 1
    await done(session, manager)
