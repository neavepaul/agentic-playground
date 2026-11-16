"""Chat prompts sized for the model that will read them: roomy for Groq, small for the local model,
and a reply that falls back to the local model is read from the small one."""
import json

import httpx
import pytest

from app.chat.context import message_tokens
from app.config import Settings
from app.llm.base import ModelError
from app.llm.groq import HostedChat
from tests.test_chat import StreamingModel, make_session, reply
from tests.test_groq import groq
from tests.test_hosted_chat import Local, chat_with, said, sse

BIG = [{"role": "system", "content": "sys"}, {"role": "user", "content": "big " * 50}]
SMALL = [{"role": "system", "content": "sys"}, {"role": "user", "content": "small"}]


class Recording(Local):
    def __init__(self):
        super().__init__()
        self.seen = []

    async def stream_text(self, messages, max_tokens=300, temperature=None):
        self.seen.append(messages)
        async for chunk in super().stream_text(messages, max_tokens, temperature):
            yield chunk


# ── HostedChat ──────────────────────────────────────────────────────────────
async def test_groq_reads_the_big_prompt_and_the_local_model_never_sees_it():
    bodies = []

    def respond(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, content=sse("Hello."))

    local = Recording()
    client, chat = await chat_with(respond, local)
    try:
        out = "".join([c async for c in chat.stream_text(BIG, max_tokens=200, local_messages=SMALL)])
        assert out == "Hello." and bodies[0]["messages"][-1]["content"].startswith("big")
        assert local.seen == []
    finally:
        await client.close()


async def test_a_fallback_to_the_local_model_rebuilds_from_the_small_prompt():
    def respond(request):
        return httpx.Response(429, headers={"retry-after": "600"})

    local = Recording()
    client, chat = await chat_with(respond, local)
    try:
        out = "".join([c async for c in chat.stream_text(BIG, max_tokens=200, local_messages=SMALL)])
        assert out == "Local words." and local.seen == [SMALL] and chat.last_metrics["provider"] == "local"
    finally:
        await client.close()


async def test_without_an_alternate_the_local_model_gets_the_same_messages():
    local = Recording()
    client, chat = await chat_with(lambda request: httpx.Response(429, headers={"retry-after": "600"}), local)
    try:
        await said(chat, BIG)
        assert local.seen == [BIG]
    finally:
        await client.close()


async def test_a_prompt_only_groq_can_hold_is_never_sent_to_the_local_model():
    local = Recording()
    client, chat = await chat_with(lambda request: httpx.Response(429, headers={"retry-after": "600"}), local)
    try:
        with pytest.raises(ModelError, match="too large for the local model"):
            "".join([c async for c in chat.stream_text(BIG, max_tokens=200, local_messages=False)])
        assert local.seen == []
    finally:
        await client.close()


# ── in a conversation ───────────────────────────────────────────────────────
class Alternates(StreamingModel):
    """A chat client like HostedChat: takes two prompts, and says who answered."""
    prompt_alternatives = True

    def __init__(self, replies, provider="groq"):
        super().__init__(replies)
        self.provider, self.alternates = provider, []

    async def stream_text(self, messages, max_tokens=300, temperature=None, local_messages=None):
        self.alternates.append(local_messages)
        async for chunk in super().stream_text(messages, max_tokens, temperature):
            yield chunk
        self.last_metrics = {"provider": self.provider}


def chatting(provider="groq", replies=None, **settings):
    session, _, *_ = make_session([])
    model = Alternates(replies or [reply(f"Reply {i}. " + "Sounds lovely, tell me more about it. " * 2)
                                   for i in range(1, 12)], provider)
    session.client = model
    session.settings = Settings(chat_compaction=False, chat_history_tokens=200,
                                chat_hosted_history_tokens=2000, **settings)
    return session, model


async def talk(session, count):
    for i in range(1, count + 1):
        session.submit(f"Message number {i}: I keep thinking about my holiday plans and what we might do there.")
        await session.drain()


def turns_in(prompt):
    return len(prompt) - 2


@pytest.mark.asyncio
async def test_groq_is_given_more_history_than_the_local_model_would_read():
    session, model = chatting()
    await talk(session, 8)
    hosted, local = model.prompts[-1], model.alternates[-1]
    assert turns_in(hosted) > turns_in(local) >= 1
    assert message_tokens(local) <= session.settings.context_tokens - session.settings.chat_max_tokens
    metrics = session.history()[-1]["metrics"]
    assert metrics["prompt_for"] == "hosted" and metrics["estimated_prompt_tokens"] == message_tokens(hosted)


@pytest.mark.asyncio
async def test_a_reply_the_local_model_wrote_reports_the_small_prompt():
    session, model = chatting(provider="local")
    await talk(session, 8)
    metrics = session.history()[-1]["metrics"]
    assert metrics["prompt_for"] == "local" and metrics["estimated_prompt_tokens"] == message_tokens(model.alternates[-1])


@pytest.mark.asyncio
async def test_a_plain_local_client_still_gets_one_small_prompt():
    session, model, *_ = make_session([reply(f"Reply {i}.") for i in range(1, 9)])
    session.settings = Settings(chat_compaction=False, chat_history_tokens=200, chat_hosted_history_tokens=2000)
    await talk(session, 8)
    assert turns_in(model.prompts[-1]) < 8 and "local_messages" not in session.history()[-1]["metrics"]


@pytest.mark.asyncio
async def test_a_message_only_groq_can_hold_is_sent_without_a_local_fallback():
    session, model = chatting(replies=[reply("Read it all.")])
    session.submit("Please read this: " + "lorem ipsum dolor sit amet " * 700)   # ~19k chars: over 4K, under 8K
    await session.drain()
    assert model.alternates == [False] and session.history()[-1]["text"] == "Read it all."


@pytest.mark.asyncio
async def test_a_message_too_big_for_every_window_is_still_refused_plainly():
    session, model = chatting()
    session.submit("lorem ipsum dolor sit amet " * 2500)   # ~67k chars: over both
    await session.drain()
    assert "Too much context" in session.history()[-1]["text"] and model.prompts == []


@pytest.mark.asyncio
async def test_the_note_is_sized_for_groq_when_groq_is_the_usual_speaker():
    session, _ = chatting()
    assert session._history_budget == 2000
    plain, *_ = make_session([])
    assert plain._history_budget == plain.settings.chat_history_tokens
