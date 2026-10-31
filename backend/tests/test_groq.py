import json

import httpx
import pytest

from app.agents.coordinator import Coordinator
from app.agents.explorer import Explorer
from app.agents.schemas import CriticReview, GoalCondition
from app.config import Settings
from app.llm.base import ModelError, structured
from app.llm.factory import create_hosted_client
from app.llm.groq import GroqClient, HostedFirst, reset_seconds
from app.tasks.models import TaskContext
from app.world.tools import WorldTools
from tests.fakes import ScriptedLLM

KEY = "gsk_private_test_key"
REVIEW = '{"approved":true,"summary":"Ready"}'


class Clock:
    def __init__(self):
        self.value = 1000.0

    def __call__(self):
        return self.value


async def groq(respond, models="openai/gpt-oss-120b,qwen/qwen3.8-27b", clock=None):
    client = GroqClient(Settings(groq_api_key=KEY, groq_models=models), now=clock or Clock())
    await client.http.aclose()
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(respond), base_url="https://groq.test/v1/",
                                    headers={"Authorization": f"Bearer {KEY}"})
    return client


def ok(content=REVIEW, model="openai/gpt-oss-120b", headers=None, finish="stop"):
    return httpx.Response(200, headers=headers or {}, json={
        "model": model, "usage": {"prompt_tokens": 50, "completion_tokens": 9},
        "choices": [{"finish_reason": finish, "message": {"content": content}}]})


@pytest.mark.parametrize("value,seconds", [("7.66s", 7.66), ("2m59.56s", 179.56), ("1h2m", 3720),
                                           ("240ms", 0.24), ("", None), ("soon", None)])
def test_reset_headers_are_read_as_seconds(value, seconds):
    assert reset_seconds(value) == (pytest.approx(seconds) if seconds is not None else None)


def test_no_key_means_no_hosted_client():
    assert create_hosted_client(Settings(groq_api_key=None)) is None
    assert create_hosted_client(Settings(groq_api_key="  ")) is None


async def test_request_shape_keeps_reasoning_server_side():
    bodies = []

    def respond(request):
        assert request.headers["authorization"] == f"Bearer {KEY}"
        bodies.append(json.loads(request.content))
        return ok()

    client = await groq(respond)
    try:
        result = await structured(client, CriticReview, "Review", {})
        assert result.approved
        body = bodies[0]
        assert body["model"] == "openai/gpt-oss-120b"
        assert body["response_format"] == {"type": "json_object"}
        assert body["include_reasoning"] is False and "reasoning_format" not in body
        assert "Output JSON schema:" in body["messages"][0]["content"]
        assert client.last_metrics["provider"] == "groq" and client.last_metrics["completion_tokens"] == 9
    finally:
        await client.close()


async def test_rate_limited_model_is_skipped_until_its_reset():
    clock, seen = Clock(), []

    def respond(request):
        model = json.loads(request.content)["model"]
        seen.append(model)
        if model == "openai/gpt-oss-120b":
            return httpx.Response(429, headers={"retry-after": "30"}, json={"error": {"code": "rate_limit_exceeded"}})
        body = json.loads(request.content)
        assert body["reasoning_format"] == "hidden" and "include_reasoning" not in body
        return ok(model=model)

    client = await groq(respond, clock=clock)
    try:
        assert await client.generate([{"role": "user", "content": "JSON"}], {}) == REVIEW
        assert seen == ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
        await client.generate([{"role": "user", "content": "JSON"}], {})
        assert seen[-1] == "qwen/qwen3.8-27b" and seen.count("openai/gpt-oss-120b") == 1
        clock.value += 31
        await client.generate([{"role": "user", "content": "JSON"}], {})
        # After its reset the first model is tried again (and here is still limited).
        assert seen[-2:] == ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
    finally:
        await client.close()


async def test_model_without_enough_token_quota_is_not_called():
    seen = []

    def respond(request):
        model = json.loads(request.content)["model"]
        seen.append(model)
        # Only 900 tokens left this minute on the first model: too few for another call.
        return ok(model=model, headers={"x-ratelimit-remaining-tokens": "900", "x-ratelimit-reset-tokens": "40s",
                                        "x-ratelimit-remaining-requests": "999"})

    client = await groq(respond)
    try:
        await client.generate([{"role": "user", "content": "JSON"}], {})
        await client.generate([{"role": "user", "content": "JSON"}], {})
        assert seen == ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
    finally:
        await client.close()


async def test_refused_request_benches_the_model_but_a_bad_generation_does_not():
    seen = []

    def respond(request):
        model = json.loads(request.content)["model"]
        seen.append(model)
        if model == "openai/gpt-oss-120b":
            return httpx.Response(400, json={"error": {"code": "invalid_request_error"}})
        return httpx.Response(400, json={"error": {"code": "json_validate_failed"}})

    client = await groq(respond)
    try:
        with pytest.raises(ModelError, match="every model failed"):
            await client.generate([{"role": "user", "content": "JSON"}], {})
        with pytest.raises(ModelError):
            await client.generate([{"role": "user", "content": "JSON"}], {})
        assert seen == ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "qwen/qwen3.8-27b"]
    finally:
        await client.close()


@pytest.mark.parametrize("mode", ["auth", "timeout", "offline"])
async def test_errors_never_reveal_the_key_or_try_every_model(mode):
    seen = []

    def respond(request):
        seen.append(request)
        if mode == "auth":
            return httpx.Response(401, text=f"bad key {KEY}")
        if mode == "timeout":
            raise httpx.ReadTimeout(f"private {KEY}", request=request)
        raise httpx.ConnectError(f"private {KEY}", request=request)

    client = await groq(respond)
    try:
        with pytest.raises(ModelError) as caught:
            await client.generate([{"role": "user", "content": "JSON"}], {})
        assert KEY not in str(caught.value) and len(seen) == 1
    finally:
        await client.close()


async def test_hosted_first_caps_hosted_calls_and_falls_back_locally():
    hosted_calls = []

    def respond(request):
        hosted_calls.append(request)
        return ok() if len(hosted_calls) == 1 else httpx.Response(503)

    client = await groq(respond, models="openai/gpt-oss-120b")
    local = ScriptedLLM([{"approved": True, "summary": "Local"}, {"approved": True, "summary": "Local again"}])
    try:
        bounded = HostedFirst(client, local, limit=2, purpose="test")
        assert (await structured(bounded, CriticReview, "Review", {})).summary == "Ready"
        assert bounded.last_metrics["provider"] == "groq"
        # Groq fails: the local model answers at once, and the failed try still counts.
        assert (await structured(bounded, CriticReview, "Review", {})).summary == "Local"
        assert "provider" not in bounded.last_metrics
        # Cap reached: no further hosted request is made.
        assert (await structured(bounded, CriticReview, "Review", {})).summary == "Local again"
        assert len(hosted_calls) == 2 and bounded.used == 2
    finally:
        await client.close()


async def test_goal_planning_uses_groq_and_decisions_stay_local():
    plans = []

    def respond(request):
        payload = json.loads(json.loads(request.content)["messages"][1]["content"])
        plans.append(payload)
        return ok(json.dumps({"summary": "Find the keys.",
                              "conditions": [{"kind": "find_object", "object": "keys"}]}))

    client = await groq(respond)
    local = ScriptedLLM([])
    try:
        coordinator = Coordinator(local, hosted=client, hosted_calls=2)
        plan = await coordinator.define_goal("Find the keys.", {"rooms": {"kitchen": {"name": "Kitchen",
                                                                                     "outline": [[0, 0]]}}})
        assert plan.conditions[0].object == "keys" and local.calls == []
        # Only the goal and room names leave the machine: no geometry, memory or history.
        assert plans == [{"goal": "Find the keys.", "floor_plan": {"rooms": {"kitchen": "Kitchen"}}}]
    finally:
        await client.close()


async def test_web_research_uses_groq_but_robot_tasks_stay_local(monkeypatch):
    import app.agents.explorer as explorer_module
    # Skip robot reflexes (e.g. an opening get_status) so the model choice itself is tested.
    monkeypatch.setattr(explorer_module, "reflex_action", lambda *args: None)
    hosted_requests = []

    def respond(request):
        hosted_requests.append(request)
        return ok(json.dumps({"summary": "Searching.", "action": {"tool": "web_search", "query": "a fact"}}))

    client = await groq(respond)
    try:
        explorer = Explorer(ScriptedLLM([]), WorldTools.schemas(), hosted=client, hosted_calls=4)
        research = TaskContext(goal="Look up a fact.",
                               conditions=[GoalCondition(kind="answer", required_tools=["web_search"])])
        action = await explorer.decide(research, research.goal)
        assert action.tool == "web_search" and len(hosted_requests) == 1

        mixed = TaskContext(goal="Look up a fact and find the keys.",
                            conditions=[GoalCondition(kind="answer", required_tools=["web_search"]),
                                        GoalCondition(kind="find_object", object="keys")])
        local = ScriptedLLM([{"summary": "Searching.", "action": {"tool": "web_search", "query": "a fact"}}])
        explorer.client = local
        await explorer.decide(mixed, mixed.goal)
        assert len(hosted_requests) == 1 and len(local.calls) == 1
    finally:
        await client.close()
