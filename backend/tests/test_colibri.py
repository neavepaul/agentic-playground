import json

import httpx
import pytest

from app.agents.schemas import CriticReview
from app.config import Settings
from app.llm.base import ModelError, structured
from app.llm.colibri import ColibriClient
from app.llm.factory import create_client
from app.llm.ollama import OllamaClient


async def mock_client(respond):
    client = ColibriClient(Settings(colibri_base_url="http://test/v1"))
    await client.http.aclose()
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(respond), base_url="http://test/v1/")
    return client


async def test_colibri_contract_repair_and_private_reasoning():
    calls = []

    def respond(request):
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        calls.append(body)
        assert body["model"] == "qwen3.6-colibri"
        assert body["enable_thinking"] is False
        assert body["stream"] is False
        assert body["max_tokens"] == 1024
        assert "response_format" not in body
        assert "Output JSON schema:" in body["messages"][0]["content"]
        return httpx.Response(200, json={
            "model": "qwen3.6-colibri", "usage": {"prompt_tokens": 80, "completion_tokens": 12},
            "choices": [{"finish_reason": "stop", "message": {
                "content": "not JSON" if len(calls) == 1 else '{"approved":true,"summary":"Ready"}',
                "reasoning_content": "private scratchpad"}}]})

    client = await mock_client(respond)
    try:
        result = await structured(client, CriticReview, "Review", {})
        assert result.approved and result.summary == "Ready"
        assert len(calls) == 2
        assert "private scratchpad" not in json.dumps(calls)
        assert client.last_metrics["completion_tokens"] == 12
    finally:
        await client.close()


@pytest.mark.parametrize("mode", ["timeout", "offline", "http", "malformed", "truncated"])
async def test_colibri_errors_are_safe(mode):
    def respond(request):
        if mode == "timeout":
            raise httpx.ReadTimeout("private details", request=request)
        if mode == "offline":
            raise httpx.ConnectError("private details", request=request)
        if mode == "http":
            return httpx.Response(400, text="private details")
        if mode == "malformed":
            return httpx.Response(200, json={"choices": []})
        return httpx.Response(200, json={"choices": [{"finish_reason": "length",
                                                     "message": {"content": "private details"}}]})

    client = await mock_client(respond)
    try:
        with pytest.raises(ModelError) as error:
            await client.generate([], {})
        assert "private details" not in str(error.value)
    finally:
        await client.close()


async def test_colibri_health_and_provider_selection():
    def respond(request):
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": [{"id": "qwen3.6-colibri"}]})

    client = await mock_client(respond)
    try:
        assert await client.health() == {"reachable": True, "model_available": True,
                                         "model": "qwen3.6-colibri"}
    finally:
        await client.close()
    for provider, expected in (("colibri", ColibriClient), ("ollama", OllamaClient)):
        client = create_client(Settings(llm_provider=provider))
        try:
            assert isinstance(client, expected)
        finally:
            await client.close()


async def test_application_health_reports_colibri(monkeypatch):
    from app.main import create_app

    client = await mock_client(lambda request: httpx.Response(
        200, json={"data": [{"id": "qwen3.6-colibri"}]}))
    monkeypatch.setattr("app.main.create_client", lambda settings: client)
    app = create_app(settings=Settings(llm_provider="colibri"))
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://app") as api:
            assert (await api.get("/api/health")).json()["llm"] == {
                "provider": "colibri", "model": "qwen3.6-colibri",
                "reachable": True, "model_available": True}
    finally:
        await client.close()
