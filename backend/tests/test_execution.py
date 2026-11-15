"""Delegation lifetime and execution instrumentation.

These cover how long a delegation runs, when control returns to the Coordinator,
and that every model call and tool action is counted.
"""
import pytest

from app.config import Settings
from app.events.bus import EventBus
from app.tasks.manager import TaskManager
from tests.fakes import ScriptedLLM, tool


@pytest.fixture(autouse=True)
def offline_search(monkeypatch):
    from app.capabilities import tools
    monkeypatch.setattr(tools, "web_search", lambda query: {
        "query": query, "results": [{"title": "Source", "url": "https://example.com", "snippet": query}]})


def answer(payload):
    return {"summary": "Ready.", "action": {"tool": "finish_answer", "answer": "Found it.", "evidence_ids": [
        o["evidence_id"] for o in payload["tool_observations"] if o["success"]]}}


def manager(replies, conditions=None, **limits):
    bus = EventBus()
    return TaskManager(ScriptedLLM(replies, conditions), bus, Settings(**limits)), bus


class TestDelegationLifetime:
    async def test_direct_initial_delegation_answers_without_extra_coordinator_call(self):
        runner, _ = manager([tool("web_search", query="a fact"), answer], max_explorer_actions=40)
        task = runner.start("Look up a fact online.")
        await runner.runner
        assert task.status == "completed", task.summary
        assert task.current_plan == task.goal
        assert task.metrics.coordinator_llm_calls == 1  # Initial GoalPlan is now included.
        assert task.metrics.explorer_llm_calls == 2

    async def test_direct_initial_delegation_still_uses_coordinator_for_replanning(self):
        runner, _ = manager([
            tool("web_search", query="first"), tool("web_search", query="second"),
            {"action": "fail", "summary": "No further search in this test."},
        ], max_explorer_actions=2, max_coordinator_cycles=2)
        task = runner.start("Look up a fact online.")
        await runner.runner
        assert task.status == "failed"
        assert task.summary == "No further search in this test."
        assert task.metrics.coordinator_llm_calls == 2  # GoalPlan plus one replan.

    async def test_completed_objective_returns_control_before_the_ceiling(self):
        runner, bus = manager([tool("web_search", query="a fact"), answer], max_explorer_actions=40)
        task = runner.start("Look up a fact online.")
        await runner.runner
        assert task.status == "completed", task.summary
        assert task.tool_count < 10  # Stopped on success, not on the ceiling.
        assert not any("safety ceiling" in e["data"].get("summary", "")
                       for e in bus.history if e["type"] == "agent_message")

    async def test_hard_ceiling_still_stops_a_runaway_delegation(self):
        runner, bus = manager([
            *[tool("web_search", query=f"query {n}") for n in range(6)],
            {"action": "fail", "summary": "Stop."},
        ], max_explorer_actions=3, max_coordinator_cycles=2)
        task = runner.start("Look up a fact online.")
        await runner.runner
        # Exactly the ceiling's worth of delegated actions.
        assert task.tool_count == 3
        assert any("safety ceiling" in e["data"].get("summary", "")
                   for e in bus.history if e["type"] == "agent_message")


class TestInstrumentation:
    async def test_task_metrics_count_every_inference_and_action(self):
        runner, bus = manager([tool("web_search", query="a fact"), answer])
        task = runner.start("Look up a fact online.")
        await runner.runner
        metrics = task.metrics
        assert task.status == "completed", task.summary
        assert metrics.llm_calls == (metrics.coordinator_llm_calls + metrics.explorer_llm_calls
                                     + metrics.critic_llm_calls)
        assert metrics.tool_actions == task.tool_count
        finished = next(e for e in bus.history if e["type"] == "task_completed")
        assert finished["data"]["metrics"]["explorer_llm_calls"] == metrics.explorer_llm_calls

    def test_ollama_metrics_are_read_from_the_response_not_invented(self):
        import httpx
        from app.llm.ollama import OllamaClient

        def respond(request):
            return httpx.Response(200, json={
                "model": "test-model",
                "message": {"content": '{"approved":true,"summary":"OK"}'},
                "total_duration": 2_000_000_000, "prompt_eval_count": 120,
                "prompt_eval_duration": 500_000_000, "eval_count": 30,
                "eval_duration": 1_500_000_000})

        async def go():
            client = OllamaClient(Settings())
            await client.http.aclose()
            client.http = httpx.AsyncClient(transport=httpx.MockTransport(respond),
                                            base_url="http://test")
            await client.generate([], {})
            await client.close()
            return client.last_metrics

        import asyncio
        metrics = asyncio.run(go())
        assert metrics == {"model": "test-model", "total_seconds": 2.0, "prompt_eval_seconds": 0.5,
                           "eval_seconds": 1.5, "prompt_tokens": 120, "completion_tokens": 30}
        assert "load_seconds" not in metrics  # Absent from the response, so never reported.
