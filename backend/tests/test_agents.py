import asyncio

import httpx
import pytest
from pydantic import ValidationError

from app.agents.schemas import CoordinatorDecision, CriticReview
from app.config import Settings
from app.evaluate import mission_satisfied
from app.events.bus import EventBus
from app.llm.base import ModelError, structured
from app.llm.ollama import OllamaClient
from app.tasks.manager import TaskManager
from app.world.engine import WorldEngine
from app.world.tools import WorldTools
from tests.fakes import ScriptedLLM, WaitingLLM, complete, tool


def manager(client, **limits):
    engine, bus = WorldEngine(), EventBus()
    return TaskManager(client, WorldTools(engine, bus), bus, Settings(**limits)), engine, bus


def test_structures():
    for output in [dict(action="teleport", summary="No"), dict(action="delegate", summary="Missing task"),
                   dict(action="complete", summary="Missing evidence"),
                   dict(action="fail", summary="No", thinking="private")]:
        with pytest.raises(ValidationError):
            CoordinatorDecision.model_validate(output)
    with pytest.raises(ValidationError):
        CriticReview.model_validate({"approved": "yes", "summary": "Invalid bool"})


async def test_invalid_json_repair_and_failure():
    client = ScriptedLLM(["<think>private</think>broken", {"approved": True, "summary": "Supported."}])
    result = await structured(client, CriticReview, "Review", {})
    assert result.approved
    assert "<think>" not in str(client.calls[1][0])
    with pytest.raises(ModelError, match="twice"):
        await structured(ScriptedLLM(["oops", "oops again"]), CriticReview, "Review", {})


async def test_delivery_end_to_end_with_mock_model():
    replies = [{"action": "delegate", "summary": "Locate the charger and recipient.", "task": "Search study."},
               tool("look"), tool("move_to", room="study"), tool("look"),
               tool("talk_to", person="dad", message="Who needs the charger?"),
               tool("pick_up", object="charger"), tool("move_to", room="hall"),
               tool("move_to", room="bedroom"), tool("look"),
               {"action": "delegate", "summary": "Deliver charger.", "task": "Give charger to Neave."},
               tool("give", object="charger", person="neave"),
               {"action": "report", "summary": "Delivery action succeeded."}, complete,
               {"approved": True, "summary": "Delivery and need are evidenced."}]
    fake = ScriptedLLM(replies)
    mgr, engine, bus = manager(fake)
    task = mgr.start("Find out who needs the charger and deliver it.")
    await mgr.runner
    assert task.status == "completed"
    assert task.critic_count == 1
    assert mission_satisfied(3, task, engine.snapshot())
    assert mgr.active_id is None
    assert any(e["type"] == "object_given" for e in bus.history)
    # Initial Coordinator and Explorer inputs do not reveal hidden people/items.
    for messages, _ in fake.calls[:2]:
        assert '"people"' not in messages[1]["content"]
        assert '"objects"' not in messages[1]["content"]
        assert '"world"' not in messages[1]["content"]


@pytest.mark.parametrize("limits,replies,expected", [
    ({"max_coordinator_cycles": 1}, [{"action": "delegate", "summary": "Search.", "task": "Search."},
                                      {"action": "report", "summary": "Blocked."}], "Coordinator cycles"),
    ({"max_tool_calls": 1}, [{"action": "delegate", "summary": "Search.", "task": "Search."},
                              tool("look")], "tool calls"),
    ({"max_critic_reviews": 1}, [{"action": "consult_critic", "summary": "Review.", "plan": "Search."},
                                {"approved": True, "summary": "Good."},
                                {"action": "consult_critic", "summary": "Review.", "plan": "Search."}], "Critic reviews"),
])
async def test_limits(limits, replies, expected):
    mgr, _, _ = manager(ScriptedLLM(replies), **limits)
    task = mgr.start("Find keys.")
    await mgr.runner
    assert task.status == "failed"
    assert expected in task.summary


async def test_timeout_and_cancel():
    mgr, _, _ = manager(WaitingLLM(), task_timeout_seconds=0.02)
    task = mgr.start("Search.")
    await mgr.runner
    assert task.status == "failed" and "time limit" in task.summary
    task = mgr.start("Search again.")
    await mgr.cancel(task.id)
    assert task.status == "cancelled"
    assert mgr.active_id is None


async def test_false_completion_rejected():
    mgr, _, _ = manager(ScriptedLLM([
        {"action": "complete", "summary": "Invented success.", "evidence_ids": ["fake"]}
    ]), max_coordinator_cycles=1)
    task = mgr.start("Find charger.")
    await mgr.runner
    assert task.status == "failed"
    assert task.critic_count == 0
    assert "unknown" in task.critic_feedback[0]["summary"]


async def test_model_failure_does_not_crash_manager():
    mgr, _, _ = manager(ScriptedLLM(["not JSON", "not JSON"]))
    task = mgr.start("Find charger.")
    await mgr.runner
    assert task.status == "failed"
    assert "invalid structured output" in task.summary


async def test_ollama_contract_and_thinking_exclusion():
    def respond(request):
        import json
        body = json.loads(request.content)
        assert body["think"] is False and body["stream"] is False
        assert body["format"]["type"] == "object"
        return httpx.Response(200, json={"message": {"content": '{"approved":true,"summary":"OK"}',
                                                      "thinking": "private scratchpad"}})
    client = OllamaClient(Settings())
    await client.http.aclose()
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(respond), base_url="http://test")
    result = await structured(client, CriticReview, "Review", {})
    assert "private" not in result.model_dump_json()
    await client.close()


async def test_critic_rejection_returns_to_coordinator():
    fake = ScriptedLLM([
        {"action": "delegate", "summary": "Observe.", "task": "Look."}, tool("look"),
        {"action": "report", "summary": "Hall observed."}, complete,
        {"approved": False, "summary": "Charger has not been observed.", "suggestion": "Search rooms."},
        {"action": "fail", "summary": "Cannot establish completion within this test."}])
    mgr, _, _ = manager(fake)
    task = mgr.start("Find charger.")
    await mgr.runner
    assert task.status == "failed" and task.cycle_count == 3
    assert task.critic_feedback[0]["approved"] is False
