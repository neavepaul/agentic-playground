import json

import httpx
import pytest
from pydantic import ValidationError

from app.agents.explorer import Explorer
from app.agents.schemas import CoordinatorDecision, CriticReview, GoalCondition
from app.config import Settings
from app.events.bus import EventBus
from app.llm.base import ModelError, structured
from app.llm.ollama import OllamaClient
from app.tasks.manager import TaskManager, critic_review
from app.tasks.models import TaskContext
from tests.fakes import ScriptedLLM, WaitingLLM, tool


@pytest.fixture(autouse=True)
def offline_search(monkeypatch):
    from app.capabilities import tools
    monkeypatch.setattr(tools, "web_search", lambda query: {
        "query": query, "results": [{"title": "Source", "url": "https://example.com", "snippet": query}]})


def manager(client, **limits):
    bus = EventBus()
    return TaskManager(client, bus, Settings(**limits)), bus


def answer(payload):
    return {"summary": "Ready.", "action": {"tool": "finish_answer", "answer": "Found it.", "evidence_ids": [
        o["evidence_id"] for o in payload["tool_observations"] if o["success"]]}}


def record(task, name, args, observation, evidence_id):
    task.record(name, args, {"success": True, "observation": observation, "evidence_id": evidence_id})


def lookup_task():
    return TaskContext(goal="Look up a fact.", conditions=[GoalCondition(required_tools=["web_search"])])


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


async def test_validation_diagnostics_do_not_expose_rejected_content(caplog):
    secret = "private rejected model content"
    invalid = {"approved": secret, "summary": "Review.", secret: secret}
    client = ScriptedLLM([invalid, invalid])
    with pytest.raises(ModelError) as error:
        await structured(client, CriticReview, "Review", {})
    assert "approved: bool_type" in str(error.value)
    assert "response: extra_forbidden" in str(error.value)
    assert "approved: bool_type" in client.calls[1][0][-1]["content"]
    assert secret not in str(error.value)
    assert secret not in caplog.text
    assert secret not in str(client.calls)


async def test_new_observations_expire_advice_without_losing_audit():
    fake = ScriptedLLM([{"approved": False, "summary": "Read the source before answering."}])
    mgr, _ = manager(fake)
    task = lookup_task()
    record(task, "web_search", {"query": "a fact"}, {"query": "a fact", "results": []}, "search-id")
    await mgr.review(task, "Answer from the search results.", "plan")
    assert len(task.compact()["critic_feedback"]) == 1
    record(task, "read_webpage", {"url": "https://example.com"},
           {"url": "https://example.com", "title": "Source", "text": "Facts"}, "page-id")
    payload = task.compact()
    assert payload["critic_feedback"] == []
    assert len(task.critic_feedback) == 1  # Audit history survives.


async def test_critic_reviews_current_evidence_without_replaying_old_verdict():
    task = lookup_task()
    record(task, "web_search", {"query": "a fact"}, {"query": "a fact", "results": []}, "search-id")
    stale = {"approved": False, "summary": "The search alone does not settle it."}
    task.critic_feedback.append(stale)
    action = {"tool": "finish_answer", "arguments": {"answer": "Found it.", "evidence_ids": ["search-id"]}}
    fake = ScriptedLLM([stale, {"approved": True, "summary": "The page supports the answer."}])
    await critic_review(fake, task, "Answer.", "completion", proposed_action=action)
    before = json.loads(fake.calls[0][0][1]["content"])
    assert [a["tool"] for a in before["recent_actions"]] == ["web_search"]
    record(task, "read_webpage", {"url": "https://example.com"},
           {"url": "https://example.com", "title": "Source", "text": "Facts"}, "page-id")
    await critic_review(fake, task, "Answer.", "completion", proposed_action=action)
    after = json.loads(fake.calls[1][0][1]["content"])
    assert after["proposed_action"] == action
    assert [a["tool"] for a in after["recent_actions"]] == ["web_search", "read_webpage"]
    assert not after["required_outcomes"][0]["satisfied"]  # The answer is still only proposed.
    assert "critic_feedback" not in before and "critic_feedback" not in after
    assert task.critic_feedback == [stale]  # Preserve history outside independent review.


@pytest.mark.parametrize("invalid_field,value,error_code", [
    ("summary", None, "string_type"),
    ("summary", "x" * 301, "string_too_long"),
    ("action", {"tool": "send_email", "to": "someone"}, "union_tag_invalid"),
])
async def test_explorer_recovers_from_invalid_model_output(invalid_field, value, error_code):
    invalid = tool("web_search", query="a fact")
    invalid[invalid_field] = value
    fake = ScriptedLLM([invalid, tool("web_search", query="a fact"), answer])
    mgr, _ = manager(fake)
    task = mgr.start("Look up a fact.")
    await mgr.runner
    assert task.status == "completed", task.summary
    searches = [a for a in task.action_history if a["tool"] == "web_search"]
    assert len(searches) == 1 and searches[0]["success"]
    assert task.critic_count == 0
    assert any(f"{invalid_field}: {error_code}" in messages[-1]["content"]
               for messages, _ in fake.calls)


@pytest.mark.parametrize("limits,replies,expected", [
    ({"max_coordinator_cycles": 1, "max_explorer_actions": 1}, [tool("web_search", query="keys")],
     "Coordinator cycles"),
    ({"max_tool_calls": 1}, [tool("web_search", query="keys"), tool("web_search", query="lost keys")], "tool calls"),
    ({"max_critic_reviews": 1, "max_explorer_actions": 1}, [
        tool("web_search", query="keys"),
        {"action": "consult_critic", "summary": "Review.", "plan": "Search."},
        {"approved": True, "summary": "Good."},
        {"action": "consult_critic", "summary": "Review.", "plan": "Search."}], "Critic reviews"),
])
async def test_limits(limits, replies, expected):
    mgr, _ = manager(ScriptedLLM(replies), **limits)
    task = mgr.start("Find keys.")
    await mgr.runner
    assert task.status == "failed"
    assert expected in task.summary


async def test_timeout_and_cancel():
    mgr, _ = manager(WaitingLLM(), task_timeout_seconds=0.02)
    task = mgr.start("Search.")
    await mgr.runner
    assert task.status == "failed" and "time limit" in task.summary
    task = mgr.start("Search again.")
    await mgr.cancel(task.id)
    assert task.status == "cancelled"
    assert mgr.active_id is None


async def test_false_completion_rejected():
    # Nothing has answered the question yet. The deterministic condition check
    # rejects completion before the Critic is consulted.
    mgr, _ = manager(ScriptedLLM([
        tool("web_search", query="a fact"),
        {"action": "complete", "summary": "Invented success.", "evidence_ids": ["fake"]}
    ]), max_explorer_actions=1, max_coordinator_cycles=2)
    task = mgr.start("Look up a fact.")
    await mgr.runner
    assert task.status == "failed"
    assert task.critic_count == 0
    assert "required outcomes lack tool evidence" in task.critic_feedback[0]["summary"]


async def test_model_failure_does_not_crash_manager():
    mgr, _ = manager(ScriptedLLM(["not JSON", "not JSON"]))
    task = mgr.start("Find charger.")
    await mgr.runner
    assert task.status == "failed"
    assert "invalid structured output" in task.summary


async def test_ollama_contract_and_thinking_exclusion():
    def respond(request):
        import json
        body = json.loads(request.content)
        assert body["think"] is True and body["stream"] is False
        assert body["options"]["num_predict"] == 4096
        assert body["format"]["type"] == "object"
        return httpx.Response(200, json={"message": {"content": '{"approved":true,"summary":"OK"}',
                                                      "thinking": "private scratchpad"}})
    client = OllamaClient(Settings(decision_thinking=True))
    await client.http.aclose()
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(respond), base_url="http://test")
    result = await structured(client, CriticReview, "Review", {})
    assert "private" not in result.model_dump_json()
    await client.close()


@pytest.mark.parametrize("mode", ["offline", "timeout", "missing", "envelope"])
async def test_ollama_errors_are_safe(mode):
    def respond(request):
        if mode == "offline":
            raise httpx.ConnectError("internal details", request=request)
        if mode == "timeout":
            raise httpx.ReadTimeout("internal details", request=request)
        if mode == "missing":
            return httpx.Response(404, json={"error": "missing model"})
        return httpx.Response(200, json={"unexpected": "envelope"})
    client = OllamaClient(Settings())
    await client.http.aclose()
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(respond), base_url="http://test")
    with pytest.raises(ModelError) as exc:
        await client.generate([], {})
    assert "internal details" not in str(exc.value)
    await client.close()


async def test_report_claims_do_not_reach_the_explorer():
    task = lookup_task()
    record(task, "web_search", {"query": "a fact"}, {"query": "a fact", "results": []}, "search-id")
    task.explorer_reports.append("Every source was already checked.")
    task.critic_feedback.append({"kind": "delegation_report", "approved": False,
                                "summary": "Every source was already checked.",
                                "observed_action_count": len(task.action_history)})
    fake = ScriptedLLM([tool("web_search", query="another source")])
    await Explorer(fake).decide(task, "Continue searching.")
    payload = json.loads(fake.calls[0][0][1]["content"])
    assert "Every source was already checked" not in str(payload)
    assert task.explorer_reports and task.critic_feedback  # Audit remains intact.


async def test_failed_tool_can_be_retried_with_same_arguments():
    fake = ScriptedLLM([tool("web_search", query="dinner recipe"), tool("web_search", query="dinner recipe"), answer])
    mgr, _ = manager(fake)
    execute = mgr.capabilities.execute
    failed_once = False
    async def transient_failure(name, args, context):
        nonlocal failed_once
        if name == "web_search" and not failed_once:
            failed_once = True
            return {"success": False, "error": "Temporary search failure", "evidence_id": "search-failed"}
        return await execute(name, args, context)
    mgr.capabilities.execute = transient_failure
    task = mgr.start("Find a dinner recipe.")
    await mgr.runner
    assert task.status == "completed", task.summary
    searches = [a for a in task.action_history if a["tool"] == "web_search"]
    assert [a["success"] for a in searches] == [False, True]
    assert not any("Repeated identical" in message for message in task.action_feedback)
