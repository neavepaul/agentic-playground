from pathlib import Path
LEGACY_WORLD = Path(__file__).parent / "fixtures" / "legacy_house.json"

import asyncio

import httpx
import pytest
from pydantic import ValidationError

from app.agents.commands import observable_commands
from app.agents.schemas import CoordinatorDecision, CriticReview, GoalCondition
from app.agents.coordinator import Coordinator
from app.config import Settings
from app.evaluate import mission_satisfied
from app.events.bus import EventBus
from app.llm.base import ModelError, structured
from app.llm.ollama import OllamaClient
from app.tasks.manager import TaskManager
from app.tasks.models import TaskContext
from app.world.engine import WorldEngine
from app.world.tools import WorldTools
from tests.fakes import ScriptedLLM, WaitingLLM, complete, tool


def manager(client, **limits):
    engine, bus = WorldEngine(LEGACY_WORLD), EventBus()
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


async def test_goal_planner_repairs_omitted_delivery_condition():
    class IncompletePlanner:
        async def generate(self, messages, response_schema):
            return '{"summary":"Find the charger and recipient.","conditions":[' \
                   '{"kind":"identify_recipient","object":"charger"}]}'

    plan = await Coordinator(IncompletePlanner()).define_goal(
        "Find out who needs the charger and deliver it.")
    assert [condition.kind for condition in plan.conditions] == ["identify_recipient", "deliver"]
    assert plan.conditions[-1].object == "charger"


async def test_delivery_end_to_end_with_mock_model():
    replies = [{"action": "delegate", "summary": "Locate the charger and recipient.", "task": "Search study."},
               tool("look"), tool("move_to", room="study"), tool("look"),
               tool("talk_to", person="dad", message="Who needs the charger?"),
               tool("pick_up", object="charger"),
               tool("move_to", room="hall"),
               tool("move_to", room="bedroom"), tool("look"),
               {"action": "delegate", "summary": "Deliver charger.", "task": "Give charger to Neave."},
               tool("give", object="charger", person="neave"),
               {"approved": True, "summary": "Neave is the confirmed recipient."},
               {"action": "report", "summary": "Delivery action succeeded."}, complete,
               {"approved": True, "summary": "Delivery and need are evidenced."}]
    fake = ScriptedLLM(replies)
    mgr, engine, bus = manager(fake)
    task = mgr.start("Find out who needs the charger and deliver it.")
    await mgr.runner
    assert task.status == "completed"
    assert task.critic_count == 2
    assert task.action_history[0]["observation"] == {"room": "hall", "inventory": []}
    assert mission_satisfied(3, task, engine.snapshot())
    assert mgr.active_id is None
    assert any(e["type"] == "object_given" for e in bus.history)
    # Initial Coordinator and Explorer inputs do not reveal hidden occupants/items.
    for messages, _ in fake.calls[:2]:
        payload = messages[1]["content"]
        assert '"people": {' not in payload or '"people": {}' in payload
        assert '"objects": {' not in payload or '"objects": {}' in payload
        assert '"world"' not in messages[1]["content"]


@pytest.mark.parametrize("limits,replies,expected", [
    ({"max_coordinator_cycles": 1}, [{"action": "delegate", "summary": "Search.", "task": "Search."},
                                      {"action": "report", "summary": "Blocked."},
                                      {"approved": True, "summary": "Report acknowledged."}], "Coordinator cycles"),
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
    ], conditions=[{"kind": "visit_room", "room": "hall"}]), max_coordinator_cycles=1)
    task = mgr.start("Go to the hall.")
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
        {"action": "delegate", "summary": "Observe.", "task": "Find charger."}, tool("look"),
        tool("move_to", room="study"), tool("look"),
        {"action": "report", "summary": "Charger observed."}, complete,
        {"approved": False, "summary": "Charger has not been observed.", "suggestion": "Search rooms."},
        {"action": "fail", "summary": "Cannot establish completion within this test."}])
    mgr, _, _ = manager(fake)
    task = mgr.start("Find charger.")
    await mgr.runner
    assert task.status == "failed" and task.cycle_count == 3
    assert task.critic_feedback[0]["approved"] is False
    assert task.critic_count == 1


async def test_empty_speech_is_repaired_before_tool_execution():
    fake = ScriptedLLM([
        {"action": "delegate", "summary": "Notify Neave.", "task": "Tell Neave dinner is ready."},
        tool("look"), tool("move_to", room="bedroom"), tool("look"),
        tool("talk_to", person="neave", message=""),
        tool("talk_to", person="neave", message="Dinner is ready."),
        {"action": "report", "summary": "Neave notified."}, complete,
        {"approved": True, "summary": "Message delivery was observed."}])
    mgr, engine, _ = manager(fake)
    task = mgr.start("Find Neave and tell him dinner is ready.")
    await mgr.runner
    assert task.status == "completed"
    assert all(entry["success"] for entry in task.action_history)
    assert sum(entry["tool"] == "talk_to" for entry in task.action_history) == 1
    assert "actual nonblank words" in fake.calls[5][0][-1]["content"]
    assert mission_satisfied(2, task, engine.snapshot())


@pytest.mark.parametrize("number,actions", [
    (1, [tool("move_to", room="study"), tool("look")]),
    (2, [tool("move_to", room="bedroom"), tool("look"),
         tool("talk_to", person="neave", message="Dinner is ready.")]),
    (4, [tool("move_to", room="kitchen"), tool("look")]),
])
async def test_other_evaluation_missions(number, actions):
    from app.evaluate import MISSIONS
    fake = ScriptedLLM([
        {"action": "delegate", "summary": "Carry out mission.", "task": MISSIONS[number]},
        tool("look"), *actions, {"action": "report", "summary": "Mission actions performed."}, complete,
        {"approved": True, "summary": "Evidence matches the goal."}])
    mgr, engine, _ = manager(fake)
    task = mgr.start(MISSIONS[number])
    await mgr.runner
    assert task.status == "completed"
    assert mission_satisfied(number, task, engine.snapshot())


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


async def test_no_progress_report_is_reviewed_and_recovers():
    fake = ScriptedLLM([
        {"action": "delegate", "summary": "Search.", "task": "Find charger."},
        {"action": "report", "summary": "Charger found."},
        {"approved": False, "summary": "No observation supports finding it.", "suggestion": "Search the rooms."},
        tool("look"), tool("move_to", room="study"), tool("look"),
        {"action": "report", "summary": "Charger observed."}, complete,
        {"approved": True, "summary": "Observed charger."}])
    mgr, _, bus = manager(fake)
    task = mgr.start("Find charger.")
    await mgr.runner
    assert task.status == "completed"
    assert task.critic_count == 2
    assert not any(e["type"] == "agent_message" and e["data"].get("summary") == "Charger found."
                   for e in bus.history)


async def test_unobserved_room_report_forces_scan_before_coordinator_replan():
    fake = ScriptedLLM([
        {"action": "delegate", "summary": "Search the house.", "task": "Find the charger and its recipient."},
        tool("look"), tool("move_to", room="kitchen"), {"action": "report", "summary": "Kitchen checked."},
        tool("look"), tool("talk_to", person="mom", message="Where are the keys?"),
        {"action": "report", "summary": "Kitchen investigated."},
        {"action": "fail", "summary": "Stop test."},
    ])
    mgr, _, bus = manager(fake, max_coordinator_cycles=2)
    task = mgr.start("Find out who needs the charger and deliver it.")
    await mgr.runner
    assert task.status == "failed"
    assert [entry["tool"] for entry in task.action_history] == ["get_status", "get_map", "look", "move_to", "look", "talk_to"]
    assert not any("current room has not been observed" in message for message in task.action_feedback)
    assert not any(event["type"] == "critic_review" for event in bus.history)


async def test_explorer_rejects_repeated_read_without_new_information():
    from app.agents.explorer import Explorer
    from app.tasks.models import TaskContext
    engine, bus = WorldEngine(LEGACY_WORLD), EventBus()
    tools = WorldTools(engine, bus)
    context = TaskContext(goal="Find keys.")
    context.record("get_status", {}, tools.execute("get_status", {}))
    context.record("look", {}, tools.execute("look", {}))
    fake = ScriptedLLM([tool("look"), tool("move_to", room="kitchen")])
    choice = await Explorer(fake, tools.schemas()).decide(context, "Find keys.")
    assert choice.tool == "move_to"
    assert len(fake.calls) == 2
    assert "look" not in fake.calls[0][1]["properties"]["command_id"]["enum"]


async def test_same_message_can_be_spoken_again_without_critic_debate():
    fake = ScriptedLLM([
        {"action": "delegate", "summary": "Search.", "task": "Find keys."},
        tool("look"), tool("move_to", room="kitchen"), tool("look"),
        tool("talk_to", person="mom", message="Where are the keys?"),
        tool("talk_to", person="mom", message="Where are the keys?"),
        {"action": "report", "summary": "The keys were already observed."},
        complete, {"approved": True, "summary": "Keys observed."}])
    mgr, _, bus = manager(fake)
    task = mgr.start("Find keys.")
    await mgr.runner
    assert task.status == "completed"
    assert task.tool_count == 7 and task.critic_count == 1
    assert sum(a["tool"] == "talk_to" for a in task.action_history) == 2
    assert not task.action_feedback


def test_recipient_can_still_receive_other_messages():
    tools = WorldTools(WorldEngine(LEGACY_WORLD), EventBus())
    task = TaskContext(
        goal="Find out who needs the charger and deliver it.",
        conditions=[GoalCondition(kind="deliver", object="charger", person="")],
    )

    def act(name, **args):
        task.record(name, args, tools.execute(name, args))

    act("get_status")
    act("look")
    act("move_to", room="bedroom")
    act("look")
    act("talk_to", person="neave", message="Who needs the charger?")

    choices, _ = observable_commands(task)
    assert "talk_to:neave" in choices
    assert "move_to:hall" in choices
    assert "move_to:study" not in choices


def test_known_recipient_can_still_be_addressed_without_repeating_investigation():
    tools = WorldTools(WorldEngine(LEGACY_WORLD), EventBus())
    task = TaskContext(
        goal="Find out who needs the charger and deliver it.",
        conditions=[GoalCondition(kind="deliver", object="charger")],
    )

    def act(name, **args):
        result = tools.execute(name, args)
        task.record(name, args, result)
        return result

    act("get_status")
    act("look")
    act("move_to", room="bedroom")
    act("look")
    choices, _ = observable_commands(task)
    assert "talk_to:neave" in choices
    assert task.repeat_reason("neave", "Who needs the charger?") is None
    assert task.repeat_reason("neave", "I have your charger.") is None


def test_conversation_preserves_local_navigation():
    tools = WorldTools(WorldEngine(LEGACY_WORLD), EventBus())
    task = TaskContext(
        goal="Find out who needs the charger and deliver it.",
        conditions=[GoalCondition(kind="deliver", object="charger", person="")],
    )

    def act(name, **args):
        task.record(name, args, tools.execute(name, args))

    act("get_status")
    act("look")
    act("move_to", room="bedroom")
    act("look")
    act("talk_to", person="neave", message="Who needs the charger?")

    choices, _ = observable_commands(task)
    assert "talk_to:neave" in choices
    assert "move_to:hall" in choices


async def test_valid_evidence_ids_cannot_substitute_for_required_outcomes():
    fake = ScriptedLLM([
        {"action": "delegate", "summary": "Ask recipient.", "task": "Ask who needs charger."},
        tool("look"), tool("move_to", room="bedroom"), tool("look"),
        tool("talk_to", person="neave", message="Who needs the charger?"),
        {"action": "report", "summary": "Neave needs it."}, complete,
        {"approved": True, "summary": "This model vote must never be consulted."}])
    mgr, _, _ = manager(fake, max_coordinator_cycles=2)
    task = mgr.start("Find out who needs the charger and deliver it.")
    await mgr.runner
    assert task.status == "failed" and task.critic_count == 0
    assert task.critic_feedback[-1]["unmet_conditions"][0]["kind"] == "deliver"


async def test_pickup_does_not_wait_for_critic_approval():
    fake = ScriptedLLM([
        {"action": "delegate", "summary": "Find charger.", "task": "Find and pick up charger."},
        tool("look"), tool("move_to", room="study"), tool("look"),
        tool("talk_to", person="dad", message="Who needs the charger?"),
        tool("pick_up", object="charger"),
        {"action": "report", "summary": "Charger picked up."},
        {"action": "fail", "summary": "Stopping the test after pickup."}])
    mgr, engine, _ = manager(fake)
    task = mgr.start("Find out who needs the charger and deliver it.")
    await mgr.runner
    assert task.status == "failed" and task.critic_count == 0
    assert engine.snapshot()["objects"]["charger"]["location"] == {"kind": "robot", "id": "robot"}


async def test_delivery_search_continues_after_repeated_recipient_question():
    fake = ScriptedLLM([
        {"action": "delegate", "summary": "Find item and recipient.", "task": "Find charger and who needs it."},
        tool("look"), tool("move_to", room="bedroom"), tool("look"),
        tool("talk_to", person="neave", message="Who needs the charger?"),
        tool("talk_to", person="neave", message="WHO needs the charger?!"),
        tool("move_to", room="hall"), tool("move_to", room="study"), tool("look"),
        tool("pick_up", object="charger"),
        {"action": "delegate", "summary": "Handoff.", "task": "Take held charger to Neave."},
        tool("move_to", room="hall"), tool("move_to", room="bedroom"),
        tool("give", object="charger", person="neave"), {"approved": True, "summary": "Held and nearby."},
        {"action": "report", "summary": "Transferred."}, complete,
        {"approved": True, "summary": "Delivery evidenced."},
    ])
    mgr, engine, _ = manager(fake, max_explorer_actions=9)
    task = mgr.start("Find who needs the charger and deliver it.")
    await mgr.runner
    assert task.status == "completed", task.summary
    assert engine.snapshot()["objects"]["charger"]["location"] == {"kind": "person", "id": "neave"}
    assert sum(a["tool"] == "talk_to" for a in task.action_history) == 2
    assert task.critic_count == 2  # give and completion; pickup is deterministic
    assert not task.action_feedback


async def test_conversation_remains_bounded_by_task_limits():
    fake = ScriptedLLM([
        {"action": "delegate", "summary": "Find recipient.", "task": "Ask who needs charger."},
        tool("look"), tool("move_to", room="bedroom"), tool("look"),
        tool("talk_to", person="neave", message="Who needs the charger?"),
        tool("talk_to", person="neave", message="Who needs the charger?"),
        tool("talk_to", person="neave", message="Who needs the charger?"),
    ])
    mgr, _, _ = manager(fake, max_explorer_actions=6, max_coordinator_cycles=1)
    task = mgr.start("Find who needs the charger and deliver it.")
    await mgr.runner
    assert task.status == "failed"
    assert "Coordinator cycles" in task.summary
    assert task.critic_count == 0
    assert sum(a["tool"] == "talk_to" for a in task.action_history) == 3


@pytest.mark.parametrize("message", [None, "", "   "])
async def test_missing_or_blank_speech_never_reaches_tool(message):
    from app.agents.explorer import Explorer
    tools = WorldTools(WorldEngine(LEGACY_WORLD), EventBus())
    context = TaskContext(goal="Deliver charger", conditions=[GoalCondition(kind="deliver", object="charger")])
    for name, args in [("get_status", {}), ("move_to", {"room": "bedroom"}), ("look", {})]:
        context.record(name, args, tools.execute(name, args))
    invalid = {"command_id": "talk_to:neave", "summary": "Ask about charger."}
    if message is not None:
        invalid["message"] = message
    fake = ScriptedLLM([invalid, {"command_id": "talk_to:neave", "message": "Who needs the charger?", "summary": "Ask."}])
    choice = await Explorer(fake, tools.schemas()).decide(context, "Ask Neave.")
    assert choice.arguments["message"] == "Who needs the charger?"
    assert len(fake.calls) == 2
    assert "message" in fake.calls[0][1]["required"]
    assert fake.calls[0][1]["properties"]["message"]["minLength"] == 1
    assert all(a["tool"] != "talk_to" for a in context.action_history)


async def test_persistently_empty_speech_fails_before_any_speech_action():
    fake = ScriptedLLM([
        {"action": "delegate", "summary": "Notify.", "task": "Tell Neave dinner is ready."},
        tool("look"), tool("move_to", room="bedroom"), tool("look"),
        tool("talk_to", person="neave", message=""),
        tool("talk_to", person="neave", message="   "),
    ])
    mgr, _, _ = manager(fake)
    task = mgr.start("Tell Neave dinner is ready.")
    await mgr.runner
    assert task.status == "failed" and "invalid structured output twice" in task.summary
    assert all(a["tool"] != "talk_to" for a in task.action_history)
    assert task.critic_count == 0


async def test_failed_speech_can_be_retried_with_same_valid_words():
    fake = ScriptedLLM([
        {"action": "delegate", "summary": "Notify.", "task": "Tell Neave dinner is ready."},
        tool("look"), tool("move_to", room="bedroom"), tool("look"),
        tool("talk_to", person="neave", message="Dinner is ready."),
        tool("talk_to", person="neave", message="Dinner is ready."),
        {"action": "report", "summary": "Message delivered."}, complete,
        {"approved": True, "summary": "Notification evidenced."},
    ])
    mgr, _, _ = manager(fake)
    execute = mgr.tools.execute
    failed_once = False
    def transient_failure(name, args, task_id=None):
        nonlocal failed_once
        if name == "talk_to" and not failed_once:
            failed_once = True
            return {"success": False, "error": "Temporary speech failure", "evidence_id": "speech-failed"}
        return execute(name, args, task_id)
    mgr.tools.execute = transient_failure
    task = mgr.start("Tell Neave dinner is ready.")
    await mgr.runner
    assert task.status == "completed", task.summary
    talks = [a for a in task.action_history if a["tool"] == "talk_to"]
    assert [a["success"] for a in talks] == [False, True]
    assert not any("Repeated identical" in message for message in task.action_feedback)
