from pathlib import Path
LEGACY_WORLD = Path(__file__).parent / "fixtures" / "legacy_house.json"

import asyncio

import httpx
import pytest
from pydantic import ValidationError

from app.agents.explorer import observable_commands
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


def test_model_entity_ids_are_canonical_without_rewriting_speech():
    from app.agents.schemas import SpokenNeed

    condition = GoalCondition(kind="place_object", object=" Sensor Pack ", room="Work Room")
    assert (condition.object, condition.room) == ("sensor_pack", "work_room")
    need = SpokenNeed(object="Sensor Pack", person=" BEA ", quote="Bea needs the Sensor Pack.")
    assert (need.object, need.person) == ("sensor_pack", "bea")
    assert need.quote == "Bea needs the Sensor Pack."
    with pytest.raises(ValidationError):
        SpokenNeed(object=" ", person="Bea", quote="Original speech")
    with pytest.raises(ValidationError):
        GoalCondition(kind="find_person", person=123)


def test_mixed_case_goal_allows_observed_recipient_handoff():
    tools = WorldTools(WorldEngine(LEGACY_WORLD), EventBus())
    task = TaskContext(goal="Bring the charger to Neave.", conditions=[
        GoalCondition(kind="deliver", object="Charger", person="Neave")])
    for name, args in [("get_status", {}), ("move_to", {"room": "study"}),
                       ("look", {}), ("pick_up", {"object": "charger"}),
                       ("move_to", {"room": "hall"}), ("move_to", {"room": "bedroom"}),
                       ("look", {})]:
        result = tools.execute(name, args)
        assert result["success"]
        task.record(name, args, result)
    delivery = task.delivery_state()[0]
    assert delivery["recipient_last_seen"] == "bedroom"
    assert delivery["missing_prerequisites"] == ["give_object"]
    commands, _ = observable_commands(task)
    assert "give:charger:neave" in commands
    assert "move_to:hall" in commands  # The model still chooses the action.
    args = {"object": "charger", "person": "neave"}
    result = tools.execute("give", args)
    assert result["success"]
    task.record("give", args, result)
    assert task.delivery_state()[0]["delivered"]
    assert task.compact()["required_outcomes"][0]["satisfied"]


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


async def test_mixed_command_schema_requires_speech_only_for_talking():
    from app.agents.explorer import Explorer
    tools = WorldTools(WorldEngine(LEGACY_WORLD), EventBus())
    task = TaskContext(goal="Find charger", conditions=[GoalCondition(kind="find_object", object="charger")])
    for name, args in [("get_status", {}), ("move_to", {"room": "bedroom"}), ("look", {})]:
        task.record(name, args, tools.execute(name, args))
    fake = ScriptedLLM([tool("talk_to", person="neave", message=""), tool("move_to", room="hall")])
    decision = await Explorer(fake, tools.schemas()).decide(task, "Search for charger")
    schema = fake.calls[0][1]
    speech, other = schema["anyOf"]
    for branch in (speech, other):
        assert "summary" in branch["required"]
        assert "summary" in branch["properties"]
        assert branch["additionalProperties"] is False
    assert speech["properties"]["command_id"]["enum"] == ["talk_to:neave"]
    assert "message" in speech["required"]
    assert speech["properties"]["message"]["minLength"] == 1
    assert speech["properties"]["message"]["pattern"] == r"\S"
    assert "move_to:hall" in other["properties"]["command_id"]["enum"]
    assert "talk_to:neave" not in other["properties"]["command_id"]["enum"]
    assert "message" not in other["required"]
    assert "speech_message_required" in fake.calls[1][0][-1]["content"]
    assert decision.tool == "move_to"


async def test_new_observations_expire_advice_without_losing_inventory():
    fake = ScriptedLLM([{"approved": False, "summary": "Locate Neave in the unobserved rooms."}])
    mgr, _, _ = manager(fake)
    task = TaskContext(goal="Deliver charger to Neave.",
                       conditions=[GoalCondition(kind="deliver", object="charger", person="neave")])
    for name, args in [("get_status", {}), ("move_to", {"room": "study"}),
                       ("look", {}), ("pick_up", {"object": "charger"})]:
        mgr.call_tool(task, name, args)
    await mgr.review(task, "Locate recipient.", "plan")
    assert len(task.compact()["critic_feedback"]) == 1
    for name, args in [("move_to", {"room": "hall"}), ("move_to", {"room": "bedroom"}), ("look", {})]:
        mgr.call_tool(task, name, args)
    raw_scan = task.action_history[-1]["observation"]
    assert raw_scan["held_objects"] == []
    payload = task.compact()
    assert payload["critic_feedback"] == []
    assert len(task.critic_feedback) == 1  # Audit history survives.
    assert payload["robot_status"]["inventory"] == ["charger"]
    view = payload["current_room_observation"]
    assert view["robot_inventory"] == ["charger"]
    assert view["objects_held_by_people"] == [] and "held_objects" not in view
    assert any(p["id"] == "neave" for p in view["people"])
    assert "give:charger:neave" in observable_commands(task)[0]
    mgr.call_tool(task, "give", {"object": "charger", "person": "neave"})
    view = task.compact()["current_room_observation"]
    assert view["robot_inventory"] == []
    assert view["objects_held_by_people"] == [{"object": "charger", "person": "neave"}]


async def test_critic_reviews_current_transfer_without_replaying_old_verdict():
    import json
    from app.tasks.manager import critic_review

    tools = WorldTools(WorldEngine(LEGACY_WORLD), EventBus())
    task = TaskContext(goal="Deliver charger to Neave.",
                       conditions=[GoalCondition(kind="deliver", object="charger", person="neave")])
    for name, args in [("get_status", {}), ("move_to", {"room": "study"}),
                       ("look", {}), ("pick_up", {"object": "charger"})]:
        task.record(name, args, tools.execute(name, args))
    stale = {"approved": False, "summary": "Need to locate Neave in the house to deliver."}
    task.critic_feedback.append(stale)
    action = {"tool": "give", "arguments": {"object": "charger", "person": "neave"}}
    fake = ScriptedLLM([stale, {"approved": True, "summary": "Recipient now visible and item held."}])
    await critic_review(fake, task, "Transfer charger.", "object_transfer", proposed_action=action)
    before = json.loads(fake.calls[0][0][1]["content"])
    assert all(p["id"] != "neave" for p in before["current_room_observation"]["people"])
    for name, args in [("move_to", {"room": "hall"}), ("move_to", {"room": "bedroom"}), ("look", {})]:
        task.record(name, args, tools.execute(name, args))
    await critic_review(fake, task, "Transfer charger.", "object_transfer", proposed_action=action)
    after = json.loads(fake.calls[1][0][1]["content"])
    assert after["proposed_action"] == action
    assert after["robot_status"] == {"room": "bedroom", "inventory": ["charger"]}
    assert any(p["id"] == "neave" for p in after["current_room_observation"]["people"])
    assert not after["required_outcomes"][0]["satisfied"]  # Transfer is still proposed.
    assert "critic_feedback" not in before and "critic_feedback" not in after
    assert task.critic_feedback == [stale]  # Preserve history outside independent review.


@pytest.mark.parametrize("object_id,source_room", [("charger", "study"), ("laptop", "bedroom")])
async def test_ready_handoff_context_survives_search_delegation(object_id, source_room):
    import json
    from app.agents.explorer import Explorer, reflex_action

    tools = WorldTools(WorldEngine(LEGACY_WORLD), EventBus())
    task = TaskContext(goal=f"Deliver {object_id} to Neave.",
                       conditions=[GoalCondition(kind="deliver", object=object_id, person="neave"), GoalCondition(kind="find_person", person="neave")])

    def act(name, **args):
        result = tools.execute(name, args)
        assert result["success"]
        task.record(name, args, result)

    def handoffs() -> list[str]:
        commands, _ = observable_commands(task)
        return [id for id, command in commands.items() if command.get("tool") == "give"]

    async def payload_for(reply):
        fake = ScriptedLLM([reply])
        await Explorer(fake, tools.schemas()).decide(task, "Search bedroom and kitchen for Neave")
        assert fake.calls, "this decision was expected to need the model"
        return json.loads(fake.calls[0][0][1]["content"])

    act("get_status")
    act("look")
    payload = await payload_for(tool("move_to", room=source_room))
    assert payload["ready_handoffs"] == []  # Nothing held and no recipient here.

    act("move_to", room=source_room)
    act("look")
    assert handoffs() == []  # Seeing an item is not holding it.
    commands, view = observable_commands(task)
    assert reflex_action(task, commands, view).tool == "pick_up"

    act("pick_up", object=object_id)
    act("move_to", room="hall")
    payload = await payload_for(tool("move_to", room="bedroom"))
    assert payload["ready_handoffs"] == []  # Recipient is not here.
    assert len(payload["recent_actions"]) <= 8
    assert all("observation" not in entry for entry in payload["recent_actions"])
    assert payload["task_memory"]["objects"][object_id]["location"] == {"kind": "robot", "id": "robot"}
    assert "move_to:bedroom" in payload["commands"]

    act("move_to", room="bedroom")
    act("look")
    assert handoffs() == [f"give:{object_id}:neave"]
    commands, view = observable_commands(task)
    assert "talk_to:neave" in commands  # Clarification stays possible.
    handoff = reflex_action(task, commands, view)
    assert handoff.tool == "give" and handoff.source == "reflex"

    act("give", object=object_id, person="neave")
    assert handoffs() == []


@pytest.mark.parametrize("invalid_field,value,error_code", [
    ("message", None, "string_type"),
    ("summary", "x" * 301, "string_too_long"),
    ("command_id", "give:charger:dad", "literal_error"),
])
async def test_handoff_recovers_from_invalid_model_output(invalid_field, value, error_code):
    # Uses LEGACY_WORLD so the scripted tool sequence is stable. The test
    # exercises the repair mechanism for invalid Explorer output — the specific
    # world entities are incidental. The corrupted reply lands on a navigation
    # decision, which is the kind of step that still requires the model.
    invalid = {"command_id": "move_to:hall", "message": "", "summary": "Head back toward Neave."}
    invalid[invalid_field] = value
    fake = ScriptedLLM([
        {"action": "delegate", "summary": "Find and deliver.", "task": "Find charger and recipient, then deliver."},
        tool("move_to", room="study"),
        tool("talk_to", person="dad", message="Who needs the charger?"),
        invalid, tool("move_to", room="hall"),
        {"approved": True, "summary": "Held charger and observed recipient."},
        complete,
        {"approved": True, "summary": "Successful transfer evidenced."},
    ])
    engine, bus = WorldEngine(LEGACY_WORLD), EventBus()
    mgr = TaskManager(fake, WorldTools(engine, bus), bus, Settings(max_explorer_actions=20))
    task = mgr.start("Find out who needs the charger and deliver it.")
    await mgr.runner
    assert task.status == "completed", task.summary
    assert engine.snapshot()["objects"]["charger"]["location"] == {"kind": "person", "id": "neave"}
    transfers = [a for a in task.action_history if a["tool"] == "give"]
    assert len(transfers) == 1 and transfers[0]["success"]
    assert task.critic_count == 2
    assert any(f"{invalid_field}: {error_code}" in messages[-1]["content"]
               for messages, _ in fake.calls)


async def test_goal_planner_repairs_omitted_delivery_condition():
    class IncompletePlanner:
        async def generate(self, messages, response_schema):
            return '{"summary":"Find the charger and recipient.","conditions":[' \
                   '{"kind":"identify_recipient","object":"charger"}]}'

    plan = await Coordinator(IncompletePlanner()).define_goal(
        "Find out who needs the charger and deliver it.")
    assert [condition.kind for condition in plan.conditions] == ["identify_recipient", "deliver"]
    assert plan.conditions[-1].object == "charger"


async def test_goal_planner_cannot_invent_unknown_recipient():
    class InventedRecipientPlanner:
        async def generate(self, messages, response_schema):
            return '{"summary":"Deliver charger.","conditions":[' \
                   '{"kind":"deliver","object":"charger","person":"neave"}]}'

    plan = await Coordinator(InventedRecipientPlanner()).define_goal(
        "Find out who needs the charger and deliver it.")
    assert len(plan.conditions) == 1
    assert plan.conditions[0].kind == "deliver"
    assert plan.conditions[0].object == "charger"
    assert plan.conditions[0].person == ""


async def test_delivery_end_to_end_with_mock_model():
    # Only genuinely ambiguous steps are scripted. Scanning an unseen room,
    # acquiring the visible required object, walking a committed route and
    # handing over to a present recipient are all executed without inference.
    replies = [{"action": "delegate", "summary": "Locate the charger and recipient.", "task": "Search study."},
               tool("move_to", room="study"),
               tool("talk_to", person="dad", message="Who needs the charger?"),
               tool("move_to", room="hall"),
               {"approved": True, "summary": "Neave is the confirmed recipient."},
               complete,
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
    assert [a["tool"] for a in task.action_history] == [
        "get_status", "get_map", "look", "move_to", "look", "pick_up",
        "talk_to", "move_to", "move_to", "look", "give"]
    # Nine embodied actions cost three Explorer inferences, not nine.
    assert task.metrics.explorer_llm_calls == 3
    assert task.metrics.reflex_actions == 5 and task.metrics.route_actions == 1
    assert task.metrics.coordinator_handoffs == 1
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


async def test_critic_rejection_returns_to_coordinator():
    fake = ScriptedLLM([
        {"action": "delegate", "summary": "Observe.", "task": "Find charger."},
        tool("move_to", room="study"), complete,
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
        tool("move_to", room="bedroom"),
        tool("talk_to", person="neave", message=""),
        tool("talk_to", person="neave", message="Dinner is ready."),
        complete,
        {"approved": True, "summary": "Message delivery was observed."}])
    mgr, engine, _ = manager(fake)
    task = mgr.start("Find Neave and tell him dinner is ready.")
    await mgr.runner
    assert task.status == "completed"
    assert all(entry["success"] for entry in task.action_history)
    assert sum(entry["tool"] == "talk_to" for entry in task.action_history) == 1
    assert any("actual nonblank words" in messages[-1]["content"] for messages, _ in fake.calls)
    assert mission_satisfied(2, task, engine.snapshot())


@pytest.mark.parametrize("number,actions", [
    (1, [tool("move_to", room="study")]),
    (2, [tool("move_to", room="bedroom"),
         tool("talk_to", person="neave", message="Dinner is ready.")]),
    (4, [tool("move_to", room="kitchen")]),
])
async def test_other_evaluation_missions(number, actions):
    from app.evaluate import MISSIONS
    fake = ScriptedLLM([
        {"action": "delegate", "summary": "Carry out mission.", "task": MISSIONS[number]},
        *actions, complete,
        {"approved": True, "summary": "Evidence matches the goal."}])
    mgr, engine, _ = manager(fake)
    task = mgr.start(MISSIONS[number])
    await mgr.runner
    assert task.status == "completed"
    assert mission_satisfied(number, task, engine.snapshot())
    # Arrival scans are reflexive, so each mission needs one navigation inference.
    assert task.metrics.explorer_llm_calls == len(actions)


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
    # The first delegation scans the start room reflexively, so its report rests on
    # real evidence. The second reports having found the charger without acting at
    # all: that claim must be challenged and recovered from inside the delegation.
    fake = ScriptedLLM([
        {"action": "delegate", "summary": "Search.", "task": "Find charger."},
        {"action": "report", "summary": "Start room scanned."},
        {"action": "delegate", "summary": "Keep searching.", "task": "Find charger."},
        {"action": "report", "summary": "Charger found."},
        {"approved": False, "summary": "No observation supports finding it.", "suggestion": "Search the rooms."},
        {"action": "report", "summary": "I will search now."},
        tool("move_to", room="study"),
        complete, {"approved": True, "summary": "Observed charger."}])
    mgr, _, bus = manager(fake)
    task = mgr.start("Find charger.")
    await mgr.runner
    assert task.status == "completed"
    assert task.critic_count == 2
    # Recovery happened inside the second delegation: no extra planning cycle.
    assert task.cycle_count == 3
    assert any("Report rejected" in message for message in task.action_feedback)
    assert not any(e["type"] == "agent_message" and e["data"].get("summary") == "Charger found."
                   for e in bus.history)


async def test_repeated_rejected_reports_stop_across_delegations():
    delegate = {"action": "delegate", "summary": "Search.", "task": "Find requested object."}
    report = {"action": "report", "summary": "Cannot find it."}
    fake = ScriptedLLM([delegate, delegate, delegate, report,
        {"approved": False, "summary": "No supporting observations."},
        delegate, report, delegate, report])
    mgr, _, _ = manager(fake, max_explorer_actions=1)
    task = mgr.start("Find charger.")
    await mgr.runner
    assert task.status == "failed"
    assert "unchanged decision loop" in task.summary
    assert task.critic_count == 1
    assert task.consecutive_report_rejections == 3
    # Bootstrap plus the one reflex scan of the start room; recovery chooses no tool.
    assert task.tool_count == 3
    assert TaskContext(goal="Another task").consecutive_report_rejections == 0
    mgr.call_tool(task, "look", {})
    assert task.consecutive_report_rejections == 0


async def test_report_claims_do_not_override_observed_search_coverage():
    import json
    from app.agents.explorer import Explorer
    tools = WorldTools(WorldEngine(LEGACY_WORLD), EventBus())
    task = TaskContext(goal="Find requested object.")
    for name in ["get_status", "get_map", "look"]:
        task.record(name, {}, tools.execute(name, {}))
    task.explorer_reports.append("All rooms were searched.")
    task.critic_feedback.append({"kind": "delegation_report", "approved": False,
                                "summary": "All rooms were searched.",
                                "observed_action_count": len(task.action_history)})
    fake = ScriptedLLM([tool("move_to", room="study")])
    await Explorer(fake, tools.schemas()).decide(task, "Continue searching.")
    payload = json.loads(fake.calls[0][0][1]["content"])
    assert "All rooms were searched" not in str(payload)
    assert payload["floor_plan"]["rooms"]["hall"]["observed"]
    assert not payload["floor_plan"]["rooms"]["study"]["observed"]
    assert "study" in payload["known_but_unobserved_rooms"]
    assert task.explorer_reports and task.critic_feedback  # Audit remains intact.


async def test_unobserved_room_report_forces_scan_before_coordinator_replan():
    # Entering a room now scans it reflexively, so a report can never be raised
    # from an unperceived room in the first place. The guard stays as a backstop.
    fake = ScriptedLLM([
        {"action": "delegate", "summary": "Search the house.", "task": "Find the charger and its recipient."},
        tool("move_to", room="kitchen"),
        tool("talk_to", person="mom", message="Where are the keys?"),
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


async def test_distinct_followup_can_be_spoken_without_critic_debate():
    # Two questions to the same person that pursue different information are both
    # allowed: only rephrasings of an answered question are rejected.
    fake = ScriptedLLM([
        {"action": "delegate", "summary": "Search.", "task": "Find Neave and notify him."},
        tool("move_to", room="kitchen"),
        tool("talk_to", person="mom", message="Where is Neave?"),
        tool("talk_to", person="mom", message="When did you last see him?"),
        tool("move_to", room="hall"),
        tool("talk_to", person="neave", message="Dinner is ready."),
        complete, {"approved": True, "summary": "Neave was notified."}])
    mgr, _, bus = manager(fake)
    task = mgr.start("Find Neave and tell him dinner is ready.")
    await mgr.runner
    assert task.status == "completed", task.summary
    assert task.critic_count == 1
    assert sum(a["tool"] == "talk_to" for a in task.action_history) == 3
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
    talk = task.action_history[-1]
    task.memory.remember_conversation(
        "neave", "thread_1", "Determine who requested the item", True,
        talk["observation"]["message"], talk["observation"]["response"], talk["evidence_id"],
    )

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
    talk = task.action_history[-1]
    task.memory.remember_conversation(
        "neave", "thread_1", "Determine who requested the item", True,
        talk["observation"]["message"], talk["observation"]["response"], talk["evidence_id"],
    )

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


async def test_delivery_search_rejects_rephrased_recipient_question_and_continues():
    # Each scripted move only names the next hop; intermediate rooms, arrival
    # scans, the pickup and the final handoff are all executed without inference.
    fake = ScriptedLLM([
        {"action": "delegate", "summary": "Find item and recipient.", "task": "Find charger and who needs it."},
        tool("move_to", room="bedroom"),
        tool("talk_to", person="neave", message="Who needs the charger?"),
        tool("talk_to", person="neave", message="WHO needs the charger?!"),
        tool("move_to", room="hall"), tool("move_to", room="hall"), tool("move_to", room="hall"),
        {"approved": True, "summary": "Held and nearby."}, complete,
        {"approved": True, "summary": "Delivery evidenced."},
    ])
    mgr, engine, _ = manager(fake, max_explorer_actions=20)
    task = mgr.start("Find who needs the charger and deliver it.")
    await mgr.runner
    assert task.status == "completed", task.summary
    assert engine.snapshot()["objects"]["charger"]["location"] == {"kind": "person", "id": "neave"}
    assert sum(a["tool"] == "talk_to" for a in task.action_history) == 1
    assert task.critic_count == 2  # give and completion; pickup is deterministic
    assert any("Conversation move rejected" in message for message in task.action_feedback)
    # The rephrasing was caught locally, so it cost no conversation inference.
    assert task.metrics.conversation_llm_calls == 1
    assert task.metrics.reflex_actions + task.metrics.route_actions > task.metrics.explorer_llm_calls


async def test_conversation_remains_bounded_by_task_limits():
    fake = ScriptedLLM([
        {"action": "delegate", "summary": "Find recipient.", "task": "Ask who needs charger."},
        tool("look"), tool("move_to", room="bedroom"), tool("look"),
        tool("talk_to", person="neave", message="Who needs the charger?"),
        tool("talk_to", person="neave", message="Who needs the charger?"),
        tool("talk_to", person="neave", message="Who needs the charger?"),
    ])
    mgr, _, _ = manager(fake, max_explorer_actions=5, max_coordinator_cycles=1)
    task = mgr.start("Find who needs the charger and deliver it.")
    await mgr.runner
    assert task.status == "failed"
    assert "Coordinator cycles" in task.summary
    assert task.critic_count == 0
    assert sum(a["tool"] == "talk_to" for a in task.action_history) == 1
    assert sum("Conversation move rejected" in message for message in task.action_feedback) == 1


async def test_conversation_recovery_survives_redelegation_and_preserves_followups():
    def leave(context):
        assert "talk_to:mom" not in context["commands"]
        assert "move_to:hall" in context["commands"]
        return tool("move_to", room="hall")

    fake = ScriptedLLM([
        {"action": "delegate", "summary": "Search.", "task": "Find recipient."},
        tool("move_to", room="kitchen"),
        tool("talk_to", person="mom", message="Do you need the charger?"),
        tool("talk_to", person="mom", message="Do you need the charger?"),
        leave, tool("move_to", room="hall"),
        tool("talk_to", person="dad", message="Who needs the charger?"),
        {"action": "report", "summary": "Recipient identified."},
        {"action": "fail", "summary": "End test."},
    ])
    mgr, _, _ = manager(fake, max_explorer_actions=20)
    task = mgr.start("Find who needs the charger and deliver it.")
    await mgr.runner
    assert task.summary == "End test."
    talks = [a for a in task.action_history if a["tool"] == "talk_to"]
    assert len(talks) == 2
    mgr.call_tool(task, "move_to", {"room": "hall"})
    mgr.call_tool(task, "move_to", {"room": "kitchen"})
    assert "talk_to:mom" in observable_commands(task)[0]
    assert not task.conversation_rejections
    assert not TaskContext(goal="New task").conversation_rejections


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
    assert "message" in fake.calls[0][1]["anyOf"][0]["required"]
    assert fake.calls[0][1]["anyOf"][0]["properties"]["message"]["minLength"] == 1
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
