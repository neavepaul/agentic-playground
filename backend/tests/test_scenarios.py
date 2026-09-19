import json
from copy import deepcopy

import pytest
from pydantic import ValidationError

from app.agents.commands import observable_commands
from app.agents.conversation import interpret_conversation
from app.agents.schemas import ConversationMeaning, GoalCondition, SpokenNeed
from app.config import Settings
from app.events.bus import EventBus
from app.tasks.manager import TaskManager
from app.tasks.models import TaskContext
from app.world.engine import WorldEngine
from app.world.initial_state import DEFAULT_WORLD
from app.world.models import World
from app.world.tools import WorldTools
from tests.fakes import ScriptedLLM, complete, tool


def alternate_world(tmp_path):
    data = {
        "name": "Unfamiliar workshop", "robot": {"room": "entry"},
        "rooms": {
            "entry": {"id": "entry", "name": "Entry", "connections": ["workshop"]},
            "workshop": {"id": "workshop", "name": "Workshop", "connections": ["entry"]}},
        "people": {"alice": {"id": "alice", "name": "Alice", "room": "workshop",
            "dialogue": [{"topics": ["medicine"], "response": "I need the medicine, please."}]}},
        "objects": {"medicine": {"id": "medicine", "name": "Medicine",
            "location": {"kind": "room", "id": "workshop"}}}}
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(data))
    return path, data


def test_sketch_topology_and_observation_boundary():
    engine = WorldEngine()
    plan = engine.get_map()
    assert set(plan["rooms"]) == {"hall", "kitchen", "bedroom_corridor", "master_bedroom",
                                  "second_bedroom", "entrance", "office"}
    assert plan["rooms"]["office"]["connections"] == ["entrance"]
    assert "master_bedroom" not in plan["rooms"]["hall"]["connections"]
    assert len(plan["doors"]) == 6
    encoded = json.dumps(plan)
    for secret in ["neave", "charger", "dialogue", "people", "objects", "messages"]:
        assert secret not in encoded
    assert engine.look()["people"] == []
    assert engine.look()["objects"] == []
    assert "charger" not in json.dumps(engine.get_status())
    engine.move_to("entrance")
    engine.move_to("office")
    response = engine.talk_to("dad", "Who needs the charger?")
    assert "Neave" in response["response"]
    assert "facts" not in response and "dialogue" not in response


def test_json_reset_reloads_and_instances_do_not_share_state(tmp_path):
    path, data = alternate_world(tmp_path)
    a, b = WorldEngine(path), WorldEngine(path)
    a.move_to("workshop")
    a.pick_up("medicine")
    assert b.snapshot()["objects"]["medicine"]["location"]["kind"] == "room"
    data["objects"]["medicine"]["location"]["id"] = "entry"
    path.write_text(json.dumps(data))
    a.reset()
    assert a.look()["objects"][0]["id"] == "medicine"


@pytest.mark.parametrize("mutate", [
    lambda w: w["robot"].update(room="missing"),
    lambda w: w["people"]["neave"].update(room="missing"),
    lambda w: w["objects"]["charger"]["location"].update(id="missing"),
    lambda w: w["rooms"]["office"].update(connections=["hall"]),
    lambda w: w["rooms"]["hall"].update(id="different"),
    lambda w: w["doors"][0].update(rooms=["office", "master_bedroom"]),
])
def test_invalid_world_references_fail_at_load(mutate):
    data = json.loads(DEFAULT_WORLD.read_text())
    mutate(data)
    with pytest.raises(ValidationError):
        World.model_validate(data)


async def test_new_names_delivery_and_followup_notification(tmp_path):
    path, _ = alternate_world(tmp_path)
    class WorkshopLLM(ScriptedLLM):
        async def generate(self, messages, response_schema):
            if response_schema.get("title") == "ConversationMeaning":
                payload = json.loads(messages[1]["content"])
                response = payload["conversation"]["response"]
                return json.dumps({"needs": [{"object": "medicine", "person": "alice", "quote": response}]
                    if response == "I need the medicine, please." else []})
            return await super().generate(messages, response_schema)
    fake = WorkshopLLM([
        {"action": "delegate", "summary": "Explore and help.", "task": "Find who needs medicine and deliver it."},
        tool("look"), tool("move_to", room="workshop"), tool("look"),
        tool("talk_to", person="alice", message="Who needs the medicine?"),
        tool("pick_up", object="medicine"), {"approved": True, "summary": "Requested item."},
        tool("give", object="medicine", person="alice"), {"approved": True, "summary": "Spoken request."},
        tool("talk_to", person="alice", message="The taxi is here."),
        {"action": "report", "summary": "Delivered and notified."}, complete,
        {"approved": True, "summary": "Evidence supports both outcomes."},
    ], conditions=[{"kind": "deliver", "object": "medicine"},
                   {"kind": "notify", "person": "alice", "message": "The taxi is here"}])
    engine, bus = WorldEngine(path), EventBus()
    manager = TaskManager(fake, WorldTools(engine, bus), bus, Settings())
    task = manager.start("Find who needs medicine, deliver it, and tell Alice the taxi is here.")
    await manager.runner
    assert task.status == "completed", task.summary
    assert engine.snapshot()["objects"]["medicine"]["location"] == {"kind": "person", "id": "alice"}
    assert len(engine.snapshot()["people"]["alice"]["messages"]) == 2
    assert task.interpreted_needs[0]["quote"] == "I need the medicine, please."
    # No remote occupants or dialogue scripts in the first model decision.
    first_input = fake.calls[0][0][1]["content"]
    assert '"dialogue"' not in first_input and '"people"' not in first_input


def test_transcript_claims_require_real_source_and_exact_quote(tmp_path):
    path, _ = alternate_world(tmp_path)
    tools = WorldTools(WorldEngine(path), EventBus())
    task = TaskContext(goal="Find medicine", conditions=[GoalCondition(kind="deliver", object="medicine")])
    for name, args in [("get_status", {}), ("move_to", {"room": "workshop"}), ("look", {}),
                       ("talk_to", {"person": "alice", "message": "Hello"})]:
        result = tools.execute(name, args)
        task.record(name, args, result)
    fabricated = ConversationMeaning(needs=[SpokenNeed(object="medicine", person="alice", quote="I need medicine")])
    task.remember_meaning(result["evidence_id"], fabricated)
    task.remember_meaning("invented", fabricated)
    assert not task.interpreted_needs
    assert "give:medicine:alice" not in observable_commands(task)[0]
