import json

import pytest
from pydantic import ValidationError

from app.agents.commands import observable_commands
from app.agents.conversation import classify_conversation_move
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


@pytest.mark.parametrize("item,recipient", [("sensor_pack", "bea"), ("parcel", "omar")])
async def test_delivery_generalizes_after_unhelpful_source(tmp_path, item, recipient):
    path, data = alternate_world(tmp_path)
    data = json.loads(json.dumps(data).replace("medicine", item).replace("alice", recipient).replace("Alice", recipient))
    data["people"]["witness"] = {"id": "witness", "name": "Witness", "room": "entry"}
    path.write_text(json.dumps(data))

    class ScenarioLLM(ScriptedLLM):
        async def generate(self, messages, response_schema):
            if response_schema.get("title") == "ConversationMeaning":
                talk = json.loads(messages[1]["content"])["conversation"]
                return json.dumps({"needs": [{"object": item, "person": recipient, "quote": talk["response"]}]
                                  if talk["person"] == recipient else [], "thread_resolved": True})
            return await super().generate(messages, response_schema)

    model = ScenarioLLM([
        {"action": "delegate", "summary": "Explore and deliver.", "task": "Find the requested item and its recipient."},
        tool("look"), tool("talk_to", person="witness", message=f"Who needs the {item}?"),
        tool("talk_to", person="witness", message=f"Who needs the {item}?"),
        tool("move_to", room="workshop"), tool("look"),
        tool("talk_to", person=recipient, message=f"Do you need the {item}?"),
        tool("pick_up", object=item), tool("give", object=item, person=recipient),
        {"approved": True, "summary": "Observed item held and recipient present."},
        {"action": "report", "summary": "Delivered."}, complete,
        {"approved": True, "summary": "Transfer evidenced."},
    ], conditions=[{"kind": "deliver", "object": item}])
    engine, bus = WorldEngine(path), EventBus()
    manager = TaskManager(model, WorldTools(engine, bus), bus, Settings(max_explorer_actions=20))
    task = manager.start(f"Find who needs the {item} and deliver it.")
    await manager.runner
    assert task.status == "completed", task.summary
    assert engine.snapshot()["objects"][item]["location"] == {"kind": "person", "id": recipient}
    assert sum(a["tool"] == "talk_to" and a["arguments"]["person"] == "witness" for a in task.action_history) == 1


def test_sketch_topology_and_observation_boundary():
    engine = WorldEngine()
    plan = engine.get_map()
    assert set(plan["rooms"]) == {"hall", "kitchen", "bedroom_corridor", "master_bedroom",
                                  "second_bedroom", "entrance", "office"}
    assert plan["rooms"]["office"]["connections"] == ["entrance"]
    assert "master_bedroom" not in plan["rooms"]["hall"]["connections"]
    assert len(plan["doors"]) == 6
    encoded = json.dumps(plan)
    for secret in ["neave", "minu", "moira", "paul", "medicine", "dialogue", "people", "objects", "messages"]:
        assert secret not in encoded
    hall_view = engine.look()
    assert hall_view["people"][0]["id"] == "moira"
    assert "dialogue" not in json.dumps(hall_view)
    assert hall_view["objects"] == []
    assert "medicine" not in json.dumps(engine.get_status())
    engine.move_to("entrance")
    engine.move_to("office")
    response = engine.talk_to("paul", "Do you need the medicine?")
    assert "medicine" in response["response"]
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


def test_dialogue_matches_natural_words_for_multiword_identifiers(tmp_path):
    path, data = alternate_world(tmp_path)
    data["people"]["alice"]["dialogue"][0]["topics"] = ["sensor_pack"]
    data["people"]["alice"]["dialogue"][0]["response"] = "I need the sensor pack."
    path.write_text(json.dumps(data))
    engine = WorldEngine(path)
    engine.move_to("workshop")
    assert engine.talk_to("alice", "Who needs the sensor pack?")["response"] == "I need the sensor pack."
    assert engine.talk_to("alice", "Who needs the sensor_pack?")["response"] == "I need the sensor pack."


@pytest.mark.parametrize("mutate", [
    lambda w: w["robot"].update(room="missing"),
    lambda w: w["people"]["neave"].update(room="missing"),
    lambda w: w["objects"]["medicine"]["location"].update(id="missing"),
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
    first_input = json.loads(fake.calls[0][0][1]["content"])
    assert first_input["task_memory"]["people"] == {}
    assert first_input["task_memory"]["objects"] == {}
    assert "dialogue" not in json.dumps(first_input)


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


def test_api_uses_configured_json_and_failed_reset_preserves_world(tmp_path):
    from fastapi.testclient import TestClient
    from app.main import create_app
    from tests.fakes import WaitingLLM
    path, _ = alternate_world(tmp_path)
    with TestClient(create_app(client=WaitingLLM(), settings=Settings(world_file=path))) as client:
        initial = client.get("/api/world").json()["world"]
        assert initial["name"] == "Unfamiliar workshop"
        path.write_text('{"invalid": true}')
        response = client.post("/api/world/reset")
        assert response.status_code == 422
        assert client.get("/api/world").json()["world"] == initial


def test_can_scan_again_after_returning_to_a_known_room(tmp_path):
    path, _ = alternate_world(tmp_path)
    tools = WorldTools(WorldEngine(path), EventBus())
    task = TaskContext(goal="Explore")
    for name, args in [("get_status", {}), ("look", {}),
                       ("move_to", {"room": "workshop"}), ("move_to", {"room": "entry"})]:
        task.record(name, args, tools.execute(name, args))
    assert "look" in observable_commands(task)[0]


def test_task_memory_preserves_medicine_plan_after_recent_history_eviction():
    engine, bus = WorldEngine(), EventBus()
    tools = WorldTools(engine, bus)
    task = TaskContext(
        goal="Find who needs the medicine and deliver it.",
        conditions=[GoalCondition(kind="deliver", object="medicine")],
    )

    def act(name, **args):
        result = tools.execute(name, args)
        task.record(name, args, result)
        return result

    act("get_status")
    act("look")
    act("move_to", room="entrance")
    act("move_to", room="office")
    act("look")
    conversation = act("talk_to", person="paul", message="Do you need the medicine?")
    task.remember_meaning(conversation["evidence_id"], ConversationMeaning(needs=[
        SpokenNeed(object="medicine", person="paul", quote="I need the medicine, please.")
    ]))
    task.remember_conversation_thread(
        "paul", "thread_1", "Determine who requested the item", True,
        conversation["observation"]["message"], conversation["observation"]["response"],
        conversation["evidence_id"],
    )
    act("move_to", room="entrance")
    act("move_to", room="hall")
    act("move_to", room="bedroom_corridor")
    act("move_to", room="second_bedroom")
    act("look")
    for room in ["bedroom_corridor", "hall", "kitchen", "hall", "entrance", "hall"]:
        act("move_to", room=room)

    payload = task.compact()
    memory = payload["task_memory"]
    assert len(payload["recent_actions"]) == 8
    assert memory["objects"]["medicine"]["location"] == {"kind": "room", "id": "second_bedroom"}
    assert memory["people"]["paul"]["location"] == "office"
    assert memory["people"]["minu"]["location"] == "second_bedroom"
    assert memory["people"]["paul"]["conversation_threads"]["thread_1"]["turns"][-1]["response"] == \
        "I need the medicine, please."
    assert memory["reported_needs_unverified"][0]["person"] == "paul"
    assert "office" in memory["visited_rooms"]


async def test_semantic_conversation_threads_block_rephrasing_but_allow_new_purposes():
    class SemanticClassifier:
        async def generate(self, messages, response_schema):
            payload = json.loads(messages[1]["content"])
            assert response_schema.get("title") == "ConversationMove"
            message = payload["proposed_message"].casefold()
            if "there" in message:
                return json.dumps({
                    "existing_thread_id": "thread_1",
                    "thread_summary": "Determine arrival time",
                    "advances_thread": False,
                })
            return json.dumps({
                "existing_thread_id": "",
                "thread_summary": "Determine which entrance to use",
                "advances_thread": True,
            })

    task = TaskContext(goal="Coordinate a visit.")
    task.memory.set_person("dad", "office", "person-evidence", "Dad")
    task.remember_conversation_thread(
        "dad", "thread_1", "Determine arrival time", True,
        "What time should I arrive?", "Come at six.", "conversation-evidence",
    )

    repeated = await classify_conversation_move(
        SemanticClassifier(), task, "dad", "When do you want me there?"
    )
    assert repeated.existing_thread_id == "thread_1"
    assert repeated.advances_thread is False

    new_topic = await classify_conversation_move(
        SemanticClassifier(), task, "dad", "Which entrance should I use?"
    )
    assert new_topic.existing_thread_id == ""
    assert new_topic.advances_thread is True


async def test_first_conversation_cannot_be_rejected_as_repeated():
    class NoClassificationExpected:
        async def generate(self, messages, response_schema):
            raise AssertionError("There is no prior conversation to classify a repeat against.")

    task = TaskContext(goal="Ask the local person for directions.")
    task.memory.set_person("witness", "entry", "observed", "Witness")
    move = await classify_conversation_move(NoClassificationExpected(), task, "witness", "Where is the exit?")
    assert move.advances_thread
    assert move.existing_thread_id == ""
    assert task.memory.people["witness"].conversation_threads == {}
