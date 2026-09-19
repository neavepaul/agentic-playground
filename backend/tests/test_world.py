from pathlib import Path
LEGACY_WORLD = Path(__file__).parent / "fixtures" / "legacy_house.json"

import pytest

from app.events.bus import EventBus
from app.world.engine import WorldEngine
from app.world.tools import TOOL_REGISTRY, WorldTools


@pytest.fixture
def setup():
    engine, bus = WorldEngine(LEGACY_WORLD), EventBus()
    return engine, bus, WorldTools(engine, bus)


def test_movement_and_isolation(setup):
    engine, bus, tools = setup
    assert tools.execute("move_to", {"room": "kitchen"})["success"]
    assert not tools.execute("move_to", {"room": "study"})["success"]
    observation = tools.execute("look", {})["observation"]
    assert observation["people"] == [{"id": "mom", "name": "Mom"}]
    assert [o["id"] for o in observation["objects"]] == ["keys"]
    assert "charger" not in str(observation)
    assert "people" not in tools.execute("get_status", {})["observation"]
    assert any(e["type"] == "tool_failed" for e in bus.history)


def test_pickup_give_drop_reset(setup):
    engine, _, tools = setup
    assert not tools.execute("pick_up", {"object": "charger"})["success"]
    tools.execute("move_to", {"room": "study"})
    assert tools.execute("pick_up", {"object": "charger"})["success"]
    assert not tools.execute("pick_up", {"object": "charger"})["success"]
    assert not tools.execute("give", {"object": "charger", "person": "neave"})["success"]
    assert engine.snapshot()["objects"]["charger"]["location"]["kind"] == "robot"
    assert tools.execute("drop", {"object": "charger"})["success"]
    tools.execute("pick_up", {"object": "charger"})
    tools.execute("move_to", {"room": "hall"})
    tools.execute("move_to", {"room": "bedroom"})
    assert tools.execute("give", {"object": "charger", "person": "neave"})["success"]
    assert engine.snapshot()["objects"]["charger"]["location"] == {"kind": "person", "id": "neave"}
    assert engine.snapshot()["robot"]["inventory"] == []
    engine.reset()
    assert engine.snapshot() == WorldEngine(LEGACY_WORLD).snapshot()


def test_registry_validation_and_conversation(setup):
    _, _, tools = setup
    assert set(TOOL_REGISTRY) == {"look", "get_map", "get_status", "move_to", "talk_to", "pick_up", "drop", "give"}
    for tool, args in [("teleport", {}), ("move_to", {}), ("look", {"secret": True}),
                       ("move_to", {"room": 4}), ("give", {"object": "keys", "person": "mom"})]:
        assert not tools.execute(tool, args)["success"]
    assert not tools.execute("talk_to", {"person": "dad", "message": "charger?"})["success"]
    tools.execute("move_to", {"room": "study"})
    assert "Neave" in tools.execute("talk_to", {"person": "dad", "message": "Who needs the charger?"})["observation"]["response"]


def test_nonportable_and_failed_actions_leave_world_unchanged(setup):
    engine, _, tools = setup
    tools.execute("move_to", {"room": "bedroom"})
    engine._world.objects["laptop"].portable = False
    before = engine.snapshot()
    for tool, args in [("pick_up", {"object": "laptop"}),
                       ("drop", {"object": "keys"}),
                       ("give", {"object": "keys", "person": "neave"}),
                       ("talk_to", {"person": "neave", "message": "   "})]:
        assert not tools.execute(tool, args)["success"]
        assert engine.snapshot() == before


def test_snapshots_cannot_mutate_authoritative_world(setup):
    engine, _, tools = setup
    view = engine.snapshot()
    view["objects"]["charger"]["location"]["id"] = "hall"
    obs = tools.execute("look", {})["observation"]
    obs["connections"].clear()
    assert engine.snapshot()["objects"]["charger"]["location"]["id"] == "study"
    assert tools.execute("move_to", {"room": "study"})["success"]
