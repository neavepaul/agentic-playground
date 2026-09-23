from pathlib import Path
LEGACY_WORLD = Path(__file__).parent / "fixtures" / "legacy_house.json"

from app.agents.explorer import observable_commands
from app.events.bus import EventBus
from app.tasks.models import TaskContext
from app.agents.schemas import GoalCondition
from app.world.engine import WorldEngine
from app.world.tools import WorldTools


def test_unobserved_exits_do_not_hide_route_to_known_object():
    tools = WorldTools(WorldEngine(LEGACY_WORLD), EventBus())
    task = TaskContext(goal="Find who needs charger and deliver it.",
                       conditions=[GoalCondition(kind="deliver", object="charger")])
    for name, args in [("get_status", {}), ("move_to", {"room": "study"}), ("look", {}),
                       ("move_to", {"room": "hall"}), ("look", {})]:
        task.record(name, args, tools.execute(name, args))
    commands, _ = observable_commands(task)
    assert "move_to:study" in commands  # Known object, despite unknown recipient.
    assert "move_to:kitchen" in commands  # Exploration remains a choice.


def test_command_choices_use_only_observations_and_reconcile_ownership():
    tools = WorldTools(WorldEngine(LEGACY_WORLD), EventBus())
    task = TaskContext(goal="Arbitrary goal text is not parsed by the command builder.",
                       conditions=[GoalCondition(kind="deliver", object="charger", person="neave")])

    def act(name, **args):
        task.record(name, args, tools.execute(name, args))

    act("get_status")
    choices, view = observable_commands(task)
    assert set(choices) == {"look"}
    assert view is None and "charger" not in str(choices)
    act("look")
    choices, _ = observable_commands(task)
    assert set(choices) == {"report", "move_to:kitchen", "move_to:study", "move_to:bedroom"}
    act("move_to", room="study")
    assert set(observable_commands(task)[0]) == {"look"}
    act("look")
    choices, _ = observable_commands(task)
    assert "pick_up:charger" in choices and "talk_to:dad" in choices
    assert "move_to:bedroom" not in choices and "give:charger:dad" not in choices
    act("pick_up", object="charger")
    choices, view = observable_commands(task)
    assert "pick_up:charger" not in choices and "give:charger:dad" not in choices
    assert not view["objects"]
    act("drop", object="charger")
    choices, view = observable_commands(task)
    assert "pick_up:charger" in choices and "give:charger:dad" not in choices
    assert len(view["objects"]) == 1
    act("pick_up", object="charger")
    act("move_to", room="hall")
    act("move_to", room="bedroom")
    act("look")
    assert "give:charger:neave" in observable_commands(task)[0]
    assert "pick_up:laptop" not in observable_commands(task)[0]
    act("give", object="charger", person="neave")
    choices, view = observable_commands(task)
    assert "give:charger:neave" not in choices and "pick_up:charger" not in choices
    assert {"object": "charger", "person": "neave"} in view["held_objects"]
    act("move_to", room="hall")
    act("move_to", room="study")
    assert "pick_up:charger" not in observable_commands(task)[0]


def test_local_investigation_does_not_hide_other_valid_actions():
    tools = WorldTools(WorldEngine(LEGACY_WORLD), EventBus())
    task = TaskContext(goal="Find who needs the charger and deliver it.",
                       conditions=[GoalCondition(kind="deliver", object="charger")])
    for name, args in [("get_status", {}), ("move_to", {"room": "kitchen"}), ("look", {})]:
        task.record(name, args, tools.execute(name, args))
    choices, _ = observable_commands(task)
    assert "talk_to:mom" in choices
    assert "report" in choices


def test_delivery_requires_observation_but_does_not_force_questions():
    tools = WorldTools(WorldEngine(LEGACY_WORLD), EventBus())
    task = TaskContext(goal="Find who needs the charger and deliver it.",
                       conditions=[GoalCondition(kind="deliver", object="charger")])

    task.record("get_status", {}, tools.execute("get_status", {}))
    choices, view = observable_commands(task)
    assert view is None and set(choices) == {"look"}

    for name, args in [("look", {}), ("move_to", {"room": "kitchen"}), ("look", {})]:
        task.record(name, args, tools.execute(name, args))
    choices, _ = observable_commands(task)
    assert set(choices) == {"report", "move_to:hall", "talk_to:mom"}


def test_pending_delivery_keeps_all_observed_exits_available():
    tools = WorldTools(WorldEngine(), EventBus())
    task = TaskContext(goal="Neave needs the charger. Find it and bring it to him.",
                       conditions=[GoalCondition(kind="deliver", object="charger")])

    def act(name, **args):
        result = tools.execute(name, args)
        assert result["success"]
        task.record(name, args, result)

    act("get_status")
    act("look")
    act("move_to", room="bedroom_corridor")
    act("look")

    choices, _ = observable_commands(task)
    assert "report" in choices
    assert "move_to:hall" in choices
    assert "move_to:master_bedroom" in choices
    assert "move_to:second_bedroom" in choices


def test_pick_up_of_required_object_is_marked_priority():
    """pick_up is flagged priority when it directly satisfies an unmet acquire_object prerequisite."""
    tools = WorldTools(WorldEngine(), EventBus())
    task = TaskContext(goal="Deliver medicine to paul.",
                       conditions=[GoalCondition(kind="deliver", object="medicine", person="paul")])

    def act(name, **args):
        result = tools.execute(name, args)
        assert result["success"], f"{name} failed: {result}"
        task.record(name, args, result)

    act("get_status")
    act("get_map")
    # Robot starts in hall; move to second_bedroom where medicine lives.
    act("look")
    act("move_to", room="bedroom_corridor")
    act("look")
    act("move_to", room="second_bedroom")
    act("look")

    choices, _ = observable_commands(task)
    # Medicine is visible on the floor. Since it's not held, acquire_object is unmet.
    assert "pick_up:medicine" in choices
    assert choices["pick_up:medicine"].get("priority") is True, "pick_up:medicine must be marked priority"


def test_pick_up_is_not_priority_when_already_held():
    """After picking up the required object, pick_up disappears from commands entirely."""
    tools = WorldTools(WorldEngine(), EventBus())
    task = TaskContext(goal="Deliver medicine to paul.",
                       conditions=[GoalCondition(kind="deliver", object="medicine", person="paul")])

    def act(name, **args):
        result = tools.execute(name, args)
        assert result["success"], f"{name} failed: {result}"
        task.record(name, args, result)

    act("get_status")
    act("get_map")
    act("look")
    act("move_to", room="bedroom_corridor")
    act("look")
    act("move_to", room="second_bedroom")
    act("look")
    act("pick_up", object="medicine")

    choices, _ = observable_commands(task)
    assert "pick_up:medicine" not in choices


def test_non_delivery_pick_up_is_not_marked_priority():
    """pick_up commands for irrelevant objects carry no priority flag."""
    tools = WorldTools(WorldEngine(), EventBus())
    # Goal only cares about keys; medicine visible in second_bedroom is irrelevant.
    task = TaskContext(goal="Get the keys.",
                       conditions=[GoalCondition(kind="hold_object", object="keys")])

    def act(name, **args):
        result = tools.execute(name, args)
        task.record(name, args, result)

    act("get_status")
    act("get_map")
    act("look")
    act("move_to", room="bedroom_corridor")
    act("look")
    act("move_to", room="second_bedroom")
    act("look")

    choices, _ = observable_commands(task)
    # Medicine is visible but irrelevant to the keys goal — not offered as a command at all.
    assert "pick_up:medicine" not in choices
