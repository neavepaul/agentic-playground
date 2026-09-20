from pathlib import Path
LEGACY_WORLD = Path(__file__).parent / "fixtures" / "legacy_house.json"

from app.agents.commands import observable_commands
from app.events.bus import EventBus
from app.tasks.models import TaskContext
from app.agents.schemas import GoalCondition
from app.world.engine import WorldEngine
from app.world.tools import WorldTools


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


def test_delivery_does_not_allow_report_when_local_investigation_is_available():
    tools = WorldTools(WorldEngine(LEGACY_WORLD), EventBus())
    task = TaskContext(goal="Find who needs the charger and deliver it.",
                       conditions=[GoalCondition(kind="deliver", object="charger")])
    for name, args in [("get_status", {}), ("move_to", {"room": "kitchen"}), ("look", {})]:
        task.record(name, args, tools.execute(name, args))
    choices, _ = observable_commands(task)
    assert "talk_to:mom" in choices
    assert "report" not in choices


def test_delivery_requires_look_then_local_questions_before_navigation():
    tools = WorldTools(WorldEngine(LEGACY_WORLD), EventBus())
    task = TaskContext(goal="Find who needs the charger and deliver it.",
                       conditions=[GoalCondition(kind="deliver", object="charger")])

    task.record("get_status", {}, tools.execute("get_status", {}))
    choices, view = observable_commands(task)
    assert view is None and set(choices) == {"look"}

    for name, args in [("look", {}), ("move_to", {"room": "kitchen"}), ("look", {})]:
        task.record(name, args, tools.execute(name, args))
    choices, _ = observable_commands(task)
    assert set(choices) == {"talk_to:mom"}
