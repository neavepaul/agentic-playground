from app.agents.commands import observable_commands
from app.events.bus import EventBus
from app.tasks.models import TaskContext
from app.world.engine import WorldEngine
from app.world.tools import WorldTools


def test_command_choices_use_only_observations_and_reconcile_ownership():
    tools = WorldTools(WorldEngine(), EventBus())
    task = TaskContext(goal="Arbitrary goal, deliberately not examined by command builder.")

    def act(name, **args):
        task.record(name, args, tools.execute(name, args))

    act("get_status")
    choices, view = observable_commands(task)
    assert set(choices) == {"look", "report"}
    assert view is None and "charger" not in str(choices)
    act("look")
    choices, _ = observable_commands(task)
    assert set(choices) == {"report", "move_to:kitchen", "move_to:study", "move_to:bedroom"}
    act("move_to", room="study")
    assert set(observable_commands(task)[0]) == {"look", "report"}
    act("look")
    choices, _ = observable_commands(task)
    assert "pick_up:charger" in choices and "talk_to:dad" in choices
    assert "move_to:bedroom" not in choices and "give:charger:dad" not in choices
    act("pick_up", object="charger")
    choices, view = observable_commands(task)
    assert "pick_up:charger" not in choices and "give:charger:dad" in choices
    assert not view["objects"]
    act("drop", object="charger")
    choices, view = observable_commands(task)
    assert "pick_up:charger" in choices and "give:charger:dad" not in choices
    assert len(view["objects"]) == 1
    act("pick_up", object="charger")
    act("move_to", room="hall")
    act("move_to", room="bedroom")
    act("look")
    act("give", object="charger", person="neave")
    choices, view = observable_commands(task)
    assert "give:charger:neave" not in choices and "pick_up:charger" not in choices
    assert {"object": "charger", "person": "neave"} in view["held_objects"]
    act("move_to", room="hall")
    act("move_to", room="study")
    assert "pick_up:charger" not in observable_commands(task)[0]
