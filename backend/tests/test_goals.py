from pathlib import Path
LEGACY_WORLD = Path(__file__).parent / "fixtures" / "legacy_house.json"

from app.agents.schemas import GoalCondition, ConversationMeaning, SpokenNeed
from app.events.bus import EventBus
from app.tasks.goals import check_conditions
from app.tasks.models import TaskContext
from app.world.engine import WorldEngine
from app.world.tools import WorldTools


def setup(conditions):
    tools = WorldTools(WorldEngine(LEGACY_WORLD), EventBus())
    task = TaskContext(goal="Test conditions", conditions=conditions)

    def act(name, **args):
        result = tools.execute(name, args)
        assert result["success"]
        task.record(name, args, result)

    act("get_status")
    return task, act


def test_laptop_delivery_cannot_satisfy_charger_goal():
    task, act = setup([GoalCondition(kind="deliver", object="charger")])
    act("move_to", room="bedroom")
    act("look")
    act("talk_to", person="neave", message="Who needs the charger?")
    source = task.action_history[-1]
    task.remember_meaning(source["evidence_id"], ConversationMeaning(needs=[SpokenNeed(
        object="charger", person="neave", quote=source["observation"]["response"])]))
    act("pick_up", object="laptop")
    act("give", object="laptop", person="neave")
    assert not check_conditions(task)[0]["satisfied"]
    act("move_to", room="hall")
    act("move_to", room="study")
    act("look")
    act("pick_up", object="charger")
    act("move_to", room="hall")
    act("move_to", room="bedroom")
    act("give", object="charger", person="neave")
    assert check_conditions(task)[0]["satisfied"]


def test_broadcast_requires_all_rooms_and_each_person():
    task, act = setup([GoalCondition(kind="notify_everyone", message="We are leaving in ten minutes.")])
    act("look")
    assert not check_conditions(task)[0]["satisfied"]
    for room, person in [("kitchen", "mom"), ("bedroom", "neave"), ("study", "dad")]:
        act("move_to", room=room)
        act("look")
        assert not check_conditions(task)[0]["satisfied"]
        act("talk_to", person=person, message="We are leaving in ten minutes.")
        act("move_to", room="hall")
    assert check_conditions(task)[0]["satisfied"]


def test_hold_and_place_conditions_follow_latest_object_location():
    task, act = setup([GoalCondition(kind="hold_object", object="keys"),
                       GoalCondition(kind="place_object", object="keys", room="study")])
    act("move_to", room="kitchen")
    act("look")
    act("pick_up", object="keys")
    assert [c["satisfied"] for c in check_conditions(task)] == [True, False]
    act("move_to", room="hall")
    act("move_to", room="study")
    act("drop", object="keys")
    assert [c["satisfied"] for c in check_conditions(task)] == [False, True]
