"""Tests for deterministic pathfinding and navigation hints in TaskContext."""
from app.tasks.models import TaskContext
from app.agents.schemas import GoalCondition
from app.events.bus import EventBus
from app.world.engine import WorldEngine
from app.world.tools import WorldTools


# house.json topology:
#   hall <-> kitchen
#   hall <-> bedroom_corridor <-> master_bedroom
#                              <-> second_bedroom
#   hall <-> entrance <-> office


def _task_with_map() -> TaskContext:
    """Create a TaskContext bootstrapped with floor plan from house.json."""
    tools = WorldTools(WorldEngine(), EventBus())
    task = TaskContext(goal="test")
    for name, args in [("get_status", {}), ("get_map", {})]:
        task.record(name, args, tools.execute(name, args))
    return task, tools


class TestFindPath:
    def test_adjacent_room_single_hop(self):
        task, _ = _task_with_map()
        path = task._find_path("hall", "kitchen")
        assert path == ["kitchen"]

    def test_two_hop_path(self):
        task, _ = _task_with_map()
        path = task._find_path("hall", "master_bedroom")
        assert path == ["bedroom_corridor", "master_bedroom"]

    def test_three_hop_path(self):
        task, _ = _task_with_map()
        path = task._find_path("kitchen", "office")
        # kitchen -> hall -> entrance -> office
        assert path == ["hall", "entrance", "office"]

    def test_same_room_returns_empty(self):
        task, _ = _task_with_map()
        assert task._find_path("hall", "hall") == []

    def test_unknown_start_returns_empty(self):
        task, _ = _task_with_map()
        assert task._find_path("nowhere", "hall") == []

    def test_unknown_target_returns_empty(self):
        task, _ = _task_with_map()
        assert task._find_path("hall", "nowhere") == []

    def test_path_is_shortest(self):
        task, _ = _task_with_map()
        # second_bedroom is 3 hops from entrance:
        # entrance -> hall -> bedroom_corridor -> second_bedroom
        path = task._find_path("entrance", "second_bedroom")
        assert path == ["hall", "bedroom_corridor", "second_bedroom"]
        assert len(path) == 3

    def test_no_floor_plan_returns_empty(self):
        task = TaskContext(goal="test")
        assert task._find_path("a", "b") == []


class TestNavigationHints:
    def test_hints_include_unobserved_rooms(self):
        task, tools = _task_with_map()
        # Only observe hall; all other rooms are unobserved.
        task.record("look", {}, tools.execute("look", {}))
        hints = task._navigation_hints()
        assert "explore:kitchen" in hints
        assert hints["explore:kitchen"]["next_hop"] == "kitchen"
        assert hints["explore:master_bedroom"]["next_hop"] == "bedroom_corridor"

    def test_observed_rooms_excluded_from_hints(self):
        task, tools = _task_with_map()
        task.record("look", {}, tools.execute("look", {}))
        task.record("move_to", {"room": "kitchen"}, tools.execute("move_to", {"room": "kitchen"}))
        task.record("look", {}, tools.execute("look", {}))
        hints = task._navigation_hints()
        assert "explore:kitchen" not in hints
        assert "explore:hall" not in hints

    def test_hints_include_last_seen_object_location(self):
        task, tools = _task_with_map()
        task.conditions = [GoalCondition(kind="deliver", object="medicine", person="paul")]
        # Observe hall then move to bedroom_corridor then second_bedroom where medicine lives.
        for name, args in [("look", {}), ("move_to", {"room": "bedroom_corridor"}),
                           ("look", {}), ("move_to", {"room": "second_bedroom"}), ("look", {})]:
            task.record(name, args, tools.execute(name, args))
        # Robot is now in second_bedroom where medicine is. Move back to hall.
        task.record("move_to", {"room": "bedroom_corridor"}, tools.execute("move_to", {"room": "bedroom_corridor"}))
        task.record("move_to", {"room": "hall"}, tools.execute("move_to", {"room": "hall"}))
        hints = task._navigation_hints()
        # Medicine was last seen in second_bedroom; from hall the next hop is bedroom_corridor.
        assert "object:medicine" in hints
        assert hints["object:medicine"]["next_hop"] == "bedroom_corridor"

    def test_hints_include_last_seen_recipient_location(self):
        task, tools = _task_with_map()
        task.conditions = [GoalCondition(kind="deliver", object="charger", person="neave")]
        # Observe hall then bedroom_corridor then master_bedroom where neave lives.
        for name, args in [("look", {}), ("move_to", {"room": "bedroom_corridor"}),
                           ("look", {}), ("move_to", {"room": "master_bedroom"}), ("look", {})]:
            task.record(name, args, tools.execute(name, args))
        # Move robot back to hall.
        task.record("move_to", {"room": "bedroom_corridor"}, tools.execute("move_to", {"room": "bedroom_corridor"}))
        task.record("move_to", {"room": "hall"}, tools.execute("move_to", {"room": "hall"}))
        hints = task._navigation_hints()
        # Neave last seen in master_bedroom; from hall next hop is bedroom_corridor.
        assert "recipient:neave" in hints
        assert hints["recipient:neave"]["next_hop"] == "bedroom_corridor"

    def test_no_hints_when_no_floor_plan(self):
        task = TaskContext(goal="test")
        assert task._navigation_hints() == {}

    def test_no_hints_when_no_current_room(self):
        task = TaskContext(goal="test")
        # Floor plan present but no robot_room.
        task.floor_plan = {"rooms": {"a": {"connections": ["b"]}, "b": {"connections": ["a"]}}}
        assert task._navigation_hints() == {}

    def test_compact_includes_navigation_hints(self):
        task, tools = _task_with_map()
        task.record("look", {}, tools.execute("look", {}))
        payload = task.compact()
        assert "navigation_hints" in payload
        # With all rooms unobserved except hall, hints should be non-empty.
        assert len(payload["navigation_hints"]) > 0
