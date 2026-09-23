import json

import pytest
from pydantic import ValidationError

from app.world.clock import WorldClock
from app.world.engine import WorldEngine, _in_range
from app.world.models import World


# ── WorldClock unit tests ──────────────────────────────────────────────────────

def test_clock_starts_at_configured_hour():
    clock = WorldClock(speed=0, start_hour=14.5)
    assert abs(clock.hour() - 14.5) < 0.01


def test_clock_time_str_format():
    clock = WorldClock(speed=0, start_hour=9.0)
    assert clock.time_str() == "09:00"

    clock2 = WorldClock(speed=0, start_hour=13.5)
    assert clock2.time_str() == "13:30"


def test_clock_wraps_at_midnight():
    clock = WorldClock(speed=0, start_hour=23.5)
    # hour() is in [0, 24) — 23.5 is well-defined
    assert 23.0 <= clock.hour() < 24.0


# ── _in_range helper ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("from_h,to_h,hour,expected", [
    (9, 17, 12, True),    # midday is inside office hours
    (9, 17, 9,  True),    # from is inclusive
    (9, 17, 17, False),   # to is exclusive
    (9, 17, 8,  False),   # before range
    (9, 17, 20, False),   # after range
    (22, 6, 23, True),    # midnight-spanning: evening side
    (22, 6, 3,  True),    # midnight-spanning: early morning side
    (22, 6, 22, True),    # midnight-spanning: from is inclusive
    (22, 6, 6,  False),   # midnight-spanning: to is exclusive
    (22, 6, 10, False),   # outside midnight-spanning range
])
def test_in_range(from_h, to_h, hour, expected):
    assert _in_range(from_h, to_h, hour) == expected


# ── Schedule-driven NPC positions ──────────────────────────────────────────────

def _engine_at_hour(hour: float, world_file=None) -> WorldEngine:
    clock = WorldClock(speed=0, start_hour=hour)
    return WorldEngine(world_file, clock=clock)


def test_scheduled_paul_is_in_office_during_work_hours():
    engine = _engine_at_hour(10.0)
    engine.move_to("entrance")
    engine.move_to("office")
    view = engine.look()
    assert any(p["id"] == "paul" for p in view["people"])


def test_scheduled_paul_is_in_kitchen_in_evening():
    engine = _engine_at_hour(19.0)
    engine.move_to("entrance")
    engine.move_to("hall")
    engine.move_to("kitchen")
    view = engine.look()
    assert any(p["id"] == "paul" for p in view["people"])


def test_scheduled_paul_is_in_master_bedroom_at_night():
    engine = _engine_at_hour(23.0)
    engine.move_to("bedroom_corridor")
    engine.move_to("master_bedroom")
    view = engine.look()
    assert any(p["id"] == "paul" for p in view["people"])


def test_paul_not_in_office_outside_work_hours():
    engine = _engine_at_hour(23.0)
    engine.move_to("entrance")
    engine.move_to("office")
    view = engine.look()
    assert not any(p["id"] == "paul" for p in view["people"])


def test_midnight_spanning_schedule_wraps_correctly():
    """Paul's master_bedroom schedule spans 22:00-09:00; verify at 02:00."""
    engine = _engine_at_hour(2.0)
    engine.move_to("bedroom_corridor")
    engine.move_to("master_bedroom")
    view = engine.look()
    assert any(p["id"] == "paul" for p in view["people"])


def test_engine_without_clock_uses_static_rooms():
    """Without a clock, everyone stays in their static room from world.json."""
    engine = WorldEngine()  # no clock
    engine.move_to("entrance")
    engine.move_to("office")
    view = engine.look()
    assert any(p["id"] == "paul" for p in view["people"])


def test_talk_to_respects_schedule(tmp_path):
    data = {
        "name": "Clock test", "robot": {"room": "entry"},
        "rooms": {
            "entry": {"id": "entry", "name": "Entry", "connections": ["lounge"]},
            "lounge": {"id": "lounge", "name": "Lounge", "connections": ["entry"]},
        },
        "people": {
            "alice": {
                "id": "alice", "name": "Alice", "room": "lounge",
                "dialogue": [{"topics": ["hello"], "response": "Hi there!"}],
                "schedule": [
                    {"from_hour": 9, "to_hour": 18, "room": "lounge"},
                    {"from_hour": 18, "to_hour": 9, "room": "entry"},
                ],
            }
        },
        "objects": {},
    }
    path = tmp_path / "world.json"
    path.write_text(json.dumps(data))

    daytime = WorldEngine(path, WorldClock(speed=0, start_hour=12))
    daytime.move_to("lounge")
    result = daytime.talk_to("alice", "hello")
    assert result["response"] == "Hi there!"

    nighttime = WorldEngine(path, WorldClock(speed=0, start_hour=20))
    # alice is in entry at night, not lounge
    with pytest.raises(Exception, match="alice is not in the current room"):
        nighttime.move_to("lounge")
        nighttime.talk_to("alice", "hello")


def test_get_status_includes_time_when_clock_present():
    engine = WorldEngine(clock=WorldClock(speed=0, start_hour=14.0))
    status = engine.get_status()
    assert status["time"] == "14:00"


def test_get_status_has_no_time_without_clock():
    engine = WorldEngine()
    assert "time" not in engine.get_status()


# ── Schedule model validation ───────────────────────────────────────────────────

def test_schedule_with_invalid_room_fails_validation(tmp_path):
    data = {
        "name": "Bad schedule", "robot": {"room": "entry"},
        "rooms": {"entry": {"id": "entry", "name": "Entry", "connections": []}},
        "people": {
            "bob": {
                "id": "bob", "name": "Bob", "room": "entry",
                "schedule": [{"from_hour": 9, "to_hour": 17, "room": "nonexistent"}],
            }
        },
        "objects": {},
    }
    with pytest.raises(ValidationError, match="unknown room"):
        World.model_validate(data)


def test_schedule_with_equal_hours_fails_validation():
    from app.world.models import ScheduleEntry
    with pytest.raises(ValidationError, match="must differ"):
        ScheduleEntry(from_hour=9, to_hour=9, room="office")
