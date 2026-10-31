"""Kip perceives the room he is in: fixtures, people and what they are doing, and changes."""
from app.chat.session import ChatSession
from app.config import Settings
from app.events.bus import EventBus
from app.memory.recall import MemoryStore
from app.persona.engine import Persona
from app.tasks.manager import TaskManager
from app.world.clock import WorldClock
from app.world.engine import WorldEngine
from app.world.tools import WorldTools
from tests.fakes import ScriptedLLM


def engine_at(hour: float) -> tuple[WorldEngine, WorldClock]:
    clock = WorldClock(speed=0, start_hour=hour)
    return WorldEngine(clock=clock), clock


def set_hour(clock: WorldClock, hour: float) -> None:
    clock._origin_sim = hour * 3600.0  # a stopped clock (speed 0) set to a new hour


def test_look_shows_fixtures_activities_and_the_room_but_the_map_does_not():
    engine, _ = engine_at(17.5)
    engine.move_to("kitchen")
    view = engine.look()
    assert view["description"].startswith("Small and warm")
    assert {"id": "stove", "name": "Stove", "state": "pans bubbling"} in view["features"]
    assert {p["id"]: p.get("activity") for p in view["people"]} == {
        "moira": "cooking", "paul": "making tea and snacking"}
    room = engine.get_map()["rooms"]["kitchen"]
    assert "features" not in room and "description" not in room


def test_feature_states_follow_the_hour_and_wrap_midnight():
    engine, clock = engine_at(8)
    engine.move_to("kitchen")
    states = lambda: {f["id"]: f.get("state") for f in engine.look()["features"]}
    assert states()["kettle"] == "steaming"
    set_hour(clock, 13)
    assert states()["kettle"] == "cold"
    engine.move_to("hall")
    engine.move_to("bedroom_corridor")
    set_hour(clock, 2)
    assert {f["id"]: f.get("state") for f in engine.look()["features"]}["night_light"] == "glowing dimly"


def test_surroundings_cover_only_the_current_room():
    engine, _ = engine_at(10)
    around = engine.surroundings()
    assert around["room"] == "hall" and around["room_name"] == "Hall" and around["time"] == "10:00"
    assert [p["id"] for p in around["people"]] == ["moira"]
    # Nobody in another room, and no schedules or dialogue tables, reach the senses.
    assert "paul" not in str(around) and "schedule" not in str(around) and "dialogue" not in str(around)


def test_presence_changes_report_arrivals_departures_and_activities_but_not_kips_own_moves():
    engine, clock = engine_at(16.9)
    engine.move_to("kitchen")
    assert engine.presence_changes() == []  # first look around: a baseline, not news
    set_hour(clock, 17.1)                   # Paul's schedule brings him to the kitchen
    changes = engine.presence_changes()
    assert [(c["change"], c["person"]) for c in changes] == [("arrived", "paul")]
    assert changes[0]["summary"] == "Paul came into the kitchen, making tea and snacking."
    assert engine.presence_changes() == []  # reported once
    set_hour(clock, 18.1)                   # Moira heads to bed
    assert [(c["change"], c["person"]) for c in engine.presence_changes()] == [("left", "moira")]
    engine.move_to("hall")                  # Kip's own move: who is there was found, not noticed
    assert engine.presence_changes() == []


def test_people_answer_off_topic_talk_in_their_own_words():
    engine, _ = engine_at(10)
    engine.move_to("bedroom_corridor")
    engine.move_to("second_bedroom")
    assert engine.talk_to("minu", "Nice weather today")["response"] == \
        "Not sure. Want to see what I'm drawing instead?"
    assert engine.talk_to("minu", "What are you drawing?")["response"].startswith("It's a robot.")


def test_chat_status_carries_live_surroundings_and_recent_noticing():
    clock = WorldClock(speed=0, start_hour=16.9)
    engine, bus = WorldEngine(clock=clock), EventBus()
    model = ScriptedLLM([])
    manager = TaskManager(model, WorldTools(engine, bus), bus, Settings())
    session = ChatSession(model, bus, manager, engine, Settings(), store=MemoryStore(":memory:"),
                          persona=Persona())
    WorldTools(engine, bus).execute("move_to", {"room": "kitchen"})
    around = session._status()["around_me_now"]
    assert around["room"] == "Kitchen" and "Moira (reading the newspaper)" not in around["people"]
    assert "Kettle: steaming" in around["fixtures"] and "Keys" in around["things"]
    assert "just_noticed" not in session._status()

    engine.presence_changes()
    set_hour(clock, 17.1)
    for change in engine.presence_changes():
        bus.emit("presence_noticed", "kip", **change)
    status = session._status()
    assert status["just_noticed"] == ["Paul came into the kitchen, making tea and snacking."]
    assert "Paul (making tea and snacking)" in status["around_me_now"]["people"]

    # Walking away: what was noticed in the kitchen is no longer "just" noticed here.
    WorldTools(engine, bus).execute("move_to", {"room": "hall"})
    assert "just_noticed" not in session._status()
