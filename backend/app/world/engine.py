import re
from pathlib import Path

from .clock import WorldClock
from .models import Location, World

_DEFAULT_WORLD = Path(__file__).resolve().parents[2] / "worlds" / "house.json"


def _load_world(path) -> World:
    return World.model_validate_json(Path(path or _DEFAULT_WORLD).read_text(encoding="utf-8"))


def _in_range(from_hour: float, to_hour: float, hour: float) -> bool:
    """True when hour falls in [from_hour, to_hour), wrapping midnight."""
    if from_hour < to_hour:
        return from_hour <= hour < to_hour
    return hour >= from_hour or hour < to_hour


class WorldError(ValueError):
    pass


class WorldEngine:
    """Deterministic simulator. Only the tool service calls these operations."""

    def __init__(self, world_file=None, clock: WorldClock | None = None) -> None:
        self.world_file = world_file
        self._world = _load_world(world_file)
        self._clock = clock
        self._initial_rooms_until = {}
        # Bumped by every reset, so decisions made against an older world can be discarded.
        self.generation = 0
        # What Kip last perceived around him: (room, {person: activity}). See presence_changes().
        self._perceived: tuple[str, dict[str, str]] | None = None
        # Deliberate (commit aa30271, tests in test_clock.py): a fresh start follows the
        # schedule immediately, while a reset restores house.json rooms until the next
        # schedule boundary.

    def _hold_initial_rooms(self) -> None:
        """After a reset, people stay in their house.json rooms until their next schedule boundary."""
        self._initial_rooms_until.clear()
        if not self._clock:
            return
        now = self._clock.simulated_seconds()
        for person in self._world.people.values():
            boundaries = [hour * 3600 for entry in person.schedule
                          for hour in (entry.from_hour, entry.to_hour)]
            if boundaries:
                delay = min((boundary - now) % 86400 or 86400 for boundary in boundaries)
                self._initial_rooms_until[person.id] = now + delay

    def _effective_room(self, person) -> str:
        """Return the room a person is in, applying their schedule if a clock is present."""
        if self._clock is None or not person.schedule:
            return person.room
        if self._clock.simulated_seconds() < self._initial_rooms_until.get(person.id, -1):
            return person.room
        hour = self._clock.hour()
        for entry in person.schedule:
            if _in_range(entry.from_hour, entry.to_hour, hour):
                return entry.room
        return person.room

    def _activity(self, person) -> str:
        """What a person is visibly doing now: their schedule entry's, else their default."""
        if self._clock is not None and person.schedule and                 self._clock.simulated_seconds() >= self._initial_rooms_until.get(person.id, -1):
            hour = self._clock.hour()
            for entry in person.schedule:
                if _in_range(entry.from_hour, entry.to_hour, hour) and entry.activity:
                    return entry.activity
        return person.activity

    def _feature_state(self, feature) -> str:
        if self._clock is not None:
            hour = self._clock.hour()
            for timed in feature.states:
                if _in_range(timed.from_hour, timed.to_hour, hour):
                    return timed.state
        return feature.state

    def snapshot(self) -> dict:
        result = self._world.snapshot()
        if self._clock:
            for pid, person in self._world.people.items():
                result["people"][pid]["room"] = self._effective_room(person)
        return result

    def reset(self) -> None:
        self._world = _load_world(self.world_file)
        self.generation += 1
        self._perceived = None
        if self._clock:
            self._clock.reset()
        self._hold_initial_rooms()

    def advance_clock(self, steps: int = 1) -> None:
        """Signal that one agent turn elapsed. Only action-driven clocks react."""
        if self._clock is not None:
            self._clock.advance(steps)

    def simulated_time(self) -> str | None:
        return self._clock.time_str() if self._clock else None

    @property
    def clock(self) -> WorldClock | None:
        return self._clock

    def people_schedules(self) -> dict:
        """Return each person's schedule as serialisable dicts, keyed by person id."""
        return {
            pid: [{"from_hour": e.from_hour, "to_hour": e.to_hour, "room": e.room}
                  for e in person.schedule]
            for pid, person in self._world.people.items()
            if person.schedule
        }

    def get_map(self) -> dict:
        return self._world.floor_plan()

    def get_status(self) -> dict:
        result = {"room": self._world.robot.room, "inventory": self.snapshot()["robot"]["inventory"]}
        if self._clock:
            result["time"] = self._clock.time_str()
            if self._clock.now() is not None:  # the device clock knows the date and timezone too
                result["today"] = self._clock.date_str()
        return result

    def look(self) -> dict:
        w = self._world
        room = w.robot.room
        place = w.rooms[room]
        view = {"room": room, "connections": place.connections[:],
                "people": [{"id": p.id, "name": p.name, **({"activity": a} if (a := self._activity(p)) else {})}
                           for p in w.people.values() if self._effective_room(p) == room],
                "objects": [{"id": o.id, "name": o.name, "portable": o.portable}
                            for o in w.objects.values() if o.location == Location(kind="room", id=room)],
                "held_objects": [{"object": o.id, "person": o.location.id}
                                 for o in w.objects.values() if o.location.kind == "person"
                                 and self._effective_room(w.people[o.location.id]) == room]}
        if place.description:
            view["description"] = place.description
        if place.features:
            view["features"] = [{"id": f.id, "name": f.name,
                                 **({"state": state} if (state := self._feature_state(f)) else {}),
                                 **({"description": f.description} if f.description else {})}
                                for f in place.features]
        return view

    def surroundings(self) -> dict:
        """What Kip perceives where he stands, without spending an action.

        A body in a room sees and hears that room: the same view as look, and only
        that room. Nothing about other rooms, schedules or the full snapshot.
        """
        view = self.look()
        view["room_name"] = self._world.rooms[view["room"]].name
        if self._clock:
            view["time"] = self._clock.time_str()
        return view

    def presence_changes(self) -> list[dict]:
        """People arriving, leaving or changing what they do in Kip's room since the last check.

        Moving to another room re-baselines quietly: who was already there was found,
        not noticed arriving, and look reports them. Pure state comparison; no model.
        """
        room = self._world.robot.room
        present = {p.id: self._activity(p) for p in self._world.people.values()
                   if self._effective_room(p) == room}
        previous, self._perceived = self._perceived, (room, present)
        if previous is None or previous[0] != room:
            return []
        before, name = previous[1], lambda pid: self._world.people[pid].name
        where = self._world.rooms[room].name
        changes = []
        for pid in present.keys() - before.keys():
            doing = f", {present[pid]}" if present[pid] else ""
            changes.append({"change": "arrived", "person": pid, "room": room, "activity": present[pid],
                            "summary": f"{name(pid)} came into the {where.lower()}{doing}."})
        for pid in before.keys() - present.keys():
            changes.append({"change": "left", "person": pid, "room": room,
                            "summary": f"{name(pid)} left the {where.lower()}."})
        for pid in present.keys() & before.keys():
            if present[pid] != before[pid] and present[pid]:
                changes.append({"change": "activity", "person": pid, "room": room, "activity": present[pid],
                                "summary": f"{name(pid)} is now {present[pid]}."})
        return sorted(changes, key=lambda change: (change["person"], change["change"]))

    def move_to(self, room: str) -> dict:
        w = self._world
        origin = w.robot.room
        if room not in w.rooms[origin].connections:
            raise WorldError(f"{room} is not directly connected to {origin}.")
        w.robot.room = room
        return {"from_room": origin, "room": room}

    def pick_up(self, object: str) -> dict:
        item = self._world.objects.get(object)
        if item is None or item.location != Location(kind="room", id=self._world.robot.room):
            raise WorldError(f"{object} is not visible on the floor in the current room.")
        if not item.portable:
            raise WorldError(f"{object} is not portable.")
        item.location = Location(kind="robot", id="robot")
        return {"object": object, "room": self._world.robot.room}

    def _held(self, object: str):
        item = self._world.objects.get(object)
        if item is None or item.location.kind != "robot":
            raise WorldError(f"The robot is not holding {object}.")
        return item

    def _nearby(self, person: str):
        npc = self._world.people.get(person)
        if npc is None or self._effective_room(npc) != self._world.robot.room:
            raise WorldError(f"{person} is not in the current room.")
        return npc

    def drop(self, object: str) -> dict:
        self._held(object).location = Location(kind="room", id=self._world.robot.room)
        return {"object": object, "room": self._world.robot.room}

    def give(self, object: str, person: str) -> dict:
        item = self._held(object)
        self._nearby(person)
        item.location = Location(kind="person", id=person)
        return {"object": object, "person": person, "room": self._world.robot.room}

    def talk_to(self, person: str, message: str) -> dict:
        npc = self._nearby(person)
        npc.messages.append(message)
        npc.messages[:] = npc.messages[-50:]
        text = message.lower()
        words = set(re.findall(r"\w+", text.replace("_", " ")))
        response = npc.fallback_response or "I heard your message. I don't have any more information about that."
        for dialogue in npc.dialogue:
            if any(set(re.findall(r"\w+", topic.lower().replace("_", " "))) <= words for topic in dialogue.topics):
                response = dialogue.response
                break
        return {"person": person, "message": message, "response": response,
                "room": self._world.robot.room}
