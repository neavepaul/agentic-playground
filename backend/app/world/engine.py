from .initial_state import initial_world
from .models import Location


class WorldError(ValueError):
    pass


class WorldEngine:
    """Deterministic simulator. Only the tool service calls these operations."""

    def __init__(self, world_file=None) -> None:
        self.world_file = world_file
        self._world = initial_world(world_file)

    def snapshot(self) -> dict:
        return self._world.snapshot()

    def reset(self) -> None:
        self._world = initial_world(self.world_file)

    def get_map(self) -> dict:
        return self._world.floor_plan()

    def get_status(self) -> dict:
        return {"room": self._world.robot.room,
                "inventory": self.snapshot()["robot"]["inventory"]}

    def look(self) -> dict:
        w = self._world
        room = w.robot.room
        return {"room": room, "connections": w.rooms[room].connections[:],
                "people": [{"id": p.id, "name": p.name} for p in w.people.values() if p.room == room],
                "objects": [{"id": o.id, "name": o.name, "portable": o.portable}
                            for o in w.objects.values() if o.location == Location(kind="room", id=room)],
                "held_objects": [{"object": o.id, "person": o.location.id}
                                 for o in w.objects.values() if o.location.kind == "person"
                                 and w.people[o.location.id].room == room]}

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
        if npc is None or npc.room != self._world.robot.room:
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
        import re
        words = set(re.findall(r"\w+", text.replace("_", " ")))
        response = "I heard your message. I don't have any more information about that."
        for dialogue in npc.dialogue:
            if any(set(re.findall(r"\w+", topic.lower().replace("_", " "))) <= words for topic in dialogue.topics):
                response = dialogue.response
                break
        return {"person": person, "message": message, "response": response,
                "room": self._world.robot.room}
