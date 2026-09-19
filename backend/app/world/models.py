import re
from typing import Literal

from pydantic import BaseModel, Field, ConfigDict, model_validator


class ScenarioModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Location(ScenarioModel):
    kind: Literal["room", "robot", "person"]
    id: str


class WorldObject(ScenarioModel):
    id: str
    name: str
    location: Location
    portable: bool = True


class Dialogue(ScenarioModel):
    topics: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def nonempty_topics(self):
        if any(not re.search(r"\w", topic) for topic in self.topics):
            raise ValueError("Dialogue topics must contain words.")
        return self
    response: str = Field(min_length=1)


class Person(ScenarioModel):
    id: str
    name: str
    room: str
    messages: list[str] = Field(default_factory=list)
    dialogue: list[Dialogue] = Field(default_factory=list)


class Room(ScenarioModel):
    id: str
    name: str
    connections: list[str]
    outline: list[tuple[float, float]] = Field(default_factory=list)
    anchor: tuple[float, float] = (0, 0)
    color: str = "#d3dfc4"


class Door(ScenarioModel):
    rooms: tuple[str, str]
    position: tuple[float, float]
    width: float = Field(default=0.6, gt=0)


class Robot(ScenarioModel):
    room: str = "hall"


class World(ScenarioModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "House"
    robot: Robot
    rooms: dict[str, Room]
    people: dict[str, Person]
    objects: dict[str, WorldObject]

    doors: list[Door] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_references(self):
        for collection in (self.rooms, self.people, self.objects):
            for key, value in collection.items():
                if key != value.id or not re.fullmatch(r"[a-z][a-z0-9_]{0,39}", key):
                    raise ValueError("Entity keys must match valid tool IDs.")
        if self.robot.room not in self.rooms:
            raise ValueError("Robot must start in an existing room.")
        for room in self.rooms.values():
            if len(set(room.connections)) != len(room.connections):
                raise ValueError("Room connections must not repeat.")
            if room.outline and len(room.outline) < 3:
                raise ValueError("Room outlines need at least three points.")
            for other in room.connections:
                if other == room.id or other not in self.rooms or room.id not in self.rooms[other].connections:
                    raise ValueError("Room connections must exist and be reciprocal.")
        for person in self.people.values():
            if person.room not in self.rooms:
                raise ValueError("Person references an unknown room.")
        for item in self.objects.values():
            valid = {"room": self.rooms, "person": self.people, "robot": {"robot": True}}
            if item.location.id not in valid[item.location.kind]:
                raise ValueError("Object references an unknown location.")
        for door in self.doors:
            a, b = door.rooms
            if a not in self.rooms or b not in self.rooms[a].connections:
                raise ValueError("Door must join connected rooms.")
        return self

    def floor_plan(self) -> dict:
        # A preloaded map contains no occupants, object locations or NPC knowledge.
        return {"rooms": {key: room.model_dump() for key, room in self.rooms.items()},
                "doors": [door.model_dump() for door in self.doors]}

    def snapshot(self) -> dict:
        result = self.model_dump()
        # Inventory is a derived view, never a second mutable location.
        result["robot"]["inventory"] = [
            o.id for o in self.objects.values() if o.location.kind == "robot"
        ]
        return result
