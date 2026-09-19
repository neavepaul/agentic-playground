from typing import Literal

from pydantic import BaseModel, Field


class Location(BaseModel):
    kind: Literal["room", "robot", "person"]
    id: str


class WorldObject(BaseModel):
    id: str
    name: str
    location: Location
    portable: bool = True


class Person(BaseModel):
    id: str
    name: str
    room: str
    messages: list[str] = Field(default_factory=list)


class Room(BaseModel):
    id: str
    name: str
    connections: list[str]


class Robot(BaseModel):
    room: str = "hall"


class World(BaseModel):
    robot: Robot
    rooms: dict[str, Room]
    people: dict[str, Person]
    objects: dict[str, WorldObject]

    def snapshot(self) -> dict:
        result = self.model_dump()
        # Inventory is a derived view, never a second mutable location.
        result["robot"]["inventory"] = [
            o.id for o in self.objects.values() if o.location.kind == "robot"
        ]
        return result
