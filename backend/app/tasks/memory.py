"""Task-scoped observation reducer. Agent prose never updates physical beliefs."""
from typing import Literal

from pydantic import BaseModel, Field

from app.world.models import Location
from .goals import normalize


class RememberedObject(BaseModel):
    name: str = ""
    location: Location | None = None
    portable: bool = True
    last_observed_room: str | None = None
    evidence_id: str = ""
    revision: int = 0


class ConversationTurn(BaseModel):
    message: str
    response: str
    evidence_id: str


class ConversationThread(BaseModel):
    summary: str
    resolved: bool = False
    turns: list[ConversationTurn] = Field(default_factory=list)


class RememberedPerson(BaseModel):
    name: str = ""
    location: str | None = None
    evidence_id: str = ""
    revision: int = 0
    conversation_threads: dict[str, ConversationThread] = Field(default_factory=dict)
    notifications: dict[str, str] = Field(default_factory=dict)


class RememberedRoom(BaseModel):
    connections: list[str]
    evidence_id: str


class ReportedNeed(BaseModel):
    object: str
    person: str
    speaker: str
    quote: str
    evidence_id: str
    status: Literal["interpretation_of_speech"] = "interpretation_of_speech"


class TaskMemory(BaseModel):
    robot_room: str | None = None
    simulated_time: str | None = None
    visited_rooms: set[str] = Field(default_factory=set)
    rooms: dict[str, RememberedRoom] = Field(default_factory=dict)
    people: dict[str, RememberedPerson] = Field(default_factory=dict)
    objects: dict[str, RememberedObject] = Field(default_factory=dict)
    reported_needs: dict[str, ReportedNeed] = Field(default_factory=dict)

    def inventory(self) -> list[str]:
        return sorted(key for key, item in self.objects.items()
                      if item.location == Location(kind="robot", id="robot"))

    def set_object(self, key: str, location: Location | None, room: str | None,
                   evidence_id: str, **attributes) -> None:
        item = self.objects.setdefault(key, RememberedObject(name=key))
        if item.location != location:
            item.revision += 1
        item.location, item.evidence_id = location, evidence_id
        if room is not None:
            item.last_observed_room = room
        for attribute, value in attributes.items():
            setattr(item, attribute, value)

    def set_person(self, key: str, room: str | None, evidence_id: str, name: str | None = None) -> None:
        person = self.people.setdefault(key, RememberedPerson(name=key))
        if person.location != room:
            person.revision += 1
        person.location, person.evidence_id = room, evidence_id
        if name:
            person.name = name

    def room_view(self, room: str | None) -> dict | None:
        if room not in self.rooms:
            return None
        return {"room": room, "connections": self.rooms[room].connections[:],
                "people": [{"id": key, "name": p.name} for key, p in self.people.items() if p.location == room],
                "objects": [{"id": key, "name": o.name, "portable": o.portable}
                            for key, o in self.objects.items() if o.location == Location(kind="room", id=room)],
                "held_objects": [{"object": key, "person": o.location.id} for key, o in self.objects.items()
                                 if o.location and o.location.kind == "person"
                                 and self.people[o.location.id].location == room]}

    def next_thread_id(self, person: str) -> str:
        known = self.people.setdefault(person, RememberedPerson(name=person))
        index = 1
        while f"thread_{index}" in known.conversation_threads:
            index += 1
        return f"thread_{index}"

    def remember_conversation(self, person: str, thread_id: str, summary: str,
                              resolved: bool, message: str, response: str,
                              evidence_id: str) -> None:
        known = self.people.setdefault(person, RememberedPerson(name=person))
        thread = known.conversation_threads.get(thread_id)
        if thread is None:
            thread = ConversationThread(summary=summary or "Conversation topic")
            known.conversation_threads[thread_id] = thread
        elif summary:
            thread.summary = summary
        thread.resolved = resolved
        thread.turns.append(ConversationTurn(message=message[:500], response=response[:500], evidence_id=evidence_id))
        thread.turns[:] = thread.turns[-8:]

    def observe(self, tool: str, args: dict, obs: dict, evidence_id: str,
                notification: bool) -> None:
        if tool == "get_status":
            self.robot_room = obs["room"]
            if obs.get("time"):
                self.simulated_time = obs["time"]
            self.visited_rooms.add(obs["room"])
            for key in self.inventory():
                if key not in obs["inventory"]:
                    self.set_object(key, None, None, evidence_id)
            for key in obs["inventory"]:
                self.set_object(key, Location(kind="robot", id="robot"), obs["room"], evidence_id)
        elif tool == "move_to":
            self.robot_room = obs["room"]
            self.visited_rooms.add(obs["room"])
        elif tool == "look":
            room = obs["room"]
            self.visited_rooms.add(room)
            self.rooms[room] = RememberedRoom(connections=obs["connections"], evidence_id=evidence_id)
            visible_objects = {o["id"] for o in obs["objects"]} | {o["object"] for o in obs["held_objects"]}
            # A new complete room scan supersedes old beliefs, including absence.
            for key, item in self.objects.items():
                local = item.location == Location(kind="room", id=room)
                if item.location and item.location.kind == "person":
                    local = self.people[item.location.id].location == room
                if local and key not in visible_objects:
                    self.set_object(key, None, None, evidence_id)
            visible_people = {p["id"] for p in obs["people"]}
            for key, person in self.people.items():
                if person.location == room and key not in visible_people:
                    self.set_person(key, None, evidence_id)
            for person in obs["people"]:
                self.set_person(person["id"], room, evidence_id, person["name"])
            for item in obs["objects"]:
                self.set_object(item["id"], Location(kind="room", id=room), room, evidence_id,
                                name=item["name"], portable=item["portable"])
            for item in obs["held_objects"]:
                self.set_object(item["object"], Location(kind="person", id=item["person"]), room, evidence_id)
        elif tool in {"pick_up", "drop", "give"}:
            kind, owner = {"pick_up": ("robot", "robot"), "drop": ("room", obs["room"]),
                           "give": ("person", obs.get("person", ""))}[tool]
            if tool == "give":
                self.set_person(owner, obs["room"], evidence_id)
            self.set_object(args["object"], Location(kind=kind, id=owner), obs["room"], evidence_id)
        elif tool == "talk_to":
            key = obs["person"]
            self.set_person(key, obs["room"], evidence_id)
            if notification:
                self.people[key].notifications[normalize(obs["message"])] = evidence_id

    def prompt(self) -> dict:
        people = {}
        for key, person in self.people.items():
            people[key] = {
                "name": person.name,
                "location": person.location,
                "evidence_id": person.evidence_id,
                "conversation_threads": {
                    thread_id: {
                        "summary": thread.summary,
                        "resolved": thread.resolved,
                        "turns": [turn.model_dump() for turn in thread.turns[-4:]],
                    }
                    for thread_id, thread in person.conversation_threads.items()
                },
            }
        return {"visited_rooms": sorted(self.visited_rooms),
                "observed_rooms": {key: room.model_dump() for key, room in self.rooms.items()},
                "people": people,
                "objects": {key: item.model_dump(exclude={"revision"}) for key, item in self.objects.items()},
                "reported_needs_unverified": [claim.model_dump() for claim in self.reported_needs.values()]}
