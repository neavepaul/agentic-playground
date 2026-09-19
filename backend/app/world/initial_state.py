from .models import Location, Person, Robot, Room, World, WorldObject


def initial_world() -> World:
    graph = {"hall": ["kitchen", "bedroom", "study"],
             "kitchen": ["hall"], "bedroom": ["hall"], "study": ["hall"]}
    return World(
        robot=Robot(),
        rooms={id: Room(id=id, name=id.title(), connections=links) for id, links in graph.items()},
        people={id: Person(id=id, name=id.title(), room=room) for id, room in
                [("mom", "kitchen"), ("neave", "bedroom"), ("dad", "study")]},
        objects={id: WorldObject(id=id, name=id.title(), location=Location(kind="room", id=room))
                 for id, room in [("keys", "kitchen"), ("laptop", "bedroom"), ("charger", "study")]},
    )
