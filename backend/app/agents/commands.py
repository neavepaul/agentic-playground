from app.tasks.models import TaskContext
from app.tasks.goals import known_recipients


def observable_commands(context: TaskContext) -> tuple[dict, dict | None]:
    """Scope tools to observed targets and the Coordinator's fixed goal conditions.

    This never reads simulator state or interprets natural-language goal text.
    """
    room = context.robot_status.get("room")
    inventory = context.robot_status.get("inventory", [])
    movable = {c.object for c in context.conditions if c.kind in {"hold_object", "deliver", "place_object"}}
    needs = known_recipients(context)
    deliveries = {(c.object, c.person or needs.get(c.object)) for c in context.conditions if c.kind == "deliver"}
    view = context.memory.room_view(room)
    commands = {"report": {"description": "Return discoveries or a blockage to Coordinator."},
                "look": {"tool": "look", "arguments": {},
                         "description": "Scan the current room and refresh local visual observations."}}
    if view is None:
        commands["look"] = {"tool": "look", "arguments": {},
                            "description": "Observe this room; its contents and exits are not yet known."}
        return commands, None

    # Permit fresh scans on return; don't immediately repeat an unchanged scan.
    previous = context.action_history[-1] if context.action_history else None
    if previous and previous["success"] and previous["tool"] == "look":
        commands.pop("look", None)

    def add(tool: str, arguments: dict, description: str) -> None:
        id = ":".join([tool, *arguments.values()])
        commands[id] = {"tool": tool, "arguments": arguments, "description": description}

    for exit in view["connections"]:
        add("move_to", {"room": exit}, f"Move from {room} to the connected room {exit}.")
    for item in view["objects"]:
        if item.get("portable", True) and item["id"] in movable:
            add("pick_up", {"object": item["id"]}, f"Take visible {item['id']} into inventory.")
    for person in view["people"]:
        add("talk_to", {"person": person["id"]},
            f"Ask {person['id']} an unanswered question or convey a requested notification. "
            "Do not use speech to announce your plan; put that in summary. This transfers no objects.")
    for item in inventory:
        if item not in movable:
            continue
        add("drop", {"object": item}, f"Place held {item} in this room.")
        for person in view["people"]:
            if (item, person["id"]) in deliveries:
                add("give", {"object": item, "person": person["id"]},
                    f"Transfer held {item} to {person['id']} in this room.")
    return commands, view
