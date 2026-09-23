"""Deterministic verification of goal conditions using tool evidence only."""
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import TaskContext


def normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def known_recipients(context: "TaskContext") -> dict[str, str]:
    # Grounded language interpretations remain fallible; no simulator needs are read.
    return {claim["object"]: claim["person"] for claim in context.interpreted_needs}


def check_conditions(context: "TaskContext") -> list[dict]:
    actions = [a for a in context.action_history if a["success"]]
    looks = [a["observation"] for a in actions if a["tool"] == "look"]
    talks = [a["observation"] for a in actions if a["tool"] == "talk_to"]
    needs = known_recipients(context)

    def location(object_id: str):
        update = context.discoveries.get("object:" + object_id)
        if update:
            obs = update["observation"]
            return {"pick_up": ("robot", "robot"), "drop": ("room", obs["room"]),
                    "give": ("person", obs.get("person", ""))}[update["tool"]]
        for view in reversed(looks):
            if any(o["id"] == object_id for o in view["objects"]):
                return "room", view["room"]
            for item in view["held_objects"]:
                if item["object"] == object_id:
                    return "person", item["person"]
        if object_id in context.robot_status.get("inventory", []):
            return "robot", "robot"
        return None

    def notified(person: str, message: str) -> bool:
        return any(talk["person"] == person and normalize(message) in normalize(talk["message"])
                   for talk in talks)

    results = []
    for condition in context.conditions:
        kind = condition.kind
        if kind == "find_object":
            met = location(condition.object) is not None
        elif kind == "find_person":
            met = any(p["id"] == condition.person for view in looks for p in view["people"])
        elif kind == "identify_recipient":
            # Satisfied by conversation evidence OR by a completed named deliver condition
            # (the identity is proven by the successful handoff itself).
            delivered_to = any(
                c.kind == "deliver" and c.object == condition.object and c.person
                and location(c.object) == ("person", c.person)
                for c in context.conditions
            )
            met = condition.object in needs or delivered_to
        elif kind == "hold_object":
            met = location(condition.object) == ("robot", "robot")
        elif kind == "deliver":
            recipient = condition.person or needs.get(condition.object)
            met = bool(recipient) and location(condition.object) == ("person", recipient)
        elif kind == "place_object":
            met = location(condition.object) == ("room", condition.room)
        elif kind == "visit_room":
            met = context.robot_status.get("room") == condition.room
        elif kind == "notify":
            met = notified(condition.person, condition.message)
        else:  # notify_everyone: all reachable rooms must first be observed.
            seen = {view["room"] for view in looks}
            known = seen | {room for view in looks for room in view["connections"]}
            people = {p["id"] for view in looks for p in view["people"]}
            met = bool(seen) and seen == known and all(notified(p, condition.message) for p in people)
        results.append({**condition.model_dump(exclude_defaults=True), "satisfied": bool(met)})
    return results
