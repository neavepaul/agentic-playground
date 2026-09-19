from copy import deepcopy
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from app.agents.schemas import GoalCondition
from .goals import check_conditions, known_recipients


class TaskContext(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    goal: str
    status: Literal["running", "completed", "failed", "cancelled"] = "running"
    summary: str = "Starting task."
    cycle_count: int = 0
    tool_count: int = 0
    critic_count: int = 0
    current_plan: str = ""
    robot_status: dict = Field(default_factory=dict)
    discoveries: dict = Field(default_factory=dict)
    action_history: list[dict] = Field(default_factory=list)
    critic_feedback: list[dict] = Field(default_factory=list)
    explorer_reports: list[str] = Field(default_factory=list)
    conditions: list[GoalCondition] = Field(default_factory=list)
    floor_plan: dict = Field(default_factory=dict)
    interpreted_needs: list[dict] = Field(default_factory=list)
    recipient: str | None = None
    recipient_location: str | None = None
    item_location: str | None = None
    carried_by: str | None = None
    delivered: bool = False
    explored_rooms: set[str] = Field(default_factory=set)
    last_progress_signature: tuple | None = None
    action_feedback: list[str] = Field(default_factory=list)

    def feedback(self, message: str) -> None:
        self.action_feedback.append(message)
        self.action_feedback[:] = self.action_feedback[-6:]

    def delivery_state(self) -> list[dict]:
        """Summarize remembered prerequisites; never consult simulator truth."""
        needs = known_recipients(self)
        inventory = self.robot_status.get("inventory", [])
        room = self.robot_status.get("room")
        result = []
        for condition in self.conditions:
            if condition.kind != "deliver":
                continue
            recipient = condition.person or needs.get(condition.object)
            location = self._object_location(condition.object)
            held = condition.object in inventory
            delivered = bool(recipient and location == ("person", recipient))
            missing = []
            if not delivered:
                if not held:
                    if location is None:
                        missing.append("locate_object")
                    missing.append("acquire_object")
                if not recipient:
                    missing.append("identify_recipient")
                elif not self._person_location(recipient):
                    missing.append("locate_recipient")
                missing.append("give_object")
            result.append({"object": condition.object, "recipient": recipient,
                           "held": held, "last_observed_location": location,
                           "current_room_observed": "room:" + str(room) in self.discoveries,
                           "observed_on_floor_here": location == ("room", room),
                           "recipient_last_seen": self._person_location(recipient) if recipient else None,
                           "missing_prerequisites": missing, "delivered": delivered})
        return result

    def remember_meaning(self, evidence_id, meaning) -> None:
        source = next((a for a in self.action_history if a["evidence_id"] == evidence_id
                       and a["success"] and a["tool"] == "talk_to"), None)
        if source is None:
            return
        for need in meaning.needs:
            if need.quote in source["observation"]["response"]:
                self.interpreted_needs.append({**need.model_dump(), "evidence_id": evidence_id})
        self.refresh_task_state()

    def _person_location(self, person: str) -> str | None:
        for entry in (v for k, v in self.discoveries.items() if k.startswith("room:") and v.get("success")):
            obs = entry["observation"]
            if any(p["id"] == person for p in obs.get("people", [])):
                return obs["room"]
        return None

    def _object_location(self, object_id: str):
        for action in reversed(self.action_history):
            if not action.get("success"):
                continue
            if action.get("tool") == "pick_up" and action["arguments"].get("object") == object_id:
                return ("robot", "robot")
            if action.get("tool") == "drop" and action["arguments"].get("object") == object_id:
                return ("room", action["observation"].get("room"))
            if action.get("tool") == "give" and action["arguments"].get("object") == object_id:
                return ("person", action["observation"].get("person"))
        for entry in reversed([v for k, v in self.discoveries.items() if k.startswith("room:") and v.get("success")]):
            obs = entry["observation"]
            for item in obs.get("held_objects", []):
                if item["object"] == object_id:
                    return ("person", item["person"])
            if any(item["id"] == object_id for item in obs.get("objects", [])):
                return ("room", obs["room"])
        if object_id in self.robot_status.get("inventory", []):
            return ("robot", "robot")
        return None

    def refresh_task_state(self) -> None:
        self.explored_rooms = {entry["observation"]["room"] for key, entry in self.discoveries.items()
                               if key.startswith("room:") and entry.get("success")}
        recipient_name = None
        target_object = None
        for condition in self.conditions:
            if condition.kind == "deliver":
                target_object = condition.object
                recipient_name = condition.person or known_recipients(self).get(target_object)
                break
        if target_object:
            self.item_location = None
            object_state = self._object_location(target_object)
            if object_state:
                self.item_location = object_state[1]
            self.recipient = recipient_name
            self.recipient_location = self._person_location(recipient_name) if recipient_name else None
            self.carried_by = "robot" if target_object in self.robot_status.get("inventory", []) else None
            self.delivered = bool(recipient_name and object_state and object_state[0] == "person" and object_state[1] == recipient_name)
        else:
            self.recipient = None
            self.recipient_location = None
            self.item_location = None
            self.carried_by = None
            self.delivered = False

    def progress_signature(self) -> tuple:
        return (
            self.robot_status.get("room"),
            tuple(sorted(self.robot_status.get("inventory", []))),
            self.recipient,
            self.recipient_location,
            self.item_location,
            self.delivered,
            tuple(sorted(self.explored_rooms)),
        )

    def record(self, tool: str, arguments: dict, result: dict) -> None:
        entry = {"tool": tool, "arguments": arguments, **result}
        self.action_history.append(entry)
        if not result["success"]:
            return
        obs = result["observation"]
        if tool == "get_status":
            self.robot_status = deepcopy(obs)
        elif tool == "get_map":
            self.floor_plan = deepcopy(obs)
        elif tool == "move_to":
            self.robot_status["room"] = obs["room"]
        elif tool == "look":
            self.discoveries["room:" + obs["room"]] = entry
        elif tool == "talk_to":
            self.discoveries["conversation:" + result["evidence_id"]] = entry
        elif tool in {"give", "drop", "pick_up"}:
            self.discoveries["object:" + arguments["object"]] = entry
            inventory = self.robot_status.setdefault("inventory", [])
            if tool == "pick_up" and arguments["object"] not in inventory:
                inventory.append(arguments["object"])
            elif tool != "pick_up" and arguments["object"] in inventory:
                inventory.remove(arguments["object"])
        self.refresh_task_state()

    def compact(self) -> dict:
        # Bounded task memory: latest room/object facts, recent conversations/actions.
        facts = [v for k, v in self.discoveries.items() if not k.startswith("conversation:")]
        conversations = [v for k, v in self.discoveries.items() if k.startswith("conversation:")]
        observed_rooms = {v["observation"]["room"] for v in facts if v["tool"] == "look"}
        known_exits = {room for v in facts if v["tool"] == "look"
                       for room in v["observation"]["connections"]}
        known_exits |= set(self.floor_plan.get("rooms", {}))
        recent = self.action_history[-8:]
        recent_ids = {entry["evidence_id"] for entry in recent}
        older_discoveries = [entry for entry in facts + conversations[-8:]
                             if entry["evidence_id"] not in recent_ids]
        return {"goal": self.goal, "current_plan": self.current_plan,
                "floor_plan": {"rooms": {key: {"name": value["name"], "connections": value["connections"]}
                                           for key, value in self.floor_plan.get("rooms", {}).items()}},
                "delivery_state_from_observations": self.delivery_state(),
                "action_feedback": self.action_feedback,
                "interpreted_needs_unverified": self.interpreted_needs,
                "required_outcomes": check_conditions(self),
                "robot_status": self.robot_status, "discoveries": older_discoveries,
                "known_but_unobserved_rooms": sorted(known_exits - observed_rooms),
                "recent_actions": recent,
                "critic_feedback": self.critic_feedback[-2:],
                "explorer_reports_unverified": self.explorer_reports[-2:],
                "cycle_count": self.cycle_count, "tool_count": self.tool_count}

    def public(self) -> dict:
        payload = self.model_dump(exclude={"action_history", "discoveries", "critic_feedback", "explorer_reports"})
        payload["explored_rooms"] = sorted(self.explored_rooms)
        payload["last_progress_signature"] = list(self.last_progress_signature) if self.last_progress_signature else None
        return payload
