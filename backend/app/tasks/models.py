from copy import deepcopy
from uuid import uuid4
from typing import Literal

from pydantic import BaseModel, Field, computed_field

from app.agents.schemas import ConversationMeaning, GoalCondition
from .goals import check_conditions, known_recipients, normalize
from .memory import ReportedNeed, TaskMemory


class TaskContext(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    goal: str
    status: Literal["running", "completed", "failed", "cancelled"] = "running"
    summary: str = "Starting task."
    cycle_count: int = 0
    tool_count: int = 0
    critic_count: int = 0
    current_plan: str = ""
    memory: TaskMemory = Field(default_factory=TaskMemory)
    # Audit evidence is task-local and bounded by the execution tool budget.
    # No planner reconstructs current entity state from this transcript.
    action_history: list[dict] = Field(default_factory=list)
    critic_feedback: list[dict] = Field(default_factory=list)
    explorer_reports: list[str] = Field(default_factory=list)
    conditions: list[GoalCondition] = Field(default_factory=list)
    floor_plan: dict = Field(default_factory=dict)
    last_progress_signature: tuple | None = None
    action_feedback: list[str] = Field(default_factory=list)

    def feedback(self, message: str) -> None:
        self.action_feedback.append(message)
        self.action_feedback[:] = self.action_feedback[-6:]

    @computed_field
    @property
    def robot_status(self) -> dict:
        return ({"room": self.memory.robot_room, "inventory": self.memory.inventory()}
                if self.memory.robot_room is not None else {})

    @computed_field
    @property
    def interpreted_needs(self) -> list[dict]:
        return [claim.model_dump() for claim in self.memory.reported_needs.values()]

    @computed_field
    @property
    def explored_rooms(self) -> list[str]:
        return sorted(self.memory.rooms)

    @computed_field
    @property
    def discoveries(self) -> dict:
        """Legacy API audit view only; never used for planning or state updates."""
        entries = {}
        for entry in self.action_history:
            if not entry["success"]:
                continue
            tool, obs = entry["tool"], entry["observation"]
            if tool == "look":
                entries["room:" + obs["room"]] = entry
            elif tool == "talk_to":
                entries["conversation:" + entry["evidence_id"]] = entry
            elif tool in {"pick_up", "drop", "give"}:
                entries["object:" + obs["object"]] = entry
        return deepcopy(entries)

    def _object_location(self, key: str) -> tuple[str, str] | None:
        item = self.memory.objects.get(key)
        return (item.location.kind, item.location.id) if item and item.location else None

    def _person_location(self, key: str) -> str | None:
        person = self.memory.people.get(key)
        return person.location if person else None

    def _first_delivery(self) -> dict:
        return next(iter(self.delivery_state()), {})

    @computed_field
    @property
    def recipient(self) -> str | None:
        return self._first_delivery().get("recipient")

    @computed_field
    @property
    def recipient_location(self) -> str | None:
        return self._first_delivery().get("recipient_last_seen")

    @computed_field
    @property
    def item_location(self) -> str | None:
        location = self._first_delivery().get("last_observed_location")
        return location[1] if location else None

    @computed_field
    @property
    def carried_by(self) -> str | None:
        return "robot" if self._first_delivery().get("held") else None

    @computed_field
    @property
    def delivered(self) -> bool:
        return self._first_delivery().get("delivered", False)

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
                           "current_room_observed": room in self.memory.rooms,
                           "observed_on_floor_here": location == ("room", room),
                           "recipient_last_seen": self._person_location(recipient) if recipient else None,
                           "missing_prerequisites": missing, "delivered": delivered})
        return result

    @staticmethod
    def _mentions(text: str, alias: str) -> bool:
        return bool(alias) and " " + normalize(alias.replace("_", " ")) + " " in " " + normalize(text.replace("_", " ")) + " "

    def is_notification(self, person: str, message: str) -> bool:
        return any(c.kind in {"notify", "notify_everyone"} and (c.kind == "notify_everyone" or c.person == person)
                   and normalize(c.message) in normalize(message) for c in self.conditions)

    def conversation_topics(self, person: str, message: str) -> list[str]:
        if self.is_notification(person, message):
            return []
        aliases = {c.object: {c.object} for c in self.conditions if c.object}
        for key, item in self.memory.objects.items():
            aliases.setdefault(key, set()).update({key, item.name})
        for known in self.memory.people.values():
            for topic in known.asked_topics:
                aliases.setdefault(topic, set()).add(topic)
        return sorted(key for key, names in aliases.items() if any(self._mentions(message, name) for name in names))

    def repeat_reason(self, person: str, message: str) -> str | None:
        """Optional diagnostic only; this never authorizes or blocks speech."""
        known = self.memory.people.get(person)
        if known is None:
            return None
        topics = self.conversation_topics(person, message)
        for topic in topics:
            answer = known.asked_topics.get(topic)
            if answer and answer.state_stamp == self.memory.question_stamp(person, topic):
                return (f"{person} was already asked about {topic} (evidence {answer.evidence_id}). "
                        f"Their response was: {answer.response}. No relevant observed state changed.")
        if topics:
            return None  # A newly observed relevant change permits re-investigation.
        stamp = known.answered_messages.get(self.memory.message_key(message))
        if stamp is not None and stamp == self.memory.question_stamp(person):
            return f"{person} already received this message; no relevant observed state changed."
        return None

    def remember_meaning(self, evidence_id: str, meaning: ConversationMeaning) -> None:
        source = next((a for a in self.action_history if a["evidence_id"] == evidence_id
                       and a["success"] and a["tool"] == "talk_to"), None)
        if source is None:
            self.feedback("Conversation interpretation rejected: no successful speech evidence.")
            return
        obs = source["observation"]
        for need in meaning.needs:
            object_name = self.memory.objects.get(need.object)
            person_name = self.memory.people.get(need.person)
            object_grounded = self._mentions(need.quote, need.object) or bool(
                object_name and self._mentions(need.quote, object_name.name))
            person_grounded = (self._mentions(need.quote, need.person) or bool(
                person_name and self._mentions(need.quote, person_name.name)) or
                (need.person == obs["person"] and self._mentions(need.quote, "I")))
            if need.quote not in obs["response"] or not object_grounded or not person_grounded:
                self.feedback("Conversation interpretation rejected: quote or named entities lack speech evidence.")
                continue
            # Relations are explicitly unverified interpretations of real quoted speech.
            # They never create people, objects or physical locations.
            key = "|".join([obs["person"], need.object, need.person])
            self.memory.reported_needs[key] = ReportedNeed(**need.model_dump(), speaker=obs["person"], evidence_id=evidence_id)

    def progress_signature(self) -> tuple:
        return (self.memory.robot_room, tuple(self.memory.inventory()),
                tuple((k, v.revision) for k, v in sorted(self.memory.objects.items())),
                tuple((k, v.revision) for k, v in sorted(self.memory.people.items())),
                tuple(sorted(self.memory.rooms)))

    def record(self, tool: str, arguments: dict, result: dict) -> None:
        entry = deepcopy({"tool": tool, "arguments": arguments, **result})
        self.action_history.append(entry)
        if not result["success"]:
            return
        obs = entry["observation"]
        if tool == "get_map":
            self.floor_plan = deepcopy(obs)
            return
        topics = self.conversation_topics(obs["person"], obs["message"]) if tool == "talk_to" else []
        notification = tool == "talk_to" and self.is_notification(obs["person"], obs["message"])
        self.memory.observe(tool, entry["arguments"], obs, result["evidence_id"], topics, notification)

    def compact(self) -> dict:
        known_rooms = set(self.floor_plan.get("rooms", {})) | set(self.memory.rooms)
        known_rooms |= {exit for room in self.memory.rooms.values() for exit in room.connections}
        outcomes = check_conditions(self)
        memory = self.memory.prompt()
        memory["completed_outcomes"] = [c for c in outcomes if c["satisfied"]]
        return deepcopy({"goal": self.goal, "current_plan": self.current_plan,
                "floor_plan": {"rooms": {key: {"name": value["name"], "connections": value["connections"]}
                                           for key, value in self.floor_plan.get("rooms", {}).items()}},
                "task_memory": memory,
                "delivery_state_from_observations": self.delivery_state(),
                "action_feedback": self.action_feedback[-6:],
                "required_outcomes": outcomes, "robot_status": self.robot_status,
                "known_but_unobserved_rooms": sorted(known_rooms - set(self.memory.rooms)),
                "recent_actions": self.action_history[-8:],
                "critic_feedback": self.critic_feedback[-2:],
                "explorer_reports_unverified": self.explorer_reports[-2:],
                "cycle_count": self.cycle_count, "tool_count": self.tool_count})

    def public(self) -> dict:
        payload = self.model_dump(mode="json", exclude={"memory", "action_history", "discoveries", "critic_feedback", "explorer_reports"})
        memory = self.memory.prompt()
        memory["completed_outcomes"] = [condition for condition in check_conditions(self)
                                         if condition["satisfied"]]
        payload["task_memory"] = memory
        return payload
