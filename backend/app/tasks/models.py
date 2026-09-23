from collections import deque
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
    conversation_rejections: dict[str, int] = Field(default_factory=dict)
    consecutive_report_rejections: int = 0
    long_term_memory: dict = Field(default_factory=dict)
    self_model: dict = Field(default_factory=dict)

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

    def _find_path(self, start: str, target: str) -> list[str]:
        """BFS shortest path over the floor_plan graph.

        Returns the list of rooms from start (exclusive) to target (inclusive),
        so result[0] is the immediate next hop. Returns [] when start == target
        or either room is unknown or the target is unreachable.
        """
        rooms = self.floor_plan.get("rooms", {})
        if not rooms or start == target or start not in rooms or target not in rooms:
            return []
        queue: deque[list[str]] = deque([[start]])
        seen: set[str] = {start}
        while queue:
            path = queue.popleft()
            for neighbor in rooms.get(path[-1], {}).get("connections", []):
                if neighbor not in seen:
                    if neighbor == target:
                        return path[1:] + [neighbor]
                    seen.add(neighbor)
                    queue.append(path + [neighbor])
        return []

    def _navigation_hints(self) -> dict:
        """Deterministic next-hop hints toward exploration and delivery targets.

        Uses BFS over the floor_plan so the LLM never has to re-derive graph
        traversal from natural language. Each hint gives the immediate next room
        to enter on the shortest route to a target.
        """
        current = self.robot_status.get("room")
        if not current or not self.floor_plan.get("rooms"):
            return {}
        observed = set(self.memory.rooms)
        hints: dict = {}
        for room_id in self.floor_plan["rooms"]:
            if room_id not in observed:
                path = self._find_path(current, room_id)
                if path:
                    hints[f"explore:{room_id}"] = {"target": room_id, "reason": "unobserved",
                                                    "next_hop": path[0], "full_path": path}
        ltm_people = self.long_term_memory.get("people", {})
        ltm_objects = self.long_term_memory.get("objects", {})
        for delivery in self.delivery_state():
            obj_id = delivery["object"]
            loc = delivery.get("last_observed_location")
            if loc and loc[0] == "room" and loc[1] != current:
                path = self._find_path(current, loc[1])
                if path:
                    hints[f"object:{obj_id}"] = {
                        "target": loc[1], "reason": "last_observed_object_location",
                        "next_hop": path[0], "full_path": path}
            elif not loc:
                # No task-scoped observation yet; fall back to long-term memory.
                ltm_room = ltm_objects.get(obj_id, {}).get("last_seen_room")
                already_scanned = any(a["success"] and a["tool"] == "look"
                                      and a["observation"]["room"] == ltm_room
                                      for a in self.action_history)
                if ltm_room and ltm_room != current and not already_scanned:
                    path = self._find_path(current, ltm_room)
                    if path:
                        hints[f"object:{obj_id}"] = {
                            "target": ltm_room, "reason": "long_term_memory_last_seen",
                            "next_hop": path[0], "full_path": path,
                            "confidence": "prior_observation_verify_with_look"}

            recipient = delivery.get("recipient")
            recipient_loc = delivery.get("recipient_last_seen")
            if recipient and recipient_loc and recipient_loc != current:
                path = self._find_path(current, recipient_loc)
                if path:
                    hints[f"recipient:{recipient}"] = {
                        "target": recipient_loc, "reason": "last_seen_recipient",
                        "next_hop": path[0], "full_path": path}
            elif recipient and not recipient_loc:
                # Not seen this task; fall back to long-term memory.
                ltm = ltm_people.get(recipient, {})
                ltm_room = ltm.get("last_seen_room") or (ltm.get("typical_rooms") or [None])[0]
                # Suppress the hint only for rooms scanned AFTER the object was picked up.
                # Pre-pickup looks are stale by the time delivery starts — the recipient
                # may have moved in the interim, so we should not exclude those rooms.
                last_pickup = next(
                    (i for i in range(len(self.action_history) - 1, -1, -1)
                     if self.action_history[i]["success"]
                     and self.action_history[i]["tool"] == "pick_up"
                     and self.action_history[i]["arguments"].get("object") == obj_id),
                    -1
                )
                rooms_scanned_post_pickup = {
                    a["observation"]["room"]
                    for a in self.action_history[last_pickup + 1:]
                    if a["success"] and a["tool"] == "look"
                }
                already_checked = ltm_room in rooms_scanned_post_pickup
                if ltm_room and ltm_room != current and not already_checked:
                    path = self._find_path(current, ltm_room)
                    if path:
                        hints[f"recipient:{recipient}"] = {
                            "target": ltm_room, "reason": "long_term_memory_last_seen",
                            "next_hop": path[0], "full_path": path,
                            "confidence": "prior_observation_verify_with_look"}
                # LTM room already verified empty — fall back to schedule prediction.
                if f"recipient:{recipient}" not in hints:
                    schedules = self.long_term_memory.get("_schedules", {})
                    rec_schedule = schedules.get(recipient, [])
                    time_str = self.robot_status.get("time", "")
                    if rec_schedule and time_str:
                        try:
                            h, m = time_str.split(":")
                            hour = float(h) + float(m) / 60.0
                            predicted_room = None
                            for entry in rec_schedule:
                                fh, th = entry["from_hour"], entry["to_hour"]
                                in_range = (fh <= hour < th) if fh < th else (hour >= fh or hour < th)
                                if in_range:
                                    predicted_room = entry["room"]
                                    break
                            if (predicted_room and predicted_room != current
                                    and predicted_room not in rooms_scanned_post_pickup):
                                path = self._find_path(current, predicted_room)
                                if path:
                                    hints[f"recipient:{recipient}"] = {
                                        "target": predicted_room,
                                        "reason": "schedule_predicted_location",
                                        "next_hop": path[0], "full_path": path,
                                        "confidence": "schedule_prediction_verify_with_look"}
                        except (ValueError, TypeError, AttributeError):
                            pass
        return hints

    def progress_signature(self) -> tuple:
        return (self.memory.robot_room, tuple(self.memory.inventory()),
                tuple((k, v.revision) for k, v in sorted(self.memory.objects.items())),
                tuple((k, v.revision) for k, v in sorted(self.memory.people.items())),
                tuple(sorted(self.memory.rooms)))

    def record(self, tool: str, arguments: dict, result: dict) -> None:
        entry = deepcopy({"tool": tool, "arguments": arguments, **result})
        self.action_history.append(entry)
        if not result["success"]:
            person = arguments.get("person")
            if tool in {"talk_to", "give"} and person and result.get("error") == f"{person} is not in the current room.":
                self.memory.set_person(person, None, result["evidence_id"])
                self.memory.rooms.pop(self.memory.robot_room, None)
            return
        self.consecutive_report_rejections = 0
        if tool in {"move_to", "pick_up", "drop", "give"}:
            self.conversation_rejections.clear()
        elif tool == "talk_to":
            self.conversation_rejections.pop(arguments["person"], None)
        obs = entry["observation"]
        if tool == "get_map":
            self.floor_plan = deepcopy(obs)
            return
        notification = tool == "talk_to" and self.is_notification(obs["person"], obs["message"])
        self.memory.observe(tool, entry["arguments"], obs, result["evidence_id"], notification)

    def compact(self) -> dict:
        known_rooms = set(self.floor_plan.get("rooms", {})) | set(self.memory.rooms)
        known_rooms |= {exit for room in self.memory.rooms.values() for exit in room.connections}
        outcomes = check_conditions(self)
        memory = self.memory.prompt()
        long_term_memory = deepcopy(self.long_term_memory)
        scanned_rooms = {a["observation"]["room"] for a in self.action_history
                         if a["success"] and a["tool"] == "look"}
        for item in long_term_memory.get("objects", {}).values():
            if item.get("last_seen_room") in scanned_rooms:
                item.pop("last_seen_room", None)
        memory["completed_outcomes"] = [c for c in outcomes if c["satisfied"]]
        room_view = self.memory.room_view(self.robot_status.get("room"))
        if room_view is not None:
            room_view["objects_held_by_people"] = room_view.pop("held_objects")
            room_view["robot_inventory"] = self.memory.inventory()
        return deepcopy({"goal": self.goal, "current_plan": self.current_plan,
                "floor_plan": {"rooms": {key: {"name": value["name"], "connections": value["connections"],
                                               "observed": key in self.memory.rooms}
                                           for key, value in self.floor_plan.get("rooms", {}).items()}},
                "task_memory": memory,
                "current_room_observation": room_view,
                "delivery_state_from_observations": self.delivery_state(),
                "navigation_hints": self._navigation_hints(),
                "long_term_memory": long_term_memory,
                "self_model": self.self_model,
                "action_feedback": self.action_feedback[-6:],
                "required_outcomes": outcomes, "robot_status": self.robot_status,
                "known_but_unobserved_rooms": sorted(known_rooms - set(self.memory.rooms)),
                "recent_actions": self.action_history[-8:],
                "critic_feedback": [f for f in self.critic_feedback
                                    if f.get("observed_action_count") == len(self.action_history)
                                    and f.get("kind") != "delegation_report"][-2:],
                "explorer_reports_unverified": self.explorer_reports[-2:],
                "cycle_count": self.cycle_count, "tool_count": self.tool_count})

    def public(self) -> dict:
        payload = self.model_dump(mode="json", exclude={"memory", "action_history", "discoveries", "critic_feedback", "explorer_reports"})
        memory = self.memory.prompt()
        memory["completed_outcomes"] = [condition for condition in check_conditions(self)
                                         if condition["satisfied"]]
        payload["task_memory"] = memory
        return payload
