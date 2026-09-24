from collections import deque
from copy import deepcopy
from uuid import uuid4
from typing import Literal

from pydantic import BaseModel, Field, computed_field

from app.agents.schemas import ConversationMeaning, GoalCondition
from .goals import check_conditions, known_recipients, normalize
from .memory import ReportedNeed, TaskMemory


class Intention(BaseModel):
    """A committed short-lived subgoal that survives routine execution steps.

    Routine hops toward `destination` are executed without re-deriving strategy.
    The intention is dropped when the target is found, the destination is
    reached and scanned, evidence contradicts it, or progress stalls.
    """
    kind: Literal["search_person", "search_object", "explore", "goto"] = "goto"
    target: str = ""
    destination: str = ""
    route: list[str] = Field(default_factory=list)
    reason: str = ""


class TaskMetrics(BaseModel):
    """Per-task execution counters used to compare inference cost across runs."""
    llm_calls: int = 0
    coordinator_llm_calls: int = 0
    explorer_llm_calls: int = 0
    critic_llm_calls: int = 0
    conversation_llm_calls: int = 0
    llm_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reflex_actions: int = 0
    route_actions: int = 0
    tool_actions: int = 0
    useful_observations: int = 0
    repeat_room_visits: int = 0
    stall_events: int = 0
    coordinator_handoffs: int = 0
    simulated_start: str = ""
    simulated_end: str = ""


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
    intention: Intention | None = None
    stall_count: int = 0
    # A room scanned this many actions ago is eligible for re-search: occupants move.
    search_stale_after: int = 12
    metrics: TaskMetrics = Field(default_factory=TaskMetrics)

    def feedback(self, message: str) -> None:
        self.action_feedback.append(message)
        self.action_feedback[:] = self.action_feedback[-6:]

    @computed_field
    @property
    def robot_status(self) -> dict:
        if self.memory.robot_room is None:
            return {}
        status = {"room": self.memory.robot_room, "inventory": self.memory.inventory()}
        # Only worlds with a clock report a time; unclocked worlds stay timeless.
        if self.memory.simulated_time:
            status["time"] = self.memory.simulated_time
        return status

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

    def scan_log(self) -> dict[str, int]:
        """Room -> action index of its most recent successful scan.

        The action index is the task's logical clock: deterministic, monotonic and
        independent of inference latency, so evidence age is reproducible.
        """
        log: dict[str, int] = {}
        for index, entry in enumerate(self.action_history):
            if entry["success"] and entry["tool"] == "look":
                log[entry["observation"]["room"]] = index
        return log

    def _clock_index(self) -> int:
        """Index of the latest action: the task's logical 'now'."""
        return max(len(self.action_history) - 1, 0)

    def _entity_room(self, kind: str, target: str) -> str | None:
        return (self._person_location(target) if kind == "person"
                else (lambda loc: loc[1] if loc and loc[0] == "room" else None)(self._object_location(target)))

    def locate_evidence(self, kind: str, target: str) -> dict:
        """Resolve where a target is, ranking evidence by kind and freshness.

        Priority: fresh direct sighting > fresh empty scan (contradiction) >
        reported speech > older direct sighting > historical tendency. Rooms whose
        last scan is older than `search_stale_after` stop counting as searched, so
        historical priors revive once search evidence expires.
        """
        now = self._clock_index()
        scans = self.scan_log()
        fresh = {room for room, index in scans.items() if now - index <= self.search_stale_after}
        observed = self._entity_room(kind, target)
        if observed:
            return {"room": observed, "source": "direct_observation",
                    "age_actions": now - scans.get(observed, now), "confidence": "observed"}
        store = self.long_term_memory.get("people" if kind == "person" else "objects", {}).get(target, {})
        historical = store.get("last_seen_room") or next(iter(store.get("typical_rooms") or []), None)
        scheduled = self._scheduled_room(target) if kind == "person" else None
        for room, source in ((scheduled, "schedule_prediction"), (historical, "long_term_memory")):
            # A fresh empty scan outranks any prior: we looked, and nobody was there.
            if room and room not in fresh:
                return {"room": room, "source": source,
                        "age_actions": now - scans.get(room, 0) if room in scans else None,
                        "confidence": "prior_verify_with_look"}
        return {"room": None, "source": "unknown", "age_actions": None, "confidence": "unknown"}

    def _scheduled_room(self, person: str) -> str | None:
        """Room this person is expected in at the current simulated hour, if known."""
        schedule = self.long_term_memory.get("_schedules", {}).get(person) or []
        time_str = self.robot_status.get("time") or ""
        if not schedule or ":" not in time_str:
            return None
        try:
            hours, minutes = time_str.split(":")
            hour = float(hours) + float(minutes) / 60.0
        except (ValueError, TypeError):
            return None
        for entry in schedule:
            start, end = entry["from_hour"], entry["to_hour"]
            if (start <= hour < end) if start < end else (hour >= start or hour < end):
                return entry["room"]
        return None

    def pursuit_targets(self) -> list[tuple[str, str]]:
        """(kind, id) pairs the robot must still travel to, located or not.

        A target whose room is already known is still pursued: knowing where the
        object is does not put it in the robot's hands.
        """
        targets: list[tuple[str, str]] = []
        for delivery in self.delivery_state():
            if delivery["delivered"]:
                continue
            if not delivery["held"]:
                targets.append(("object", delivery["object"]))
            if delivery.get("recipient"):
                targets.append(("person", delivery["recipient"]))
        for condition in self.conditions:
            if condition.kind == "find_object" and not self._object_location(condition.object):
                targets.append(("object", condition.object))
            if condition.kind in {"find_person", "notify"} and not self._person_location(condition.person):
                targets.append(("person", condition.person))
        return list(dict.fromkeys(targets))

    def unresolved_targets(self) -> list[tuple[str, str]]:
        """Pursuit targets whose current room is still unknown, so search applies."""
        return [(kind, target) for kind, target in self.pursuit_targets()
                if not self._entity_room(kind, target)]

    def search_state(self) -> list[dict]:
        """Explicit search coverage per unresolved target.

        Makes 'which rooms have I already cleared, and how stale is that' a
        first-class fact instead of something the model must recall from history.
        """
        current = self.robot_status.get("room")
        rooms = self.floor_plan.get("rooms", {})
        now = self._clock_index()
        scans = self.scan_log()
        result = []
        for kind, target in self.unresolved_targets():
            searched = {room: {"age_actions": now - index, "stale": now - index > self.search_stale_after}
                        for room, index in scans.items()}
            frontier = [room for room in rooms
                        if room not in scans or now - scans[room] > self.search_stale_after]
            if current:
                frontier.sort(key=lambda room: (len(self._find_path(current, room) or [99]), room))
            evidence = self.locate_evidence(kind, target)
            result.append({"target": target, "kind": kind,
                           "best_location_guess": evidence["room"],
                           "evidence_source": evidence["source"],
                           "searched_rooms": searched,
                           "unsearched_or_stale_rooms": frontier,
                           "active_destination": self.intention.destination
                           if self.intention and self.intention.target == target else None})
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
        """Deterministic next-hop hints toward exploration and task targets.

        Uses BFS over the floor_plan so the LLM never has to re-derive graph
        traversal from natural language. Each hint gives the immediate next room
        to enter on the shortest route, plus the evidence it rests on so stale
        priors are visibly weaker than fresh sightings.
        """
        current = self.robot_status.get("room")
        if not current or not self.floor_plan.get("rooms"):
            return {}
        hints: dict = {}
        for room_id in self.floor_plan["rooms"]:
            if room_id not in self.memory.rooms:
                path = self._find_path(current, room_id)
                if path:
                    hints[f"explore:{room_id}"] = {"target": room_id, "reason": "unobserved",
                                                    "next_hop": path[0], "full_path": path}
        prefix = {"object": "object", "person": "recipient"}
        for kind, target in self.pursuit_targets():
            evidence = self.locate_evidence(kind, target)
            room = evidence["room"]
            # Nothing credible left to aim at: fall back to the nearest room whose
            # search coverage is missing or expired, so search never stalls silently.
            if not room or room == current:
                coverage = next((s for s in self.search_state() if s["target"] == target), None)
                candidates = (coverage or {}).get("unsearched_or_stale_rooms", [])
                room = next((r for r in candidates if r != current), None)
                evidence = {"source": "search_frontier", "confidence": "unsearched_or_stale"}
            if not room or room == current:
                continue
            path = self._find_path(current, room)
            if path:
                hints[f"{prefix[kind]}:{target}"] = {
                    "target": room, "reason": evidence["source"], "next_hop": path[0],
                    "full_path": path, "confidence": evidence.get("confidence", "unknown")}
        return hints

    def progress_signature(self) -> tuple:
        """Semantic task state only.

        Deliberately excludes the robot's room: moving is not an achievement, so
        hall->kitchen->hall->kitchen without discoveries produces one unchanging
        signature and is detected as a stall.
        """
        return (tuple(self.memory.inventory()),
                tuple((k, v.revision) for k, v in sorted(self.memory.objects.items())),
                tuple((k, v.revision) for k, v in sorted(self.memory.people.items())),
                tuple(sorted(self.memory.rooms)),
                len(self.memory.reported_needs),
                # An accepted conversation turn is new information by construction:
                # the conversation layer already rejected repeats before execution.
                sum(len(thread.turns) for person in self.memory.people.values()
                    for thread in person.conversation_threads.values()))

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
        self._update_intention(tool, obs)

    def _update_intention(self, tool: str, obs: dict) -> None:
        """Advance or retire the active intention after a successful action."""
        plan = self.intention
        if plan is None:
            return
        if tool == "move_to":
            # Consume the hop we just took; anything else means we left the route.
            if plan.route and plan.route[0] == obs["room"]:
                plan.route.pop(0)
            elif obs["room"] not in plan.route:
                self.intention = None
                return
        # Reached the target, or reached and scanned the destination. Merely
        # learning where the target is does not end the journey to it.
        here = self.robot_status.get("room")
        found = plan.target and self._entity_room(
            "person" if plan.kind == "search_person" else "object", plan.target) == here
        arrived = here == plan.destination and plan.destination in self.memory.rooms
        if found or arrived or (not plan.route and plan.destination in self.memory.rooms):
            self.intention = None

    def adopt_intention(self, next_hop: str) -> Intention | None:
        """Commit to the destination implied by moving to `next_hop`.

        The chosen move identifies which navigation hint the agent is acting on;
        the rest of that route then executes without further inference.
        """
        hints = self._navigation_hints()
        kinds = {"recipient": "search_person", "object": "search_object", "explore": "explore"}
        ranked = sorted((h for key, h in hints.items() if h["next_hop"] == next_hop),
                        key=lambda h: h["reason"] == "unobserved")
        for key, hint in hints.items():
            if hint["next_hop"] != next_hop or (ranked and hint is not ranked[0]):
                continue
            prefix, _, target = key.partition(":")
            self.intention = Intention(kind=kinds.get(prefix, "goto"),
                                       target="" if prefix == "explore" else target,
                                       destination=hint["target"], route=list(hint["full_path"]),
                                       reason=hint["reason"])
            return self.intention
        return None

    def authoritative_targets(self) -> dict:
        """Object/recipient IDs fixed by the goal conditions themselves.

        These outrank anything in long-term memory: a remembered association
        between some other person and the same object must never redirect a task
        whose recipient the goal already names.
        """
        recipients = sorted({c.person for c in self.conditions if c.kind == "deliver" and c.person})
        return {"objects": sorted({c.object for c in self.conditions if c.object}),
                "recipients": recipients,
                "recipient_is_fixed_by_goal": bool(recipients),
                "note": "Long-term memory is supporting context only; it cannot change these."}

    def _scoped_long_term_memory(self) -> dict:
        """Long-term memory with claims that could override the explicit task removed."""
        store = deepcopy(self.long_term_memory)
        # Schedules are machine-read by locate_evidence; they are noise in the prompt.
        store.pop("_schedules", None)
        now = self._clock_index()
        fresh = {room for room, index in self.scan_log().items() if now - index <= self.search_stale_after}
        for item in store.get("objects", {}).values():
            # A room we just scanned is authoritative about what is not in it.
            if item.get("last_seen_room") in fresh:
                item.pop("last_seen_room", None)
        targets = self.authoritative_targets()
        if targets["recipient_is_fixed_by_goal"]:
            named = set(targets["recipients"])
            contested = set(targets["objects"])
            for person_id, person in store.get("people", {}).items():
                if person_id not in named:
                    person["known_needs"] = [n for n in person.get("known_needs", []) if n not in contested]
        return store

    def compact(self) -> dict:
        known_rooms = set(self.floor_plan.get("rooms", {})) | set(self.memory.rooms)
        known_rooms |= {exit for room in self.memory.rooms.values() for exit in room.connections}
        outcomes = check_conditions(self)
        memory = self.memory.prompt()
        long_term_memory = self._scoped_long_term_memory()
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
                "search_coverage": self.search_state(),
                "active_intention": self.intention.model_dump() if self.intention else None,
                "authoritative_targets": self.authoritative_targets(),
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
