"""Persistent memory that survives across tasks.

Observations made during a task are task-scoped and ephemeral. This store
holds durable facts: where people tend to be, where objects are usually
found, and what each person has needed in the past. It is advisory only —
the robot must still verify everything through live observation.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field


class PersonMemory(BaseModel):
    name: str
    location_tally: dict[str, int] = Field(default_factory=dict)
    last_seen_room: str | None = None
    known_needs: list[str] = Field(default_factory=list)
    delivery_counts: dict[str, int] = Field(default_factory=dict)


class ObjectMemory(BaseModel):
    name: str
    last_seen_room: str | None = None


class TaskRecord(BaseModel):
    task_id: str
    goal: str
    outcome: str
    timestamp: str
    summary: str


class PersistentMemory(BaseModel):
    version: int = 1
    people: dict[str, PersonMemory] = Field(default_factory=dict)
    objects: dict[str, ObjectMemory] = Field(default_factory=dict)
    recent_tasks: list[TaskRecord] = Field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> PersistentMemory:
        """Load from disk; return a blank instance if the file is missing or corrupt."""
        if path.exists():
            try:
                return cls.model_validate_json(path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return cls()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    def update_from_task(self, context) -> None:
        """Extract durable facts from a finished task and merge them in.

        Called after every task regardless of outcome so partial explorations
        still contribute location knowledge even when the task fails.
        """
        for person_id, person in context.memory.people.items():
            pm = self.people.setdefault(person_id, PersonMemory(name=person.name or person_id))
            if person.name:
                pm.name = person.name
            if person.location:
                pm.last_seen_room = person.location
                pm.location_tally[person.location] = pm.location_tally.get(person.location, 0) + 1

        for obj_id, obj in context.memory.objects.items():
            om = self.objects.setdefault(obj_id, ObjectMemory(name=obj.name or obj_id))
            if obj.name:
                om.name = obj.name
            if obj.last_observed_room:
                om.last_seen_room = obj.last_observed_room

        # Successful deliveries reveal recurring needs.
        for delivery in context.delivery_state():
            if delivery.get("delivered") and delivery.get("recipient"):
                pm = self.people.setdefault(delivery["recipient"],
                                            PersonMemory(name=delivery["recipient"]))
                obj = delivery["object"]
                if obj not in pm.known_needs:
                    pm.known_needs.append(obj)
                pm.delivery_counts[obj] = pm.delivery_counts.get(obj, 0) + 1

        self.recent_tasks.append(TaskRecord(
            task_id=context.id,
            goal=context.goal,
            outcome=context.status,
            timestamp=datetime.now(timezone.utc).isoformat(),
            summary=context.summary,
        ))
        self.recent_tasks[:] = self.recent_tasks[-20:]

    def prompt(self) -> dict:
        """Compact payload for the LLM. Advisory only — stale by definition."""
        people = {}
        for pid, p in self.people.items():
            typical = sorted(p.location_tally, key=lambda r: -p.location_tally[r])[:2]
            people[pid] = {
                "name": p.name,
                "last_seen_room": p.last_seen_room,
                "typical_rooms": typical,
                "known_needs": p.known_needs,
            }
        return {
            "note": "Prior observations — verify with look before acting on them.",
            "people": people,
            "objects": {oid: {"name": o.name, "last_seen_room": o.last_seen_room}
                        for oid, o in self.objects.items()},
            "recent_tasks": [{"goal": t.goal, "outcome": t.outcome, "summary": t.summary}
                             for t in self.recent_tasks[-5:]],
        }
