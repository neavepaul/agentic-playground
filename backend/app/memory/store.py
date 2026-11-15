"""Persistent memory that survives across tasks and restarts.

Evidence gathered during a task is task-scoped. This store holds what lasts: Kip's
personality, his recent errands and his own track record. It is advisory only.

It lives as one document in the state table of kip.db.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from app.memory.durable import UnreadableStoreError
from app.persona.state import PersonaState


STATE_KEY = "persistent"


class SelfModel(BaseModel):
    """Kip's record of his own performance across tasks."""
    tool_failure_counts: dict[str, int] = Field(default_factory=dict)


class TaskRecord(BaseModel):
    task_id: str
    goal: str
    outcome: str
    timestamp: str
    summary: str


class PersistentMemory(BaseModel):
    version: int = 1
    recent_tasks: list[TaskRecord] = Field(default_factory=list)
    self_model: SelfModel = Field(default_factory=SelfModel)
    persona: PersonaState = Field(default_factory=PersonaState)
    # Monotonic count of finished tasks; recent_tasks is capped, this is not.
    tasks_recorded: int = 0
    _load_error: str = PrivateAttr(default="")
    _target: object = PrivateAttr(default=None)  # where this memory saves to

    @property
    def task_total(self) -> int:
        """Finished tasks ever recorded (never capped like recent_tasks)."""
        return max(self.tasks_recorded, len(self.recent_tasks))

    @model_validator(mode="after")
    def _count_existing_history(self):
        # Older saved state lacks the counter: start from the history it does contain.
        self.tasks_recorded = max(self.tasks_recorded, len(self.recent_tasks))
        return self

    @property
    def load_error(self) -> str:
        return self._load_error

    @classmethod
    def load(cls, store) -> PersistentMemory:
        """Load from kip.db's state table (a MemoryStore).

        Missing data starts a fresh memory. Unreadable data is NOT treated as empty: the
        app runs with a blank in-memory copy, the error is reported, and saving is refused
        so Kip's real memories stay untouched until fixed.
        """
        memory = cls()
        where = f"kip.db state '{STATE_KEY}'"
        text = store.get_state(STATE_KEY)
        if text is not None:
            try:
                memory = cls.model_validate_json(text)
            except Exception as exc:
                logging.getLogger("agentic_friend.memory").error(
                    "Persistent memory in %s could not be read (%s). It will not be overwritten.",
                    where, type(exc).__name__)
                memory._load_error = f"{where} could not be read ({type(exc).__name__}); saving is disabled."
        memory._target = store
        return memory

    def save(self) -> bool:
        """Save to the store this memory was loaded from.

        A memory that was never loaded (a bare PersistentMemory() in tests) has nowhere to
        go and is not saved. The app always loads it, so it always has a store.
        """
        if self._load_error:
            raise UnreadableStoreError(self._load_error)
        if self._target is None:
            return False
        self._target.set_state({STATE_KEY: self.model_dump_json(indent=2)})
        return True

    def update_from_task(self, context) -> None:
        """Record a finished task and any tool failures in it, whatever the outcome."""
        for entry in context.action_history:
            if not entry["success"] and not entry.get("refused"):
                self.self_model.tool_failure_counts[entry["tool"]] = (
                    self.self_model.tool_failure_counts.get(entry["tool"], 0) + 1)
        self.recent_tasks.append(TaskRecord(
            task_id=context.id,
            goal=context.goal,
            outcome=context.status,
            timestamp=datetime.now(timezone.utc).isoformat(),
            summary=context.summary,
        ))
        self.recent_tasks[:] = self.recent_tasks[-20:]
        self.tasks_recorded += 1

    def self_model_prompt(self) -> dict:
        """Compact self-model payload for the Coordinator. Empty when no task history exists."""
        if not self.recent_tasks:
            return {}
        by_outcome: dict[str, int] = {}
        for t in self.recent_tasks:
            by_outcome[t.outcome] = by_outcome.get(t.outcome, 0) + 1
        return {
            "note": "Kip's own performance history — use to plan more honestly.",
            "task_summary": {
                "total": len(self.recent_tasks),
                "completed": by_outcome.get("completed", 0),
                "failed": by_outcome.get("failed", 0),
                "cancelled": by_outcome.get("cancelled", 0),
            },
            "tool_failures": dict(
                sorted(self.self_model.tool_failure_counts.items(), key=lambda x: -x[1])[:5]
            ),
        }

    def prompt(self) -> dict:
        """Compact payload for the LLM. Advisory only."""
        return {
            "note": "Recent errands and how they went.",
            "recent_tasks": [{"goal": t.goal, "outcome": t.outcome, "summary": t.summary}
                             for t in self.recent_tasks[-5:]],
        }
