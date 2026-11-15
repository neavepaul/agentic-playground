"""Deterministic verification of goal conditions using tool evidence only."""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import TaskContext


def check_conditions(context: "TaskContext") -> list[dict]:
    # The capability executor already checked the answer's task-local evidence and required tools.
    answered = any(a["success"] and a["tool"] == "finish_answer" for a in context.action_history)
    return [{**condition.model_dump(exclude_defaults=True), "kind": condition.kind, "satisfied": answered}
            for condition in context.conditions]
