import re

from app.llm.base import LLMClient, ModelError, structured
from app.tasks.models import TaskContext
from .prompts import COORDINATOR, GOAL_PLANNER
from .schemas import CoordinatorDecision, GoalCondition, GoalPlan


class Coordinator:
    def __init__(self, client: LLMClient, digital_enabled: bool = False) -> None:
        self.client = client
        self.digital_enabled = digital_enabled

    async def decide(self, context: TaskContext) -> CoordinatorDecision:
        return await structured(self.client, CoordinatorDecision, COORDINATOR, context.compact(),
                                metrics=context.metrics, role="coordinator")

    @staticmethod
    def _requires_recipient_discovery(goal: str) -> bool:
        return bool(re.search(
            r"\b(?:find(?:\s+out)?|determine|discover|identify)\s+who\s+needs\b|\bwhoever\s+needs\b",
            goal,
            re.I,
        ))

    async def define_goal(self, goal: str, floor_plan: dict | None = None) -> GoalPlan:
        prompt = GOAL_PLANNER
        if self.digital_enabled:
            from .prompts import DIGITAL_GOAL_PLANNER
            prompt += DIGITAL_GOAL_PLANNER
        # Goal interpretation needs room names/IDs, not renderer geometry or routes.
        # Keep the original observed map intact for navigation and the frontend.
        room_names = {room_id: room.get("name", room_id)
                      for room_id, room in (floor_plan or {}).get("rooms", {}).items()}
        unknown_recipient = self._requires_recipient_discovery(goal)
        # A narrow consistency check for an explicit delivery clause. This rejects
        # prerequisite-only plans without hardcoding an object, recipient or route.
        explicit_delivery = re.search(r"(?:^\s*(?:please\s+)?|\b(?:and|then)\s+)deliver\b", goal, re.I)
        for attempt in range(2):
            plan = await structured(self.client, GoalPlan, prompt,
                                    {"goal": goal, "floor_plan": {"rooms": room_names}},
                                    role="coordinator")
            if unknown_recipient:
                plan = plan.model_copy(update={
                    "conditions": [
                        condition.model_copy(update={"person": ""})
                        if condition.kind == "deliver" else condition
                        for condition in plan.conditions
                    ]
                })
            prerequisite_only = all(c.kind in {"find_object", "find_person", "identify_recipient", "visit_room"}
                                    for c in plan.conditions)
            if not (explicit_delivery and prerequisite_only):
                return plan
            prompt += "\nYour prior plan omitted DELIVERY. Define the FINAL outcome for the ENTIRE goal: " \
                      "a deliver condition for the requested object, not just finding it or identifying its recipient."
        object_id = next((condition.object for condition in plan.conditions if condition.object), None)
        if explicit_delivery and object_id:
            person = (
                "" if unknown_recipient
                else next((condition.person for condition in plan.conditions if condition.person), "")
            )
            return plan.model_copy(update={
                "summary": plan.summary + " Delivery is required and will be verified after a successful handoff.",
                "conditions": [*plan.conditions, GoalCondition(kind="deliver", object=object_id, person=person)],
            })
        raise ModelError("Could not define complete goal conditions: the delivery requirement was omitted.")
