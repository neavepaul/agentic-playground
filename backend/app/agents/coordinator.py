import re

from app.llm.base import LLMClient, ModelError, structured
from app.tasks.models import TaskContext
from .prompts import COORDINATOR, GOAL_PLANNER
from .schemas import CoordinatorDecision, GoalPlan


class Coordinator:
    def __init__(self, client: LLMClient) -> None:
        self.client = client

    async def decide(self, context: TaskContext) -> CoordinatorDecision:
        return await structured(self.client, CoordinatorDecision, COORDINATOR, context.compact())

    async def define_goal(self, goal: str) -> GoalPlan:
        prompt = GOAL_PLANNER
        # A narrow consistency check for an explicit delivery clause. This rejects
        # prerequisite-only plans without hardcoding an object, recipient or route.
        explicit_delivery = re.search(r"(?:^\s*(?:please\s+)?|\b(?:and|then)\s+)deliver\b", goal, re.I)
        for attempt in range(2):
            plan = await structured(self.client, GoalPlan, prompt, {"goal": goal})
            prerequisite_only = all(c.kind in {"find_object", "find_person", "identify_recipient", "visit_room"}
                                    for c in plan.conditions)
            if not (explicit_delivery and prerequisite_only):
                return plan
            prompt += "\nYour prior plan omitted DELIVERY. Define the FINAL outcome for the ENTIRE goal: " \
                      "a deliver condition for the requested object, not just finding it or identifying its recipient."
        raise ModelError("Could not define complete goal conditions: the delivery requirement was omitted.")
