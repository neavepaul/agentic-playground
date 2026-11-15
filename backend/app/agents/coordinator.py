from app.llm.base import LLMClient, structured
from app.tasks.models import TaskContext
from .prompts import COORDINATOR, GOAL_PLANNER, PYTHON_PROGRAM
from .schemas import CoordinatorDecision, GoalCondition, GoalPlan, InitialCapability, PythonProgram


class Coordinator:
    def __init__(self, client: LLMClient, hosted=None, hosted_calls: int = 0) -> None:
        self.client = client
        # Goal planning is one high-value call per errand: worth a hosted model when
        # one is configured. It sees only the goal, never memory.
        self.hosted, self.hosted_calls = hosted, hosted_calls

    async def decide(self, context: TaskContext) -> CoordinatorDecision:
        return await structured(self.client, CoordinatorDecision, COORDINATOR, context.compact(),
                                metrics=context.metrics, role="coordinator")

    def _planner(self):
        if self.hosted is not None and self.hosted_calls:
            from app.llm.groq import HostedFirst
            return HostedFirst(self.hosted, self.client, self.hosted_calls, "goal planning")
        return self.client

    async def define_goal(self, goal: str, metrics=None) -> GoalPlan:
        return await structured(self._planner(), GoalPlan, GOAL_PLANNER, {"goal": goal},
                                role="coordinator", metrics=metrics, compact_schema=True)

    async def python_plan(self, goal: str, metrics=None) -> GoalPlan:
        """A request that only asks for Python: the plan is known, the model writes just the code."""
        program = await structured(self._planner(), PythonProgram, PYTHON_PROGRAM, {"request": goal},
                                   role="coordinator", metrics=metrics, compact_schema=True)
        return GoalPlan(summary="Run the program.", conditions=[GoalCondition(kind="answer", required_tools=["run_python"])],
                        initial_actions=[InitialCapability(tool="run_python", arguments={"code": program.code})])
