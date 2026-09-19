from app.llm.base import LLMClient, structured
from app.tasks.models import TaskContext
from .prompts import COORDINATOR
from .schemas import CoordinatorDecision


class Coordinator:
    def __init__(self, client: LLMClient) -> None:
        self.client = client

    async def decide(self, context: TaskContext) -> CoordinatorDecision:
        return await structured(self.client, CoordinatorDecision, COORDINATOR, context.compact())
