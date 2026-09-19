from app.llm.base import LLMClient, structured
from app.tasks.models import TaskContext
from .prompts import EXPLORER
from .schemas import ExplorerDecision


class Explorer:
    def __init__(self, client: LLMClient, tool_schemas: dict) -> None:
        self.client = client
        self.tool_schemas = tool_schemas

    async def decide(self, context: TaskContext, task: str) -> ExplorerDecision:
        return await structured(self.client, ExplorerDecision, EXPLORER,
                                {**context.compact(), "delegated_task": task, "tools": self.tool_schemas})
