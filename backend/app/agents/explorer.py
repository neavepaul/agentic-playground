from typing import Literal

from pydantic import Field, create_model

from app.llm.base import LLMClient, structured
from app.tasks.models import TaskContext
from .prompts import EXPLORER
from .schemas import ExplorerDecision


class Explorer:
    def __init__(self, client: LLMClient, tool_schemas: dict) -> None:
        self.client = client
        self.tool_schemas = tool_schemas

    async def decide(self, context: TaskContext, task: str) -> ExplorerDecision:
        allowed = list(self.tool_schemas)
        # Two identical reads cannot add information in this deterministic world.
        # Constrain only that no-op; route, conversation and action choices remain the model's.
        if context.action_history:
            last = context.action_history[-1]
            if last["success"] and last["tool"] in {"look", "get_status"}:
                allowed.remove(last["tool"])
                if last["tool"] == "look":
                    allowed.remove("get_status")
        step_schema = create_model(
            "ExplorerStep", __base__=ExplorerDecision,
            tool=(Literal[tuple(["", *allowed])], Field(default="")),
        )
        return await structured(self.client, step_schema, EXPLORER,
                                {**context.compact(), "delegated_task": task,
                                 "allowed_tools_this_step": allowed,
                                 "tools": {name: self.tool_schemas[name] for name in allowed}})
