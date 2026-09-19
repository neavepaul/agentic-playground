from typing import Literal

from pydantic import create_model

from app.llm.base import LLMClient, structured
from app.tasks.models import TaskContext
from .commands import observable_commands
from .prompts import EXPLORER
from .schemas import CommandChoice, ExplorerDecision


class Explorer:
    def __init__(self, client: LLMClient, tool_schemas: dict) -> None:
        self.client = client
        self.tool_names = set(tool_schemas)

    async def decide(self, context: TaskContext, task: str) -> ExplorerDecision:
        commands, view = observable_commands(context)
        commands = {id: command for id, command in commands.items()
                    if id == "report" or command["tool"] in self.tool_names}
        schema = create_model("ExplorerCommand", __base__=CommandChoice,
                              command_id=(Literal[tuple(commands)], ...))
        choice = await structured(self.client, schema, EXPLORER,
                                  {**context.compact(), "delegated_task": task,
                                   "current_room_observation": view, "commands": commands})
        if choice.command_id == "report":
            return ExplorerDecision(action="report", summary=choice.summary)
        command = commands[choice.command_id]
        arguments = dict(command["arguments"])
        if command["tool"] == "talk_to":
            arguments["message"] = choice.message
        return ExplorerDecision(action="tool", tool=command["tool"],
                                arguments=arguments, summary=choice.summary)
