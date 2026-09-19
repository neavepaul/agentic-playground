from typing import Literal

from pydantic import create_model

from app.llm.base import LLMClient, ModelError, structured
from app.tasks.goals import normalize
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
        answered = [a for a in context.action_history if a["success"] and a["tool"] == "talk_to"]
        for attempt in range(2):
            choice = await structured(self.client, schema, EXPLORER,
                                      {**context.compact(), "delegated_task": task,
                                       "current_room_observation": view, "commands": commands})
            command = commands[choice.command_id]
            duplicate = next((a for a in answered
                              if command.get("tool") == "talk_to"
                              and a["arguments"]["person"] == command["arguments"]["person"]
                              and normalize(a["arguments"]["message"]) == normalize(choice.message)), None)
            if duplicate is None:
                break
            context.feedback(
                f"Repeated message rejected: {duplicate['arguments']['person']} already answered "
                f"this message (evidence {duplicate['evidence_id']}). Read that response. "
                "Choose a different useful action; a different message to the same person is allowed. "
                "If a requested object is neither held nor observed here, search elsewhere using the map.")
            if attempt:
                raise ModelError("Explorer repeated an answered message after correction; stopped without another Critic loop.")
        if choice.command_id == "report":
            return ExplorerDecision(action="report", summary=choice.summary)
        command = commands[choice.command_id]
        arguments = dict(command["arguments"])
        if command["tool"] == "talk_to":
            arguments["message"] = choice.message
        return ExplorerDecision(action="tool", tool=command["tool"],
                                arguments=arguments, summary=choice.summary)
