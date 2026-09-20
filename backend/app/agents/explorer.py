from typing import Literal

from pydantic import ConfigDict, Field, create_model

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
        commands, _ = observable_commands(context)
        commands = {id: command for id, command in commands.items()
                    if id == "report" or command["tool"] in self.tool_names}
        # Give commands already require a held goal object and an observed,
        # identified recipient. Surface this affordance without a second memory
        # store, simulator access, or taking action on the model's behalf.
        ready_handoffs = [id for id, command in commands.items() if command.get("tool") == "give"]
        fields = {"command_id": (Literal[tuple(commands)], ...)}
        if all(command.get("tool") == "talk_to" for command in commands.values()):
            fields["message"] = (str, Field(min_length=1, max_length=500, pattern=r"\S",
                                            description="Required actual words to say to the person."))
        speaking = [id for id, command in commands.items() if command.get("tool") == "talk_to"]
        other = [id for id in commands if id not in speaking]
        branches = []
        if speaking:
            branches.append({"type": "object", "properties": {"command_id": {"enum": speaking},
                                            "message": {"type": "string", "minLength": 1,
                                                        "maxLength": 500, "pattern": r"\S"}},
                             "required": ["command_id", "message"]})
        if other:
            branches.append({"type": "object", "properties": {"command_id": {"enum": other}}, "required": ["command_id"]})
        def constrain_speech(schema: dict) -> None:
            # Some constrained decoders compile anyOf branches independently of
            # sibling properties. Each branch must carry the entire object schema.
            for branch in branches:
                branch["properties"] = {**schema["properties"], **branch["properties"]}
                branch["required"] = list(dict.fromkeys([*schema.get("required", []), *branch["required"]]))
                branch["additionalProperties"] = False
            schema["anyOf"] = branches

        schema = create_model("ExplorerCommand", __base__=CommandChoice,
                              __config__=ConfigDict(json_schema_extra=constrain_speech), **fields)
        choice = await structured(self.client, schema, EXPLORER,
                                  {**context.compact(), "delegated_task": task,
                                   "ready_handoffs": ready_handoffs,
                                   "commands": commands},
                                  repair_hint="For talk_to, include message with the actual nonblank words to speak. "
                                              "Putting those words in summary does not supply message.")
        if choice.command_id == "report":
            return ExplorerDecision(action="report", summary=choice.summary)
        command = commands[choice.command_id]
        arguments = dict(command["arguments"])
        if command["tool"] == "talk_to":
            arguments["message"] = choice.message
        return ExplorerDecision(action="tool", tool=command["tool"],
                                arguments=arguments, summary=choice.summary)
