from typing import Literal
import logging

from pydantic import ConfigDict, Field, create_model

from app.llm.base import LLMClient, structured
from app.tasks.models import TaskContext
from app.tasks.goals import known_recipients
from .prompts import EXPLORER
from .schemas import CommandChoice, ExplorerDecision


def observable_commands(context: TaskContext) -> tuple[dict, dict | None]:
    """Scope tools to observed targets and the Coordinator's fixed goal conditions.

    This never reads simulator state or interprets natural-language goal text.
    """
    room = context.robot_status.get("room")
    inventory = context.robot_status.get("inventory", [])
    movable = {c.object for c in context.conditions if c.kind in {"hold_object", "deliver", "place_object"}}
    needs = known_recipients(context)
    deliveries = {(c.object, c.person or needs.get(c.object)) for c in context.conditions if c.kind == "deliver"}
    view = context.memory.room_view(room)
    commands: dict = {"report": {"description": "Return discoveries or a blockage to Coordinator."},
                      "look": {"tool": "look", "arguments": {},
                               "description": "Scan the current room and refresh local visual observations."}}
    if view is None:
        commands["look"] = {"tool": "look", "arguments": {},
                            "description": "Observe this room; its contents and exits are not yet known."}
        if any(condition.kind == "deliver" for condition in context.conditions):
            commands.pop("report", None)
        return commands, None

    # Permit fresh scans on return; don't immediately repeat an unchanged scan.
    previous = context.action_history[-1] if context.action_history else None
    if previous and previous["success"] and previous["tool"] == "look":
        commands.pop("look", None)

    def add(tool: str, arguments: dict, description: str, priority: bool = False) -> None:
        id = ":".join([tool, *arguments.values()])
        entry: dict = {"tool": tool, "arguments": arguments, "description": description}
        if priority:
            entry["priority"] = True
        commands[id] = entry

    for exit in view["connections"]:
        add("move_to", {"room": exit}, f"Move from {room} to the connected room {exit}.")

    # Identify objects whose pick_up directly satisfies an unmet acquire_object prerequisite
    # so the affordance layer can surface this as a priority action rather than one equal choice.
    unmet_acquires = {d["object"] for d in context.delivery_state()
                      if "acquire_object" in d.get("missing_prerequisites", [])}
    # When a delivery recipient still needs to be located, promote look to priority so the
    # robot scans the current room before moving away. This prevents hall↔room loops where
    # the model moves without ever verifying whether the recipient has arrived here.
    needs_recipient_scan = any("locate_recipient" in d.get("missing_prerequisites", [])
                               for d in context.delivery_state())
    if needs_recipient_scan and "look" in commands:
        commands["look"]["priority"] = True
        commands["look"]["description"] = (
            "PRIORITY — Recipient location unknown. Scan this room first to check "
            "whether they are present before navigating elsewhere."
        )

    for item in view["objects"]:
        if item.get("portable", True) and item["id"] in movable:
            is_priority = item["id"] in unmet_acquires
            desc = (f"PRIORITY — {item['id']} is the required object and is visible here. "
                    "Pick it up to satisfy the acquire_object prerequisite."
                    if is_priority else f"Take visible {item['id']} into inventory.")
            add("pick_up", {"object": item["id"]}, desc, priority=is_priority)

    for person in view["people"]:
        if context.conversation_rejections.get(person["id"], 0) >= 1:
            continue
        add("talk_to", {"person": person["id"]},
            f"Speak to {person['id']} with a useful unanswered question or message. "
            "Do not use speech to announce your plan; put that in summary. This transfers no objects.")
    for item in inventory:
        if item not in movable:
            continue
        add("drop", {"object": item}, f"Place held {item} in this room.")
        for person in view["people"]:
            if (item, person["id"]) in deliveries:
                add("give", {"object": item, "person": person["id"]},
                    f"Transfer held {item} to {person['id']} in this room.")
    if context.conversation_rejections:
        physical = {key: command for key, command in commands.items()
                    if command.get("tool") in {"move_to", "pick_up", "drop", "give"}}
        if physical:
            commands = physical
    return commands, view


class Explorer:
    def __init__(self, client: LLMClient, tool_schemas: dict) -> None:
        self.client = client
        self.tool_names = set(tool_schemas)

    async def decide(self, context: TaskContext, task: str) -> ExplorerDecision:
        commands, _ = observable_commands(context)
        commands = {id: command for id, command in commands.items()
                    if id == "report" or command["tool"] in self.tool_names}
        logging.getLogger("agentic_friend.decisions").info(
            "Explorer context: task=%s room=%s inventory=%s commands=%s delivery=%s",
            context.id, context.robot_status.get("room"), context.robot_status.get("inventory", []),
            list(commands), context.delivery_state(),
        )
        # Give commands already require a held goal object and an observed,
        # identified recipient. Surface this affordance without a second memory
        # store, simulator access, or taking action on the model's behalf.
        ready_handoffs = [id for id, command in commands.items() if command.get("tool") == "give"]
        # Priority actions directly satisfy an immediately actionable prerequisite
        # (e.g. pick_up when the required object is visible and not yet held).
        # Surface them explicitly so the LLM does not treat them as equal choices.
        priority_actions = [id for id, command in commands.items() if command.get("priority")]
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
        payload = context.compact()
        # Reports are model claims, not observations. The Coordinator receives
        # them as handoffs; replaying them to Explorer reinforces false discoveries.
        payload.pop("explorer_reports_unverified", None)
        # Current facts already live in task_memory. Retain the recent action
        # sequence and failures without replaying maps and stale room snapshots.
        payload["recent_actions"] = [
            {key: action[key] for key in ("tool", "arguments", "success", "error") if key in action}
            for action in context.action_history[-8:]
        ]
        choice = await structured(self.client, schema, EXPLORER,
                                  {**payload, "delegated_task": task,
                                   "ready_handoffs": ready_handoffs,
                                   "priority_actions": priority_actions,
                                   "commands": commands},
                                  repair_hint="For talk_to, include message with the actual nonblank words to speak. "
                                              "Putting those words in summary does not supply message.")
        logging.getLogger("agentic_friend.decisions").info(
            "Explorer choice: task=%s command=%s summary=%s", context.id, choice.command_id, choice.summary)
        if choice.command_id == "report":
            return ExplorerDecision(action="report", summary=choice.summary)
        command = commands[choice.command_id]
        arguments = dict(command["arguments"])
        if command["tool"] == "talk_to":
            arguments["message"] = choice.message
        return ExplorerDecision(action="tool", tool=command["tool"],
                                arguments=arguments, summary=choice.summary)
