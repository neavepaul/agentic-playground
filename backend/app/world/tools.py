import logging
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.events.bus import EventBus
from .engine import WorldEngine, WorldError

Id = Annotated[str, Field(min_length=1, max_length=40, pattern=r"^[a-z][a-z0-9_]*$")]


class Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class MoveArguments(Arguments):
    room: Id


class ObjectArguments(Arguments):
    object: Id


class GiveArguments(ObjectArguments):
    person: Id


class TalkArguments(Arguments):
    person: Id
    message: str = Field(min_length=1, max_length=500)


# Explicit allowlist: model strings are never resolved as Python attributes.
TOOL_REGISTRY = {
    "get_map": (Arguments, WorldEngine.get_map, "map_observed"),
    "get_status": (Arguments, WorldEngine.get_status, "status_observed"),
    "look": (Arguments, WorldEngine.look, "room_observed"),
    "move_to": (MoveArguments, WorldEngine.move_to, "robot_moved"),
    "pick_up": (ObjectArguments, WorldEngine.pick_up, "object_picked_up"),
    "drop": (ObjectArguments, WorldEngine.drop, "object_dropped"),
    "give": (GiveArguments, WorldEngine.give, "object_given"),
    "talk_to": (TalkArguments, WorldEngine.talk_to, "person_spoken_to"),
}


class WorldTools:
    def __init__(self, engine: WorldEngine, bus: EventBus) -> None:
        self._engine = engine
        self._bus = bus

    @staticmethod
    def schemas() -> dict:
        return {name: spec[0].model_json_schema() for name, spec in TOOL_REGISTRY.items()}

    def execute(self, tool: str, arguments: dict, task_id: str | None = None) -> dict:
        logging.getLogger("agentic_friend.tools").info("Explorer -> %s(%s)", tool, arguments)
        try:
            if tool not in TOOL_REGISTRY:
                raise WorldError(f"Unknown tool: {tool}")
            schema, operation, event_type = TOOL_REGISTRY[tool]
            args = schema.model_validate(arguments)
            observation = operation(self._engine, **args.model_dump())
            result = {"success": True, "observation": observation}
        except (ValidationError, WorldError) as exc:
            error = ("Invalid arguments: " + "; ".join(
                f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
                if isinstance(exc, ValidationError) else str(exc))
            result = {"success": False, "error": error}
            event_type = "tool_failed"
        event = self._bus.emit(event_type, "explorer", tool=tool, arguments=arguments,
                               task_id=task_id, **result)
        # Full state goes ONLY to visualizer subscribers, never into tool results.
        if result["success"] and tool in {"move_to", "pick_up", "drop", "give", "talk_to"}:
            self._bus.emit("world_updated", world=self._engine.snapshot())
        return {**result, "evidence_id": event["id"]}
