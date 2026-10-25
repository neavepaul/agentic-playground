import asyncio
import logging

from pydantic import ValidationError

from app.events.bus import EventBus
from .schemas import ACTION_SCHEMAS
from .python import run_python
from .web import read_webpage, web_search

DIGITAL_TOOLS = frozenset(ACTION_SCHEMAS)


def validation_reasons(exc: ValidationError, skip: int = 0) -> str:
    """Field-level reasons the model can act on, without echoing input values."""
    reasons = []
    for error in exc.errors():
        field = ".".join(str(part) for part in error["loc"][skip:]) or "input"
        reasons.append(f"{field}: {error['msg']}")
    return "; ".join(reasons)[:400]


def required_tool_progress(context) -> dict:
    required = {tool for c in context.conditions if c.kind == "answer" for tool in c.required_tools}
    evidence = {}
    for entry in context.action_history:
        if entry["success"] and entry["tool"] in DIGITAL_TOOLS - {"finish_answer"}:
            evidence.setdefault(entry["tool"], []).append(entry["evidence_id"])
    missing = sorted(required - evidence.keys())
    return {"missing_tools": missing, "can_finish": bool(evidence) and not missing,
            "successful_evidence_ids": {tool: ids[-6:] for tool, ids in evidence.items()}}


class CapabilityTools:
    def __init__(self, bus: EventBus, enabled: bool = True):
        self.bus, self.enabled = bus, enabled

    async def execute(self, tool: str, arguments: dict, context) -> dict:
        try:
            if not self.enabled or tool not in ACTION_SCHEMAS:
                raise ValueError("This capability is unavailable.")
            args = ACTION_SCHEMAS[tool].model_validate({**arguments, "tool": tool})
            async with asyncio.timeout(30):
                if tool == "web_search":
                    observation = await asyncio.to_thread(web_search, args.query)
                elif tool == "read_webpage":
                    observation = await read_webpage(args.url, args.start_char)
                elif tool == "run_python":
                    observation = await run_python(args.code, args.data)
                else:
                    observation = self.finish_answer(args, context)
            result = {"success": True, "observation": observation}
            event_type = "capability_observed"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logging.getLogger("agentic_friend.capabilities").info("%s failed: %s", tool, type(exc).__name__)
            if isinstance(exc, ValidationError):
                error = "Invalid tool arguments: " + validation_reasons(exc)
            elif isinstance(exc, (ValueError, TimeoutError)):
                error = str(exc)[:400] or "Tool time limit reached."
            elif type(exc).__module__.startswith("pydantic_monty"):
                error = (type(exc).__name__ + ": " + str(exc))[:500]
            else:
                # Provider/runtime exceptions can contain environment details.
                error = f"{tool} failed ({type(exc).__name__}). Check the input, simplify it, or try another source."
            result, event_type = {"success": False, "error": error}, "tool_failed"
        event = self.bus.emit(event_type, "explorer", task_id=context.id,
                              tool=tool, arguments=arguments, **result)
        return {**result, "evidence_id": event["id"]}

    @staticmethod
    def finish_answer(args, context) -> dict:
        by_id = {a["evidence_id"]: a for a in context.action_history
                 if a["success"] and a["tool"] in DIGITAL_TOOLS - {"finish_answer"}}
        if any(key not in by_id for key in args.evidence_ids):
            raise ValueError("Answer must cite successful search, page or Python evidence from this task.")
        evidence = [by_id[key] for key in args.evidence_ids]
        required = {tool for c in context.conditions if c.kind == "answer" for tool in c.required_tools}
        missing = required - {a["tool"] for a in evidence}
        if missing:
            raise ValueError("Answer is missing cited tool evidence for: " + ", ".join(sorted(missing))
                             + ". Run any unfinished tools and cite their successful evidence IDs.")
        sources = []
        for item in evidence:
            obs = item["observation"]
            candidates = obs.get("results", []) if item["tool"] == "web_search" else (
                [obs] if item["tool"] == "read_webpage" else [])
            for source in candidates:
                if not any(s["url"] == source["url"] for s in sources):
                    sources.append({"title": source.get("title", source["url"]), "url": source["url"]})
        return {"answer": args.answer, "sources": sources, "evidence_ids": args.evidence_ids}
