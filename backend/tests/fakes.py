import asyncio
import json


class ScriptedLLM:
    def __init__(self, replies, conditions=None):
        self.replies = iter(replies)
        self.calls = []
        self.conditions = conditions

    async def generate(self, messages, response_schema):
        if response_schema.get("title") == "GoalPlan":
            conditions = self.conditions
            if conditions is None:
                conditions = [{"kind": "answer", "required_tools": ["web_search"]}]
            return json.dumps({"summary": "Required outcomes defined.", "conditions": conditions})
        self.calls.append((messages, response_schema))
        reply = next(self.replies)
        if callable(reply):
            reply = reply(json.loads(messages[1]["content"]))
        return reply if isinstance(reply, str) else json.dumps(reply)


class WaitingLLM:
    async def generate(self, messages, response_schema):
        await asyncio.sleep(100)


def tool(name, **arguments):
    """A capability choice, as the Explorer's model returns it."""
    return {"summary": f"Using {name}.", "action": {"tool": name, **arguments}}


def complete(context):
    return {"action": "complete", "summary": "Requested actions completed.",
            "evidence_ids": [a["evidence_id"] for a in context["recent_actions"] if a["success"]]}
