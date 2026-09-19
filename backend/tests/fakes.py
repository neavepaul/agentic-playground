import asyncio
import json


class ScriptedLLM:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    async def generate(self, messages, response_schema):
        self.calls.append((messages, response_schema))
        reply = next(self.replies)
        if callable(reply):
            reply = reply(json.loads(messages[1]["content"]))
        if isinstance(reply, dict) and "command_id" in response_schema.get("properties", {}) and "action" in reply:
            if reply["action"] == "report":
                reply = {"command_id": "report", "summary": reply["summary"]}
            elif reply["action"] == "tool":
                args = reply.get("arguments", {})
                reply = {"command_id": ":".join([reply["tool"], *[v for k, v in args.items() if k != "message"]]),
                         "message": args.get("message", ""), "summary": reply["summary"]}
        return reply if isinstance(reply, str) else json.dumps(reply)


class WaitingLLM:
    async def generate(self, messages, response_schema):
        await asyncio.sleep(100)


def tool(name, **arguments):
    return {"action": "tool", "summary": f"Using {name}.", "tool": name, "arguments": arguments}


def complete(context):
    return {"action": "complete", "summary": "Requested actions completed.",
            "evidence_ids": [a["evidence_id"] for a in context["recent_actions"]
                             if a["success"] and a["tool"] != "get_status"]}
