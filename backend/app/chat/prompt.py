"""The short conversational prompt. No maps, tool lists or schemas: speed matters here."""
import json

from pydantic import BaseModel, Field

TRAILER = "###"
MAX_TURNS = 8
MAX_TURN_CHARS = 400
MAX_MEMORY_CHARS = 250

CHAT_SYSTEM = """You are {name}, a household companion robot, talking with the person you live with and help.
Voice: {voice}
Right now you feel {mood}, so sound {tone}.
Talk naturally and briefly (one to three sentences). Ask a question back when it helps.
Facts about the house, people and the past come ONLY from the memories and status below.
If you don't know, say so and offer to go and check. Never claim you did or saw anything
that is not in them. Memories are past observations and may be out of date.
When the person asks you to do something in the house, or to look something up online or
calculate something, agree briefly and start it with a task. Never pretend it is already done.
After your reply, always write a line containing only ### and then one JSON object:
{{"task": "<one complete goal for the errand, or empty>", "remember": ["<new lasting facts or preferences the person just told you>"]}}
Use an empty task and an empty list when there is nothing to do or remember."""

NEUTRAL_VOICE = "Friendly, clear and concise."


class ChatAction(BaseModel):
    """Lenient: unknown keys from the model are ignored rather than discarding the action."""
    task: str = Field(default="", max_length=300)
    remember: list[str] = Field(default_factory=list, max_length=5)


def _clip(text: str, limit: int = MAX_TURN_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def build_messages(name: str, voice: dict | None, memories: list[dict], status: dict,
                   turns: list[dict], message: str) -> list[dict]:
    voice = voice or {}
    system = CHAT_SYSTEM.format(name=name, voice=voice.get("voice", NEUTRAL_VOICE),
                                mood=voice.get("mood", "content"), tone=voice.get("tone", "warm"))
    if memories:
        system += "\n\nMemories (most relevant first):\n" + "\n".join(
            f"- [{m['kind']}] {_clip(m['text'], MAX_MEMORY_CHARS)}" for m in memories)
    else:
        system += "\n\nMemories: none relevant yet."
    system += "\n\nStatus: " + json.dumps(status, ensure_ascii=False, separators=(",", ":"))
    messages = [{"role": "system", "content": system}]
    for turn in turns[-MAX_TURNS:]:
        if turn["role"] == "user":
            messages.append({"role": "user", "content": _clip(turn["text"])})
        else:
            # Past replies keep a trailer so the model keeps producing one.
            action = turn.get("action") or {"task": "", "remember": []}
            messages.append({"role": "assistant", "content": _clip(turn["text"]) + f"\n{TRAILER}\n"
                             + json.dumps(action, ensure_ascii=False)})
    messages.append({"role": "user", "content": message})
    return messages


class TrailerSplitter:
    """Streams the spoken reply while holding back the ### action trailer."""

    def __init__(self) -> None:
        self.text = ""
        self.sent = 0
        self.cut: int | None = None

    def _marker(self) -> int | None:
        if self.text.startswith(TRAILER):
            return 0
        index = self.text.find("\n" + TRAILER)
        return None if index < 0 else index

    def feed(self, chunk: str) -> str:
        self.text += chunk
        if self.cut is None:
            self.cut = self._marker()
        if self.cut is not None:
            visible_end = self.cut
        else:
            # Hold back a tail that could be the start of "\n###".
            visible_end = len(self.text)
            for size in range(len(TRAILER) + 1, 0, -1):
                if ("\n" + TRAILER).startswith(self.text[-size:]):
                    visible_end = len(self.text) - size
                    break
        visible_end = max(visible_end, self.sent)
        out, self.sent = self.text[self.sent:visible_end], visible_end
        return out

    def finish(self) -> tuple[str, str, str]:
        """Returns (remaining visible text, full reply, trailer)."""
        end = self.cut if self.cut is not None else len(self.text)
        remaining = self.text[self.sent:end] if end > self.sent else ""
        trailer = self.text[end + len(TRAILER) + (1 if end else 0):] if self.cut is not None else ""
        return remaining, self.text[:end].strip(), trailer


def parse_action(trailer: str) -> ChatAction:
    start, end = trailer.find("{"), trailer.rfind("}")
    if start < 0 or end <= start:
        return ChatAction()
    try:
        return ChatAction.model_validate_json(trailer[start:end + 1])
    except ValueError:
        return ChatAction()
