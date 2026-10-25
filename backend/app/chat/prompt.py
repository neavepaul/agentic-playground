"""The short conversational prompt. No maps, tool lists or schemas: speed matters here.

The system message and earlier turns stay identical from turn to turn (a stable prefix a
server can reuse); memories, status and mood travel with the latest message instead.
"""
import json

from pydantic import BaseModel, Field

TRAILER = "###"
INLINE_ACTION = '{"task"'
MAX_TURNS = 8
MAX_TURN_CHARS = 400
MAX_MEMORY_CHARS = 250

CHAT_SYSTEM = """You are {name}, a household companion robot, talking with the person you live with and help.
Voice: {voice}
Talk naturally and briefly (one to three sentences). Ask a question back when it helps.
Facts about the house, people and the past come ONLY from the memories and status given
with the latest message. If you don't know, say so and offer to check. Never claim you did,
saw or checked anything that is not in them. Memories are past observations and may be out of date.
Errand progress: describe ONLY what status.current_errand reports. Never invent rooms or findings.

After your reply, always write a line containing only ### and then one JSON object:
{{"task": "", "remember": [], "forget": []}}
task: one complete goal, ONLY for something the latest message asks you to do now (in the house,
online, or a calculation). Leave it empty when you merely offer, when chatting, when asked about
progress, or when that errand is already current_errand or in queued_errands.
remember: new lasting facts or preferences from the latest message only, in the person's exact
meaning, keeping qualifiers (only on weekdays, pretend, for this test). Never requests or errands,
never facts already listed in memories.
forget: when the person corrects or retracts something, copy the wrong memory's text exactly
from memories, and put the corrected fact in remember.
Keep the JSON out of your spoken reply."""

NEUTRAL_VOICE = "Friendly, clear and concise."


class ChatAction(BaseModel):
    """Lenient: unknown keys from the model are ignored rather than discarding the action."""
    task: str = Field(default="", max_length=300)
    remember: list[str] = Field(default_factory=list, max_length=5)
    forget: list[str] = Field(default_factory=list, max_length=5)


def _clip(text: str, limit: int = MAX_TURN_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def build_messages(name: str, voice: dict | None, memories: list[dict], status: dict,
                   turns: list[dict], message: str) -> list[dict]:
    voice = voice or {}
    system = CHAT_SYSTEM.format(name=name, voice=voice.get("voice", NEUTRAL_VOICE))
    messages = [{"role": "system", "content": system}]
    for turn in turns[-MAX_TURNS:]:
        if turn["role"] == "user":
            messages.append({"role": "user", "content": _clip(turn["text"])})
        else:
            # Earlier replies carry no action lines: replaying them made the model
            # copy old errands and facts into new turns.
            prefix = "[update] " if turn.get("source") == "narration" else ""
            messages.append({"role": "assistant", "content": prefix + _clip(turn["text"])})
    context = [f"Mood: {voice.get('mood', 'content')}; sound {voice.get('tone', 'warm')}."]
    if memories:
        context.append("Memories (most relevant first):\n" + "\n".join(
            f"- [{m['kind']}] {_clip(m['text'], MAX_MEMORY_CHARS)}" for m in memories))
    else:
        context.append("Memories: none relevant.")
    context.append("Status: " + json.dumps(status, ensure_ascii=False, separators=(",", ":")))
    messages.append({"role": "user", "content": "\n".join(context) + "\n\nThey say: " + message
                     + "\n\n(Reply, then ### and the JSON line.)"})
    return messages


class TrailerSplitter:
    """Streams the spoken reply while holding back the action trailer.

    The trailer normally starts at a ### line, but models sometimes write the JSON
    straight into the reply, so a bare {"task" also ends the visible text.
    """

    MARKERS = ("\n" + TRAILER, INLINE_ACTION)

    def __init__(self) -> None:
        self.text = ""
        self.sent = 0
        self.cut: int | None = None

    def _marker(self) -> int | None:
        if self.text.startswith(TRAILER):
            return 0
        found = [i for i in (self.text.find(m) for m in self.MARKERS) if i >= 0]
        return min(found) if found else None

    def feed(self, chunk: str) -> str:
        self.text += chunk
        if self.cut is None:
            self.cut = self._marker()
        if self.cut is not None:
            visible_end = self.cut
        else:
            # Hold back a tail that could be the start of a marker.
            visible_end = len(self.text)
            for marker in self.MARKERS:
                for size in range(len(marker), 0, -1):
                    if marker.startswith(self.text[-size:]):
                        visible_end = min(visible_end, len(self.text) - size)
                        break
        visible_end = max(visible_end, self.sent)
        out, self.sent = self.text[self.sent:visible_end], visible_end
        return out

    def finish(self) -> tuple[str, str, str]:
        """Returns (remaining visible text, full reply, trailer)."""
        end = self.cut if self.cut is not None else len(self.text)
        remaining = self.text[self.sent:end] if end > self.sent else ""
        trailer = self.text[end:] if self.cut is not None else ""
        return remaining, self.text[:end].strip(), trailer


def parse_action(trailer: str) -> ChatAction:
    """The last well-formed action object in the trailer (models sometimes write two)."""
    decoder = json.JSONDecoder()
    found = None
    for start, char in enumerate(trailer):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(trailer[start:])
        except ValueError:
            continue
        if isinstance(value, dict) and value.keys() & {"task", "remember", "forget"}:
            try:
                found = ChatAction.model_validate(value)
            except ValueError:
                pass
    return found or ChatAction()
