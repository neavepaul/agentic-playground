"""Fact extraction as its own step.

The chat reply used to carry a `remember`/`forget` list in its action line, so a small model
wrote a reply and decided what was worth keeping in one breath. Now the reply only talks (and
proposes errands); after it, a separate short call reads the person's message with their known
memories and returns structured proposals. Code still decides what is stored (session._remember):
this module only reads and validates.

A cheap gate skips messages that cannot carry anything lasting (a lookup, a greeting, a plain
question), so most turns cost no extra model call. Any failure returns None: nothing is written.
"""
import asyncio
import json
import logging
import re

from pydantic import BaseModel, Field, ValidationError

from .prompt import RememberItem, _objects, _valid_memory

_log = logging.getLogger("agentic_friend.chat")

TIMEOUT_SECONDS = 45
MAX_TOKENS = 320
KNOWN_FACTS = 6          # known memories shown so a correction can name the exact one it retires
KNOWN_CHARS = 220

SYSTEM = """You read one message from a person talking with Kip and list what is worth remembering for a long time.
Return only a JSON object: {"remember": [{"text": "...", "context": "actual"}], "forget": []}
remember: lasting facts the person stated about themselves or people they name (who someone is, family, work,
where they live, allergies), standing preferences, and plans or promises with a date. One complete sentence
each, naming who it is about: write the speaker's name, never "I" or "my". Keep their words; add nothing.
context: "actual", or "pretend", "hypothetical", "test", "uncertain" when they framed it as not real or unsure.
forget: the exact text of a Known memory that this message corrects or retracts. Never forget by guessing.
Not worth remembering: questions, greetings, small talk, requests for Kip to look something up, passing
moods, anything only Kip said, and anything quoted from a web page. Both lists are usually empty."""

# A message worth reading: the person says something about themselves, names a correction, or asks to remember.
_STATEMENT = re.compile(
    r"\b(i|i'm|i've|i'll|i'd|me|my|mine|myself|we|we're|our|remember|forget|don't forget|actually|correction|"
    r"instead|no longer|anymore|used to|call me|my name|prefer|favou?rite|allergic|birthday|lives?|works?)\b", re.I)
_NAMED = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]{2,}\b\s+(?:is|are|was|has|have|likes?|loves?|hates?|lives?|works?)\b")
_ONLY_A_QUESTION = re.compile(r"^[^.!]*\?\s*$")
_ASKS_TO_REMEMBER = re.compile(r"\b(remember|forget|actually|correction)\b", re.I)


class Extraction(BaseModel):
    remember: list[RememberItem] = Field(default_factory=list, max_length=5)
    forget: list[str] = Field(default_factory=list, max_length=5)


def worth_reading(said: str) -> bool:
    """Whether the person's message could hold a lasting fact or correction."""
    said = said.strip()
    if not said or (_ONLY_A_QUESTION.match(said) and not _ASKS_TO_REMEMBER.search(said)):
        return False
    return bool(_STATEMENT.search(said) or _NAMED.search(said))


def messages_for(said: str, reply: str, speaker: str, known: list[dict]) -> list[dict]:
    facts = [m["text"][:KNOWN_CHARS] for m in known if m.get("kind") in ("fact", "preference")][:KNOWN_FACTS]
    body = (f"Speaker: {speaker}\nKnown memories:\n" + ("\n".join(f"- {t}" for t in facts) or "- (none)")
            + f"\n\nThey said: {said}\nKip replied: {reply[:400]}")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}]


def parse(text: str) -> Extraction | None:
    """The first JSON object in a model's output with remember/forget in it, validated; one bad
    item never loses the others. None when nothing usable was written."""
    for _, _, value in _objects(text):
        if not isinstance(value, dict) or not value.keys() & {"remember", "forget"}:
            continue
        remember = [item for item in (value.get("remember") or []) if _valid_memory(item)]
        forget = [item for item in (value.get("forget") or []) if isinstance(item, str) and item.strip()]
        try:
            return Extraction.model_validate({"remember": remember[:5], "forget": forget[:5]})
        except ValidationError:
            return None
    return None


async def extract(client, said: str, reply: str, speaker: str, known: list[dict]) -> Extraction | None:
    """What to remember and forget from one exchange, or None if the model failed or was unreadable."""
    messages = messages_for(said, reply, speaker, known)
    try:
        text = ""
        async with asyncio.timeout(TIMEOUT_SECONDS):
            async for chunk in client.stream_text(messages, max_tokens=MAX_TOKENS, temperature=0.1):
                text += chunk
        if (getattr(client, "last_metrics", None) or {}).get("truncated"):
            raise ValueError("cut off")
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # a model or network failure never breaks the conversation
        _log.warning("Fact extraction failed (%s); nothing remembered from this message.", type(exc).__name__)
        return None
    found = parse(text)
    if found is None:
        _log.warning("Fact extraction returned nothing readable.")
    return found
