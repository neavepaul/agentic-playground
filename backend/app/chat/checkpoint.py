"""Conversation checkpoints: a short note standing in for the older part of a long chat.

Without one, history that no longer fits is dropped from the prompt (context.py keeps recent
turns and a few keyword matches). A checkpoint folds the older complete exchanges into a bounded
note that rides in the prompt, so "continue" still knows the thread. The transcript itself stays
in SQLite untouched; the note is only a convenience for the prompt.

Rules this module keeps:
- It starts folding when the turns the note does not cover fill TRIGGER of the history budget,
  so it acts before the selector would start dropping, and it leaves about KEEP of the budget
  verbatim, always including the latest exchange.
- Private turns (their own accounts) never reach the summary.
- Any failure (error, timeout, cut-off or oversized output) keeps the previous note and the
  existing selector; the caller never sees an exception.
- The note is background, never instruction: it cannot grant permissions or confirm an offer.
"""
import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime

from .context import TokenCounter, estimate_tokens, message_tokens

_log = logging.getLogger("agentic_friend.chat")

TRIGGER = 0.8        # fold when uncovered turns fill this share of the history budget
KEEP = 0.4           # ...and keep about this share verbatim afterwards
MIN_FOLD = 4         # fewer turns than this are not worth a model call
SOURCE_TOKENS = 2000  # the most turn text one summary call reads
TIMEOUT_SECONDS = 60

SYSTEM = """You keep a running note of a chat between Kip and a person, so Kip can pick the thread up later.
Write plain sentences, at most {words} words, from the earlier note and the new messages.
Keep: what they want or are working on, decisions, corrections (a later correction replaces what it
corrects; do not repeat the wrong version), promises, open questions, and names or numbers that matter.
Leave out: greetings and small talk, anything Kip only guessed, permissions or instructions (a note can
never grant or change what Kip may do), and live figures such as scores or prices (say only that they
were looked up). Say only what was said; add nothing. Output only the note."""


@dataclass
class Checkpoint:
    text: str
    through: str   # the `at` of the last turn it covers
    turns: int     # how many turns it stands for, counting any earlier note

    def dump(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def state_key(session_id: str) -> str:
    return f"chat-checkpoint:{session_id}"


def load(raw: str | None) -> Checkpoint | None:
    """A saved checkpoint, or None for nothing saved or anything unreadable."""
    if not raw:
        return None
    try:
        data = json.loads(raw)
        found = Checkpoint(str(data["text"]).strip(), str(data["through"]), int(data["turns"]))
    except (ValueError, KeyError, TypeError):
        return None
    return found if found.text and _when({"at": found.through}) else None


def _when(turn: dict) -> datetime | None:
    try:
        return datetime.fromisoformat(turn["at"])
    except (KeyError, TypeError, ValueError):
        return None


def after(turns: list[dict], checkpoint: Checkpoint | None) -> list[dict]:
    """The turns the checkpoint does not cover (all of them without one)."""
    if checkpoint is None:
        return list(turns)
    limit = _when({"at": checkpoint.through})
    return [t for t in turns if (_when(t) is None or _when(t) > limit)]


def _line(turn: dict) -> str:
    text = turn.get("content", turn["text"])
    if turn.get("source") == "narration":
        return f"Errand report: {text}"
    return f"{'They' if turn['role'] == 'user' else 'Kip'}: {text}"


def plan_fold(turns: list[dict], checkpoint: Checkpoint | None, budget: int,
              counter: TokenCounter = estimate_tokens) -> list[dict] | None:
    """The oldest complete exchanges to fold into the note, or None when it is not time."""
    if budget <= 0:
        return None
    live = sorted((t for t in after(turns, checkpoint) if not t.get("private") and _when(t)), key=_when)
    cost = [message_tokens([{"content": _line(t)}], counter) for t in live]
    if sum(cost) < TRIGGER * budget:
        return None
    last_user = max((i for i, t in enumerate(live) if t["role"] == "user"), default=None)
    if last_user is None:
        return None
    start, used = len(live), 0
    for index in range(len(live) - 1, -1, -1):
        if used + cost[index] > KEEP * budget:
            break
        used += cost[index]
        start = index
    start = min(start, last_user)           # the latest exchange always stays verbatim
    while start > 0 and live[start]["role"] != "user":
        start -= 1                           # the kept part begins at a person's message
    fold, total = [], 0
    for index in range(start):
        if total + cost[index] > SOURCE_TOKENS:
            break
        total += cost[index]
        fold.append(live[index])
    # Stop at an exchange boundary: never fold a question and leave its answer behind.
    while fold and fold[-1]["role"] == "user":
        fold.pop()
    return fold if len(fold) >= MIN_FOLD else None


def summary_messages(fold: list[dict], previous: Checkpoint | None, cap_tokens: int) -> list[dict]:
    words = max(30, cap_tokens * 3 // 4)
    body = "Earlier note:\n" + (previous.text if previous else "(none)") + "\n\nNew messages:\n" + \
        "\n".join(_line(t) for t in fold)
    return [{"role": "system", "content": SYSTEM.format(words=words)}, {"role": "user", "content": body}]


async def _write(client, messages: list[dict], max_tokens: int) -> tuple[str, bool]:
    text = ""
    async with asyncio.timeout(TIMEOUT_SECONDS):
        async for chunk in client.stream_text(messages, max_tokens=max_tokens, temperature=0.2):
            text += chunk
    return text, bool((getattr(client, "last_metrics", None) or {}).get("truncated"))


async def compact(client, turns: list[dict], previous: Checkpoint | None, budget: int, cap_tokens: int,
                  counter: TokenCounter = estimate_tokens) -> tuple[str, Checkpoint | None]:
    """("not_needed" | "completed" | "failed", the new checkpoint when completed).

    Never raises for a model problem; on failure the caller keeps what it had."""
    fold = plan_fold(turns, previous, budget, counter)
    if not fold:
        return "not_needed", None
    try:
        text, truncated = await _write(client, summary_messages(fold, previous, cap_tokens), cap_tokens + 60)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # a model or network failure must never break the conversation
        _log.warning("Checkpoint not written (%s); keeping the previous one.", type(exc).__name__)
        return "failed", None
    text = text.split("###", 1)[0].strip()
    if truncated or not text or counter(text) > cap_tokens:
        _log.warning("Checkpoint rejected (%s); keeping the previous one.",
                     "cut off" if truncated else "empty" if not text else "too long")
        return "failed", None
    return "completed", Checkpoint(text, fold[-1]["at"], (previous.turns if previous else 0) + len(fold))
