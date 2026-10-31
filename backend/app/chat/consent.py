"""Deterministic consent for errands that come up in conversation.

The chat model may *propose* an errand; this module decides whether anything may start.
An errand starts only when the person's own message asks for it, or when they clearly
agree to one specific offer Kip made. Refusals and hedges never authorize.

What a started errand may then *do* is not decided here from the words of a goal: it is a
Grant (app/tasks/permissions.py) checked against a tool registry right before each tool
runs. A request grants every kind of action; a direct question grants only read-only
lookups and calculations, so it can never move, handle things or talk to anyone else.
"""
import re
import time
from dataclasses import dataclass, field
from typing import Literal
from uuid import uuid4

Reply = Literal["yes", "no", "unclear"]

# Any of these means the person is not giving a plain go-ahead.
_REFUSAL = re.compile(
    r"\b(no|nope|nah|not|n't|never|stop|cancel|wait|hold on|hang on|later|nevermind|never mind|"
    r"rather not|without|instead|but|unless|maybe|perhaps|first|before|after)\b")
_AFFIRMATION = re.compile(
    r"^(yes|yeah|yep|yup|sure|ok|okay|alright|all right|please do|please|go ahead|do it|go for it|"
    r"sounds good|absolutely|of course|definitely|that would be (great|lovely|nice|helpful)|great|"
    r"thanks|thank you)\b")

_ACTION_VERBS = (r"find|bring|fetch|get|grab|take|check|look for|look in|look around|search|calculate|"
                 r"compute|use|run|read|deliver|give|go to|go and|put|place|carry|remind|notify|inform|"
                 r"ask|work out|figure out|count|tell")
# "tell me / ask me / let me" are conversation, not errands.
_NOT_ERRAND = re.compile(r"^(tell|ask) (me|us)\b")
_CLAUSE_START = re.compile(r"(?:^|[.!?;:]\s+|,\s*|\band\s+|\bthen\s+)"
                           r"(?:(?:also|just|now|quickly|kip|please)[,]?\s+)*(" + _ACTION_VERBS + r")\b(.*)")
_POLITE = re.compile(r"\b(please|could you|can you|would you|will you|would you mind|i need you to|"
                     r"i want you to|i'd like you to)\s+(?:(?:also|just|quickly|go and|now)\s+)*"
                     r"(" + _ACTION_VERBS + r")\b(.*)")
# Negation that applies to the request itself ("Don't…", "Never…"). A leading "No," that
# refuses Kip's offer does not cancel a new request after it ("No, please find the keys").
_NEGATED_ACTION = re.compile(r"\W*(don'?t|do not|never|stop|please don'?t)\b")


def _normal(text: str) -> str:
    text = text.casefold().replace("’", "'")
    text = re.sub(r"n't\b", " n't", text)  # don't -> do n't, so "n't" is a refusal word
    return " ".join(text.split())


def classify_reply(text: str) -> Reply:
    """How a message answers a pending offer. Only a plain go-ahead is "yes"."""
    lowered = _normal(text)
    if _REFUSAL.search(lowered):
        return "no"
    if "?" in text or not _AFFIRMATION.search(lowered.strip(" .,!")):
        return "unclear"
    return "yes" if len(lowered.split()) <= 10 else "unclear"


def is_request(text: str) -> bool:
    """Does the message itself ask Kip to do something (not just talk about it)?"""
    lowered = " ".join(text.casefold().replace("’", "'").split())
    # Judged clause by clause: "Don't go into the kitchen; find my keys" forbids one thing
    # and asks for another. A negated clause never counts as asking.
    for clause in re.split(r"[.;!?]+|,|\bbut\b", lowered):
        clause = clause.strip()
        if not clause or _NEGATED_ACTION.match(clause) or re.search(
                r"\b(don'?t|do not|never)\s+(\w+\s+)?(" + _ACTION_VERBS + r")\b", clause):
            continue
        for pattern in (_CLAUSE_START, _POLITE):
            for match in pattern.finditer(clause):
                verb, rest = match.group(1) if pattern is _CLAUSE_START else match.group(2), \
                    match.group(2) if pattern is _CLAUSE_START else match.group(3)
                if not _NOT_ERRAND.match(f"{verb}{rest}".strip()):
                    return True
    return False


_QUESTION_START = re.compile(r"^(?:(?:so|ok|okay|hey|kip|and|but)[, ]+)*(what|what's|whats|who|who's|whose|when|"
                             r"where|which|why|how|how's|is|are|was|were|does|do|did|can|could|will|would|"
                             r"should|any idea|tell me|do you know)\b")


def is_question(text: str) -> bool:
    """A direct question, by its form: it asks for information, not for an action."""
    lowered = " ".join(text.casefold().replace("’", "'").split())
    return lowered.rstrip().endswith("?") or bool(_QUESTION_START.match(lowered))


# ── task identity ──────────────────────────────────────────────────────────────
_FILLER = {"the", "a", "an", "please"}


def task_identity(goal: str) -> tuple[str, ...]:
    """Canonical identity of an errand: ordered content words, negations always kept.

    "Find the keys." == "find keys", but "Tell Neave dinner is ready" and
    "Tell Neave dinner is NOT ready" stay different errands.
    """
    # False negatives are safer than merging different instructions. Recipients,
    # ownership, direction, timing, negations and numeric punctuation carry intent.
    # In particular, never stem names (James/James's) or erase "me" versus "us".
    words = re.findall(r"[+-]?\d+(?:[.,:/-]\d+)*|[\w']+|[^\w\s.,!?;]", _normal(goal))
    return tuple(word for word in words if word not in _FILLER)


# ── offers ─────────────────────────────────────────────────────────────────────
@dataclass
class Offer:
    goal: str
    reply_turn_id: str
    generation: int
    id: str = field(default_factory=lambda: str(uuid4()))
    status: str = "pending"  # pending | consumed | declined | expired | superseded | deferred
    created_at: float = field(default_factory=time.monotonic)
    authorized_by: str | None = None  # the user turn that agreed
    effects: frozenset[str] | None = None  # what a yes authorizes; None: whatever the offered goal needs

    @property
    def identity(self) -> tuple[str, ...]:
        return task_identity(self.goal)

    def public(self) -> dict:
        return {"id": self.id, "goal": self.goal, "status": self.status,
                "age_seconds": round(time.monotonic() - self.created_at, 1)}
