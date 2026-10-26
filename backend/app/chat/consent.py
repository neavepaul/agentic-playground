"""Deterministic consent for errands that come up in conversation.

The chat model may *propose* an errand; this module decides whether anything may start.
An errand starts only when the person's own message asks for it, or when they clearly
agree to one specific offer Kip made. Refusals and hedges never authorize. A direct
question authorizes only a digital lookup that answers it (a web search or a calculation):
it changes nothing in the house. A question never starts a physical errand.
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
    if _NEGATED_ACTION.match(lowered) or re.search(r"\b(don'?t|do not|never)\s+(\w+\s+)?(" + _ACTION_VERBS + r")\b",
                                                    lowered):
        return False
    for pattern in (_CLAUSE_START, _POLITE):
        for match in pattern.finditer(lowered):
            verb, rest = match.group(1) if pattern is _CLAUSE_START else match.group(2), \
                match.group(2) if pattern is _CLAUSE_START else match.group(3)
            if not _NOT_ERRAND.match(f"{verb}{rest}".strip()):
                return True
    return False


_QUESTION_START = re.compile(r"^(?:(?:so|ok|okay|hey|kip|and|but)[, ]+)*(what|what's|whats|who|who's|whose|when|"
                             r"where|which|why|how|how's|is|are|was|were|does|do|did|can|could|will|would|"
                             r"should|any idea|tell me|do you know)\b")
# Lookups that only read the web or run a calculation.
_LOOKUP_GOAL = re.compile(r"\b(search|look up|look online|online|web|internet|website|webpage|google|"
                          r"calculate|compute|convert|work out|python|program)\b")
# Anything that moves, fetches, or speaks to someone is physical, even inside a lookup.
_PHYSICAL_GOAL = re.compile(r"\b(bring|fetch|grab|take|give|deliver|carry|put|place|move|go to|go and|visit|"
                            r"check the|check on|pick up|drop|tell|ask|notify|inform|remind|room|kitchen|"
                            r"bedroom|office|hall)\b")
_NO_LOOKUP = re.compile(r"\b(don'?t|do not|never|no need to|without)\s+(\w+\s+)?(look|search|google|check|calculate)")


def answers_question(text: str, goal: str) -> bool:
    """Is this message a direct question, and the goal only the digital lookup that answers it?"""
    lowered = " ".join(text.casefold().replace("’", "'").split())
    if not goal or _NO_LOOKUP.search(lowered):
        return False
    if not (lowered.rstrip().endswith("?") or _QUESTION_START.match(lowered)):
        return False
    planned = goal.casefold().replace("’", "'")
    return bool(_LOOKUP_GOAL.search(planned)) and not _PHYSICAL_GOAL.search(planned)


# ── task identity ──────────────────────────────────────────────────────────────
_FILLER = {"the", "a", "an", "my", "our", "please", "for", "me", "us", "now", "just", "kip", "some",
           "go", "and", "can", "could", "you", "would", "will", "to"}
_NEGATIONS = {"no", "not", "never", "n't", "without", "nobody", "nothing", "none"}


def task_identity(goal: str) -> tuple[str, ...]:
    """Canonical identity of an errand: ordered content words, negations always kept.

    "Find the keys." == "find keys", but "Tell Neave dinner is ready" and
    "Tell Neave dinner is NOT ready" stay different errands.
    """
    words = re.findall(r"[\w']+", _normal(goal))
    canon = []
    for word in words:
        if word in _NEGATIONS:
            canon.append("not")
        elif word not in _FILLER:
            canon.append(word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word)
    return tuple(canon)


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

    @property
    def identity(self) -> tuple[str, ...]:
        return task_identity(self.goal)

    def public(self) -> dict:
        return {"id": self.id, "goal": self.goal, "status": self.status,
                "age_seconds": round(time.monotonic() - self.created_at, 1)}
