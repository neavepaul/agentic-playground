"""What a message is doing, read deterministically from the person's own words.

A message can change state (grant or revoke a permission, turn an offer down, stop work)
without asking for anything new. Treating those as fresh instructions is how old requests
got replayed. This module only classifies; session.py decides what happens.
"""
import re
from dataclasses import dataclass, field

from app.tasks.permissions import explicitly_requested_tools
from .consent import is_question, is_request

_DECLINES = re.compile(r"^\W*(no|nope|nah|not now|not right now|no thanks|no thank you|leave it|skip it|"
                       r"don'?t bother)\b")
_CANCEL = re.compile(r"^\W*(?:ok(?:ay)?[, ]+|please[, ]+|kip[, ]+)*(?:stop|cancel|abort|never ?mind|forget it)"
                     r"(?:\s+(?:that|it|this|now|the (?:errand|task|search|lookup|check)|what you'?re doing|everything))*"
                     r"(?:[, ]+please)?\W*$")
_WANTS_OFFER = re.compile(r"\b(offer to|could you offer|suggest|wait for my (?:ok|okay|go-ahead|confirmation|yes)|"
                          r"ask (?:me )?(?:first|before)|check with me (?:first|before)|before (?:you )?go(?:ing)?\b)")
_ABOUT_PAST = re.compile(r"\b(what (?:did|have) you (?:do|done|been doing|been up to)|what have you done|"
                         r"did you (?:\w+ ){0,2}(?:speak|talk|go|check|run|read|see|find|look|tell|visit)|"
                         r"have you (?:\w+ ){0,2}(?:spoken|talked|been|checked|run|read|seen|found|told|visited))\b")
_INTRODUCES = re.compile(r"\b(?:it'?s|this is|i'?m|i am|me,)\s+([a-z][a-z'-]+)\b")


@dataclass
class Intent:
    request: bool = False          # asks for an action ("find my keys")
    question: bool = False         # asks for information ("where are my keys?")
    granted: list[str] = field(default_factory=list)   # restrictions lifted by this message
    revoked: list[str] = field(default_factory=list)   # restrictions added by this message
    declines: bool = False         # opens by turning something down ("no thanks")
    cancel: bool = False           # stop the current work
    wants_offer: bool = False      # "offer to…", "wait for my confirmation before going"
    about_past: bool = False       # "what did you do today?"
    needs_tools: list[str] = field(default_factory=list)  # tools their own words ask to run

    @property
    def changes_state_only(self) -> bool:
        """Only permissions or a refusal: nothing new is being asked for."""
        return bool(self.granted or self.revoked or self.declines or self.cancel) and not (
            self.request or self.needs_tools or self.question or self.wants_offer)

    @property
    def small_talk(self) -> bool:
        return not (self.request or self.question or self.needs_tools or self.wants_offer or self.cancel
                    or self.granted or self.revoked)


def read_intent(text: str, granted: list[str] = (), revoked: list[str] = ()) -> Intent:
    lowered = " ".join(text.casefold().replace("’", "'").split())
    wants_offer = bool(_WANTS_OFFER.search(lowered))
    cancel = bool(_CANCEL.match(lowered))
    # "You can use Python again" grants; it does not ask for anything to be done.
    needs = explicitly_requested_tools(text)
    request = is_request(text) and not cancel
    if granted and not needs and request:
        request = bool(re.search(r"[.;!?]\s+\S", text.strip()))  # a request after the grant, not the grant itself
        request = request and is_request(re.split(r"[.;!?]\s+", text.strip(), maxsplit=1)[-1])
    return Intent(request=request and not wants_offer, question=not request and is_question(text),
                  granted=list(granted), revoked=list(revoked), declines=bool(_DECLINES.match(lowered)),
                  cancel=cancel, wants_offer=wants_offer, about_past=bool(_ABOUT_PAST.search(lowered)),
                  needs_tools=needs)


def introduced_as(text: str, people: dict) -> str | None:
    """A known person introducing themselves ("Hi Kip, it's Neave")."""
    names = {str(person.get("name", key)).casefold(): key for key, person in people.items()}
    names.update({key.casefold(): key for key in people})
    for match in _INTRODUCES.finditer(text.casefold()):
        if match.group(1) in names:
            return names[match.group(1)]
    return None
