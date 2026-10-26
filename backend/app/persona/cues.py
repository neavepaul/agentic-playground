"""What a person's own words say about how to treat them. Pure, cheap and conservative.

Only explicit wording counts: Kip never infers a feeling nobody expressed. A cue that
is negated ("I'm not tired") is ignored.
"""
import re
from dataclasses import dataclass

_PATTERNS = {
    # They describe being worn out, low or unwell.
    "distress": r"exhausted|knackered|worn out|so tired|really tired|rough day|bad day|awful day|long day|"
                r"stressed|anxious|overwhelmed|upset|lonely|miserable|depressed|sad|crying|"
                r"feel(?:ing)? (?:sick|ill|awful|terrible|rubbish|down|low)|in pain|headache|migraine",
    # Topics where jokes are out of place whatever anyone's mood.
    "serious": r"died|passed away|death|funeral|hospital|emergency|ambulance|cancer|diagnos\w*|surgery|"
               r"bleeding|can'?t breathe|chest pain|overdose|suicid\w*|self[- ]harm|on fire|smoke alarm|has fallen",
    # Kip's humour landed.
    "laughter": r"ha(?:ha)+|hehe|lol|lmao|rofl|that'?s funny|good one|you'?re funny|made me laugh|😂|🤣|😄",
    # Kip's humour did not land, or is not wanted.
    "humour_unwanted": r"not funny|stop joking|no jokes|be serious|not the time|stop being sarcastic|"
                       r"(?:don'?t|do not) be sarcastic|cut the (?:jokes|sarcasm)|enough with the jokes",
    # They want space; asking again would be pestering.
    "boundary": r"leave me alone|not now|i need (?:some )?(?:space|quiet|peace)|stop asking|"
                r"(?:don'?t|do not) (?:check on|bother|disturb) me|i'?m (?:busy|working)|be quiet|shh+",
    "praise": r"thanks|thank you|well done|good job|nice work|great job|nicely done|you'?re the best|good robot",
    "criticism": r"idiot|stupid|useless|dumb|you'?re wrong|that'?s wrong|bad robot|you messed up",
}
_COMPILED = {name: re.compile(r"(?<![\w'])(?:" + pattern + r")(?![\w'])") for name, pattern in _PATTERNS.items()}
_NEGATION = re.compile(r"\b(?:not|never|no|\w+n't)\W+(?:\w+\W+){0,2}$")
_NEGATABLE = {"distress", "serious"}


@dataclass(frozen=True)
class Cues:
    distress: bool = False
    serious: bool = False
    laughter: bool = False
    humour_unwanted: bool = False
    boundary: bool = False
    praise: bool = False
    criticism: bool = False
    evidence: str = ""  # the words that raised concern, quoted as said

    @property
    def sensitive(self) -> bool:
        return self.distress or self.serious


def read_cues(text: str) -> Cues:
    lowered = " ".join(text.casefold().replace("’", "'").split())
    found, evidence = {}, ""
    for name, pattern in _COMPILED.items():
        for match in pattern.finditer(lowered):
            if name in _NEGATABLE and _NEGATION.search(lowered[:match.start()]):
                continue
            found[name] = True
            if name in _NEGATABLE and not evidence:
                evidence = match.group(0)
            break
    return Cues(**found, evidence=evidence)
