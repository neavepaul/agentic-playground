"""What a task is allowed to do, checked where tools actually run.

Every executable tool has one effect, from this backend registry, never from model text.
A task carries a Grant: the effects the person authorized, and the words that
authorized it. Restrictions are what people asked Kip not to do; they are live and shared,
so a later "don't use Python" also binds tasks that are already queued or running.

The manager asks deny() right before every dispatch. Planning and the Explorer use the
same rules to avoid choosing a forbidden action, but the dispatch check is the guarantee.
"""
from __future__ import annotations

import re
from typing import Iterable

from pydantic import BaseModel, Field

# ── the registry ────────────────────────────────────────────────────────────────
INTROSPECTION = "introspection"            # own status and map: no effect on anything
DIGITAL_LOOKUP = "digital_lookup"          # reads public web content
CALCULATION = "calculation"                # restricted Python, no host access
REPLY = "reply"                            # answering the person Kip is talking to
OBSERVATION = "physical_observation"       # looking around where the robot is
NAVIGATION = "physical_navigation"         # moving through the house
MANIPULATION = "object_manipulation"       # picking up, dropping, handing over
COMMUNICATION = "third_party_communication"  # speaking to someone else in the house

TOOL_EFFECTS = {
    "get_status": INTROSPECTION, "get_map": INTROSPECTION,
    "look": OBSERVATION, "move_to": NAVIGATION,
    "pick_up": MANIPULATION, "drop": MANIPULATION, "give": MANIPULATION,
    "talk_to": COMMUNICATION,
    "web_search": DIGITAL_LOOKUP, "read_webpage": DIGITAL_LOOKUP,
    "run_python": CALCULATION, "finish_answer": REPLY,
}
EVERYTHING = frozenset(TOOL_EFFECTS.values())
READ_ONLY = frozenset({INTROSPECTION, DIGITAL_LOOKUP, CALCULATION, REPLY})
PHYSICAL = frozenset({OBSERVATION, NAVIGATION, MANIPULATION})

_HOUSE = frozenset({NAVIGATION, OBSERVATION})
CONDITION_EFFECTS = {
    "find_object": _HOUSE, "find_person": _HOUSE, "visit_room": _HOUSE,
    "identify_recipient": _HOUSE | {COMMUNICATION},
    "hold_object": _HOUSE | {MANIPULATION}, "place_object": _HOUSE | {MANIPULATION},
    "deliver": _HOUSE | {MANIPULATION},
    "notify": _HOUSE | {COMMUNICATION}, "notify_everyone": _HOUSE | {COMMUNICATION},
    "answer": frozenset({REPLY}),
}

PLAIN = {
    DIGITAL_LOOKUP: "looking things up online", CALCULATION: "running a calculation",
    OBSERVATION: "looking around the house", NAVIGATION: "moving around the house",
    MANIPULATION: "picking things up or handing them over",
    COMMUNICATION: "talking to someone else", INTROSPECTION: "checking my own status", REPLY: "answering you",
}
TOOL_PLAIN = {"run_python": "run Python", "read_webpage": "read webpages", "web_search": "search online"}


def plain(effects: Iterable[str]) -> str:
    words = [PLAIN[e] for e in sorted(effects) if e in PLAIN]
    if len(words) <= 1:
        return words[0] if words else "nothing"
    return ", ".join(words[:-1]) + " and " + words[-1]


def condition_effects(conditions) -> set[str]:
    """What a planned outcome needs, from its typed kind and required tools."""
    needed: set[str] = set()
    for condition in conditions:
        needed |= CONDITION_EFFECTS.get(condition.kind, EVERYTHING)  # unknown kinds need everything
        needed |= {TOOL_EFFECTS[tool] for tool in getattr(condition, "required_tools", []) or []}
    return needed


# ── restrictions: what people asked Kip not to do ───────────────────────────────
_CUE = re.compile(r"\b(don'?t|do not|never|no|without|avoid|stop|mustn'?t|shouldn'?t|not allowed to)\b")
_LIFT = re.compile(r"\b(you can|you may|feel free to|go ahead and|it'?s (?:fine|ok|okay) to|ok(?:ay)? to|you'?re allowed to)\b")
_FILLER = re.compile(r"^(?:(?:you|please|even|ever|any|bother|try|need|want|to|be|using|use of)\s+)*")
_ACTION = re.compile(r"^(use|using|run|running|read|reading|search|searching|go|going|enter|entering|open|opening|"
                     r"browse|browsing|google|look|looking|touch|touching|talk|talking|tell|telling|speak|leave|"
                     r"move|moving|visit|bother|access|check|head|walk|step|set foot|pick|hand|give|carry)\b")
# What each kind of wording rules out, mapped onto the registry.
_SUBJECTS = [
    (re.compile(r"\b(python|code|programs?|scripts?)\b"), {"tools": {"run_python"}}),
    (re.compile(r"\b(web ?pages?|pages?|websites?|sites?|links?|articles?)\b"), {"tools": {"read_webpage"}}),
    (re.compile(r"\b(search(?:ing)? (?:online|the web|the internet)|web ?search(?:es)?|google|search engines?)\b"),
     {"tools": {"web_search"}}),
    # Generic: only when nothing more specific about the web was said.
    (re.compile(r"\b(internet|online|the web|web)\b"), {"tools": {"web_search", "read_webpage"}, "generic": True}),
    (re.compile(r"\b(talk(?:ing)? to|speak(?:ing)? to|tell(?:ing)?|bother(?:ing)?|disturb(?:ing)?) "
                r"(?:anyone|anybody|people|everyone|the others)\b"), {"effects": {COMMUNICATION}}),
    (re.compile(r"\b(touch(?:ing)?|pick(?:ing)? (?:anything|things) up|handl(?:e|ing)|mov(?:e|ing) (?:anything|things))\b"),
     {"effects": {MANIPULATION}}),
    (re.compile(r"\b(leave (?:this|the) room|go anywhere|wander|move around)\b"), {"effects": {NAVIGATION}}),
]
_INTO_ROOM = r"\b(go|going|enter|entering|head|walk|step|set foot|get|move|visit|into|in|inside)\b"


class Restrictions:
    """What people in this conversation asked Kip not to do, until they say otherwise."""

    def __init__(self) -> None:
        self.tools: set[str] = set()
        self.effects: set[str] = set()
        self.rooms: set[str] = set()

    def __bool__(self) -> bool:
        return bool(self.tools or self.effects or self.rooms)

    @staticmethod
    def _clauses(text: str, cue: re.Pattern):
        lowered = " ".join(text.casefold().replace("’", "'").split())
        for match in cue.finditer(lowered):
            clause = re.split(r"[.;!?]|,\s*(?:but|and|so|then)\b|\bbut\b", lowered[match.end():], maxsplit=1)[0]
            yield match.group(1), _FILLER.sub("", clause.strip())

    @staticmethod
    def _subjects(text: str, at_start: bool) -> dict:
        found = {"tools": set(), "effects": set(), "rooms": set()}
        for pattern, target in _SUBJECTS:
            if target.get("generic") and found["tools"]:
                continue
            if pattern.match(text) if at_start else pattern.search(text):
                found["tools"] |= target.get("tools", set())
                found["effects"] |= target.get("effects", set())
        return found

    def _targets(self, clause: str, bare: bool, rooms: dict[str, str]) -> dict:
        nothing = {"tools": set(), "effects": set(), "rooms": set()}
        if bare:
            # "no Python", "without the internet": the thing must follow the word directly.
            # "No, use Python" and "No, go into the kitchen" answer something else.
            if not clause[:1].isalnum():
                return nothing
            return self._subjects(re.sub(r"^(?:(?:more|using|the|any)\s+)*", "", clause), at_start=True)
        if not _ACTION.match(clause):
            return nothing  # "I don't know if the web is down"
        found = self._subjects(clause, at_start=False)
        if re.search(_INTO_ROOM, clause):
            spaced = clause.replace("_", " ")
            for room_id, name in rooms.items():
                names = {room_id.replace("_", " "), name.casefold()}
                if any(re.search(r"\b" + re.escape(n) + r"\b", spaced) for n in names if n):
                    found["rooms"].add(room_id)
        return found

    def update(self, text: str, rooms: dict[str, str] | None = None) -> bool:
        """Apply prohibitions and lifted prohibitions in the person's own words."""
        rooms = rooms or {}
        before = (set(self.tools), set(self.effects), set(self.rooms))
        for cue, clause in self._clauses(text, _LIFT):
            found = self._targets(clause, False, rooms)
            self.tools -= found["tools"]
            self.effects -= found["effects"]
            self.rooms -= found["rooms"]
        for cue, clause in self._clauses(text, _CUE):
            found = self._targets(clause, cue in {"no", "without"}, rooms)
            self.tools |= found["tools"]
            self.effects |= found["effects"]
            self.rooms |= found["rooms"]
        return before != (self.tools, self.effects, self.rooms)

    def forbids_tool(self, tool: str) -> bool:
        return tool in self.tools or TOOL_EFFECTS.get(tool) in self.effects

    def describe(self) -> list[str]:
        notes = [TOOL_PLAIN[t] for t in sorted(self.tools)]
        notes += [PLAIN[e] for e in sorted(self.effects)]
        notes += [f"go into the {r.replace('_', ' ')}" for r in sorted(self.rooms)]
        return notes

    def mentions_only_forbidden_tools(self, goal: str) -> bool:
        """A goal that names its tools, all of which are forbidden (e.g. "...using Python")."""
        named = self._subjects(goal.casefold(), at_start=False)["tools"]
        return bool(named) and all(self.forbids_tool(tool) for tool in named)


# ── grants: what this task was authorized to do ────────────────────────────────
class Grant(BaseModel):
    effects: list[str] = Field(default_factory=lambda: sorted(EVERYTHING))
    source: str = "api"            # request | question | offer | autonomy | api
    authorized_text: str = ""      # the words that authorized it: the person's own, or the offer they accepted

    @classmethod
    def full(cls, source: str, text: str = "") -> Grant:
        return cls(effects=sorted(EVERYTHING), source=source, authorized_text=text)

    @classmethod
    def read_only(cls, text: str) -> Grant:
        return cls(effects=sorted(READ_ONLY), source="question", authorized_text=text)

    def names(self, *words: str) -> bool:
        text = self.authorized_text.casefold().replace("_", " ")
        return any(w and re.search(r"\b" + re.escape(w.casefold().replace("_", " ")) + r"\b", text) for w in words)

    def deny(self, tool: str, arguments: dict, restrictions: Restrictions | None = None,
             objects: dict | None = None) -> str:
        """Why this exact call may not run now, or "" when it may."""
        effect = TOOL_EFFECTS.get(tool)
        if effect is None:
            return f"{tool} is not a registered tool"
        restrictions = restrictions or Restrictions()
        if tool in restrictions.tools:
            return f"you asked me not to {TOOL_PLAIN.get(tool, tool)}"
        if effect in restrictions.effects:
            return f"you asked me to avoid {PLAIN[effect]}"
        room = arguments.get("room") if tool == "move_to" else None
        if room and room in restrictions.rooms:
            return f"you asked me not to go into the {room.replace('_', ' ')}"
        if effect not in self.effects:
            return f"this errand was only cleared for {plain(set(self.effects) - {INTROSPECTION, REPLY})}"
        if effect == MANIPULATION:
            thing = arguments.get("object", "")
            item = (objects or {}).get(thing) or {}
            # Consequential handling (medicine and the like) needs the item named by whoever authorized it.
            if item.get("sensitive") and not self.names(thing, item.get("name", "")):
                return f"handling the {item.get('name', thing)} needs someone to ask me for it by name"
        return ""

    def uncovered(self, conditions, restrictions: Restrictions | None = None) -> set[str]:
        """Effects the plan needs that this grant, or a standing restriction, rules out."""
        restrictions = restrictions or Restrictions()
        needed = condition_effects(conditions) - {INTROSPECTION}
        return {e for e in needed if e not in self.effects or e in restrictions.effects}


# A goal that explicitly scopes itself to the web or a calculation. Only used to decide
# whether a question may *start* a read-only task; the grant is what limits it.
_DIGITAL_SCOPE = re.compile(r"\b(online|the web|web ?search|internet|website|webpage|google|python|look up|"
                            r"calculat\w*|comput\w*|convert\w*)\b|https?://")


def digital_scope(goal: str) -> bool:
    return bool(_DIGITAL_SCOPE.search(goal.casefold()))



# ── tools a person explicitly asks to have run ─────────────────────────────────
_URL = re.compile(r"https?://[^\s,;)]+|\bwww\.[^\s,;)]+|\b(?:read|open|visit|summari[sz]e|what'?s on|what is on|look at)"
                  r"\b[^.?!]{0,25}?\b[\w-]+\.(?:com|org|net|io|dev|ai|in|co\.uk)\b")
_RUN_PYTHON = re.compile(r"\b(?:run|runs|running|use|using|execute|executing|with|in|via|through)\b(?:\s+\S+){0,2}?\s+python\b"
                         r"|\bpython\b(?:\s+\S+){0,3}?\s+(?:run|execute)\b")
_WEB_SEARCH = re.compile(r"\b(?:search|look (?:it |that |this )?up|google|browse|find)\b[^.?!]{0,30}?\b(?:online|the web|web|"
                         r"internet)\b|\bweb search\b|\bgoogle\b|\bsearch online\b")
_NEGATED_CLAUSE = re.compile(r"^\W*(?:don'?t|do not|never|no|without|avoid|stop|not)\b")


def explicitly_requested_tools(text: str) -> list[str]:
    """Tools the person's own words ask to have run ("use Python to…", "read https://…").

    Negated clauses ("don't run Python") and permission grants ("you can use Python
    again") are not requests to run anything.
    """
    found: set[str] = set()
    lowered = " ".join(text.casefold().replace("’", "'").split())
    for clause in re.split(r"(?<!\w\.\w)[.;!?](?=\s|$)|,\s*(?:but|and then)\b|\bbut\b", lowered):
        clause = clause.strip()
        if not clause or _NEGATED_CLAUSE.match(clause) or _LIFT.search(clause):
            continue
        if _RUN_PYTHON.search(clause):
            found.add("run_python")
        if _URL.search(clause):
            found.add("read_webpage")
        if _WEB_SEARCH.search(clause):
            found.add("web_search")
    return sorted(found)
