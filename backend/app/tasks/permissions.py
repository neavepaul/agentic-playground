"""What a task is allowed to do, checked where tools actually run.

Every executable tool has one effect, from this backend registry, never from model text.
A task carries a Grant: the effects the person authorized, and the words that
authorized it. Restrictions are what the person asked Kip not to do; they are live and
shared, so a later "don't use Python" also binds tasks that are already queued or running.

The manager asks deny() right before every dispatch. Planning and the Explorer use the
same rules to avoid choosing a forbidden action, but the dispatch check is the guarantee.
"""
from __future__ import annotations

import re
from typing import Iterable

from pydantic import BaseModel, Field

# ── the registry ────────────────────────────────────────────────────────────────
DIGITAL_LOOKUP = "digital_lookup"          # reads public web content
CALCULATION = "calculation"                # restricted Python, no host access
REPLY = "reply"                            # answering the person Kip is talking to

TOOL_EFFECTS = {
    "web_search": DIGITAL_LOOKUP, "read_webpage": DIGITAL_LOOKUP,
    "run_python": CALCULATION, "finish_answer": REPLY,
}
EVERYTHING = frozenset(TOOL_EFFECTS.values())
# Every tool Kip has only reads or computes; a question may use all of them.
READ_ONLY = EVERYTHING

CONDITION_EFFECTS = {"answer": frozenset({REPLY})}

PLAIN = {DIGITAL_LOOKUP: "looking things up online", CALCULATION: "running a calculation",
         REPLY: "answering you"}
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
_FILLER = re.compile(r"^(?:(?:you|please|even|ever|any|bother|try|need|want|to|be|dare|use of)\s+)*")
_ACTION = re.compile(r"^(use|using|run|running|read|reading|search|searching|go|going|open|opening|"
                     r"browse|browsing|google|look|looking|access|check|visit|do|doing|make|making|"
                     r"touch|touching)\b")
# "Stop using Python", "avoid the internet", "never touch python": these may name the thing straight away.
_DIRECT_CUES = {"stop", "avoid", "never"}
# "Stay offline", "keep it offline": no cue word, but plainly no internet.
_OFFLINE = re.compile(r"\b(?:stay|keep(?: it)?|remain|work|be|go)\s+offline\b")
# A message that lifts and also forbids or defers ("I know you can run code, but please don't",
# "you can use python later, not now", "ok to use python? no, don't") lifts nothing.
_HOLD = re.compile(r"\b(don'?t|do not|never|not now|later|avoid|stop|mustn'?t|shouldn'?t|not allowed|without)\b")
# What each kind of wording rules out, mapped onto the registry.
_SUBJECTS = [
    (re.compile(r"\b(python|code|programs?|scripts?)\b"), {"tools": {"run_python"}}),
    (re.compile(r"\b(web ?pages?|pages?|websites?|sites?|links?|articles?)\b"), {"tools": {"read_webpage"}}),
    (re.compile(r"\b(search(?:ing)? (?:online|the web|the internet)|web ?search(?:es)?|google|search engines?)\b"),
     {"tools": {"web_search"}}),
    # Generic: only when nothing more specific about the web was said.
    (re.compile(r"\b(internet|online|the web|web)\b"), {"tools": {"web_search", "read_webpage"}, "generic": True}),
]


class Restrictions:
    """What the person asked Kip not to do, until they say otherwise."""

    def __init__(self) -> None:
        self.tools: set[str] = set()

    def __bool__(self) -> bool:
        return bool(self.tools)

    @staticmethod
    def _clauses(text: str, cue: re.Pattern):
        lowered = " ".join(text.casefold().replace("’", "'").split())
        for match in cue.finditer(lowered):
            clause = re.split(r"[.;!?]|,\s*(?:but|and|so|then)\b|\bbut\b", lowered[match.end():], maxsplit=1)[0]
            yield match.group(1), _FILLER.sub("", clause.strip())

    @staticmethod
    def _subjects(text: str, at_start: bool) -> set[str]:
        found: set[str] = set()
        for pattern, target in _SUBJECTS:
            if target.get("generic") and found:
                continue
            if pattern.match(text) if at_start else pattern.search(text):
                found |= target["tools"]
        return found

    def _targets(self, clause: str, bare: bool) -> set[str]:
        if bare:
            # "no Python", "without the internet": the thing must follow the word directly.
            # "No, use Python" answers something else.
            if not clause[:1].isalnum():
                return set()
            return self._subjects(re.sub(r"^(?:(?:more|using|the|any)\s+)*", "", clause), at_start=True)
        if not _ACTION.match(clause):
            return set()  # "I don't know if the web is down"
        return self._subjects(clause, at_start=False)

    def _direct(self, clause: str) -> set[str]:
        """After stop/avoid/never: an action ("stop using python") or the thing itself ("avoid the internet")."""
        return self._targets(clause, False) or self._subjects(
            re.sub(r"^(?:(?:more|using|the|any|all)\s+)*", "", clause), at_start=True)

    def update(self, text: str) -> bool:
        """Apply prohibitions and lifted prohibitions in the person's own words."""
        before = set(self.tools)
        lowered = " ".join(text.casefold().replace("’", "'").split())
        if not _HOLD.search(lowered):  # a message that also forbids or defers lifts nothing
            for cue, clause in self._clauses(lowered, _LIFT):
                self.tools -= self._targets(clause, False)
        for cue, clause in self._clauses(text, _CUE):
            if cue in _DIRECT_CUES:
                self.tools |= self._direct(clause)
            else:
                self.tools |= self._targets(clause, cue in {"no", "without"})
        if _OFFLINE.search(lowered):
            self.tools |= {"web_search", "read_webpage"}
        return before != self.tools

    def forbids_tool(self, tool: str) -> bool:
        return tool in self.tools

    def describe(self) -> list[str]:
        return [TOOL_PLAIN[t] for t in sorted(self.tools)]

    def mentions_only_forbidden_tools(self, goal: str) -> bool:
        """A goal that names its tools, all of which are forbidden (e.g. "...using Python")."""
        named = self._subjects(goal.casefold(), at_start=False)
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

    def deny(self, tool: str, arguments: dict, restrictions: Restrictions | None = None) -> str:
        """Why this exact call may not run now, or "" when it may."""
        effect = TOOL_EFFECTS.get(tool)
        if effect is None:
            return f"{tool} is not a registered tool"
        restrictions = restrictions or Restrictions()
        if tool in restrictions.tools:
            return f"you asked me not to {TOOL_PLAIN.get(tool, tool)}"
        if effect not in self.effects:
            return f"this errand was only cleared for {plain(set(self.effects) - {REPLY})}"
        return ""

    def uncovered(self, conditions) -> set[str]:
        """Effects the plan needs that this grant does not cover."""
        return {e for e in condition_effects(conditions) if e not in self.effects}


# A goal that explicitly scopes itself to the web or a calculation. Only used to decide
# whether a question may *start* a read-only task; the grant is what limits it.
_DIGITAL_SCOPE = re.compile(r"\b(online|the web|web ?(?:search|lookup)|internet|website|webpage|google|python|look ?up|"
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
