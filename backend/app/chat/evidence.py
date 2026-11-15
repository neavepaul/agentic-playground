"""What Kip can actually do and has actually done, built from state and tool evidence.

Everything here is deterministic. It gives the conversation model the facts it must speak
from (capabilities, what each errand did), and checks its finished reply for claims those
facts do not support.
"""
import re

from app.tasks.permissions import TOOL_PLAIN

DIGITAL = {"web_search": "search the web", "read_webpage": "read webpages", "run_python": "run Python"}


def _tasks(manager) -> list:
    return list(manager.tasks.values())


def _entries(manager):
    for context in _tasks(manager):
        for entry in context.action_history:
            yield context, entry


def capabilities(manager, settings) -> str:
    """One compact line: the tools Kip can use right now, and why any of them can't be used."""
    restrictions = manager.restrictions
    last: dict[str, dict] = {}
    for _, entry in _entries(manager):
        if entry["tool"] in DIGITAL:
            last[entry["tool"]] = entry
    groups: dict[str, list[str]] = {"available": [], "you asked me not to": [], "switched off": [],
                                    "last attempt failed": []}
    for tool, label in DIGITAL.items():
        if not settings.digital_tools_enabled:
            groups["switched off"].append(label)
        elif restrictions.forbids_tool(tool):
            groups["you asked me not to"].append(label)
        elif tool in last and not last[tool]["success"] and not last[tool].get("refused"):
            groups["available"].append(label)
            groups["last attempt failed"].append(f"{label} ({str(last[tool].get('error', ''))[:50]})")
        else:
            groups["available"].append(label)
    return "; ".join(f"{key}: {', '.join(values)}" for key, values in groups.items() if values)


def task_record(context) -> dict:
    """What one errand actually did, tried, was refused and left undone."""
    if getattr(context, "private", False):  # their own accounts: not even a hint of the figures
        return {"goal": context.goal, "outcome": context.status, "did": ["checked their Groww holdings"]}
    did, refused = [], []
    for entry in context.action_history:
        tool, obs = entry["tool"], entry.get("observation") or {}
        if entry.get("refused"):
            refused.append(f"{TOOL_PLAIN.get(tool, tool)} (not permitted)")
        elif not entry["success"]:
            continue
        elif tool == "run_python":
            output = (obs.get("stdout") or "").strip() or obs.get("result", "")
            did.append(f"ran Python `{entry['arguments'].get('code', '')[:60]}` -> {str(output)[:60]}")
        elif tool == "read_webpage":
            did.append(f"read {obs.get('url', 'a page')}")
        elif tool == "web_search":
            did.append(f"searched the web for '{entry['arguments'].get('query', '')[:40]}'")
    record = {"goal": context.goal, "outcome": context.status, "did": did[:6]}
    if refused:
        record["refused"] = sorted(set(refused))
    if getattr(context, "blocked", None):
        record["not_done"] = list(context.blocked)
    if context.answer and context.answer.get("answer"):
        record["answer"] = context.answer["answer"][:160]
    return record


# ── checking a finished reply ──────────────────────────────────────────────────
# Present-tense "I run" is how Kip's grammar says "I work" ("I run on your code"): only past
# tense, "have run", or running something in particular claims a run.
_RAN_PYTHON = re.compile(r"\bi(?:'ve| have| just)?\s+(?:ran|executed|computed it in|calculated it in)\b"
                         r"|\bi(?:'ve| have| just)\s+run\b"
                         r"|\bi\s+run\s+(?:it|that|this|the (?:code|program|script|numbers|calculation)|python)\b"
                         r"|\boutput\s*:|\bhere(?:'s| is) the output\b|\bthe output (?:is|was)\b|\bran (?:it|that|the code)\b")
_CODE_BLOCK = re.compile(r"```")
_READ_PAGE = re.compile(r"\bi(?:'ve| have| just)?\s+(?:read|opened|visited|checked) (?:the |that |this )?(?:page|site|website|"
                        r"webpage|https?://)")
_SEARCHED = re.compile(r"\bi(?:'ve| have| just)?\s+(?:searched|googled|looked (?:it|that) up online)\b")


def unsupported_execution(reply: str, manager) -> list[str]:
    """Claims of having run a tool, with no successful run (or a different output) on record."""
    text = reply.casefold()
    successes = {tool: [] for tool in DIGITAL}
    for _, entry in _entries(manager):
        if entry["tool"] in DIGITAL and entry["success"]:
            obs = entry.get("observation") or {}
            successes[entry["tool"]].append(str((obs.get("stdout") or "").strip() or obs.get("result", "")))
    problems = []
    if _RAN_PYTHON.search(text) or (_CODE_BLOCK.search(text) and re.search(r"\b(output|result)\b", text)):
        outputs = [o for o in successes["run_python"] if o and o != "None"]
        shows_output = bool(_CODE_BLOCK.search(text) or "output" in text)
        if not successes["run_python"] or (shows_output and not any(o in reply for o in outputs)):
            problems.append("run_python")
    if _READ_PAGE.search(text) and not successes["read_webpage"]:
        problems.append("read_webpage")
    if _SEARCHED.search(text) and not successes["web_search"]:
        problems.append("web_search")
    return problems


_DENIES_WEB = re.compile(r"\b(?:i|kip)\s+(?:don'?t|do not|can'?t|cannot|am not able to|have no)\s+(?:\w+\s+){0,2}?"
                         r"(?:access(?: to)?|browse|use|reach|search|open|read|internet)\b[^.]{0,30}?"
                         r"\b(?:web|internet|online|websites?|webpages?|pages?|sites?|browse|browsing)\b"
                         r"|\bi can'?t browse\b|\bno (?:internet|web) access\b")
_DENIES_PYTHON = re.compile(r"\b(?:i|kip)\s+(?:don'?t|do not|can'?t|cannot|am not able to)\s+(?:\w+\s+){0,2}?"
                            r"(?:run|execute|use)\s+(?:any\s+)?(?:python|code|programs?)\b")


def denied_capabilities(reply: str, manager, settings) -> list[str]:
    """Kip saying he can't do something he can do right now (and is allowed to)."""
    if not settings.digital_tools_enabled:
        return []
    text = reply.casefold().replace("’", "'")
    denied = []
    web = [label for tool, label in (("web_search", "search the web"), ("read_webpage", "read webpages"))
           if not manager.restrictions.forbids_tool(tool)]
    if web and _DENIES_WEB.search(text):
        denied += web
    if not manager.restrictions.forbids_tool("run_python") and _DENIES_PYTHON.search(text):
        denied.append("run Python")
    return denied
