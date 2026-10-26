"""What Kip can actually do and has actually done, built from state and tool evidence.

Everything here is deterministic. It gives the conversation model the facts it must speak
from (capabilities, scans, who was where and how we know), and checks its finished reply
for claims those facts do not support.
"""
import re

from app.tasks.permissions import TOOL_PLAIN

DIGITAL = {"web_search": "search the web", "read_webpage": "read webpages", "run_python": "run Python"}
HOUSE = "move around the house, look, pick up and hand over things, talk to people"


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
    rooms = sorted(restrictions.rooms)
    groups["available"].append(HOUSE + (f" (not the {', '.join(r.replace('_', ' ') for r in rooms)})" if rooms else ""))
    return "; ".join(f"{key}: {', '.join(values)}" for key, values in groups.items() if values)


def room_scans(manager) -> dict[str, dict]:
    """The latest time Kip actually looked in each room, and exactly what was there."""
    scans: dict[str, dict] = {}
    for _, entry in _entries(manager):
        if entry["tool"] == "look" and entry["success"]:
            obs = entry["observation"]
            scans[obs["room"]] = {
                "people": [p["id"] for p in obs.get("people", [])],
                "objects": [o["id"] for o in obs.get("objects", [])] + [h["object"] for h in obs.get("held_objects", [])],
                "at": entry.get("observed_at", "")}
    return scans


def _place(room: str) -> str:
    return room.replace("_", " ")


def last_seen(manager, world: dict) -> dict:
    """Where each person was last seen, and how Kip knows: a fresh look, or an older record."""
    scans = room_scans(manager)
    people = world.get("people", {})
    view = {}
    for pid, person in people.items():
        name = person.get("name", pid)
        fresh = [(room, scan) for room, scan in scans.items() if pid in scan["people"]]
        if fresh:
            room, scan = fresh[-1]
            view[name] = f"I saw them in the {_place(room)}" + (f" at {scan['at']}" if scan["at"] else "")
            continue
        record = manager.persistent.people.get(pid) if manager.persistent else None
        if record and record.last_seen_room:
            note = f"older record only: last seen in the {_place(record.last_seen_room)}"
            scan = scans.get(record.last_seen_room)
            if scan is not None:
                note += f"; not there when I looked" + (f" at {scan['at']}" if scan["at"] else "")
            view[name] = note
        else:
            view[name] = "unknown"
    return view


def describe_scan(room: str, scan: dict, world: dict) -> str:
    names = {**{k: v.get("name", k) for k, v in world.get("people", {}).items()},
             **{k: v.get("name", k).lower() for k, v in world.get("objects", {}).items()}}
    people = [names.get(p, p) for p in scan["people"]]
    things = [f"the {names.get(o, o)}" for o in scan["objects"]]
    seen = people + things
    contents = (", ".join(seen[:-1]) + " and " + seen[-1]) if len(seen) > 1 else (seen[0] if seen else "")
    return f"the {_place(room)}: " + (contents if contents else "nobody there, and nothing I could pick out")


def task_record(context, world: dict) -> dict:
    """What one errand actually did, tried, was refused and left undone."""
    did, refused, scans = [], [], {}
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
        elif tool == "look":
            scans[obs["room"]] = {"people": [p["id"] for p in obs.get("people", [])],
                                  "objects": [o["id"] for o in obs.get("objects", [])]}
        elif tool == "talk_to":
            did.append(f"spoke to {obs.get('person')}")
        elif tool == "give":
            did.append(f"gave the {obs.get('object')} to {obs.get('person')}")
    did += ["looked in " + describe_scan(room, scan, world) for room, scan in scans.items()]
    record = {"goal": context.goal, "outcome": context.status, "did": did[:6]}
    if refused:
        record["refused"] = sorted(set(refused))
    if getattr(context, "blocked", None):
        record["not_done"] = list(context.blocked)
    if context.answer and context.answer.get("answer"):
        record["answer"] = context.answer["answer"][:160]
    return record


# ── checking a finished reply ──────────────────────────────────────────────────
_RAN_PYTHON = re.compile(r"\bi(?:'ve| have| just)?\s+(?:ran|run|executed|computed it in|calculated it in)\b"
                         r"|\boutput\s*:|\bhere(?:'s| is) the output\b|\bthe output (?:is|was)\b|\bran (?:it|that|the code)\b")
_CODE_BLOCK = re.compile(r"```")
_READ_PAGE = re.compile(r"\bi(?:'ve| have| just)?\s+(?:read|opened|visited|checked) (?:the |that |this )?(?:page|site|website|"
                        r"webpage|https?://)")
_SEARCHED = re.compile(r"\bi(?:'ve| have| just)?\s+(?:searched|googled|looked (?:it|that) up online)\b")
_HEDGE = re.compile(r"\b(?:usually|normally|often|typically|tends to|used to|earlier|before|last seen|might|may|"
                    r"maybe|probably|perhaps|could be|older|record)\b")
_PRESENCE = re.compile(r"\b(?:was|is|were|'s|been)\s+(?:\w+\s+){0,2}?(?:in|at|inside)\b")


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


def contradicted_presence(reply: str, manager, world: dict) -> list[str]:
    """'Paul was in the office' when Kip's latest look at the office found no Paul."""
    scans = room_scans(manager)
    people = {pid: {pid.casefold(), str(p.get("name", pid)).casefold()} for pid, p in world.get("people", {}).items()}
    rooms = {rid: {rid.replace("_", " ").casefold(), str(r.get("name", rid)).casefold()}
             for rid, r in world.get("rooms", {}).items()}
    found, last_person = [], None
    for sentence in re.split(r"(?<=[.!?])\s+", reply.casefold().replace("’", "'")):
        named = [pid for pid, names in people.items()
                 if any(re.search(r"\b" + re.escape(n) + r"\b", sentence) for n in names)]
        if named:
            last_person = named[-1]
        subject = named or ([last_person] if last_person and re.search(r"\b(he|she|they)\b", sentence) else [])
        if not subject or not _PRESENCE.search(sentence) or _HEDGE.search(sentence):
            continue
        for rid, names in rooms.items():
            if rid in scans and any(re.search(r"\b" + re.escape(n) + r"\b", sentence) for n in names):
                for pid in subject:
                    if pid not in scans[rid]["people"]:
                        found.append((pid, rid))
    return found
