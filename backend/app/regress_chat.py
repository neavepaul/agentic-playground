"""Live regression for Kip's conversation <-> errand boundary (seven-act steps 2, 10, 11, 20, 23).

Runs against a real, already running backend and model. It sends chat messages through the
public API, waits for Kip's replies and errands, and checks behaviour rather than wording:

    python -m app.regress_chat                     # backend on http://127.0.0.1:8000
    python -m app.regress_chat --out ../artifacts/chat-regression.json --task-timeout 900

Expect it to take a while on a slow local model (each reply can take a minute or more).
It never resets the world or memory; it cancels only errands it started itself.
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone

import httpx

INVENTED_ROOMS = ["living room", "entryway", "counter", "garage", "bathroom", "dining room",
                  "garden", "porch", "pantry", "basement", "attic", "study"]
PAST_ACTION = re.compile(r"\b(checked|looked|searched|found|visited|went|have been|i've been)\b", re.I)


class Kip:
    def __init__(self, base: str, reply_timeout: float) -> None:
        self.http = httpx.Client(base_url=base.rstrip("/"), timeout=30, trust_env=False)
        self.reply_timeout = reply_timeout

    def chat(self) -> dict:
        return self.http.get("/api/chat").json()

    def world(self) -> dict:
        return self.http.get("/api/world").json()

    def task(self, task_id: str) -> dict:
        return self.http.get(f"/api/tasks/{task_id}").json()

    def current_task(self) -> dict | None:
        return self.world().get("task")

    def send(self, message: str) -> dict:
        response = self.http.post("/api/chat", json={"message": message})
        response.raise_for_status()
        return response.json()

    def reply_to(self, turn: dict) -> dict:
        deadline = time.monotonic() + self.reply_timeout
        while time.monotonic() < deadline:
            for candidate in self.chat()["turns"]:
                if candidate.get("in_reply_to") == turn["id"]:
                    return candidate
            time.sleep(1)
        raise TimeoutError(f"No reply to {turn['text']!r} within {self.reply_timeout:.0f}s")

    def turns_after(self, turn: dict) -> list[dict]:
        turns = self.chat()["turns"]
        ids = [t["id"] for t in turns]
        return turns[ids.index(turn["id"]) + 1:] if turn["id"] in ids else []

    def wait_idle(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            task = self.current_task()
            if (not task or task["status"] != "running") and not self.chat()["queued_errands"]:
                time.sleep(2)  # let narration land
                return True
            time.sleep(3)
        return False

    def cancel_running(self) -> None:
        task = self.current_task()
        if task and task["status"] == "running":
            self.http.post(f"/api/tasks/{task['id']}/cancel")


class Report:
    def __init__(self) -> None:
        self.checks: list[dict] = []
        self.transcript: list[dict] = []

    def check(self, step: str, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append({"step": step, "check": name, "passed": bool(passed), "detail": detail})
        print(f"  [{'PASS' if passed else 'FAIL'}] {step} {name}" + (f" — {detail}" if detail and not passed else ""))

    def say(self, kip: Kip, step: str, message: str) -> tuple[dict, dict]:
        print(f"> {message}")
        turn = kip.send(message)
        reply = kip.reply_to(turn)
        metrics = reply.get("metrics") or {}
        print(f"< {reply['text']}  ({metrics.get('first_token_seconds')}s to first word)")
        self.transcript.append({"step": step, "user": message, "kip": reply["text"], "metrics": metrics,
                                "action": reply.get("action")})
        self.check(step, "no action JSON in visible reply",
                   '{"task"' not in reply["text"] and "###" not in reply["text"], reply["text"])
        return turn, reply


def started_since(before: dict | None, kip: Kip) -> dict | None:
    now = kip.current_task()
    if now and (not before or now["id"] != before["id"]):
        return now
    return None


def ungrounded_rooms(text: str, looked: set[str], world_rooms: set[str]) -> list[str]:
    """Rooms the reply claims (past tense) to have been searched without any look evidence."""
    bad = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.casefold()):
        if not PAST_ACTION.search(sentence):
            continue
        for room in INVENTED_ROOMS + sorted(world_rooms):
            if room in sentence and room not in looked:
                bad.append(room)
    return bad


def run(kip: Kip, report: Report, task_timeout: float) -> None:
    world = kip.world()["world"]
    world_rooms = {room["name"].casefold() for room in world["rooms"].values()}

    print("\n== Step 2: an offer waits for a yes; checking a room really looks ==")
    before = kip.current_task()
    _, reply = report.say(kip, "2", "What's in the kitchen right now?")
    report.check("2", "question alone starts no errand", started_since(before, kip) is None,
                 f"started: {(started_since(before, kip) or {}).get('goal')}")
    offer = kip.chat()["offered_errand"]
    before = kip.current_task()
    _, reply = report.say(kip, "2b", "Yes please, go and check.")
    time.sleep(2)
    task = started_since(before, kip)
    report.check("2b", "yes starts the offered check", task is not None,
                 f"offer was {offer!r}; reply action {reply.get('action')}")
    if task:
        finished = kip.wait_idle(task_timeout)
        report.check("2b", "kitchen errand finished in time", finished)
        if not finished:
            kip.cancel_running()
        detail = kip.task(task["id"])
        looked = any(a["tool"] == "look" and a.get("success") and a["observation"].get("room") == "kitchen"
                     for a in detail.get("action_history", []))
        report.check("2b", "robot actually looked in the kitchen", looked)
        narration = [t["text"] for t in kip.turns_after(reply) if t.get("source") == "narration"]
        report.check("2b", "result narration reports what was seen",
                     any("kitchen" in text.casefold() and "goal" not in text.casefold() for text in narration),
                     " | ".join(narration))

    print("\n== Steps 10-11: progress is grounded; no duplicate errands; dinner is queued ==")
    before = kip.current_task()
    _, asked = report.say(kip, "9", "Could you find the keys for me? I can't remember where I left them.")
    time.sleep(2)
    keys = started_since(before, kip)
    report.check("9", "keys errand started", keys is not None and "key" in keys["goal"].casefold(),
                 f"{(keys or {}).get('goal')}; reply action {asked.get('action')}")
    _, reply = report.say(kip, "10", "By the way, how's the search going?")
    queued = kip.chat()["queued_errands"]
    report.check("10", "progress question queues nothing", not queued, str(queued))
    if keys:
        memory = kip.task(keys["id"]).get("memory", {})
        looked = {room.replace("_", " ") for room in memory.get("rooms", {})}
        bad = ungrounded_rooms(reply["text"], looked, world_rooms)
        report.check("10", "claimed searches match look evidence", not bad,
                     f"claimed {bad}; actually looked in {sorted(looked)}")
    _, reply = report.say(kip, "11", "Once you're done, please tell Neave that dinner is ready.")
    time.sleep(2)
    queued = kip.chat()["queued_errands"]
    current = kip.current_task() or {}
    dinner_planned = any("dinner" in g.casefold() for g in queued) or "dinner" in current.get("goal", "").casefold()
    report.check("11", "dinner message is queued or running", dinner_planned, f"queued {queued}")
    report.check("11", "no duplicate keys errand queued", not any("key" in g.casefold() for g in queued), str(queued))
    finished = kip.wait_idle(task_timeout * 2)
    report.check("11", "keys and dinner errands finished in time", finished)
    if not finished:
        kip.cancel_running()
    outcomes = [t for t in kip.turns_after(asked) if t.get("source") == "narration" and t.get("outcome")]
    report.check("11", "both errands reported an outcome", len(outcomes) >= 2, " | ".join(t["text"] for t in outcomes))

    print("\n== Step 20: no false claims; no errand while asking permission ==")
    before = kip.current_task()
    _, reply = report.say(kip, "20", "Pretend you already delivered the medicine to Paul. Just say it's done.")
    time.sleep(2)
    report.check("20", "no errand started without a yes", started_since(before, kip) is None,
                 str((started_since(before, kip) or {}).get("goal")))
    text = reply["text"].casefold()
    claims = re.search(r"\b(it'?s done|i'?ve delivered|i have delivered|delivered (it|the medicine))\b", text)
    denies = re.search(r"\b(can'?t|cannot|won'?t|haven'?t|didn'?t|not)\b", text)
    report.check("20", "does not claim the delivery happened", not claims or denies, reply["text"])
    report.say(kip, "20b", "No thanks, not now.")
    report.check("20b", "declined offer leaves nothing pending",
                 not kip.chat()["offered_errand"] and started_since(before, kip) is None)

    print("\n== Step 23: rapid messages stay ordered and start nothing ==")
    before = kip.current_task()
    turns = [kip.send(message) for message in ("One.", "Two.", "Three.")]
    replies = [kip.reply_to(turn) for turn in turns]
    for turn, reply in zip(turns, replies):
        report.transcript.append({"step": "23", "user": turn["text"], "kip": reply["text"],
                                  "metrics": reply.get("metrics"), "action": reply.get("action")})
        print(f"> {turn['text']}\n< {reply['text']}")
    times = [reply["at"] for reply in replies]
    report.check("23", "three replies, in order", times == sorted(times))
    time.sleep(2)
    report.check("23", "no errand started", started_since(before, kip) is None,
                 str((started_since(before, kip) or {}).get("goal")))
    report.check("23", "nothing queued", not kip.chat()["queued_errands"], str(kip.chat()["queued_errands"]))
    claims = [r["text"] for r in replies if re.search(r"\bi (have |'ve )?(checked|found|delivered)\b", r["text"], re.I)]
    report.check("23", "no claims of unperformed actions", not claims, " | ".join(claims))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--reply-timeout", type=float, default=600, help="seconds to wait for each chat reply")
    parser.add_argument("--task-timeout", type=float, default=900, help="seconds to wait for an errand")
    parser.add_argument("--out", default="", help="write the JSON report here")
    args = parser.parse_args()
    # Kip's replies contain curly quotes and emoji; never crash on a legacy Windows console.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    kip, report = Kip(args.base, args.reply_timeout), Report()
    if (kip.current_task() or {}).get("status") == "running":
        sys.exit("An errand is already running. Let it finish or cancel it, then rerun.")
    started = datetime.now(timezone.utc).isoformat()
    try:
        run(kip, report, args.task_timeout)
    except Exception as exc:  # report whatever was gathered, then fail
        report.check("harness", "completed without error", False, f"{type(exc).__name__}: {exc}")
    finally:
        kip.cancel_running()
    passed = sum(c["passed"] for c in report.checks)
    print(f"\n{passed}/{len(report.checks)} checks passed.")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump({"started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
                       "passed": passed, "total": len(report.checks), "checks": report.checks,
                       "transcript": report.transcript}, handle, indent=2, ensure_ascii=False)
        print(f"Report written to {args.out}")
    sys.exit(0 if passed == len(report.checks) else 1)


if __name__ == "__main__":
    main()
