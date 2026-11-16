"""What a conversation does with "remind me…", "set a timer", "what reminders do I have?" and "cancel…".

One helper per conversation (one person): it reads the message (when.py), acts on the scheduler, and
returns the reply as text, so the whole exchange needs no model and answers at once. It only ever touches
this person's own jobs. A request it cannot finish ("remind me to call mom": when?) is asked about, and the
next message completes it.
"""
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from app.scheduler import LimitError, Scheduler
from .when import (Request, Schedule, read_request, read_schedule, say_length, say_repeat, say_time)

LOOKUP_PREFIX = "Look up online: "
# A scheduled check runs unattended and lands in a chat, so it never reads someone's own accounts.
_ACCOUNTS = re.compile(r"\b(?:groww|portfolio|holdings?|demat|my (?:stocks?|shares|investments?|mutual funds?|"
                       r"balance|bank|account|net worth|returns?))\b", re.I)


@dataclass
class _Pending:
    kind: str
    what: str = ""
    schedule: Schedule | None = None


class Reminders:
    def __init__(self, scheduler: Scheduler, owner: str, session_id: str, speaker: str | None,
                 now: Callable[[], datetime]) -> None:
        self.scheduler, self.owner, self.session_id, self.speaker, self.now = scheduler, owner, session_id, speaker, now
        self._pending: _Pending | None = None

    # ── entry point ─────────────────────────────────────────────────────────
    def handle(self, text: str) -> str | None:
        """The reply to a message about reminders, or None when it is about something else."""
        now = self.now()
        pending, self._pending = self._pending, None
        request = read_request(text, now)
        if request is None and pending is not None:
            request = self._complete(pending, text, now)
        if request is None:
            return None
        if request.action == "list":
            return self._list(now)
        if request.action == "cancel":
            return self._cancel(request, now)
        if request.action == "ask_when":
            self._pending = _Pending(request.kind, request.what)
            if request.kind == "timer" and not request.what:
                return "How long? Say '5 minutes'."
            if request.note:
                return request.note if "?" in request.note else f"{request.note} When?"
            if request.kind == "timer":
                return "How long? Say '5 minutes'."
            return ("When? Say 'in 20 minutes', 'at 6pm' or 'tomorrow at 9'." if request.what
                    else "Remind you of what, and when?")
        if request.action == "ask_what":
            self._pending = _Pending(request.kind, schedule=request.schedule)
            return "Remind you of what?"
        return self._set(request, now)

    def _complete(self, pending: _Pending, text: str, now: datetime) -> Request | None:
        """The answer to "When?" (a time) or "Remind you of what?" (the words)."""
        if pending.schedule is None:
            schedule, rest, problem = read_schedule(text, now)
            if schedule is None and not problem:
                # "10 minutes" on its own, answering "How long?" or "When?": a length of time from now.
                schedule, rest, problem = read_schedule(f"in {text.strip()}", now)
                if schedule is not None and rest:
                    schedule = None          # more than a length of time ("in call mom"): not that
            if schedule is None:
                if problem:
                    self._pending = pending
                    return Request("ask_when", kind=pending.kind, what=pending.what, note=problem)
                return None                                   # not an answer: carry on as normal
            what = pending.what or re.sub(r"^\W*(?:to|that|about)\b", " ", rest).strip(" .,;:!?-")
            if not what and pending.kind != "timer":
                self._pending = _Pending(pending.kind, schedule=schedule)
                return Request("ask_what", kind=pending.kind, schedule=schedule)
            return Request("set", kind=pending.kind, what=what, schedule=schedule)
        words = " ".join(text.split())
        if not words or len(words) > 160 or words.endswith("?"):
            return None
        what = re.sub(r"^(?:to|that|about)\s+", "", words, flags=re.I).strip(" .,;:!?-")
        return Request("set", kind=pending.kind, what=what, schedule=pending.schedule) if what else None

    # ── setting ─────────────────────────────────────────────────────────────
    def _set(self, request: Request, now: datetime) -> str:
        schedule, kind, what = request.schedule, request.kind, request.what
        if kind == "lookup" and _ACCOUNTS.search(what):
            return "I won't schedule checks on your accounts. Ask me when you want them."
        text = f"{LOOKUP_PREFIX}{what}" if kind == "lookup" else what
        try:
            job = self.scheduler.add(self.owner, self.session_id, kind, text, schedule.first, schedule.repeat,
                                     self.speaker)
        except LimitError as exc:
            return str(exc)
        when = say_time(schedule.first, now)
        if kind == "timer":
            length = schedule.first.timestamp() - now.timestamp()
            return f"Timer set. {say_length(length)}." + (f" For: {what}." if what else "")
        repeat = say_repeat(schedule.repeat)
        where = f"{repeat}, first {when}" if repeat else when
        if kind == "lookup":
            return f"Set. {where}. I check: {what}."
        return f"Set. {where}. I remind you: {what}." if what else f"Set. {where}."

    # ── looking and cancelling ──────────────────────────────────────────────
    def _label(self, job) -> str:
        if job.kind == "timer":
            return "timer" + (f" ({job.text})" if job.text else "")
        if job.kind == "lookup":
            return "check: " + job.text.removeprefix(LOOKUP_PREFIX)
        return job.text or "reminder"

    def _line(self, index: int, job, now: datetime) -> str:
        due = datetime.fromtimestamp(job.due, now.tzinfo)
        repeat = say_repeat(job.repeat)
        when = f"{repeat}, next {say_time(due, now)}" if repeat else say_time(due, now)
        if job.kind == "timer":
            return f"{index}. {self._label(job)}, {say_length(job.due - now.timestamp())} left"
        return f"{index}. {self._label(job)}, {when}"

    def _list(self, now: datetime) -> str:
        jobs = self.scheduler.active(self.owner)
        if not jobs:
            return "Nothing set."
        lines = [self._line(i, job, now) for i, job in enumerate(jobs, 1)]
        return f"You have {len(jobs)}: " + " ".join(f"{line}." for line in lines)

    def _cancel(self, request: Request, now: datetime) -> str:
        jobs = self.scheduler.active(self.owner)
        if not jobs:
            return "Nothing set, so nothing to cancel."
        if request.noun:   # "the timer" is a timer, "that reminder" a reminder: never the other kind
            kind = request.noun
            jobs = [job for job in jobs if job.kind == kind]
            if not jobs:
                return f"No {'check' if kind == 'lookup' else kind} set."
        if request.all:
            gone = self.scheduler.cancel(self.owner, [job.id for job in jobs])
            return f"Cancelled all {len(gone)}." if len(gone) != 1 else f"Cancelled: {self._label(gone[0])}."
        words = set(request.match.split())
        if words:
            matches = [job for job in jobs if words <= set(re.findall(r"[\w']+", (job.text + " " + job.kind).casefold()))
                       or any(w in job.text.casefold() for w in words)]
        else:
            matches = jobs
        if not matches:
            return "No reminder like that. Say 'what reminders do I have?' to see them."
        if len(matches) > 1 and not words:
            return ("Which one? " + " ".join(f"{self._line(i, job, now)}." for i, job in enumerate(matches, 1))
                    + " Say its name, or 'cancel all'.")
        if len(matches) > 1:
            gone = self.scheduler.cancel(self.owner, [job.id for job in matches])
            return f"Cancelled {len(gone)}: " + "; ".join(self._label(job) for job in gone) + "."
        gone = self.scheduler.cancel(self.owner, [matches[0].id])
        return f"Cancelled: {self._label(gone[0])}." if gone else "Could not cancel that."


LATE_SECONDS = 120   # later than this is said out loud


def reminder_message(job, late: float) -> str:
    """What Kip says when a job is due. A late one (he was off, or it was queued) says how late."""
    delay = f" (late by {say_length(late)})" if late > LATE_SECONDS else ""
    if job.kind == "timer":
        return f"Timer done{': ' + job.text if job.text else ''}.{delay}"
    if job.kind == "lookup":
        return f"Scheduled check: {job.text.removeprefix(LOOKUP_PREFIX)}. On it.{delay}"
    return (f"Time. Reminder: {job.text}.{delay}" if job.text else f"Time.{delay}")
