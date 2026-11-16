"""Reminders, timers and scheduled checks: things Kip does later, or again and again, on his own.

Every job lives in kip.db, so a restart (or a power cut) forgets nothing. A job is not an errand and takes
nothing the conversation needs: no errand slot, no reply lock. Any number can be waiting at once; each one
that comes due is delivered by a task of its own, so a slow delivery never holds up another timer, and a
timer going off while Kip is mid-reply or running a lookup simply arrives. The loop sleeps until the next
job is due (and wakes at once when a sooner one is added), so a 10-second timer fires on time.

Delivery is a callback, `deliver(job, late_seconds)`, given by the app: it knows which conversation and
person the job belongs to. A one-off job is finished only after its delivery succeeded (retried twice
after a failure); a repeating job moves to its next time first, so a crash never makes it fire in a burst.
"""
import asyncio
import logging
import sqlite3
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from uuid import uuid4

_log = logging.getLogger("agentic_friend.scheduler")

MAX_ACTIVE = 30          # jobs one person may have waiting
RETRY_SECONDS = 30
MAX_ATTEMPTS = 3
TICK_SECONDS = 15.0      # the longest the loop sleeps, with nothing to wake it


class LimitError(ValueError):
    """Said to the person as written."""


@dataclass
class Job:
    id: str
    owner: str            # a profile id (or, with no profiles, the conversation's id)
    speaker: str | None
    session_id: str       # the conversation it was set in, where it will arrive
    kind: str             # reminder | timer | lookup
    text: str
    due: float            # epoch seconds
    repeat: str           # "", daily, weekdays, weekly:<0-6>, every:<minutes>, yearly
    created: float
    status: str = "active"
    attempts: int = 0


Deliver = Callable[[Job, float], Awaitable[None]]


def next_due(job_due: datetime, repeat: str, now: datetime) -> datetime:
    """The first time after `now` that a repeating job comes round, keeping its time of day."""
    step = None
    if repeat.startswith("every:"):
        step = timedelta(minutes=int(repeat.split(":")[1]))
    due = job_due
    for _ in range(1000):
        if step is not None:
            due = due + step
        elif repeat == "daily":
            due = due + timedelta(days=1)
        elif repeat == "weekdays":
            due = due + timedelta(days=1)
            while due.weekday() > 4:
                due = due + timedelta(days=1)
        elif repeat.startswith("weekly:"):
            due = due + timedelta(days=7)
        elif repeat == "yearly":
            try:
                due = due.replace(year=due.year + 1)
            except ValueError:               # 29 February
                due = due.replace(year=due.year + 1, day=28)
        else:
            raise ValueError(f"unknown repeat {repeat!r}")
        if due > now:
            return due
    return now + timedelta(days=1)


class Scheduler:
    def __init__(self, store, now: Callable[[], datetime], deliver: Deliver, tick: float = TICK_SECONDS) -> None:
        self.store, self.now, self.deliver, self.tick = store, now, deliver, tick
        self._firing: set[str] = set()
        self._tasks: set[asyncio.Task] = set()
        self._wake = asyncio.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = False

    # ── storage ─────────────────────────────────────────────────────────────
    def _db(self) -> sqlite3.Connection:
        db = self.store._db
        if not self._ready:
            with self.store._lock:
                db.executescript("""
                    CREATE TABLE IF NOT EXISTS schedule (
                        id TEXT PRIMARY KEY, owner TEXT NOT NULL, speaker TEXT, session_id TEXT NOT NULL,
                        kind TEXT NOT NULL, text TEXT NOT NULL, due REAL NOT NULL, repeat TEXT NOT NULL DEFAULT '',
                        created REAL NOT NULL, status TEXT NOT NULL DEFAULT 'active', attempts INTEGER NOT NULL DEFAULT 0);
                    CREATE INDEX IF NOT EXISTS schedule_due ON schedule(status, due);
                    CREATE INDEX IF NOT EXISTS schedule_owner ON schedule(owner, status);
                """)
            self._ready = True
        return db

    @staticmethod
    def _job(row) -> Job:
        return Job(row["id"], row["owner"], row["speaker"], row["session_id"], row["kind"], row["text"], row["due"],
                   row["repeat"], row["created"], row["status"], row["attempts"])

    def add(self, owner: str, session_id: str, kind: str, text: str, due: datetime, repeat: str = "",
            speaker: str | None = None) -> Job:
        with self.store._lock:
            db = self._db()
            waiting = db.execute("SELECT COUNT(*) FROM schedule WHERE owner = ? AND status = 'active'",
                                 (owner,)).fetchone()[0]
            if waiting >= MAX_ACTIVE:
                raise LimitError(f"You already have {MAX_ACTIVE} reminders and timers waiting. Cancel some first.")
            job = Job(str(uuid4()), owner, speaker, session_id, kind, text.strip(), due.timestamp(), repeat,
                      self.now().timestamp())
            with db:
                db.execute("INSERT INTO schedule(id, owner, speaker, session_id, kind, text, due, repeat, created) "
                           "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                           (job.id, job.owner, job.speaker, job.session_id, job.kind, job.text, job.due, job.repeat,
                            job.created))
        self._poke()
        return job

    def active(self, owner: str) -> list[Job]:
        with self.store._lock:
            rows = self._db().execute("SELECT * FROM schedule WHERE owner = ? AND status = 'active' ORDER BY due",
                                      (owner,)).fetchall()
        return [self._job(row) for row in rows]

    def cancel(self, owner: str, job_ids: list[str]) -> list[Job]:
        """Cancel these jobs if they are this person's. Returns the ones that were."""
        mine = [job for job in self.active(owner) if job.id in set(job_ids)]
        with self.store._lock:
            db = self._db()
            with db:
                db.executemany("UPDATE schedule SET status = 'cancelled' WHERE id = ? AND owner = ?",
                               [(job.id, owner) for job in mine])
        return mine

    def _next_time(self) -> float | None:
        with self.store._lock:
            row = self._db().execute("SELECT MIN(due) FROM schedule WHERE status = 'active'").fetchone()
        return row[0] if row else None

    # ── running ─────────────────────────────────────────────────────────────
    def _poke(self) -> None:
        """Wake the loop: a sooner job may have been added (from any thread)."""
        if self._loop is not None and not self._loop.is_closed():
            try:
                self._loop.call_soon_threadsafe(self._wake.set)
            except RuntimeError:
                pass

    def dispatch(self) -> int:
        """Start delivering every job that is due now. Returns how many. Never waits for a delivery."""
        now = self.now()
        stamp = now.timestamp()
        with self.store._lock:
            rows = self._db().execute("SELECT * FROM schedule WHERE status = 'active' AND due <= ? ORDER BY due",
                                      (stamp,)).fetchall()
        started = 0
        for row in rows:
            job = self._job(row)
            if job.id in self._firing:
                continue
            late = max(0.0, stamp - job.due)
            if job.repeat:
                # Move on first: a crash mid-delivery costs one occurrence, never a burst of them.
                following = next_due(datetime.fromtimestamp(job.due, now.tzinfo), job.repeat, now).timestamp()
                with self.store._lock:
                    with self._db() as db:
                        db.execute("UPDATE schedule SET due = ? WHERE id = ?", (following, job.id))
            self._firing.add(job.id)
            task = asyncio.get_running_loop().create_task(self._fire(job, late), name=f"reminder-{job.id[:8]}")
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            started += 1
        return started

    async def _fire(self, job: Job, late: float) -> None:
        try:
            await self.deliver(job, late)
        except asyncio.CancelledError:
            raise
        except Exception:
            _log.exception("Could not deliver %s %s", job.kind, job.id[:8])
            if not job.repeat:
                self._retry(job)
        else:
            if not job.repeat:
                self._set_status(job.id, "done")
        finally:
            self._firing.discard(job.id)

    def _retry(self, job: Job) -> None:
        attempts = job.attempts + 1
        with self.store._lock:
            with self._db() as db:
                if attempts >= MAX_ATTEMPTS:
                    db.execute("UPDATE schedule SET status = 'failed', attempts = ? WHERE id = ?", (attempts, job.id))
                else:
                    db.execute("UPDATE schedule SET due = ?, attempts = ? WHERE id = ?",
                               (self.now().timestamp() + RETRY_SECONDS, attempts, job.id))

    def _set_status(self, job_id: str, status: str) -> None:
        with self.store._lock:
            with self._db() as db:
                db.execute("UPDATE schedule SET status = ? WHERE id = ? AND status = 'active'", (status, job_id))

    async def run(self) -> None:
        """The loop: deliver what is due, then sleep until the next job (or a new, sooner one)."""
        self._loop = asyncio.get_running_loop()
        while True:
            try:
                self.dispatch()
            except Exception:
                _log.exception("Scheduler pass failed")
            upcoming = self._next_time()
            wait = self.tick if upcoming is None else min(self.tick, max(0.02, upcoming - self.now().timestamp()))
            self._wake.clear()
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=wait)
            except asyncio.TimeoutError:
                pass

    async def close(self) -> None:
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
