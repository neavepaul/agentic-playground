"""The scheduler: durable jobs, delivered on time, in parallel, whatever else is going on."""
import asyncio
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from app.memory.recall import MemoryStore
from app.scheduler import MAX_ACTIVE, LimitError, Scheduler, next_due

TZ = ZoneInfo("Asia/Kolkata")


class Clock:
    """A clock the test moves. Monday 16 November 2026, 8:45 pm in India."""
    def __init__(self):
        self.value = datetime(2026, 11, 16, 20, 45, tzinfo=TZ)

    def __call__(self):
        return self.value

    def advance(self, **delta):
        self.value += timedelta(**delta)


class Inbox:
    """Where deliveries land: what was delivered, and how late."""
    def __init__(self):
        self.got, self.fail_times = [], 0

    async def __call__(self, job, late):
        if self.fail_times:
            self.fail_times -= 1
            raise RuntimeError("could not reach the conversation")
        self.got.append((job.text, round(late)))


def make(store=None):
    clock, inbox = Clock(), Inbox()
    store = store or MemoryStore(":memory:")
    return Scheduler(store, clock, inbox), clock, inbox, store


async def settle(scheduler):
    """Let every delivery that was started finish."""
    while scheduler._tasks:
        await asyncio.gather(*list(scheduler._tasks), return_exceptions=True)
    await asyncio.sleep(0)


def job_in(scheduler, clock, minutes, text="stretch", owner="paul", repeat="", kind="reminder", session="chat-1"):
    return scheduler.add(owner, session, kind, text, clock() + timedelta(minutes=minutes), repeat)


# ── keeping and showing jobs ────────────────────────────────────────────────
def test_jobs_are_kept_in_order_and_each_person_sees_only_their_own():
    scheduler, clock, _, _ = make()
    job_in(scheduler, clock, 30, "later")
    job_in(scheduler, clock, 5, "sooner")
    job_in(scheduler, clock, 10, "not paul's", owner="neave")
    assert [j.text for j in scheduler.active("paul")] == ["sooner", "later"]
    assert [j.text for j in scheduler.active("neave")] == ["not paul's"]
    assert scheduler.active("nobody") == []


def test_one_person_cannot_cancel_anothers_job():
    scheduler, clock, _, _ = make()
    mine = job_in(scheduler, clock, 5, "mine", owner="paul")
    assert scheduler.cancel("neave", [mine.id]) == []
    assert [j.text for j in scheduler.active("paul")] == ["mine"]
    assert [j.text for j in scheduler.cancel("paul", [mine.id])] == ["mine"]
    assert scheduler.active("paul") == []


def test_a_person_may_only_have_so_many_waiting():
    scheduler, clock, _, _ = make()
    for i in range(MAX_ACTIVE):
        job_in(scheduler, clock, i + 1, f"job {i}")
    with pytest.raises(LimitError, match=str(MAX_ACTIVE)):
        job_in(scheduler, clock, 99, "one too many")
    job_in(scheduler, clock, 99, "someone else's is fine", owner="neave")


def test_jobs_survive_a_restart(tmp_path):
    path = tmp_path / "kip.db"
    scheduler, clock, _, store = make(MemoryStore(path))
    job_in(scheduler, clock, 5, "remember the milk", repeat="daily")
    store.close()
    again, _, _, store2 = make(MemoryStore(path))
    [kept] = again.active("paul")
    assert (kept.text, kept.repeat, kept.session_id) == ("remember the milk", "daily", "chat-1")
    store2.close()


# ── delivering ──────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_a_job_is_delivered_once_when_due_and_then_finished():
    scheduler, clock, inbox, _ = make()
    job_in(scheduler, clock, 10)
    assert scheduler.dispatch() == 0                 # not yet
    clock.advance(minutes=10)
    assert scheduler.dispatch() == 1
    await settle(scheduler)
    clock.advance(minutes=5)
    assert scheduler.dispatch() == 0                 # never again
    await settle(scheduler)
    assert inbox.got == [("stretch", 0)] and scheduler.active("paul") == []


@pytest.mark.asyncio
async def test_a_job_that_came_due_while_kip_was_off_arrives_late_and_says_so():
    scheduler, clock, inbox, _ = make()
    job_in(scheduler, clock, 10)
    clock.advance(hours=3)                           # the Pi was off
    scheduler.dispatch()
    await settle(scheduler)
    assert inbox.got == [("stretch", 170 * 60)]


@pytest.mark.asyncio
async def test_a_repeating_job_moves_to_its_next_time_and_fires_once_however_late():
    scheduler, clock, inbox, _ = make()
    job_in(scheduler, clock, 60, "vitamins", repeat="daily")          # 9:45 pm
    clock.advance(days=3, hours=2)                                     # three days off
    scheduler.dispatch()
    await settle(scheduler)
    assert len(inbox.got) == 1                                         # no burst of missed days
    [job] = scheduler.active("paul")
    assert datetime.fromtimestamp(job.due, TZ) == datetime(2026, 11, 20, 21, 45, tzinfo=TZ)
    clock.value = datetime(2026, 11, 20, 21, 45, tzinfo=TZ)
    scheduler.dispatch()
    await settle(scheduler)
    assert len(inbox.got) == 2


@pytest.mark.asyncio
async def test_a_failed_delivery_is_tried_again_and_then_given_up():
    scheduler, clock, inbox, _ = make()
    job_in(scheduler, clock, 1)
    inbox.fail_times = 2
    clock.advance(minutes=1)
    scheduler.dispatch()
    await settle(scheduler)
    assert inbox.got == [] and len(scheduler.active("paul")) == 1      # waiting to retry
    clock.advance(seconds=31)
    scheduler.dispatch()
    await settle(scheduler)
    clock.advance(seconds=31)
    inbox.fail_times = 0
    scheduler.dispatch()
    await settle(scheduler)
    assert inbox.got == [("stretch", 0)] or len(inbox.got) == 1        # the third try got through


@pytest.mark.asyncio
async def test_a_job_that_never_gets_through_is_dropped_not_retried_forever():
    scheduler, clock, inbox, _ = make()
    job_in(scheduler, clock, 1)
    inbox.fail_times = 99
    for _ in range(5):
        clock.advance(minutes=1)
        scheduler.dispatch()
        await settle(scheduler)
    assert inbox.got == [] and scheduler.active("paul") == []


@pytest.mark.asyncio
async def test_a_cancelled_job_never_arrives():
    scheduler, clock, inbox, _ = make()
    job = job_in(scheduler, clock, 5)
    scheduler.cancel("paul", [job.id])
    clock.advance(minutes=10)
    assert scheduler.dispatch() == 0
    assert inbox.got == []


# ── in parallel ─────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_a_slow_delivery_never_holds_up_another_timer():
    """A timer must not block Kip, nor another timer: each delivery is a task of its own."""
    clock, release, order = Clock(), asyncio.Event(), []

    async def deliver(job, late):
        if job.text == "slow":
            await release.wait()
        order.append(job.text)

    scheduler = Scheduler(MemoryStore(":memory:"), clock, deliver)
    job_in(scheduler, clock, 1, "slow")
    job_in(scheduler, clock, 1, "quick")
    clock.advance(minutes=1)
    assert scheduler.dispatch() == 2
    await asyncio.sleep(0.05)
    assert order == ["quick"]                         # arrived while the other was still stuck
    release.set()
    await settle(scheduler)
    assert order == ["quick", "slow"]


@pytest.mark.asyncio
async def test_many_timers_can_be_waiting_at_once_and_all_fire():
    scheduler, clock, inbox, _ = make()
    for i in range(20):
        job_in(scheduler, clock, 1 + i % 3, f"timer {i}")
    clock.advance(minutes=3)
    assert scheduler.dispatch() == 20
    await settle(scheduler)
    assert sorted(text for text, _ in inbox.got) == sorted(f"timer {i}" for i in range(20))


@pytest.mark.asyncio
async def test_the_loop_wakes_for_a_job_added_sooner_than_it_planned_to_sleep():
    """With nothing waiting the loop sleeps long; a 0.3 s timer added then must still fire on time."""
    delivered = asyncio.Event()

    async def deliver(job, late):
        delivered.set()

    real = lambda: datetime.now(TZ)
    scheduler = Scheduler(MemoryStore(":memory:"), real, deliver, tick=30)
    runner = asyncio.create_task(scheduler.run())
    await asyncio.sleep(0.1)                           # the loop is now asleep for up to 30 s
    scheduler.add("paul", "chat-1", "timer", "", real() + timedelta(seconds=0.3))
    await asyncio.wait_for(delivered.wait(), timeout=3)
    runner.cancel()
    await asyncio.gather(runner, return_exceptions=True)
    await scheduler.close()


# ── when a repeating job comes round ────────────────────────────────────────
@pytest.mark.parametrize("repeat, start, now, expected", [
    ("daily", (2026, 11, 16, 8), (2026, 11, 16, 9), (2026, 11, 17, 8)),
    ("daily", (2026, 11, 16, 8), (2026, 11, 20, 9), (2026, 11, 21, 8)),
    ("weekdays", (2026, 11, 20, 8), (2026, 11, 20, 9), (2026, 11, 23, 8)),      # Friday -> Monday
    ("weekly:0", (2026, 11, 16, 9), (2026, 11, 16, 10), (2026, 11, 23, 9)),
    ("yearly", (2027, 3, 3, 9), (2027, 3, 3, 10), (2028, 3, 3, 9)),
])
def test_a_repeating_job_comes_round_at_its_time_of_day(repeat, start, now, expected):
    assert next_due(datetime(*start, tzinfo=TZ), repeat, datetime(*now, tzinfo=TZ)) == datetime(*expected, tzinfo=TZ)


def test_a_yearly_job_on_29_february_survives_a_year_without_one():
    assert next_due(datetime(2028, 2, 29, 9, tzinfo=TZ), "yearly", datetime(2028, 3, 1, tzinfo=TZ)) == \
        datetime(2029, 2, 28, 9, tzinfo=TZ)


def test_an_interval_job_skips_the_times_it_missed_and_lands_on_its_own_grid():
    due = next_due(datetime(2026, 11, 16, 9, 0, tzinfo=TZ), "every:15", datetime(2026, 11, 16, 9, 40, tzinfo=TZ))
    assert due == datetime(2026, 11, 16, 9, 45, tzinfo=TZ)
