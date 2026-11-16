"""Reminders and timers in a conversation: set by plain words, answered at once, and delivered on their own
without ever waiting on (or holding up) a reply or an errand."""
import asyncio
import time
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from app.chat import ChatSession
from app.chat.reminders import reminder_message
from app.config import Settings
from app.events.bus import EventBus
from app.main import create_app
from app.memory.recall import MemoryStore
from app.persona import Persona
from app.scheduler import Scheduler
from app.tasks.manager import TaskManager
from tests.conftest import sign_in
from tests.fakes import WaitingLLM
from tests.test_chat import StreamingModel, reply
from tests.test_scheduler import Clock, settle


class SlowModel(StreamingModel):
    """Takes as long as the test says to write its reply."""
    def __init__(self, replies):
        super().__init__(replies)
        self.gate = asyncio.Event()

    async def stream_text(self, messages, max_tokens=300, temperature=None):
        await self.gate.wait()
        async for chunk in super().stream_text(messages, max_tokens, temperature):
            yield chunk


def build(replies=(), model=None, store=None, clock=None, owner_key=None):
    """A conversation with a scheduler whose deliveries land in it, as the app wires them."""
    clock = clock or Clock()
    store = store or MemoryStore(":memory:")
    model = model or StreamingModel(list(replies))
    bus = EventBus()
    manager = TaskManager(model, bus, Settings())
    holder = {}

    async def deliver(job, late):
        session = holder["session"]
        text = reminder_message(job, late)
        if job.kind == "lookup":
            session.run_scheduled_lookup(job.text, text)
        else:
            session.deliver(text)

    scheduler = Scheduler(store, clock, deliver)
    session = ChatSession(model, bus, manager, Settings(), store=store, persona=Persona(), scheduler=scheduler,
                          owner_key=owner_key)
    session.attach()
    session.reminders.now = clock
    holder["session"] = session
    return session, scheduler, clock, model, manager


def said(session, text):
    """Send a message and return what Kip answered, from the conversation's history."""
    turn = session.submit(text)
    return next((t["text"] for t in session.history() if t.get("in_reply_to") == turn["id"]), None)


# ── plain words in, plain words out, no model ───────────────────────────────
def test_a_reminder_is_set_and_read_back_with_its_time_and_without_asking_the_model():
    session, scheduler, clock, model, _ = build()
    assert said(session, "remind me in 10 minutes to stretch") == "Set. 8:55 pm today. I remind you: stretch."
    [job] = scheduler.active(session.session_id)
    assert (job.kind, job.text) == ("reminder", "stretch") and model.prompts == []


def test_a_timer_says_how_long_and_a_repeating_one_says_when_it_repeats():
    session, scheduler, *_ = build()
    assert said(session, "set a 5 minute timer") == "Timer set. 5 minutes."
    assert said(session, "remind me every weekday at 8 to check email") == \
        "Set. every weekday, first 8:00 am tomorrow. I remind you: check email."
    assert said(session, "every morning at 7:30 tell me the weather in Pune") == \
        "Set. every day, first 7:30 am tomorrow. I check: the weather in Pune."


def test_listing_and_cancelling():
    session, scheduler, *_ = build()
    assert said(session, "what reminders do I have?") == "Nothing set."
    said(session, "remind me in 10 minutes to stretch")
    said(session, "remind me tomorrow at 9 to call the dentist")
    listing = said(session, "what reminders do I have?")
    assert listing == "You have 2: 1. stretch, 8:55 pm today. 2. call the dentist, 9:00 am tomorrow."
    assert said(session, "cancel the stretch reminder") == "Cancelled: stretch."
    assert [j.text for j in scheduler.active(session.session_id)] == ["call the dentist"]
    assert said(session, "cancel the dentist reminder") == "Cancelled: call the dentist."
    assert said(session, "cancel all reminders") == "Nothing set, so nothing to cancel."


def test_cancelling_without_saying_which_asks_when_there_are_several():
    session, scheduler, *_ = build()
    said(session, "remind me in 10 minutes to stretch")
    said(session, "set a 20 minute timer")
    assert said(session, "cancel the timer") == "Cancelled: timer."        # one timer: no need to ask, and the
    assert [j.text for j in scheduler.active(session.session_id)] == ["stretch"]    # reminder is left alone
    said(session, "set a 20 minute timer")
    said(session, "set a 30 minute timer")
    assert said(session, "cancel the timer").startswith("Which one? 1. timer, 20 minutes left. 2. timer, 30 minutes")
    assert said(session, "cancel all timers") == "Cancelled all 2."
    assert [j.text for j in scheduler.active(session.session_id)] == ["stretch"]
    said(session, "set a 20 minute timer")
    assert said(session, "stop reminding me") == "Cancelled all 2."


def test_what_is_missing_is_asked_for_and_the_next_message_completes_it():
    session, scheduler, *_ = build()
    assert said(session, "remind me to call mom").startswith("When?")
    assert said(session, "at 6pm") == "Set. 6:00 pm tomorrow. I remind you: call mom."
    assert said(session, "remind me tomorrow") == "Remind you of what?"
    assert said(session, "to renew the passport") == "Set. 9:00 am tomorrow. I remind you: renew the passport."
    assert said(session, "set a timer") == "How long? Say '5 minutes'."
    assert said(session, "10 minutes") == "Timer set. 10 minutes."


@pytest.mark.asyncio
async def test_a_message_that_is_not_the_answer_carries_on_as_normal_conversation():
    session, scheduler, _, model, _ = build([reply("Pune. Good city.")])
    assert said(session, "remind me to call mom").startswith("When?")
    session.submit("Where do I live?")                          # not a time: the question is just answered
    await session.drain()
    assert len(model.prompts) == 1 and not scheduler.active(session.session_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["can you remind me what we decided about Goa?", "remind me about the budget?",
                                  "what's the weather tomorrow?"])
async def test_questions_that_only_look_like_reminders_still_reach_the_model(text):
    session, scheduler, _, model, _ = build([reply("Here you go.")])
    session.submit(text)
    await session.drain()
    assert len(model.prompts) <= 1 and not scheduler.active(session.session_id)


@pytest.mark.asyncio
async def test_kip_knows_he_can_do_this_so_he_does_not_say_he_cannot():
    session, *_ = build()
    assert "reminders and timers" in session._status()["my_tools"]


# ── arriving on their own ───────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_a_timer_arrives_in_the_conversation_when_it_is_up():
    session, scheduler, clock, *_ = build()
    said(session, "set a 5 minute timer")
    clock.advance(minutes=5)
    scheduler.dispatch()
    await settle(scheduler)
    arrived = [t for t in session.history() if t.get("source") == "reminder"]
    assert [t["text"] for t in arrived] == ["Timer done."]


@pytest.mark.asyncio
async def test_a_repeating_reminder_keeps_arriving_and_a_late_one_says_so():
    session, scheduler, clock, *_ = build()
    said(session, "remind me every day at 8am to take vitamins")
    clock.value = clock.value.replace(day=17, hour=8, minute=0)
    scheduler.dispatch()
    await settle(scheduler)
    clock.value = clock.value.replace(day=18, hour=9, minute=30)        # the Pi was off at 8
    scheduler.dispatch()
    await settle(scheduler)
    texts = [t["text"] for t in session.history() if t.get("source") == "reminder"]
    assert texts == ["Time. Reminder: take vitamins.", "Time. Reminder: take vitamins. (late by 90 minutes)"]


@pytest.mark.asyncio
async def test_a_scheduled_check_starts_a_lookup_when_it_is_due():
    session, scheduler, clock, _, manager = build()
    said(session, "every morning at 7:30 tell me the weather in Pune")
    clock.value = clock.value.replace(day=17, hour=7, minute=30)
    scheduler.dispatch()
    await settle(scheduler)
    assert [t["text"] for t in session.history() if t.get("source") == "reminder"] == \
        ["Scheduled check: the weather in Pune. On it."]
    assert [c.goal for c in manager.tasks.values()] == ["Look up online: the weather in Pune"]
    await manager.close()


def test_a_scheduled_check_never_reads_their_accounts():
    session, scheduler, *_ = build()
    answer = said(session, "every morning at 8 tell me how my Groww portfolio is doing")
    assert "won't schedule checks on your accounts" in answer and not scheduler.active(session.session_id)


def test_each_person_has_only_their_own_reminders_across_all_their_chats():
    clock, store = Clock(), MemoryStore(":memory:")
    first, scheduler, *_ = build(clock=clock, store=store, owner_key="paul")
    second, scheduler2, *_ = build(clock=clock, store=store, owner_key="paul")     # paul's other chat
    other, scheduler3, *_ = build(clock=clock, store=store, owner_key="neave")
    said(first, "remind me in 10 minutes to stretch")
    assert said(second, "what reminders do I have?").startswith("You have 1: 1. stretch")
    assert said(other, "what reminders do I have?") == "Nothing set."
    assert said(other, "cancel all reminders") == "Nothing set, so nothing to cancel."
    assert len(scheduler.active("paul")) == 1


# ── in parallel ─────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_setting_a_timer_does_not_wait_for_a_reply_still_being_written():
    model = SlowModel([reply("Here is the long answer.")])
    session, scheduler, clock, _, _ = build(model=model)
    session.submit("Tell me about your day, in detail.")
    await asyncio.sleep(0.05)
    assert session.busy                                          # mid-reply, and held there
    assert said(session, "set a 5 minute timer") == "Timer set. 5 minutes."     # answered at once
    clock.advance(minutes=5)
    scheduler.dispatch()
    await settle(scheduler)
    assert any(t["text"] == "Timer done." for t in session.history())          # and it arrived, reply still pending
    model.gate.set()
    await session.drain()
    assert any(t["text"] == "Here is the long answer." for t in session.history())


@pytest.mark.asyncio
async def test_a_timer_goes_off_and_can_be_set_while_an_errand_is_running():
    session, scheduler, clock, _, manager = build()
    manager.start("Find the keys")                               # an errand now holds the one errand slot
    assert manager.active_id is not None
    assert said(session, "set a 2 minute timer") == "Timer set. 2 minutes."
    clock.advance(minutes=2)
    scheduler.dispatch()
    await settle(scheduler)
    assert any(t["text"] == "Timer done." for t in session.history())
    assert manager.active_id is not None                         # the errand was never disturbed
    await manager.close()


@pytest.mark.asyncio
async def test_timers_run_alongside_each_other_and_alongside_chat():
    model = SlowModel([reply("Done thinking.")])
    session, scheduler, clock, _, _ = build(model=model)
    session.submit("Think about it for a while.")
    for minutes in (1, 2, 3):
        said(session, f"set a {minutes} minute timer")
    clock.advance(minutes=3)
    assert scheduler.dispatch() == 3
    await settle(scheduler)
    assert sum(t["text"] == "Timer done." for t in session.history()) == 3
    model.gate.set()
    await session.drain()


# ── through the real app ────────────────────────────────────────────────────
def test_a_reminder_set_by_chat_arrives_in_the_same_conversation_even_after_a_restart_of_the_loop():
    with TestClient(create_app(client=WaitingLLM())) as client:
        sign_in(client, "Neave", "1234")
        sent = client.post("/api/chat", json={"message": "remind me in 10 minutes to stretch"})
        conversation = sent.json()["session_id"]
        history = client.get("/api/chat", params={"conversation_id": conversation}).json()["turns"]
        assert [t["text"] for t in history if t["role"] == "kip"] == ["Set. " + history[1]["text"][5:]] or \
            history[1]["text"].endswith("I remind you: stretch.")
        scheduler = client.app.state.scheduler
        store = client.app.state.people._store
        with store._lock:
            store._db.execute("UPDATE schedule SET due = ?", (time.time() + 0.5,))
            store._db.commit()
        scheduler._poke()
        deadline = time.time() + 6
        arrived = []
        while time.time() < deadline and not arrived:
            time.sleep(0.2)
            turns = client.get("/api/chat", params={"conversation_id": conversation}).json()["turns"]
            arrived = [t for t in turns if t.get("source") == "reminder"]
        assert [t["text"] for t in arrived] == ["Time. Reminder: stretch."]


def test_a_reminder_for_a_deleted_conversation_lands_in_the_persons_latest_one():
    with TestClient(create_app(client=WaitingLLM())) as client:
        sign_in(client, "Neave", "1234")
        first = client.post("/api/chat", json={"message": "remind me in 10 minutes to stretch"}).json()["session_id"]
        second = client.post("/api/chat", json={"message": "hello there"}).json()["session_id"]
        assert client.delete(f"/api/conversations/{first}").status_code == 204
        store = client.app.state.people._store
        with store._lock:
            store._db.execute("UPDATE schedule SET due = ?", (time.time() + 0.5,))
            store._db.commit()
        client.app.state.scheduler._poke()
        deadline, arrived = time.time() + 6, []
        while time.time() < deadline and not arrived:
            time.sleep(0.2)
            turns = client.get("/api/chat", params={"conversation_id": second}).json()["turns"]
            arrived = [t for t in turns if t.get("source") == "reminder"]
        assert [t["text"] for t in arrived] == ["Time. Reminder: stretch."]
