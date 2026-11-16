"""Checkpoints: older turns of a long chat folded into a note that rides in the prompt."""
import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from app.chat import checkpoint as cp
from app.config import Settings
from app.memory.recall import MemoryStore
from tests.test_chat import StreamingModel, make_session, reply, spoken

START = datetime(2026, 11, 14, 12, 0, tzinfo=timezone.utc)


def turns(count, words=40, private=()):
    return [{"id": str(i), "role": "user" if i % 2 == 0 else "kip", "text": f"turn {i} " + "word " * words,
             "at": (START + timedelta(minutes=i)).isoformat(), **({"private": True} if i in private else {})}
            for i in range(count)]


class Scripted:
    def __init__(self, *outputs):
        self.outputs, self.calls, self.last_metrics = list(outputs), [], {}

    async def stream_text(self, messages, max_tokens=300, temperature=None):
        self.calls.append(messages)
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        for start in range(0, len(out), 5):
            yield out[start:start + 5]


# ── planning ────────────────────────────────────────────────────────────────
def test_short_chats_are_left_alone():
    assert cp.plan_fold(turns(6), None, 800) is None


def test_long_chats_fold_the_oldest_exchanges_and_keep_the_latest_verbatim():
    log = turns(10)
    fold = cp.plan_fold(log, None, 800)
    assert [t["id"] for t in fold] == ["0", "1", "2", "3", "4", "5"]
    assert fold[-1]["role"] == "kip"            # never a question without its answer
    assert "9" not in {t["id"] for t in fold} and "8" not in {t["id"] for t in fold}


def test_the_note_covers_only_what_it_folded():
    log = turns(10)
    note = cp.Checkpoint("older stuff", log[5]["at"], 6)
    assert [t["id"] for t in cp.after(log, note)] == ["6", "7", "8", "9"]
    assert cp.plan_fold(log, note, 800) is None  # what is left fits: nothing more to fold yet


def test_private_turns_never_reach_the_summary():
    fold = cp.plan_fold(turns(12, private={1, 2}), None, 800)
    assert fold and not any(t.get("private") for t in fold)
    messages = cp.summary_messages(fold, None, 200)
    assert "turn 1 " not in messages[1]["content"] and "turn 0 " in messages[1]["content"]


def test_too_few_turns_are_not_worth_a_call():
    assert cp.plan_fold(turns(3, words=400), None, 800) is None


def test_a_saved_note_round_trips_and_garbage_is_ignored():
    note = cp.Checkpoint("They plan a trip.", START.isoformat(), 4)
    assert cp.load(note.dump()) == note
    assert cp.load("") is None and cp.load("not json") is None and cp.load('{"text": "x"}') is None


# ── writing ─────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_a_good_summary_becomes_the_checkpoint():
    client = Scripted("They are planning a trip to Goa in March; the budget is 40k.")
    status, note = await cp.compact(client, turns(10), None, 800, 200)
    assert status == "completed" and "Goa" in note.text and note.turns == 6
    assert note.through == turns(10)[5]["at"]
    # The model is told to write a note, not given anything beyond the folded turns.
    sent = client.calls[0][1]["content"]
    assert "turn 5 " in sent and "turn 6 " not in sent


@pytest.mark.asyncio
async def test_a_second_fold_builds_on_the_first_note():
    previous = cp.Checkpoint("They plan a trip.", turns(10)[5]["at"], 6)
    log = turns(20)
    client = Scripted("They plan a trip; flights are booked.")
    status, note = await cp.compact(client, log, previous, 800, 200)
    assert status == "completed" and note.turns > 6
    assert "They plan a trip." in client.calls[0][1]["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("output", [ModuleNotFoundError("down"), "", "word " * 400, "Fine note.\n###\n{}"])
async def test_a_bad_summary_changes_nothing_or_is_cleaned(output):
    previous = cp.Checkpoint("Old note.", turns(10)[1]["at"], 2)
    status, note = await cp.compact(Scripted(output), turns(12), previous, 800, 100)
    if output == "Fine note.\n###\n{}":
        assert status == "completed" and note.text == "Fine note."   # a stray action line is cut
    else:
        assert status == "failed" and note is None


@pytest.mark.asyncio
async def test_a_cut_off_summary_is_rejected():
    client = Scripted("They were discussing a")
    client.last_metrics = {"truncated": True}
    assert await cp.compact(client, turns(10), None, 800, 200) == ("failed", None)


@pytest.mark.asyncio
async def test_not_needed_makes_no_model_call():
    client = Scripted()
    assert await cp.compact(client, turns(4), None, 800, 200) == ("not_needed", None)
    assert client.calls == []


# ── in a conversation ───────────────────────────────────────────────────────
def say(i):
    return f"Message number {i}: I keep thinking about my holiday plans and what we might do there together."


def conversation_with(replies, **settings):
    session, model, manager, events = make_session(replies)
    session.settings = Settings(chat_history_tokens=200, **settings)
    return session, model


async def talk(session, count, start=1):
    for i in range(start, start + count):
        session.submit(say(i))
        await session.drain()


@pytest.mark.asyncio
async def test_a_long_chat_carries_a_note_and_the_latest_exchange_verbatim():
    session, model = conversation_with([reply(f"Reply {i}. " + "Sounds lovely, tell me more about it. " * 2)
                                        for i in (1, 2, 3)] + ["They are planning a holiday and weighing places.",
                                                              reply("Reply 4."), reply("Reply 5.")])
    await talk(session, 3)
    assert session.checkpoint and "holiday" in session.checkpoint.text and session.compaction == "completed"
    session.submit("Continue where we left off")
    await session.drain()
    prompt = [p for p in model.prompts if "They say: " in p[-1]["content"]][-1]   # the reply, not a summary call
    latest = prompt[-1]["content"]
    assert "Earlier in this chat" in latest and "planning a holiday" in latest
    older = " ".join(m["content"] for m in prompt[1:-1])
    assert "Message number 1:" not in older            # folded, not repeated
    assert "Message number 3:" in older and "Reply 3." in older   # the latest exchange stays word for word
    assert session.history()[-1]["metrics"]["checkpoint_tokens"] > 0
    assert session.history()[-1]["metrics"]["compaction"] == "completed"
    assert spoken(latest) == "Continue where we left off"


@pytest.mark.asyncio
async def test_the_note_survives_a_restart_and_the_transcript_stays_whole():
    store = MemoryStore(":memory:")
    session, model, *_ = make_session([reply(f"Reply {i}. " + "Sounds lovely, tell me more about it. " * 2)
                                       for i in (1, 2, 3)] + ["They are planning a holiday."], store=store)
    session.settings = Settings(chat_history_tokens=200)
    await talk(session, 3)
    assert session.checkpoint
    again = reopened(session, store)
    assert again.checkpoint == session.checkpoint
    assert len(store.conversation_turns(session.session_id)) == 6   # nothing was removed


@pytest.mark.asyncio
async def test_a_failed_summary_keeps_the_chat_working_and_does_not_retry_at_once():
    session, model = conversation_with([reply(f"Reply {i}. " + "Sounds lovely, tell me more about it. " * 2)
                                        for i in (1, 2, 3)] + [ModuleNotFoundError("offline"), reply("Reply 4.")])
    await talk(session, 3)
    assert session.checkpoint is None and session.compaction == "failed"
    session.submit(say(4))
    await session.drain()
    assert session.history()[-1]["text"] == "Reply 4."   # the chat went on; the next call was a reply, not a retry


@pytest.mark.asyncio
async def test_compaction_can_be_turned_off():
    session, model = conversation_with([reply(f"Reply {i}. " + "Sounds lovely, tell me more about it. " * 2)
                                        for i in (1, 2, 3)], chat_compaction=False)
    await talk(session, 3)
    assert session.checkpoint is None and len(model.prompts) == 3


@pytest.mark.asyncio
async def test_a_correction_drops_the_note_so_the_old_version_is_not_carried():
    store = MemoryStore(":memory:")
    session, model, *_ = make_session(
        [reply(f"Reply {i}. " + "Sounds lovely, tell me more about it. " * 2) for i in (1, 2, 3)]
        + ["They are planning a holiday in Goa."]
        + ['Noted.\n###\n{"task": "", "offer": "", "remember": [], "forget": ["They are planning a holiday in Goa"]}'],
        store=store)
    session.settings = Settings(chat_history_tokens=200)
    await talk(session, 3)
    assert session.checkpoint
    session.submit("Actually it is Kerala, not Goa")
    await session.drain()
    assert session.checkpoint is None
    again = reopened(session, store)
    assert again.checkpoint is None


@pytest.mark.asyncio
async def test_a_slow_summary_cannot_revive_a_note_dropped_meanwhile():
    session, model = conversation_with([reply("Reply 1."), reply("Reply 2."), reply("Reply 3.")])
    gate = asyncio.Event()

    class Slow(StreamingModel):
        async def stream_text(self, messages, max_tokens=300, temperature=None):
            await gate.wait()
            yield "An old note."
    session.client = Slow([])
    session._user_turns = 5
    session.turns.clear()
    session.turns.extend(turns(10))
    job = asyncio.create_task(session._compact())
    await asyncio.sleep(0)
    session.checkpoint, session._checkpoint_epoch = None, session._checkpoint_epoch + 1  # a correction lands
    gate.set()
    await job
    assert session.checkpoint is None


def reopened(session, store):
    """The same conversation opened again (after a restart)."""
    from app.chat import ChatSession
    return ChatSession(session.client, session.bus, session.manager, Settings(), store=store,
                       persona=session.persona, session_id=session.session_id)
