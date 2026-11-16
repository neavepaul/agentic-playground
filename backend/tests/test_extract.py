"""Fact extraction as its own step: the reply talks, a separate call decides what to remember."""
import json

import pytest

from app.chat import extract
from app.config import Settings
from tests.test_chat import make_session, reply


def found(remember=(), forget=()):
    return json.dumps({"remember": [{"text": t, "context": "actual"} if isinstance(t, str) else t for t in remember],
                       "forget": list(forget)})


def with_extraction(replies, **settings):
    session, model, *_ = make_session(replies)
    session.settings = Settings(**{"chat_extraction": True, **settings})
    return session, model


def facts(session):
    return [m["text"] for m in session.store.recent(20) if m["kind"] in ("fact", "preference")
            and m["superseded_by"] is None]


# ── the gate ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("said", [
    "My name is Neave and I work as an engineer.", "I'm allergic to peanuts.", "Remember that Paul likes chess.",
    "Actually it's Paul, not Neave.", "Paul is my brother.", "What's my favourite fruit? Remember I said mango."])
def test_messages_that_may_hold_a_lasting_fact_are_read(said):
    assert extract.worth_reading(said)


@pytest.mark.parametrize("said", ["hi", "What's the weather in Pune today?", "Who is playing tonight?",
                                  "Look up the latest cricket score.", "thanks", "ok cool", ""])
def test_everything_else_costs_no_model_call(said):
    assert not extract.worth_reading(said)


# ── reading the model's output ──────────────────────────────────────────────
def test_a_valid_answer_is_parsed():
    result = extract.parse('Sure.\n' + found(["Neave works as an engineer."], ["Neave is a student"]))
    assert [i.text for i in result.remember] == ["Neave works as an engineer."]
    assert result.forget == ["Neave is a student"]


def test_one_bad_item_never_loses_the_rest():
    raw = json.dumps({"remember": [{"text": "Good fact.", "context": "actual"},
                                   {"text": "Odd.", "context": "banana"}, {"text": ""}], "forget": [3, "Old fact"]})
    result = extract.parse(raw)
    assert [i.text for i in result.remember] == ["Good fact."] and result.forget == ["Old fact"]


@pytest.mark.parametrize("raw", ["", "nothing to add", "{not json", '{"task": "x"}', "[]"])
def test_unreadable_output_means_nothing_is_written(raw):
    assert extract.parse(raw) is None


def test_the_call_names_the_speaker_and_shows_what_is_already_known():
    known = [{"kind": "fact", "text": "Neave is allergic to peanuts."}, {"kind": "episode", "text": "A chat."}]
    system, body = [m["content"] for m in extract.messages_for("Actually it's Paul.", "Okay.", "Neave", known)]
    assert "never \"I\" or \"my\"" in system and "Speaker: Neave" in body
    assert "- Neave is allergic to peanuts." in body and "A chat." not in body


# ── in a conversation ───────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_a_fact_is_filed_by_a_separate_call_after_the_reply():
    session, model = with_extraction([reply("Nice to meet you, Neave."),
                                      found(["Neave works as an engineer."])])
    session.submit("My name is Neave and I work as an engineer.")
    await session.drain()
    assert facts(session) == ["Neave works as an engineer."]
    chat_call, extraction_call = model.prompts
    assert "remember" not in chat_call[0]["content"]          # the chat model only talks
    assert "Known memories" in extraction_call[1]["content"]  # the second call is the extractor
    assert session.history()[-1]["text"] == "Nice to meet you, Neave."


@pytest.mark.asyncio
async def test_a_message_with_nothing_lasting_makes_no_second_call():
    session, model = with_extraction([reply("Looks sunny.")])
    session.submit("What do you think about tea?")
    await session.drain()
    assert len(model.prompts) == 1 and facts(session) == []


@pytest.mark.asyncio
async def test_a_reply_that_volunteers_its_own_memories_is_not_asked_twice():
    session, model = with_extraction([reply("Noted.", remember=["Neave likes tea."])])
    session.submit("I like tea.")
    await session.drain()
    assert len(model.prompts) == 1 and facts(session) == ["Neave likes tea."]


@pytest.mark.asyncio
async def test_a_failed_extraction_leaves_the_chat_and_the_store_alone():
    session, model = with_extraction([reply("Okay."), RuntimeError("model down"), reply("Still here.")])
    session.submit("I'm allergic to peanuts.")
    await session.drain()
    assert facts(session) == [] and session.history()[-1]["text"] == "Okay."
    session.submit("What do you think about tea?")
    await session.drain()
    assert session.history()[-1]["text"] == "Still here."


@pytest.mark.asyncio
async def test_unreadable_extraction_output_writes_nothing():
    session, model = with_extraction([reply("Okay."), "I would remember that!"])
    session.submit("I'm allergic to peanuts.")
    await session.drain()
    assert facts(session) == []


@pytest.mark.asyncio
async def test_a_correction_retires_the_old_fact_and_files_the_new_one():
    session, model = with_extraction([reply("Got it."), found(["Paul is allergic to peanuts."],
                                                              ["Neave is allergic to peanuts"])])
    await session.store.add("fact", "Neave is allergic to peanuts", ["neave"], meta={"speaker": "unknown"})
    session.submit("Actually it's Paul who is allergic to peanuts, not me.")
    await session.drain()
    assert facts(session) == ["Paul is allergic to peanuts."]
    old = [m for m in session.store.recent(10) if m["text"] == "Neave is allergic to peanuts"][0]
    assert old["superseded_by"]   # kept as history, no longer answering


@pytest.mark.asyncio
async def test_extraction_can_be_turned_off():
    session, model = with_extraction([reply("Okay.")], chat_extraction=False)
    session.submit("I'm allergic to peanuts.")
    await session.drain()
    assert len(model.prompts) == 1 and facts(session) == []


@pytest.mark.asyncio
async def test_the_episode_summary_includes_what_was_just_learned():
    session, model = with_extraction([reply("Okay."), found(["Neave works as an engineer."])])
    session.submit("I work as an engineer.")
    await session.drain()
    episode = [m["text"] for m in session.store.recent(20) if m["kind"] == "episode"]
    assert any("works as an engineer" in text for text in episode)
