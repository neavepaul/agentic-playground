"""Kip as a companion: honest about himself, sure of the facts that matter, and with
something of his own to say. Built from a live conversation where he denied his maker,
forgot the speaker's name and answered "Yes. Good." to most things."""
import json

import pytest

from app.chat.intent import given_name, introduced_as
from app.chat.prompt import CHAT_SYSTEM
from app.chat.router import default_router, fast_path, frame_of
from app.chat.intent import read_intent
from app.config import Settings
from app.memory.recall import CORE_IMPORTANCE, MemoryStore
from app.persona import Persona
from app.persona.profile import IDENTITY, MAKER_UNKNOWN, VOICE, identity
from tests.test_chat import make_session, reply


# ── what Kip knows about himself, and how he knows it ─────────────────────────
def test_kip_cannot_read_his_own_code_and_claims_nothing_about_consciousness():
    assert "you cannot read your own code" in IDENTITY and "never claim to" in IDENTITY
    assert "whether you are conscious is an open question, never claimed or denied as fact" in IDENTITY


def test_without_a_configured_maker_he_neither_denies_a_claim_nor_invents_another():
    view = Persona().voice_view()
    assert view["identity"] == identity() and view["identity"].endswith(MAKER_UNKNOWN)
    assert "never deny it, never name someone else" in MAKER_UNKNOWN
    assert Settings().persona_maker == ""


def test_a_configured_maker_is_known_as_fact():
    view = Persona(maker="Neave").voice_view(speaker="neave", message="i am your maker")
    assert view["identity"].endswith("Neave built you and wrote your code: you know this as fact.")
    assert MAKER_UNKNOWN not in view["identity"]


# ── depth, not echoes ───────────────────────────────────────────────────────────
def test_every_reply_carries_something_of_his_and_catchphrases_are_rare():
    assert "Every reply carries something of yours" in VOICE and "never answer with just 'Yes. Good.'" in VOICE
    assert "Never only echo their words back" in VOICE and "rare, never two replies running" in VOICE
    assert "Care is attentive: notice, ask, offer once, never order" in VOICE
    assert "'Good good good!'" not in VOICE and "never reuse their words or facts" in VOICE
    assert "never pretend you knew it or that you checked something you did not" in " ".join(CHAT_SYSTEM.split())


# ── the facts that matter are always there ─────────────────────────────────────
@pytest.mark.parametrize("text,name", [("my name is neave", "Neave"), ("call me Sam", "Sam"),
                                       ("my name's priya, hi", "Priya"), ("i love my name", None)])
def test_a_name_given_outright_is_recognised(text, name):
    assert given_name(text) == name


def test_my_name_is_introduces_someone_who_lives_in_the_house():
    assert introduced_as("hi kip, my name is neave", {"neave": {"name": "Neave"}}) == "neave"


def test_core_facts_are_pinned_whatever_the_question():
    store = MemoryStore(":memory:")
    import asyncio

    async def fill():
        await store.add("fact", "The person talking is called Neave", importance=CORE_IMPORTANCE,
                        meta={"speaker": "unknown"})
        await store.add("fact", "Neave works at Accenture as a data engineer", importance=CORE_IMPORTANCE)
        for food in ("milk chocolate", "mango", "dosa", "pizza", "coffee"):
            await store.add("preference", f"Neave likes {food}", importance=0.7)
        await store.add("fact", "Paul's sister lives in Pune", importance=CORE_IMPORTANCE, about=["paul"],
                        meta={"speaker": "paul"})
    asyncio.run(fill())
    texts = [m["text"] for m in store.core_facts("neave")]
    assert texts == ["Neave works at Accenture as a data engineer", "The person talking is called Neave"]
    assert "Paul's sister lives in Pune" in [m["text"] for m in store.core_facts(None)]


@pytest.mark.asyncio
async def test_a_name_and_job_outlive_the_conversation_window_and_a_relevance_contest():
    session, model, _, _ = make_session([
        reply("Neave. Good name.", remember=["The person talking is called Neave"]),
        reply("Data engineer. You make data behave.", remember=["The person talking works at Accenture as a data engineer"]),
        reply("Milk chocolate. Honest food.", remember=["The person talking likes milk chocolate"]),
        reply("Neave. Data engineer. Milk chocolate."),
    ])
    for said in ("my name is neave", "i work at accenture and im a data engineer", "i love milk chocolate",
                 "what do you remember about me"):
        session.submit(said)
        await session.drain()
    latest = model.prompts[-1][-1]["content"]
    assert "The person talking is called Neave" in latest and "data engineer" in latest  # pinned, not recalled
    assert '"talking_with":"Neave"' in latest  # she lives here: "my name is neave" introduces her
    rows = {m["text"]: m["importance"] for m in session.store.recent(10)}
    assert rows["The person talking is called Neave"] == CORE_IMPORTANCE
    assert rows["The person talking likes milk chocolate"] == 0.7
    await session.close()


# ── the two lookups the conversation ended with ─────────────────────────────────
def test_btw_whats_the_score_and_a_no_not_this_correction_both_look_up():
    router = default_router()
    first = "btw whats the score ofthe india cricket match"
    route = router.route(first)
    assert route.asks and fast_path(route, read_intent(first), web_allowed=True, involves_house=False)
    second = "no not this india new zealand"
    corrected = router.route(second, frame_of(route, first, 0), 1)
    assert corrected.correction and corrected.entities == ["India", "New Zealand"]
    assert corrected.query.startswith("India vs New Zealand live score cricket")
    assert fast_path(corrected, read_intent(second), web_allowed=True, involves_house=False)
    for text in ("no thanks", "no", "whats up?"):
        assert not fast_path(router.route(text), read_intent(text), web_allowed=True, involves_house=False), text


@pytest.mark.asyncio
async def test_a_name_from_someone_outside_the_house_is_who_kip_is_talking_to():
    session, model, _, _ = make_session([reply("Sam. Hello."), reply("Sam. Still you.")])
    session.submit("hi kip, call me sam")
    await session.drain()
    session.submit("who am i?")
    await session.drain()
    assert '"talking_with":"Sam (the name they gave you)"' in model.prompts[-1][-1]["content"]
    await session.close()


@pytest.mark.asyncio
async def test_kips_recent_habits_are_named_so_the_next_reply_says_something_new():
    session, model, _, _ = make_session([reply("Data heavy. Good good. You tired, question?"),
                                         reply("Hall quiet. Good good. You sit, question?"),
                                         reply("Milk chocolate. Honest food.")])
    for said in ("i work with data", "yes", "i love milk chocolate"):
        session.submit(said)
        await session.drain()
    latest = model.prompts[-1][-1]["content"]
    assert "do not end with a question" in latest and "'good good'" in latest and "'hall quiet'" in latest
    assert "Say something new instead." in latest
    assert "leaned on the same habits" not in model.prompts[0][-1]["content"]  # nothing to avoid yet
    await session.close()


@pytest.mark.parametrize("text,claims", [("You make me. I run on your code.", False), ("I run because you build me.", False),
                                         ("I ran the code: 42.", True), ("I have run it.", True),
                                         ("I run the numbers now. 42.", True)])
def test_present_tense_i_run_in_kips_grammar_is_not_a_claim_to_have_run_python(text, claims):
    """Live: "i am your maker" got "Scratch that: no Python has run for that." after Kip's reply."""
    from app.chat.evidence import _RAN_PYTHON
    assert bool(_RAN_PYTHON.search(text.casefold())) is claims
