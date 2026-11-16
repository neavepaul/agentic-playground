"""Kip does not leave "I check flight prices" hanging, and does not add up budgets in his head."""
import pytest

from app.chat.evidence import unkept_promise
from app.chat.intent import read_intent
from app.chat.session import LOOKUP_PREFIX
from app.tasks.permissions import READ_ONLY
from tests.test_chat import make_session, reply


# ── what counts as a promise ────────────────────────────────────────────────
@pytest.mark.parametrize("text, expected", [
    ("Budget tight. I verify flight prices first. Vegetarian food easy.", ("verify flight prices", "lookup")),
    ("Good swap. I check specific plantation options now. One question: Do we need a guide?",
     ("check specific plantation options", "lookup")),
    ("I'll look it up for you.", ("look it up", "lookup")),
    ("Let me search for quiet beaches near Agonda.", ("search for quiet beaches near agonda", "lookup")),
    ("I will calculate the total now.", ("calculate the total", "calculation")),
    ("I find places.", ("find places", "lookup")),
])
def test_a_promise_to_look_something_up_is_found(text, expected):
    assert unkept_promise(text) == expected


@pytest.mark.parametrize("text", [
    "I check my notes. Plan set.", "I can search web for that. Want me to?", "I search web for quiet beaches, question?",
    "I find Palolem too loud.", "I check with you first.", "Nice to meet you, Claude.", "I checked it yesterday.",
    "I don't check prices.", "Should I check the price?", ""])
def test_things_that_are_not_promises_are_left_alone(text):
    assert unkept_promise(text) is None


# ── what counts as asking for arithmetic ────────────────────────────────────
@pytest.mark.parametrize("text", [
    "Can you sanity check whether we are still inside the forty thousand each?",
    "Is 11k enough for stay and scuba at 3000 a night?", "What is 15% of 2400?", "How much is it in total?",
    "Please add up the costs for three nights.", "What's 17 + 23?"])
def test_a_request_for_arithmetic_is_recognised(text):
    assert read_intent(text).calculates


@pytest.mark.parametrize("text", [
    "Our budget is around forty thousand rupees per person in total including flights.",
    "How much is a litre of milk?", "Where do I live?", "Use Python to work out 17 * 23 + 4", "Stop that.",
    "I'm planning a trip to Goa for four days."])
def test_a_statement_or_a_plain_question_is_not(text):
    assert not read_intent(text).calculates


# ── in a conversation ───────────────────────────────────────────────────────
def corrections(session):
    return [t["text"] for t in session.history() if t.get("source") == "correction"]


@pytest.mark.asyncio
async def test_an_unkept_promise_is_called_out_and_held_for_a_yes():
    session, model, manager, _ = make_session([reply("Budget tight. I check flight prices first."), reply("Looking.")])
    session.submit("Our budget is forty thousand each, including flights from Pune.")
    await session.drain()
    assert corrections(session) == ["Scratch that: I haven't started that. Say yes and I check flight prices first."
                                    .replace(" first", "")]
    assert session.offer and session.offer.goal.startswith(LOOKUP_PREFIX + "check flight prices")
    assert session.offer.effects == sorted(READ_ONLY) and not manager.tasks


@pytest.mark.asyncio
async def test_a_yes_then_runs_the_held_lookup():
    session, model, manager, _ = make_session([reply("I check flight prices."), reply("On it.")])
    session.submit("Our budget is forty thousand each, including flights from Pune.")
    await session.drain()
    session.submit("yes please")
    await session.drain()
    assert [t.goal for t in manager.tasks.values()][0].startswith(LOOKUP_PREFIX + "check flight prices")


@pytest.mark.asyncio
async def test_a_promise_with_an_errand_behind_it_is_not_called_out():
    session, model, manager, _ = make_session([reply("I check flight prices now.",
                                                     task="Look up online: flight prices Pune to Goa Dabolim")])
    session.submit("Please look up flight prices Pune to Goa online.")
    await session.drain()
    assert corrections(session) == [] and len(manager.tasks) == 1


@pytest.mark.asyncio
async def test_a_promise_to_calculate_is_called_out_without_inventing_a_job():
    session, model, manager, _ = make_session([reply("I will calculate the total now.")])
    session.submit("We have forty thousand each, with a stay of nine thousand and scuba at three thousand.")
    await session.drain()
    assert corrections(session) == ["Scratch that: I haven't started that, so nothing is done. "
                                    "Ask me again and I calculate the total."]
    assert session.offer is None and not manager.tasks


@pytest.mark.asyncio
async def test_an_arithmetic_request_tells_the_model_to_run_python_with_the_numbers():
    session, model, *_ = make_session([reply("Running it.", task="Calculate with Python: 9000 + 3000 vs 11000")])
    session.submit("Can you sanity check whether we are still inside the forty thousand each?")
    await session.drain()
    latest = model.prompts[0][-1]["content"]
    assert "put a Python calculation in task" in latest and "every number it uses" in latest
    session2, model2, *_ = make_session([reply("Hello.")])
    session2.submit("Where do I live?")
    await session2.drain()
    assert "Python calculation in task" not in model2.prompts[0][-1]["content"]


@pytest.mark.asyncio
async def test_a_proposed_calculation_runs_and_nothing_is_called_out():
    session, model, manager, _ = make_session([reply("Running it now.",
                                                     task="Calculate with Python: 9000 + 3000 against 11000")])
    session.submit("Can you sanity check whether Rohan is inside eleven thousand for stay and scuba?")
    await session.drain()
    assert corrections(session) == [] and len(manager.tasks) == 1


@pytest.mark.asyncio
async def test_a_sum_done_in_the_head_is_called_out():
    session, model, manager, _ = make_session([reply("Rohan: 9k plus 3k equals 12k. Over by 1k. Fine.")])
    session.submit("Can you sanity check whether Rohan is inside eleven thousand for stay and scuba?")
    await session.drain()
    assert any("worked that sum out in my head" in c for c in corrections(session)) and not manager.tasks


@pytest.mark.asyncio
async def test_ordinary_replies_are_never_flagged():
    session, model, manager, _ = make_session([reply("Pune. Good city."), reply("Chai it is.")])
    session.submit("Where do I live?")
    await session.drain()
    session.submit("What drink do I like?")
    await session.drain()
    assert corrections(session) == []
