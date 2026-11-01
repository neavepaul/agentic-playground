"""The intent router: what a message means, read locally, and the gates it never gets around.

Model-free: the TF-IDF router trains from the shipped examples in a fraction of a second.
"""
from datetime import date

import pytest

from app.chat.intent import read_intent
from app.chat.router import (LIVE, Frame, IntentRouter, default_router, fast_path, frame_of, read_slots)
from app.chat.router_train import cases, evaluate
from app.config import Settings
from app.memory.recall import MemoryStore
from tests.test_conversation_truth import PlanningModel, done, kip_lines, ran, session_with  # noqa: F401
from tests.test_permissions import said, settle

TODAY = date(2026, 11, 1)


@pytest.fixture(scope="module")
def router() -> IntentRouter:
    return default_router()


def conversation(router, *messages):
    """Route messages in order, each with the last information request as context."""
    frame, route = None, None
    for turn, text in enumerate(messages):
        route = router.route(text, frame, turn, TODAY)
        frame = frame_of(route, text, turn) or frame
    return route


def allowed(route, text, web=True, house=False, **flags):
    return fast_path(route, read_intent(text, **flags), web_allowed=web, involves_house=house)


# ── what a message means ───────────────────────────────────────────────────────
def test_where_did_lando_qualify_asks_for_a_grid_position_not_a_place(router):
    """The real failure: Kip answered with the circuit. "Where" never decides the aspect."""
    route = router.route("where did lando qualify in mexico?", today=TODAY)
    assert route.intent == "information.sports" and route.domain == "f1"
    assert route.entities == ["Lando Norris"] and route.focus == "qualifying position"
    assert route.time == "latest completed" and route.place == "Mexico"
    assert route.query == "Lando Norris qualifying position Mexico Grand Prix Formula 1 2026"
    assert "where" not in route.query.casefold()
    assert "a position, not a place" in route.meaning and "2026" in route.meaning
    assert allowed(route, "where did lando qualify in mexico?")


@pytest.mark.parametrize("text,focus", [("where did hamilton finish", "finishing position"),
                                        ("did charles win today", "finishing position"),
                                        ("who took pole at silverstone", "qualifying position"),
                                        ("when's the next grand prix", "schedule"),
                                        ("did chelsea beat spurs", "match result")])
def test_the_aspect_comes_from_the_sport_not_the_question_word(router, text, focus):
    assert router.route(text, today=TODAY).focus == focus


def test_names_the_lexicon_knows_are_resolved_and_ambiguous_ones_need_a_reason(router):
    assert read_slots("did max win").ambiguous == [("Max Verstappen", "f1")]
    assert read_slots("did max win").entities == []           # "max" alone is also a word
    assert router.route("did max win the race", today=TODAY).entities == ["Max Verstappen"]
    assert router.route("max is a great driver", today=TODAY).intent == "conversation.casual"


@pytest.mark.parametrize("text", ["lando is so fast", "i love watching f1 with you", "do you like cricket",
                                  "the weather is lovely today", "moira is watching the cricket in the hall",
                                  "lando crashed and i'm gutted", "what's your favourite team", "how's it going"])
def test_talk_about_sport_or_weather_is_not_a_lookup(router, text):
    assert not allowed(router.route(text, today=TODAY), text)


def test_a_typed_lookup_without_a_verb_still_counts(router):
    for text in ("ipl points table", "price of gold", "weekend forecast for bangalore"):
        route = router.route(text, today=TODAY)
        assert route.lookup_phrase and allowed(route, text), text


# ── context ───────────────────────────────────────────────────────────────────
def test_what_about_max_keeps_the_question_and_changes_the_driver(router):
    route = conversation(router, "where did lando qualify?", "what about max?")
    assert route.followup and route.entities == ["Max Verstappen"]
    assert route.focus == "qualifying position" and route.earlier == "where did lando qualify?"
    assert "Max Verstappen qualifying position" in route.query and allowed(route, "what about max?")


def test_how_did_he_finish_keeps_the_driver_and_asks_for_the_finishing_position(router):
    route = conversation(router, "where did lando qualify?", "how did he finish?")
    assert route.entities == ["Lando Norris"] and route.focus == "finishing position"


def test_and_yesterday_moves_the_weather_question_in_time(router):
    route = conversation(router, "what's the weather in delhi", "and yesterday?")
    assert route.intent == "information.weather" and route.place == "Delhi"
    assert route.time == "yesterday" and route.focus == "past weather"
    assert route.query == "weather Delhi yesterday"


def test_a_message_about_something_else_does_not_lean_on_the_last_question(router):
    assert not conversation(router, "where did lando qualify?", "i'm so tired today").followup
    assert not conversation(router, "where did lando qualify?", "he was rough on me today").followup
    # Context goes stale: four messages later "how did he finish?" points at nobody.
    frame = Frame("information.sports", "f1", "qualifying position", ["Lando Norris"], None, None, "x", 0)
    assert router.route("how did he finish?", frame, turn=4, today=TODAY).clarify


def test_a_pronoun_with_nobody_to_point_at_is_asked_about_not_guessed(router):
    route = router.route("how did he do?", today=TODAY)
    assert route.clarify == "Who you mean, question?" and not allowed(route, "how did he do?")


# ── corrections ───────────────────────────────────────────────────────────────
def test_i_meant_what_position_is_a_correction_of_the_same_question(router):
    route = conversation(router, "where did lando qualify?", "i meant what position")
    assert route.correction and route.focus == "qualifying position" and route.entities == ["Lando Norris"]
    assert allowed(route, "i meant what position")


def test_a_correction_can_change_the_aspect_and_its_no_is_not_a_refusal(router):
    text = "no, i meant the race, not qualifying"
    route = conversation(router, "where did lando qualify?", text)
    assert route.correction and route.focus == "finishing position"
    assert read_intent(text).declines and allowed(route, text)
    assert not allowed(router.route("no", today=TODAY), "no")


@pytest.mark.parametrize("text", ["what about max?", "thanks, and where did he finish?",
                                  "i meant to tell you, i'm going out"])
def test_follow_ups_and_asides_are_not_corrections(router, text):
    assert not conversation(router, "where did lando qualify?", text).correction


# ── the gates the router never gets around ─────────────────────────────────────
def test_cancellations_permissions_offers_tools_and_the_house_come_first(router):
    route = router.route("where did lando qualify?", today=TODAY)
    assert allowed(route, "where did lando qualify?")
    assert not allowed(route, "where did lando qualify?", web=False)       # web search ruled out
    assert not allowed(route, "where did lando qualify?", house=True)      # mixed with a house errand
    # "Don't read webpages; where did...": the question still stands, the search is still allowed.
    assert allowed(route, "where did lando qualify?", revoked=["read_webpage"])
    assert not allowed(route, "where did lando qualify?", web=False, revoked=["web_search"])
    assert not allowed(route, "don't search the web", revoked=["web_search"])  # nothing asked
    for text in ("stop", "search the web for lando's qualifying result",
                 "offer to check the weather but wait for my ok"):
        assert not allowed(router.route(text, today=TODAY), text), text


# ── measured quality: the evaluation set ───────────────────────────────────────
def test_the_evaluation_set_holds_its_measured_quality(router):
    report = evaluate(router, cases())
    right = lambda key: report["accuracy"][key][0] / report["accuracy"][key][1]
    assert report["false_positive_fast"] == 0          # never a lookup nobody asked for
    assert right("fast path") >= 0.95 and right("tool routing") >= 0.93 and right("intent") >= 0.9
    assert right("regression cases") == 1 and right("context resolution") == 1
    assert right("correction detection") == 1 and right("entities") >= 0.95
    assert report["accuracy"]["legacy fast path"][0] < report["accuracy"]["fast path"][0]
    assert report["latency_ms"]["p95"] < 20


# ── corrections and shadow decisions persist for offline review ───────────────
def test_corrections_are_deduplicated_and_join_training_only_once_accepted():
    store = MemoryStore(":memory:")
    corrected = {"intent": "information.sports", "focus": "qualifying position"}
    first = store.add_intent_correction("Where did Lando qualify?", {}, {}, corrected, "i meant what position")
    assert first and store.add_intent_correction("where did lando  qualify?", {}, {}, corrected, "position!") is None
    [row] = store.intent_corrections()
    assert row["seen"] == 2 and row["status"] == "unreviewed" and row["outcome"] == "pending"
    assert store.accepted_intent_examples() == []      # unverified: never training data
    store.set_intent_outcome(first, "answered")
    assert store.review_intent_correction(first, "accepted")
    assert store.accepted_intent_examples() == [("information.sports", "Where did Lando qualify?")]
    assert IntentRouter(store.accepted_intent_examples()).rows[-1] == ("information.sports",
                                                                        "Where did Lando qualify?")


def test_shadow_decisions_are_summarised_and_bounded():
    store = MemoryStore(":memory:")
    for index in range(3):
        store.log_intent_shadow(f"q{index}", {"intent": "information.sports"}, router_fast=True,
                                live_fast=index == 0)
    summary = store.intent_shadow_summary()
    assert summary["messages"] == 3 and summary["both_fast"] == 1 and summary["router_only_fast"] == 2
    assert summary["router_only_examples"] == ["q1", "q2"]


# ── in the conversation ───────────────────────────────────────────────────────
def answering(text):
    def answer(payload):
        ids = [o["evidence_id"] for o in payload["tool_observations"] if o["success"]]
        return {"summary": "Ready.", "action": {"tool": "finish_answer", "evidence_ids": ids, "answer": text}}
    return answer


@pytest.mark.asyncio
async def test_shadow_mode_records_the_router_and_changes_nothing(ran):
    model = PlanningModel([said("Not sure. I check.")])
    session, manager = session_with(model, settings=Settings(direct_initial_delegation=True,
                                                             intent_router="shadow"))
    session.submit("where did russell end up?")  # no live-fact word: the usual path replies first
    await settle(session, manager)
    assert len(model.prompts) == 1 and not manager.tasks
    summary = session.store.intent_shadow_summary()
    assert summary["router_only_fast"] == 1 and summary["router_only_examples"] == ["where did russell end up?"]
    await done(session, manager)


@pytest.mark.asyncio
async def test_on_the_router_sends_a_live_question_straight_to_a_lookup_of_what_it_means(ran):
    model = PlanningModel([], task=[answering("P6. Not his best day."), answering("Max P2.")])
    session, manager = session_with(model, settings=Settings(direct_initial_delegation=True, intent_router="on"))
    session.submit("where did russell end up?")
    await settle(session, manager)
    await manager.runner
    session.submit("what about max?")
    await settle(session, manager)
    await manager.runner
    assert model.prompts == []                            # no chat-model call for either
    first, second = list(manager.tasks.values())
    assert "they mean: Formula 1: George Russell — finishing position" in first.goal
    assert first.action_history[2]["arguments"]["query"].startswith("George Russell finishing position")
    assert "following up on: where did russell end up?" in second.goal
    assert second.action_history[2]["arguments"]["query"].startswith("Max Verstappen finishing position")
    assert kip_lines(session, "narration") == ["P6. Not his best day.", "Max P2."]
    await done(session, manager)


@pytest.mark.asyncio
async def test_on_a_pronoun_with_nobody_named_gets_a_question_back_without_a_model_call(ran):
    model = PlanningModel([])
    session, manager = session_with(model, settings=Settings(intent_router="on"))
    session.submit("how did he do in qualifying?")
    await settle(session, manager)
    assert model.prompts == [] and not manager.tasks
    assert session.history()[-1]["text"] == "Who you mean, question?"
    await done(session, manager)


@pytest.mark.asyncio
async def test_on_a_correction_is_recorded_and_the_corrected_question_looked_up(ran):
    model = PlanningModel([], task=[answering("Lando P3."), answering("Lando P6 on grid.")])
    session, manager = session_with(model, settings=Settings(direct_initial_delegation=True, intent_router="on"))
    session.submit("where did lando qualify?")
    await settle(session, manager)
    await manager.runner
    session.submit("no, i meant what position")
    await settle(session, manager)
    await manager.runner
    corrected = list(manager.tasks.values())[-1]
    assert "they corrected: no, i meant what position" in corrected.goal
    [row] = session.store.intent_corrections()
    assert row["utterance"] == "where did lando qualify?" and row["correction"] == "no, i meant what position"
    assert row["status"] == "unreviewed" and row["outcome"] == "answered"
    await done(session, manager)


@pytest.mark.asyncio
async def test_on_a_ruled_out_web_search_keeps_the_usual_reply(ran):
    model = PlanningModel([said("Alright."), said("No web. Not know where he qualify.")])
    session, manager = session_with(model, settings=Settings(intent_router="on"))
    session.submit("don't search the web")
    await settle(session, manager)
    session.submit("where did lando qualify?")
    await settle(session, manager)
    assert len(model.prompts) == 2 and not manager.tasks
    await done(session, manager)


@pytest.mark.asyncio
async def test_on_small_talk_about_sport_gets_a_reply_not_a_lookup(ran):
    model = PlanningModel([said("Cricket good. Long game. Very long.")])
    session, manager = session_with(model, settings=Settings(intent_router="on"))
    session.submit("do you like cricket?")  # the live-fact words alone would have looked this up
    await settle(session, manager)
    assert len(model.prompts) == 1 and not manager.tasks
    assert "This is live information" not in model.prompts[0][-1]["content"]
    await done(session, manager)


def test_live_intents_are_the_ones_that_change_by_the_hour():
    assert LIVE == {"information.sports", "information.weather", "information.finance", "information.news"}


# ── the run that went wrong: "when is the india new zealand game starting" ──────
def test_a_start_time_question_keeps_its_aspect_teams_and_today(router):
    route = router.route("when is the india new zealand game starting", today=TODAY)
    assert route.focus == "start time" and route.entities == ["India", "New Zealand"]
    assert route.query == "India vs New Zealand start time cricket 1 November 2026"
    assert "not the whole series" in route.meaning and "today (1 November 2026)" in route.meaning


def test_bare_follow_ups_inherit_the_match_and_real_questions_stand_alone(router):
    first = "when is the india new zealand game starting"
    for text, focus in (("what time kip?", "start time"), ("ah shit already started?", "match status")):
        route = conversation(router, first, text)
        assert route.followup and route.focus == focus and route.entities == ["India", "New Zealand"], text
        assert allowed(route, text)
    for text in ("whats up?", "what time is it?", "how are you bud?", "thanks kip"):
        route = conversation(router, first, text)
        assert not route.followup and not allowed(route, text), text


def test_kip_never_asks_permission_for_a_web_lookup():
    from app.chat.prompt import CHAT_SYSTEM
    from app.persona.profile import VOICE
    assert "never ask whether to" in CHAT_SYSTEM and "offer to check" in CHAT_SYSTEM  # the house still asks
    assert "look it up, question" not in VOICE


@pytest.mark.asyncio
async def test_the_router_is_on_by_default_and_answers_the_match_and_its_follow_up(ran):
    assert Settings().intent_router == "on"
    model = PlanningModel([], task=[answering("Twelve thirty today. Hamilton."), answering("Twelve thirty.")])
    session, manager = session_with(model, settings=Settings(direct_initial_delegation=True))
    session.submit("when is the india new zealand game starting")
    await settle(session, manager)
    await manager.runner
    session.submit("what time kip?")
    await settle(session, manager)
    await manager.runner
    assert model.prompts == []  # no chat-model reply, so nothing to ask permission or guess
    first, second = list(manager.tasks.values())
    assert first.action_history[2]["arguments"]["query"].startswith("India vs New Zealand start time cricket")
    assert "following up on: when is the india new zealand game starting" in second.goal
    assert kip_lines(session, "narration") == ["Twelve thirty today. Hamilton.", "Twelve thirty."]
    await done(session, manager)


# ── "what time is it?": the device clock answers, no model call ───────────────
@pytest.mark.parametrize("text", ["what time is it?", "whats the time kip", "what time kip?", "time please"])
def test_clock_questions_are_recognised(text):
    from app.chat.router import clock_question
    assert clock_question(text)


@pytest.mark.parametrize("text", ["what time is the match?", "what time is it in london?", "what time do you wake up"])
def test_other_time_questions_are_not_the_clock(text):
    from app.chat.router import clock_question
    assert not clock_question(text)


@pytest.mark.asyncio
async def test_the_clock_answers_the_time_but_what_time_after_a_match_question_means_the_match(ran):
    from app.world.clock import WorldClock
    model = PlanningModel([], task=[answering("Twelve thirty today. Hamilton.")])
    session, manager = session_with(model, settings=Settings(direct_initial_delegation=True))
    session.engine._clock = WorldClock(speed=0, start_hour=13.25)
    session.submit("what time is it?")
    await settle(session, manager)
    assert session.history()[-1]["text"] == "1:15 afternoon." and model.prompts == [] and not manager.tasks
    session.submit("when is the india new zealand game starting")
    await settle(session, manager)
    await manager.runner
    session.submit("what time kip?")  # about the match now, not the clock
    await settle(session, manager)
    assert len(manager.tasks) == 2 and "following up on" in list(manager.tasks.values())[-1].goal
    await done(session, manager)
