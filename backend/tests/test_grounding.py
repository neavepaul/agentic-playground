"""Answers about live facts must come from what was read, and belong to the match asked about.

Built from a live run: a fixture page listed "08:05 am IST ... Sri Lanka vs Hong Kong" and
"12:30 pm IST ... New Zealand vs India", and Kip answered "8:05 a.m. IST" for India's game.
"""
import pytest

from app.agents.schemas import GoalCondition
from app.capabilities.grounding import wrong_event
from app.capabilities.schemas import AnswerAction
from app.capabilities.tools import CapabilityTools
from app.tasks.models import TaskContext

FIXTURES = ("Cricket Matches Today 1 Nov 2026 · all times IST. Still to come today t20 07:10 am IST Not started "
            "Mission Road Ground, Mong Kok Bangladesh vs India t20 08:05 am IST Not started yet Mission Road Ground, "
            "Mong Kok Sri Lanka vs Hong Kong t20 12:30 pm IST Not started yet Seddon Park, Hamilton New Zealand vs "
            "India Captain: Glenn Phillips 9 /10 t20 12:40 pm IST Mong Kok Australia vs Afghanistan "
            "Earlier today odi 08:05 am IST Hong Kong vs Sri Lanka")
SERIES = "T20I start time 12:30 PM IST (8:00 PM local) ODI start time 7:30 AM IST (3:00 PM local)"
GOAL = ("Look up online: when is the india cricket game today? (they mean: cricket: India — start time "
        "(a clock time and day: just that, not the whole series), today.)")


def test_a_time_listed_for_another_match_is_not_the_answer():
    assert wrong_event("8:05 a.m. IST, 1 November 2026.", [FIXTURES, SERIES], ["India"], 330) == [
        "08:05 is the time of Sri Lanka vs Hong Kong, not of India"]


@pytest.mark.parametrize("answer", ["12:30 pm today. New Zealand vs India.", "12:30 pm IST.", "7:00 am UTC."])
def test_the_time_listed_with_the_match_passes_in_any_zone(answer):
    assert wrong_event(answer, [FIXTURES, SERIES], ["India"], 330) == []


def test_a_series_wide_time_with_no_fixture_named_passes_and_a_missing_time_does_not():
    assert wrong_event("12:30 pm.", [SERIES], ["India"], 330) == []
    assert wrong_event("3:15 pm.", [SERIES], ["India"], 330) == ["15:15 is in none of the pages read"]
    assert wrong_event("8:05 am.", [FIXTURES], [], 330) == []  # no team asked about: nothing to check


def test_the_explorer_cannot_finish_with_another_matchs_time():
    context = TaskContext(goal=GOAL, conditions=[GoalCondition(kind="answer", required_tools=["web_search"])])
    context.record("web_search", {"query": "India start time cricket"}, {"success": True, "evidence_id": "s1",
                   "observation": {"query": "India start time cricket", "results": [
                       {"title": "Cricket Matches Today", "url": "https://icrickets.example/today", "snippet": ""}]}})
    context.record("read_webpage", {"url": "https://icrickets.example/today"}, {"success": True, "evidence_id": "p1",
                   "observation": {"url": "https://icrickets.example/today", "title": "Today", "text": FIXTURES}})
    with pytest.raises(ValueError, match="08:05 is the time of Sri Lanka vs Hong Kong"):
        CapabilityTools.finish_answer(AnswerAction(tool="finish_answer", answer="8:05 a.m. IST.",
                                                   evidence_ids=["p1"]), context)
    done = CapabilityTools.finish_answer(AnswerAction(tool="finish_answer", answer="12:30 pm today.",
                                                      evidence_ids=["p1"]), context)
    assert done["answer"] == "12:30 pm today."


# ── a zone named in words, with its daylight saving on the day ─────────────────
LIVE_QUOTE = "First ball at Seddon Park in Hamilton is 8:00 pm Sunday 1 November: it’s played on New Zealand time."


@pytest.mark.parametrize("answer,day,ok", [
    ("12:30 pm IST, Sunday 1 November 2026.", (2026, 11, 1), True),   # NZDT, UTC+13: a live answer once rejected
    ("1:30 pm IST, Sunday 1 November 2026.", (2026, 11, 1), False),
    ("1:30 pm IST.", (2026, 5, 3), True),                              # NZST in May, UTC+12
])
def test_a_zone_the_quote_names_in_words_converts_with_daylight_saving(answer, day, ok):
    from datetime import date
    from app.capabilities.grounding import unsupported
    assert (unsupported(answer, LIVE_QUOTE, LIVE_QUOTE, 330, date(*day)) == []) is ok


def test_a_quote_naming_two_zones_lends_neither_to_a_bare_time():
    from datetime import date
    from app.capabilities.grounding import quote_zone
    assert quote_zone(LIVE_QUOTE, date(2026, 11, 1)) == "Pacific/Auckland"
    assert quote_zone("8:00 pm NZDT, which is 12:30 pm IST", date(2026, 11, 1)) is None
    assert quote_zone("Kick-off at 3 pm", date(2026, 11, 1)) is None


# ── the live run: a March final for today's score, 7 runs for 7 wickets, 4-1 for 3-2 ──
def test_a_temperature_said_in_celsius_is_backed_by_a_page_in_fahrenheit():
    """Kip says temperatures in °C; a page that gives 33 °F backs "1 °C", not an invented 7 °C."""
    from app.capabilities.grounding import unsupported
    quote = "London: Clear, 33°F, wind 5 mph"
    assert unsupported("Clear sky, about 1 °C.", quote, quote, 330) == []
    assert unsupported("Clear sky, 0.6 °C.", quote, quote, 330) == []
    assert unsupported("Clear sky, 7 °C.", quote, quote, 330) == ["7 is in nothing that was read"]


def test_a_question_about_today_rejects_a_source_dated_another_day():
    from datetime import date
    from app.capabilities.grounding import wrong_day
    today = date(2026, 11, 1)
    assert wrong_day(["IND vs NZ Cricket Scorecard, Final at Ahmedabad, March 08, 2026"], today) == \
        "the source is about 08 March 2026, not today (01 November 2026)"
    assert wrong_day(["NZ vs IND 5th T20I Match Report, November 01, 2026"], today) is None
    assert wrong_day(["Match report, 31 Oct 2026"], today) is None   # a day either side: time zones
    assert wrong_day(["no date anywhere"], today) is None


def test_the_winner_named_must_be_the_one_the_source_says_won():
    """Live: "Kimi Antonelli won ... the Mexico City Grand Prix" from "Kimi Antonelli crowned youngest
    F1 champion as Max Verstappen wins eventful Mexico City Grand Prix"."""
    from app.capabilities.grounding import wrong_winner
    quote = "Kimi Antonelli crowned youngest F1 champion as Max Verstappen wins eventful Mexico City Grand Prix"
    assert wrong_winner("Kimi Antonelli won the most recent Formula 1 race, the 2026 Mexico City Grand Prix.",
                        quote) == "the source does not say Antonelli won"
    assert wrong_winner("Max Verstappen won the Mexico City Grand Prix.", quote) is None
    assert wrong_winner("Lando Norris won latest race, São Paulo Grand Prix.",
                        "Nov 08, 2026 11:31pm UTC – Lando Norris claimed victory") is None
    assert wrong_winner("Race won by Norris. Good drive.", "Victory for Lando Norris in São Paulo") is None
    assert wrong_winner("Clear sky, 3 °C.", "Clear sky, temperature 3 °C") is None  # no winner named: no check


def test_a_latest_question_rejects_a_source_older_than_what_was_read():
    """Live: "most recent F1 race" was answered from an October Singapore report, while the São
    Paulo race of 8 November was in the results Groq saw."""
    from datetime import date
    from app.capabilities.grounding import stale_latest
    today = date(2026, 11, 14)
    read = ("Singapore GP: Verstappen wins, 5 October 2026 ... São Paulo GP results, November 08, 2026 ... "
            "Updated 14 November 2026 ... Las Vegas GP, 21 November 2026")
    assert stale_latest(["2026 Singapore Grand Prix race report, 5 October 2026"], read, today) == \
        "the source is from 05 October 2026, but what was read names 08 November 2026"
    # Today's date (page headers) and future dates (schedules) are not a later event.
    assert stale_latest(["Nov 08, 2026 11:31pm UTC: Lando Norris claimed victory"], read, today) is None
    # Undated: nothing to compare, so it passes (live, rejecting one threw out a right answer).
    assert stale_latest(["Lando Norris won the 2026 Sao Paulo Grand Prix"], read, today) is None


@pytest.mark.parametrize("answer,ok", [
    ("New Zealand beat India by 7 wickets.", True),
    ("New Zealand beat India by 7 runs.", False),              # the source says wickets
    ("India take the series 3-2.", True),
    ("Series now 4-1 New Zealand.", False),                    # no source said 4-1
    ("Starts 12:30, on 2026-11-01.", True),                    # times and dates are not scores
])
def test_result_figures_must_be_what_the_source_says(answer, ok):
    from app.capabilities.grounding import unbacked_results
    source = "New Zealand 195 for 3 beat India 191 for 8 by seven wickets (7 wkts). India take series 3-2."
    assert (unbacked_results(answer, source) == []) is ok


def test_the_page_reading_answer_cannot_turn_wickets_into_runs():
    context = TaskContext(goal="Look up online: india lost? (they mean: cricket: India vs New Zealand — match result)",
                          conditions=[GoalCondition(kind="answer", required_tools=["web_search"])])
    context.record("web_search", {"query": "India match result"}, {"success": True, "evidence_id": "s1", "observation": {
        "query": "India match result", "results": [{"title": "5th T20I Live Score", "url": "https://x.example/5",
                                                     "snippet": "New Zealand beat India by 7 wkts. India lead 3-1."}]}})
    with pytest.raises(ValueError, match="'7 runs' is not what the source says"):
        CapabilityTools.finish_answer(AnswerAction(tool="finish_answer", answer="New Zealand beat India by 7 runs.",
                                                   evidence_ids=["s1"]), context)
    done = CapabilityTools.finish_answer(AnswerAction(tool="finish_answer", answer="Yes. New Zealand won by 7 wickets.",
                                                      evidence_ids=["s1"]), context)
    assert done["answer"] == "Yes. New Zealand won by 7 wickets."
