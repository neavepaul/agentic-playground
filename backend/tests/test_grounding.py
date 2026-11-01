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
