"""The feed readers: questions read into each feed's form from example sentences, never wrong
with confidence (an unsure reading goes to the web lookup)."""
import json
from pathlib import Path

import pytest

from app.capabilities import feeds
from evaluations.slot_bench import reader_reading, score
from app.chat.router import default_router

CASES = [json.loads(line) for line in
         (Path(__file__).parents[1] / "evaluations" / "slot_cases.jsonl").read_text(encoding="utf-8").splitlines()
         if line.strip()]


@pytest.mark.parametrize("case", CASES, ids=[c["say"] for c in CASES])
def test_a_question_is_read_right_or_left_to_the_web_lookup(case):
    route = default_router().route(case["say"], None, 0)
    got, sure = reader_reading(case["domain"], route, case["say"])
    off_topic = "other" in (case["expect"].get("ask"), case["expect"].get("kind"))
    assert not sure or off_topic and got.get("ask", got.get("kind")) == "other" or not score(case["expect"], got), got


@pytest.mark.parametrize("said,ask,table", [
    ("how many points does mclaren have in the championship", "standings", "constructors"),
    ("how many points does mclaren driver lando norris have in the champoinship", "standings", "drivers"),
    ("who is the f1 drivers champion", "standings", "drivers"),
])
def test_the_f1_reader_tells_a_team_from_a_driver(said, ask, table):
    reading = feeds.reader("f1").read(said)
    assert reading.confident and (reading.form["ask"], reading.form["table"]) == (ask, table)


@pytest.mark.parametrize("said", ["are geore russel and max verstappen beefing?", "is verstappen leaving red bull"])
def test_a_question_the_f1_feed_does_not_cover_goes_to_the_web_lookup(said):
    """It once answered "George Russell finished 4th in Brazil" to whether he and Max are beefing."""
    assert not feeds.reader("f1").read(said).confident
    assert feeds.source_for({"domain": "f1", "entities": [], "said": said}) is None


def test_names_are_the_words_no_example_uses():
    reading = feeds.reader("finance").read("do i have any tata motors")
    assert reading.form["kind"] == "my holding" and reading.name == "tata motors"
    assert feeds.reader("f1").read("how many points does russel have").known == ["russell"]  # a typo


@pytest.mark.parametrize("said,domain,entities", [
    ("is kohli injured?", "cricket", ["Virat Kohli"]),
    ("did arsenal sign anyone", "football", ["Arsenal"]),
    ("are the lakers going to trade lebron", "basketball", ["Los Angeles Lakers", "LeBron James"]),
])
def test_a_sports_question_the_scoreboards_do_not_cover_goes_to_the_web_lookup(said, domain, entities):
    """These once got the side's latest score: the feeds answered anything that named a side."""
    assert feeds.source_for({"domain": domain, "entities": entities, "said": said}) is None


@pytest.mark.parametrize("said,domain,entities,focus", [
    ("what's the score in the india match", "cricket", ["India"], "score"),
    ("when do arsenal play next", "football", ["Arsenal"], "schedule"),
    ("premier league results today", "football", [], "match result"),
    ("did the lakers win", "basketball", ["Los Angeles Lakers"], "match result"),
])
def test_the_match_reader_says_which_match_is_asked_about(said, domain, entities, focus):
    q = feeds._match_read({"domain": domain, "entities": entities, "said": said})
    assert q is not None and q["focus"] == focus
    assert feeds.source_for({"domain": domain, "entities": entities, "said": said}) is not None
