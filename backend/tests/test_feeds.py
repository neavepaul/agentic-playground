"""F1 and cricket answered from results feeds (Jolpica, ESPN): exact facts in fixed words, and None
for whatever a feed does not cover, so the web lookup answers instead. Shapes are the feeds' own."""
from datetime import datetime, timedelta, timezone

import pytest

from app.capabilities import feeds

IST = timezone(timedelta(hours=5, minutes=30))
NOW = datetime(2026, 11, 14, 18, 0, tzinfo=IST)


def driver(given, family, team, position, grid=1):
    return {"position": str(position), "grid": str(grid), "Driver": {"givenName": given, "familyName": family},
            "Constructor": {"name": team}}


RESULTS = {"MRData": {"RaceTable": {"Races": [{
    "season": "2026", "round": "21", "raceName": "Brazilian Grand Prix", "date": "2026-11-08",
    "Results": [driver("Lando", "Norris", "McLaren", 1, grid=3), driver("Max", "Verstappen", "Red Bull", 2, grid=1)],
    "QualifyingResults": [driver("Max", "Verstappen", "Red Bull", 1), driver("Lando", "Norris", "McLaren", 3)]}]}}}
SEASON = {"MRData": {"RaceTable": {"Races": [
    {"round": "20", "raceName": "Mexico City Grand Prix", "date": "2026-11-01",
     "Circuit": {"circuitName": "Autódromo Hermanos Rodríguez", "Location": {"locality": "Mexico City", "country": "Mexico"}}},
    {"round": "22", "raceName": "Las Vegas Grand Prix", "date": "2026-11-21",
     "Circuit": {"circuitName": "Las Vegas Strip Circuit", "Location": {"locality": "Las Vegas", "country": "USA"}}}]}}}
NEXT = {"MRData": {"RaceTable": {"Races": [{**SEASON["MRData"]["RaceTable"]["Races"][1], "time": "04:00:00Z"}]}}}
STANDINGS = {"MRData": {"StandingsTable": {"StandingsLists": [{"season": "2026", "DriverStandings": [
    {"position": "1", "points": "390", "Driver": {"givenName": "Lando", "familyName": "Norris"}},
    {"position": "2", "points": "371", "Driver": {"givenName": "Max", "familyName": "Verstappen"}}]}]}}}


def feed(pages: dict):
    asked = []

    async def get(url):
        asked.append(url)
        return pages[url]
    return get, asked


@pytest.mark.asyncio
@pytest.mark.parametrize("question,expected", [
    ({"focus": "finishing position", "time": "latest completed"},
     "Lando Norris won the Brazilian Grand Prix on 8 November 2026, for McLaren."),
    ({"entities": ["Max Verstappen"], "focus": "finishing position", "time": "latest completed"},
     "Max Verstappen finished 2nd in the Brazilian Grand Prix on 8 November 2026."),
    ({"entities": ["Lando Norris"], "focus": "starting position"},
     "Lando Norris started 3rd on the grid at the Brazilian Grand Prix on 8 November 2026."),
])
async def test_f1_results_come_from_the_latest_race(question, expected):
    get, asked = feed({f"{feeds.F1}/2026/last/results.json": RESULTS})
    found = await feeds.answer({"domain": "f1", **question}, NOW, get)
    assert found.answer == expected and found.provider == "jolpica" and asked == [f"{feeds.F1}/2026/last/results.json"]


@pytest.mark.asyncio
async def test_f1_pole_standings_and_next_race():
    get, _ = feed({f"{feeds.F1}/2026/last/qualifying.json": RESULTS,
                   f"{feeds.F1}/current/driverStandings.json": STANDINGS, f"{feeds.F1}/current/next.json": NEXT})
    pole = await feeds.answer({"domain": "f1", "focus": "qualifying position"}, NOW, get)
    assert pole.answer == "Max Verstappen took pole in the Brazilian Grand Prix qualifying on 8 November 2026, for Red Bull."
    lando = await feeds.answer({"domain": "f1", "entities": ["Lando Norris"], "focus": "qualifying position"}, NOW, get)
    assert lando.answer == "Lando Norris qualified 3rd for the Brazilian Grand Prix on 8 November 2026."
    max_ = await feeds.answer({"domain": "f1", "entities": ["Max Verstappen"], "focus": "standings"}, NOW, get)
    assert max_.answer == "Max Verstappen is 2nd in the 2026 championship with 371 points."
    nxt = await feeds.answer({"domain": "f1", "focus": "schedule", "time": "next upcoming"}, NOW, get)
    # 04:00 UTC on 21 November is 9:30 am in India.
    assert nxt.answer == "The next race is the Las Vegas Grand Prix in Las Vegas, USA, on Saturday 21 November, 9:30 am."


@pytest.mark.asyncio
async def test_f1_a_named_race_that_has_happened_is_found_by_place():
    get, asked = feed({f"{feeds.F1}/2026.json": SEASON, f"{feeds.F1}/2026/20/results.json": RESULTS})
    await feeds.answer({"domain": "f1", "place": "Mexico", "focus": "finishing position"}, NOW, get)
    assert asked[-1] == f"{feeds.F1}/2026/20/results.json"
    # Las Vegas has not been raced yet: nothing to report, the web lookup takes over.
    assert await feeds.answer({"domain": "f1", "place": "Las Vegas", "focus": "result"}, NOW, get) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("question", [
    {"domain": "f1", "focus": "pit stops"}, {"domain": "f1", "focus": "practice times"},
    {"domain": "f1", "focus": "result", "time": "yesterday"}, {"domain": "football", "focus": "result"},
    {"domain": "f1", "entities": ["Oscar Piastri"], "focus": "finishing position"},
])
async def test_what_a_feed_does_not_cover_falls_back_to_the_web(question):
    get, _ = feed({f"{feeds.F1}/2026/last/results.json": RESULTS})
    assert await feeds.answer(question, NOW, get) is None


def match(state, home, away, date="2026-11-14T05:00:00Z", summary="", winner=None, location="Al Amerat"):
    return {"status": state, "date": date, "location": location, "link": "https://www.espncricinfo.com/x",
            "fullStatus": {"longSummary": summary},
            "competitors": [{"displayName": home[0], "score": home[1], "winner": winner == home[0]},
                            {"displayName": away[0], "score": away[1], "winner": winner == away[0]}]}


def scoreboard(*events):
    return {feeds.ESPN: {"sports": [{"leagues": [{"events": list(events)}]}]}}


@pytest.mark.asyncio
async def test_cricket_live_finished_and_upcoming_matches():
    live = match("in", ("India", "245/4 (41 ov)"), ("New Zealand", "280/8"), summary="India need 36 runs from 54 balls")
    get, _ = feed(scoreboard(live))
    found = await feeds.answer({"domain": "cricket", "entities": ["India", "New Zealand"], "focus": "score"}, NOW, get)
    assert found.answer == ("India v New Zealand is under way: India 245/4 (41 ov), New Zealand 280/8. "
                            "India need 36 runs from 54 balls.")
    done = match("post", ("India Under-19s", "339/9"), ("Pakistan Under-19s", "337/9"),
                 summary="IND Under-19 won by 1 wicket (with 0 balls remaining)", winner="India Under-19s")
    get, _ = feed(scoreboard(done))
    found = await feeds.answer({"domain": "cricket", "entities": ["India Under-19s"], "focus": "match result"}, NOW, get)
    assert found.answer == ("IND Under-19 won by 1 wicket (with 0 balls remaining) on 14 November. "
                            "India Under-19s 339/9, Pakistan Under-19s 337/9.")
    soon = match("pre", ("Sydney Sixers Women", ""), ("Melbourne Renegades Women", ""), date="2026-11-15T00:00:00Z",
                 location="North Sydney Oval, Sydney")
    get, _ = feed(scoreboard(soon))
    found = await feeds.answer({"domain": "cricket", "entities": ["Sydney Sixers Women"], "focus": "start time"}, NOW, get)
    assert found.answer == ("Sydney Sixers Women v Melbourne Renegades Women starts Sunday 15 November, 5:30 am "
                            "at North Sydney Oval, Sydney.")


@pytest.mark.asyncio
async def test_cricket_team_names_must_match_exactly():
    """"India" is not "India Under-19s": a youth final is no answer to how India did."""
    done = match("post", ("India Under-19s", "339/9"), ("Pakistan Under-19s", "337/9"), summary="IND Under-19 won")
    get, _ = feed(scoreboard(done))
    assert await feeds.answer({"domain": "cricket", "entities": ["India"], "focus": "score"}, NOW, get) is None
    assert await feeds.answer({"domain": "cricket", "entities": [], "focus": "score"}, NOW, get) is None


@pytest.mark.asyncio
async def test_a_routed_f1_question_is_answered_from_the_feed_with_no_model_call(monkeypatch):
    from tests.test_conversation_truth import PlanningModel, done, session_with
    from tests.test_permissions import settle
    from app.config import Settings

    async def fake(url):
        return RESULTS

    monkeypatch.setattr(feeds, "fetch", fake)
    model = PlanningModel([])
    session, manager = session_with(model, settings=Settings(intent_router="on"))
    session.submit("who won the most recent f1 race?")
    await settle(session, manager)
    task = next(iter(manager.tasks.values()))
    assert task.status == "completed" and task.answer["answer"].startswith("Lando Norris won the Brazilian Grand Prix")
    assert model.titles == [] and model.prompts == []  # no planning, no search, no chat reply
    assert task.action_history[0]["observation"]["provider"] == "jolpica"
    await done(session, manager)


@pytest.mark.asyncio
async def test_a_cricket_score_with_figures_passes_the_answer_checks(monkeypatch):
    """Live: the feed's "Zimbabwe 339" failed the answer check, which reads only titles and snippets,
    and the question fell through to a five-minute loop on the local model."""
    from tests.test_conversation_truth import PlanningModel, done, session_with
    from tests.test_permissions import settle
    from app.config import Settings
    live = match("in", ("Afghanistan", "159/6 (48 ov)"), ("Zimbabwe", "339"), summary="Stumps")

    async def fake(url):
        return scoreboard(live)[url]

    monkeypatch.setattr(feeds, "fetch", fake)
    model = PlanningModel([])
    session, manager = session_with(model, settings=Settings(intent_router="on"))
    session.submit("what's the score in afghanistan vs zimbabwe?")
    await settle(session, manager)
    task = next(iter(manager.tasks.values()))
    assert task.status == "completed", task.summary
    assert task.answer["answer"] == "Afghanistan v Zimbabwe is under way: Afghanistan 159/6 (48 ov), Zimbabwe 339. Stumps."
    assert model.titles == []
    await done(session, manager)


# ── CricketData (cricScore) first, ESPN when it cannot answer ─────────────────────
KEY = "00000000-0000-0000-0000-000000000000"
CRICAPI = f"{feeds.CRICAPI}?apikey={KEY}"


def row(ms, t1, t1s, t2, t2s, status, when):
    return {"ms": ms, "t1": t1, "t1s": t1s, "t2": t2, "t2s": t2s, "status": status, "dateTimeGMT": when}


CRICSCORE = {"status": "success", "info": {"hitsToday": 3, "hitsLimit": 100}, "data": [
    row("result", "India [IND]", "89/10 (26.4)", "New Zealand [NZ]", "333/10 (50)", "New Zealand won by 244 runs",
        "2026-11-13T02:00:00"),
    row("fixture", "India [IND]", "", "New Zealand [NZ]", "", "Match starts at Nov 15, 02:00 GMT", "2026-11-15T02:00:00"),
    row("live", "Afghanistan [AFG]", "159/6 (48)", "Zimbabwe [ZIM]", "339/10 (114.5)",
        "Day 2: Stumps - Afghanistan trail by 180 runs", "2026-11-13T05:30:00"),
    row("result", "India U19 [IN19]", "339/9 (50)", "Pakistan U19 [PK19]", "337/9 (50)", "India U19 won by 1 wkt",
        "2026-11-14T05:00:00"),
]}


@pytest.mark.asyncio
async def test_cricketdata_answers_results_fixtures_and_live_scores():
    """Live, 14 November: ESPN's few-day window had no senior India match; cricScore had India's
    defeat on the 13th and the next match on the 15th."""
    get, asked = feed({CRICAPI: CRICSCORE})
    india = {"domain": "cricket", "entities": ["India"], "focus": "match result", "time": "today or latest"}
    found = await feeds.answer(india, NOW, get, cricket_key=KEY)
    assert found.answer == "New Zealand won by 244 runs on 13 November. India 89/10 (26.4), New Zealand 333/10 (50)."
    assert found.provider == "cricketdata" and KEY not in found.url  # the key never reaches the evidence
    nxt = await feeds.answer({**india, "focus": "start time", "time": "today or next"}, NOW, get, cricket_key=KEY)
    assert nxt.answer == "India v New Zealand starts Sunday 15 November, 7:30 am."
    live = await feeds.answer({"domain": "cricket", "entities": ["Afghanistan", "Zimbabwe"], "focus": "score"},
                               NOW, get, cricket_key=KEY)
    assert live.answer == ("Afghanistan v Zimbabwe is under way: Afghanistan 159/6 (48), Zimbabwe 339/10 (114.5). "
                           "Day 2: Stumps - Afghanistan trail by 180 runs.")
    assert asked == [CRICAPI]  # three questions inside a minute: one of the day's 100 requests


@pytest.mark.asyncio
@pytest.mark.parametrize("cricapi", [
    {"status": "failure", "reason": "hits today exceeded hits limit"},  # the day's 100 are used
    {"status": "success", "data": []},                                  # the match is not in it
])
async def test_espn_answers_when_cricketdata_cannot(cricapi):
    live = match("in", ("India", "245/4 (41 ov)"), ("New Zealand", "280/8"), summary="India need 36 runs")
    get, asked = feed({CRICAPI: cricapi, **scoreboard(live)})
    found = await feeds.answer({"domain": "cricket", "entities": ["India"], "focus": "score"}, NOW, get,
                                cricket_key=KEY)
    assert found.provider == "espn" and asked == [CRICAPI, feeds.ESPN]


@pytest.mark.asyncio
async def test_without_a_key_only_espn_is_asked():
    live = match("in", ("India", "245/4 (41 ov)"), ("New Zealand", "280/8"), summary="India need 36 runs")
    get, asked = feed(scoreboard(live))
    found = await feeds.answer({"domain": "cricket", "entities": ["India"], "focus": "score"}, NOW, get)
    assert found.provider == "espn" and asked == [feeds.ESPN]


# ── weather from Open-Meteo, in °C ──────────────────────────────────────────────
GEO = {"results": [{"name": "London", "country": "United Kingdom", "latitude": 51.50853, "longitude": -0.12574}]}
FORECAST = {"current": {"time": "2026-11-14T17:00", "temperature_2m": 7.7, "apparent_temperature": 5.3,
                        "relative_humidity_2m": 67, "weather_code": 0, "wind_speed_10m": 4.7},
            "daily": {"time": ["2026-11-13", "2026-11-14", "2026-11-15", "2026-11-16"],
                      "weather_code": [3, 3, 61, 61], "temperature_2m_max": [8.3, 9.7, 11.0, 8.2],
                      "temperature_2m_min": [1.6, -0.4, 4.7, 5.3], "precipitation_probability_max": [0, 0, 90, 54],
                      "precipitation_sum": [0.0, 0.0, 4.8, 4.8]}}


def open_meteo():
    asked = []

    async def get(url):
        asked.append(url)
        return GEO if url.startswith(feeds.GEOCODE) else FORECAST
    return get, asked


@pytest.mark.asyncio
@pytest.mark.parametrize("focus,time,expected", [
    ("current conditions", "today", "London now: clear sky, 8 °C (feels like 5 °C), humidity 67%, wind 5 km/h. "
                                    "Today: overcast, 0 to 10 °C."),
    ("forecast", "tomorrow", "London tomorrow: light rain, 5 to 11 °C, 90% chance of rain."),
    ("past weather", "yesterday", "London yesterday: overcast, 2 to 8 °C."),
    # 14 November 2026 is a Saturday: the weekend is today and tomorrow.
    ("forecast", "this weekend", "London today: overcast, 0 to 10 °C. Tomorrow: light rain, 5 to 11 °C, "
                                 "90% chance of rain."),
])
async def test_weather_now_today_tomorrow_yesterday_and_the_weekend(focus, time, expected):
    get, asked = open_meteo()
    found = await feeds.answer({"domain": "weather", "place": "London", "focus": focus, "time": time}, NOW, get)
    assert found.answer == expected and found.provider == "open-meteo"
    assert asked[0].startswith(feeds.GEOCODE) and "timezone=auto" in asked[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("question", [
    {"place": "Delhi", "focus": "current conditions", "time": "now", "said": "what's the aqi in delhi?"},
    {"place": None, "focus": "current conditions", "time": "now", "said": "is it cold outside?"},
    {"place": "London", "focus": "past weather", "time": "last week", "said": "how was london last week?"},
])
async def test_weather_the_forecast_does_not_carry_goes_to_the_web(question):
    get, asked = open_meteo()
    assert await feeds.answer({"domain": "weather", **question}, NOW, get) is None and asked == []


@pytest.mark.asyncio
async def test_a_routed_weather_question_is_answered_from_open_meteo_with_no_model_call(monkeypatch):
    from tests.test_conversation_truth import PlanningModel, done, session_with
    from tests.test_permissions import settle
    from app.config import Settings
    get, _ = open_meteo()
    monkeypatch.setattr(feeds, "fetch", get)
    model = PlanningModel([])
    session, manager = session_with(model, settings=Settings(intent_router="on"))
    session.submit("What's the weather in London today?")
    await settle(session, manager)
    task = next(iter(manager.tasks.values()))
    assert task.status == "completed", task.summary
    assert task.answer["answer"].startswith("London now: clear sky, 8 °C") and model.titles == []
    await done(session, manager)


def test_the_clock_is_india_standard_time_whatever_the_device_says():
    from app import clock
    assert clock.now().utcoffset() == timedelta(hours=5, minutes=30) and clock.now().tzname() == "IST"
    assert "IST (UTC+05:30)" in clock.date_str()
