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
    {"place": "Delhi", "focus": "current conditions", "time": "now", "said": "what's the uv index in delhi?"},
    {"place": "Delhi", "focus": "forecast", "time": "tomorrow", "said": "will the aqi in delhi be bad tomorrow?"},
    {"place": None, "focus": "current conditions", "time": "now", "said": "is it cold outside?"},
    {"place": "London", "focus": "past weather", "time": "last week", "said": "how was london last week?"},
])
async def test_weather_the_forecast_does_not_carry_goes_to_the_web(question):
    get, asked = open_meteo()
    assert await feeds.answer({"domain": "weather", **question}, NOW, get) is None and asked == []


@pytest.mark.asyncio
@pytest.mark.parametrize("said", ["what's the aqi in delhi?", "how bad is the pollution in delhi?",
                                  "what's the PM2.5 in delhi right now?"])
async def test_air_quality_now_comes_from_open_meteo(said):
    asked = []

    async def get(url):
        asked.append(url)
        if url.startswith(feeds.GEOCODE):
            return {"results": [{"name": "Delhi", "latitude": 28.65, "longitude": 77.23}]}
        return {"current": {"time": "2026-11-15T09:00", "us_aqi": 158, "pm2_5": 45.3, "pm10": 56.0}}
    found = await feeds.answer({"domain": "weather", "place": "Delhi", "focus": "current conditions", "time": "now",
                                "said": said}, NOW, get)
    assert found.answer == "Delhi air now: AQI 158 (US scale), unhealthy. PM2.5 45 µg/m³, PM10 56 µg/m³."
    assert asked[1].startswith(feeds.AIR) and found.provider == "open-meteo"


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


# ── markets: CoinGecko, Yahoo (indices), Frankfurter (currency) ─────────────────
def markets(pages):
    asked = []

    async def get(url):
        asked.append(url)
        return next(body for prefix, body in pages.items() if url.startswith(prefix))
    return get, asked


@pytest.mark.asyncio
async def test_crypto_prices_in_rupees_and_dollars():
    get, asked = markets({feeds.CRYPTO: {"bitcoin": {"inr": 6109514, "usd": 64826, "inr_24h_change": -1.234}}})
    found = await feeds.answer({"domain": "finance", "entities": ["Bitcoin"], "focus": "price", "time": "now",
                                "said": "what's the bitcoin price?"}, NOW, get)
    assert found.answer == "Bitcoin is ₹61,09,514 ($64,826), down 1.2% in 24 hours."
    assert "ids=bitcoin" in asked[0] and found.provider == "coingecko"


@pytest.mark.asyncio
async def test_an_index_with_its_move_and_the_time_in_ist():
    meta = {"regularMarketPrice": 25144.3, "chartPreviousClose": 25044.0, "regularMarketTime": 1794650400}
    get, asked = markets({feeds.INDEX: {"chart": {"result": [{"meta": meta}]}}})
    found = await feeds.answer({"domain": "finance", "entities": ["Nifty 50"], "focus": "price", "time": "now",
                                "said": "what's the nifty at?"}, NOW, get)
    # 1794650400 is 10:00 UTC on 14 November 2026: 3:30 pm IST, when the NSE closes.
    assert found.answer == "Nifty 50 is at 25,144.30, up 0.40% on the day (as of Saturday 14 November, 3:30 pm)."
    assert asked[0].startswith(f"{feeds.INDEX}/%5ENSEI") and found.provider == "yahoo"


@pytest.mark.asyncio
@pytest.mark.parametrize("said,url,rates,expected", [
    ("what's the dollar rate?", "from=USD&to=INR", {"INR": 94.25},
     "1 US dollar is ₹94.25 (European Central Bank rate, 13 November)."),
    ("how much is a euro in rupees?", "from=EUR&to=INR", {"INR": 104.17},
     "1 euro is ₹104.17 (European Central Bank rate, 13 November)."),
    ("how many dollars is a pound?", "from=GBP&to=USD", {"USD": 1.3227},
     "1 British pound is 1.3227 US dollars (European Central Bank rate, 13 November)."),
])
async def test_exchange_rates_from_the_ecb(said, url, rates, expected):
    get, asked = markets({feeds.RATES: {"date": "2026-11-13", "rates": rates}})
    found = await feeds.answer({"domain": "finance", "entities": [], "focus": "price", "time": "now", "said": said},
                               NOW, get)
    assert found.answer == expected and url in asked[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("question", [
    {"entities": ["gold"], "said": "price of gold today", "time": "today"},
    {"entities": ["Reliance Industries"], "said": "how was reliance last week?", "time": "last week"},
    {"entities": ["Bitcoin"], "said": "what was bitcoin last week?", "time": "last week"},
])
async def test_markets_the_feeds_do_not_cover_go_to_the_web(question):
    get, asked = markets({})
    assert await feeds.answer({"domain": "finance", "focus": "price", **question}, NOW, get) is None and asked == []


# ── news: Google News RSS (India) ───────────────────────────────────────────────
def rss(*titles):
    items = "".join(f"<item><title>{t}</title><link>https://news.google.com/x</link></item>" for t in titles)
    return f'<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel><title>Top</title>{items}</channel></rss>'


@pytest.mark.asyncio
async def test_top_headlines_are_read_out_with_their_sources():
    get, asked = markets({feeds.NEWS: rss("Monsoon session to begin on Monday - The Hindu", "Markets close higher - Mint",
                                          "India beat New Zealand - ESPNcricinfo", "Fourth story - BBC")})
    found = await feeds.answer({"domain": "news", "time": "today", "said": "what's in the news?"}, NOW, get)
    assert found.answer == ("Top headlines: Monsoon session to begin on Monday (The Hindu); Markets close higher "
                            "(Mint); India beat New Zealand (ESPNcricinfo).")
    assert asked == [f"{feeds.NEWS}?hl=en-IN&gl=IN&ceid=IN:en"] and found.provider == "google-news"


@pytest.mark.asyncio
async def test_news_on_a_topic_they_name_searches_for_it():
    get, asked = markets({feeds.NEWS: rss("Polls close in Bihar - NDTV")})
    found = await feeds.answer({"domain": "news", "time": "today", "said": "any news about the elections?"}, NOW, get)
    assert found.answer == "Headlines on elections: Polls close in Bihar (NDTV)."
    assert asked == [f"{feeds.NEWS}/search?q=elections&hl=en-IN&gl=IN&ceid=IN:en"]


@pytest.mark.asyncio
async def test_yesterdays_news_goes_to_the_web():
    get, asked = markets({feeds.NEWS: rss("x - y")})
    assert await feeds.answer({"domain": "news", "time": "yesterday", "said": "what was in the news yesterday?"},
                              NOW, get) is None and asked == []


# ── football (football-data.org, then ESPN) and NBA (ESPN) ──────────────────────
FD_KEY = "fd-test-key"


def fd(status, home, away, goals, winner, when):
    return {"status": status, "utcDate": when, "homeTeam": {"name": home}, "awayTeam": {"name": away},
            "score": {"fullTime": {"home": goals[0], "away": goals[1]}, "winner": winner}, "venue": None}


FD = {"matches": [
    fd("FINISHED", "SK Slavia Praha", "Arsenal FC", (1, 2), "AWAY_TEAM", "2026-11-04T20:00:00Z"),
    fd("FINISHED", "Arsenal FC", "Hull City AFC", (1, 0), "HOME_TEAM", "2026-11-07T15:00:00Z"),
    fd("FINISHED", "Manchester United FC", "Aston Villa FC", (4, 4), "DRAW", "2026-11-08T16:30:00Z"),
    fd("TIMED", "Chelsea FC", "Arsenal FC", (None, None), None, "2026-11-18T19:45:00Z"),
]}


def espn(state, home, away, scores=("", ""), winner=None, detail="", when="2026-11-15T00:00Z"):
    return {"date": when, "status": {"type": {"state": state, "detail": detail}},
            "competitions": [{"venue": {"fullName": "Crypto.com Arena"}, "competitors": [
                {"homeAway": "home", "team": {"displayName": home}, "score": scores[0], "winner": winner == home},
                {"homeAway": "away", "team": {"displayName": away}, "score": scores[1], "winner": winner == away}]}]}


def sports_feed(pages):
    asked = []

    async def get(url, headers=None):
        asked.append((url, headers))
        return next((body for prefix, body in pages.items() if url.startswith(prefix)), {"events": []})
    return get, asked


@pytest.mark.asyncio
@pytest.mark.parametrize("question,expected", [
    ({"entities": ["Arsenal"], "focus": "match result", "time": "latest completed"},
     "Arsenal beat Hull City on 7 November. Arsenal 1, Hull City 0."),
    ({"entities": ["Manchester United"], "focus": "match result", "time": "latest completed"},
     "Manchester United and Aston Villa drew on 8 November. Manchester United 4, Aston Villa 4."),
    ({"entities": ["Arsenal"], "focus": "schedule", "time": "next upcoming"},
     "Chelsea v Arsenal starts Thursday 19 November, 1:15 am."),
])
async def test_football_from_football_data(question, expected):
    get, asked = sports_feed({feeds.FOOTBALL: FD})
    found = await feeds.answer({"domain": "football", **question}, NOW, get, football_key=FD_KEY)
    assert found.answer == expected and found.provider == "football-data"
    assert sorted(asked) == [(f"{feeds.FOOTBALL}?dateFrom=2026-11-04&dateTo=2026-11-14", {"X-Auth-Token": FD_KEY}),
                             (f"{feeds.FOOTBALL}?dateFrom=2026-11-14&dateTo=2026-11-24", {"X-Auth-Token": FD_KEY})]
    assert FD_KEY not in found.url


@pytest.mark.asyncio
async def test_espn_answers_football_without_a_key_or_when_the_match_is_missing():
    board = {"events": [espn("post", "Brighton & Hove Albion", "Brentford", ("2", "1"), "Brighton & Hove Albion",
                             when="2026-11-08T12:30Z")]}
    for key, pages in (("", {f"{feeds.ESPN_SPORTS}/soccer/eng.1/": board}),
                       (FD_KEY, {feeds.FOOTBALL: {"matches": []}, f"{feeds.ESPN_SPORTS}/soccer/eng.1/": board})):
        get, asked = sports_feed(pages)
        found = await feeds.answer({"domain": "football", "entities": ["Brentford"], "focus": "match result",
                                    "time": "latest completed"}, NOW, get, football_key=key)
        assert found.answer == "Brighton & Hove Albion beat Brentford on 8 November. Brighton & Hove Albion 2, Brentford 1."
        assert found.provider == "espn" and len([u for u, _ in asked if "/soccer/" in u]) == 7


@pytest.mark.asyncio
async def test_nba_from_espn_over_yesterday_today_and_tomorrow():
    lakers = espn("post", "Los Angeles Lakers", "Golden State Warriors", ("112", "108"), "Los Angeles Lakers")
    get, asked = sports_feed({f"{feeds.ESPN_SPORTS}/basketball/nba/scoreboard?dates=20261114": {"events": [lakers]}})
    found = await feeds.answer({"domain": "basketball", "entities": ["Los Angeles Lakers"], "focus": "match result",
                                "time": "yesterday"}, NOW, get)
    assert found.answer == ("Los Angeles Lakers beat Golden State Warriors on 15 November. "
                            "Los Angeles Lakers 112, Golden State Warriors 108.")
    assert sorted(u.rsplit("=", 1)[1] for u, _ in asked) == ["20261113", "20261114", "20261115"]


@pytest.mark.asyncio
async def test_a_result_is_dated_in_ist():
    """20:00 UTC on 4 November is 1:30 am on 5 November in India."""
    get, _ = sports_feed({feeds.FOOTBALL: {"matches": [FD["matches"][0]]}})
    found = await feeds.answer({"domain": "football", "entities": ["Arsenal"], "focus": "match result",
                                "time": "latest completed"}, NOW, get, football_key=FD_KEY)
    assert found.answer == "Arsenal beat SK Slavia Praha on 5 November. SK Slavia Praha 1, Arsenal 2."


# ── shares (Yahoo search + chart) and their Groww holdings (read-only) ──────────
@pytest.mark.asyncio
async def test_a_named_share_is_found_on_nse_and_priced_in_rupees():
    meta = {"regularMarketPrice": 1339.7, "chartPreviousClose": 1329.05, "regularMarketTime": 1794650400, "currency": "INR"}
    get, asked = markets({feeds.SEARCH: {"quotes": [{"symbol": "RELIANCE.BO"}, {"symbol": "RELIANCE.NS"}]},
                          feeds.INDEX: {"chart": {"result": [{"meta": meta}]}}})
    found = await feeds.answer({"domain": "finance", "entities": ["Reliance Industries"], "focus": "price",
                                "time": "now", "said": "reliance share price"}, NOW, get)
    assert found.answer == "Reliance Industries is ₹1,339.70, up 0.80% on the day (as of Saturday 14 November, 3:30 pm)."
    assert asked[1].startswith(f"{feeds.INDEX}/RELIANCE.NS")  # NSE before BSE


GROWW = ("groww-key", "GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ")
HOLDINGS = {"status": "SUCCESS", "payload": {"holdings": [
    {"trading_symbol": "TCS", "quantity": 10, "average_price": 2000, "tradable_exchanges": ["NSE", "BSE"]},
    {"trading_symbol": "INFY", "quantity": 20, "average_price": 1200, "tradable_exchanges": ["NSE", "BSE"]},
    {"trading_symbol": "SOLD", "quantity": 0, "average_price": 50, "tradable_exchanges": ["NSE"]}]}}
SPARK = {"spark": {"result": [
    {"symbol": "TCS.NS", "response": [{"meta": {"regularMarketPrice": 2407.5, "chartPreviousClose": 2442.4}}]},
    {"symbol": "INFY.NS", "response": [{"meta": {"regularMarketPrice": 1141.5, "chartPreviousClose": 1151.2}}]}]}}


def groww_feed(monkeypatch):
    calls = []

    async def get(url, headers=None):
        calls.append(("GET", url, headers))
        return HOLDINGS if url == feeds.GROWW_HOLDINGS_URL else SPARK

    async def post(url, body, headers):
        calls.append(("POST", url, headers))
        assert body["key_type"] == "totp" and len(body["totp"]) == 6 and headers == {"Authorization": "Bearer groww-key"}
        return {"token": "access-token", "expiry": "2026-11-16T06:00:00"}
    monkeypatch.setattr(feeds, "post", post)
    return get, calls


@pytest.mark.asyncio
async def test_their_portfolio_from_groww_holdings_priced_from_yahoo(monkeypatch):
    get, calls = groww_feed(monkeypatch)
    question = {"domain": "finance", "entities": [], "time": None, "said": "how is my portfolio doing?"}
    found = await feeds.answer(question, NOW, get, groww=GROWW)
    # 10 x 2407.50 + 20 x 1141.50 = 46,905 against 44,000 paid; the day before it was 47,448.
    assert found.answer == ("Your Groww portfolio: 2 holdings worth ₹46,905, up ₹2,905 (6.6%) on what you paid. "
                            "On the day down ₹543 (1.14%); best INFY -0.8%, worst TCS -1.4%.")
    assert found.provider == "groww"
    await feeds.answer(question, NOW, get, groww=GROWW)
    assert [c[0] for c in calls].count("POST") == 1  # the token is kept until it expires


@pytest.mark.asyncio
async def test_kip_only_ever_calls_the_groww_token_and_holdings_endpoints(monkeypatch):
    """The key can place orders: Kip must never touch anything but these two."""
    get, calls = groww_feed(monkeypatch)
    for said in ("how is my portfolio doing?", "what are my holdings worth?", "what do i hold?"):
        await feeds.answer({"domain": "finance", "entities": [], "time": "now", "said": said}, NOW, get, groww=GROWW)
    groww_urls = {url for _, url, _ in calls if url.startswith(feeds.GROWW)}
    assert groww_urls == {feeds.GROWW_TOKEN_URL, feeds.GROWW_HOLDINGS_URL}


@pytest.mark.asyncio
async def test_without_groww_keys_portfolio_questions_go_elsewhere(monkeypatch):
    get, calls = groww_feed(monkeypatch)
    assert await feeds.answer({"domain": "finance", "entities": [], "time": "now", "said": "how is my portfolio doing?"},
                              NOW, get) is None and calls == []


@pytest.mark.asyncio
async def test_an_unchanged_share_is_flat_not_up_zero_percent():
    meta = {"regularMarketPrice": 1339.7, "chartPreviousClose": 1339.7, "regularMarketTime": 1794650400, "currency": "INR"}
    get, _ = markets({feeds.SEARCH: {"quotes": [{"symbol": "RELIANCE.NS"}]}, feeds.INDEX: {"chart": {"result": [{"meta": meta}]}}})
    found = await feeds.answer({"domain": "finance", "entities": ["Reliance Industries"], "time": "now",
                                "said": "reliance share price"}, NOW, get)
    assert found.answer == "Reliance Industries is ₹1,339.70, flat on the day (as of Saturday 14 November, 3:30 pm)."


@pytest.mark.parametrize("question,keys,expected", [
    ({"domain": "f1"}, {}, "Jolpica"),
    ({"domain": "cricket", "entities": ["India"]}, {}, "ESPN"),
    ({"domain": "cricket", "entities": ["India"]}, {"cricket_key": "k"}, "CricketData"),
    ({"domain": "football", "entities": ["Arsenal"]}, {}, "ESPN"),
    ({"domain": "football", "entities": ["Arsenal"]}, {"football_key": "k"}, "football-data.org"),
    ({"domain": "basketball", "entities": ["Los Angeles Lakers"]}, {}, "ESPN"),
    ({"domain": "weather", "place": "London", "time": "now", "said": "weather in london"}, {}, "Open-Meteo"),
    ({"domain": "news", "said": "any news"}, {}, "Google News"),
    # Questions these feeds do not take go to the web lookup, so no feed is named.
    ({"domain": "basketball", "entities": [], "said": "nba scores today"}, {}, None),
    ({"domain": "football", "entities": ["Arsenal"], "focus": "standings"}, {"football_key": "k"}, None),
    ({"domain": "weather", "place": None, "said": "is it going to rain"}, {}, None),
    ({"domain": "weather", "place": "Delhi", "time": "tomorrow", "said": "aqi in delhi tomorrow"}, {}, None),
    ({"domain": "news", "time": "last week", "said": "news from last week"}, {}, None),
    ({"domain": "finance", "entities": ["Bitcoin"], "said": "bitcoin price"}, {}, "CoinGecko"),
    ({"domain": "finance", "entities": ["Nifty 50"], "said": "where is the nifty"}, {}, "Yahoo Finance"),
    ({"domain": "finance", "entities": ["Reliance Industries"], "said": "reliance share price"}, {}, "Yahoo Finance"),
    ({"domain": "finance", "entities": [], "said": "dollar to rupee rate"}, {}, "Frankfurter"),
    ({"domain": "finance", "entities": [], "said": "how is my portfolio doing?"}, {"groww": GROWW}, "Groww"),
    ({"domain": "finance", "entities": [], "said": "how is my portfolio doing?"}, {}, None),
    ({"domain": "chemistry"}, {}, None),
])
def test_the_callout_names_the_feed_that_will_be_asked(question, keys, expected):
    assert feeds.source_for(question, **keys) == expected


@pytest.mark.asyncio
async def test_portfolio_answers_stay_on_the_device(monkeypatch, caplog):
    """Shown to them, but never in a later prompt, a memory, a task record, the log or the feed's
    event: nothing of it reaches Groq or a shared log."""
    import logging
    caplog.set_level(logging.INFO)
    from tests.test_conversation_truth import PlanningModel, done, session_with
    from tests.test_permissions import settle
    from app.chat.context import select_history
    from app.chat.evidence import task_record
    from app.config import Settings
    from app.memory.recall import task_memories

    get, _ = groww_feed(monkeypatch)
    monkeypatch.setattr(feeds, "fetch", get)
    model = PlanningModel([])
    session, manager = session_with(model, settings=Settings(intent_router="on", groww_api_key=GROWW[0],
                                                             groww_api_secret=GROWW[1]))
    session.submit("how is my portfolio doing?")
    await settle(session, manager)
    task = next(iter(manager.tasks.values()))
    assert task.status == "completed" and task.private and model.prompts == []
    kip = [t for t in session.history() if t["role"] == "kip"]
    assert kip[0]["text"] == "Let me check Groww for that."
    answer = next(t for t in kip if "46,905" in t["text"])
    assert answer["private"]
    assert all("46,905" not in m["content"] for m in select_history(session.history(), "my portfolio", 4000))
    assert "46,905" not in str(task_record(task)) and task_memories(task) == []
    assert "Your Groww portfolio" not in caplog.text and "46,905" not in caplog.text
    fed = next(e for e in manager.bus.history if e["type"] == "capability_observed")
    assert "46,905" not in str(fed) and fed["data"]["private"]
    await done(session, manager)


@pytest.mark.asyncio
async def test_old_city_names_and_namesakes_find_the_big_city():
    """"Bangalore" alone geocodes to a part of Karachi; "Hyderabad" first to the one in Sindh."""
    asked = []
    places = {"Bengaluru": [{"name": "Bengaluru", "latitude": 12.97, "longitude": 77.59, "population": 8495492}],
              "Hyderabad": [{"name": "Hyderabad", "latitude": 25.39, "longitude": 68.37, "population": 1386330},
                            {"name": "Hyderabad", "latitude": 17.38, "longitude": 78.46, "population": 6809970}]}

    async def get(url):
        asked.append(url)
        if url.startswith(feeds.GEOCODE):
            return {"results": next(v for k, v in places.items() if f"name={k}" in url)}
        return FORECAST
    for place, latitude in (("Bangalore", "12.97"), ("Hyderabad", "17.38")):
        await feeds.answer({"domain": "weather", "place": place, "focus": "current conditions", "time": "now",
                            "said": f"weather in {place.lower()}"}, NOW, get)
        assert f"latitude={latitude}" in asked[-1]


@pytest.mark.asyncio
async def test_a_score_asked_before_the_game_says_when_it_starts():
    board = {"events": [{"date": "2026-11-15T21:00Z", "status": {"type": {"state": "pre"}}, "competitions": [{
        "venue": {"fullName": "Footprint Center"}, "competitors": [
            {"homeAway": "home", "team": {"displayName": "Phoenix Suns"}},
            {"homeAway": "away", "team": {"displayName": "Los Angeles Lakers"}}]}]}]}

    async def get(url):
        return board if "20261115" in url else {"events": []}
    found = await feeds.answer({"domain": "basketball", "entities": ["Los Angeles Lakers"], "focus": "score",
                                "time": "today or latest", "said": "lakers score"}, NOW.replace(day=15, hour=10), get)
    assert found.answer.startswith("Phoenix Suns v Los Angeles Lakers starts") and "Footprint Center" in found.answer
    # A result asked for is never answered with a game that has not happened.
    assert await feeds.answer({"domain": "basketball", "entities": ["Los Angeles Lakers"], "focus": "match result",
                               "time": "yesterday", "said": "did the lakers win yesterday"},
                              NOW.replace(day=15, hour=10), get) is None


@pytest.mark.asyncio
async def test_a_holding_with_no_recorded_cost_still_gets_an_answer(monkeypatch):
    """Gifted or bonus shares, or a cost Groww has not recorded: no percentage, never a crash."""
    holdings = {"status": "SUCCESS", "payload": {"holdings": [
        {"trading_symbol": "BONUS", "quantity": 1, "average_price": 0, "tradable_exchanges": ["NSE"]}]}}
    spark = {"spark": {"result": [{"symbol": "BONUS.NS", "response": [
        {"meta": {"regularMarketPrice": 100, "chartPreviousClose": 100}}]}]}}
    get, _ = groww_feed(monkeypatch)

    async def priced(url, headers=None):
        return holdings if url == feeds.GROWW_HOLDINGS_URL else spark if url.startswith(feeds.SPARK) else await get(url, headers)
    found = await feeds.answer({"domain": "finance", "entities": [], "time": "now", "said": "how is my portfolio doing?"},
                               NOW, priced, groww=GROWW)
    assert found.answer == "Your Groww portfolio: 1 holding worth ₹100, up ₹100 on what you paid. On the day up ₹0 (0.00%)."
