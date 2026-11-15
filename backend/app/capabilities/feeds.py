"""Live facts from structured feeds: Formula 1 from Jolpica (the Ergast successor, free, no key)
and cricket from CricketData.org (with a key) or ESPN's scoreboard (unofficial, no key), weather from
Open-Meteo (free, no key, °C). One request, exact facts, an answer in
fixed words: no model reads pages or writes the answer. Anything outside what a feed covers gives
None, and the usual web lookup answers instead."""
import asyncio
import logging
import re
import time
from dataclasses import dataclass
from urllib.parse import quote, urlencode
from xml.etree import ElementTree
from datetime import date, datetime, timedelta

import httpx

_log = logging.getLogger("agentic_friend.capabilities")
F1 = "https://api.jolpi.ca/ergast/f1"
ESPN = "https://site.web.api.espn.com/apis/v2/scoreboard/header?sport=cricket"
CRICAPI = "https://api.cricapi.com/v1/cricScore"
GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST = "https://api.open-meteo.com/v1/forecast"
AIR = "https://air-quality-api.open-meteo.com/v1/air-quality"
RATES = "https://api.frankfurter.app/latest"
CRYPTO = "https://api.coingecko.com/api/v3/simple/price"
INDEX = "https://query1.finance.yahoo.com/v8/finance/chart"
NEWS = "https://news.google.com/rss"
FOOTBALL = "https://api.football-data.org/v4/matches"
ESPN_SPORTS = "https://site.api.espn.com/apis/site/v2/sports"
FEEDS = {"f1": "jolpica", "cricket": "cricketdata, then espn", "weather": "open-meteo",
         "finance": "coingecko, yahoo, frankfurter", "news": "google-news",
         "football": "football-data, then espn", "basketball": "espn"}


@dataclass
class FeedAnswer:
    answer: str
    url: str
    title: str
    provider: str


async def fetch(url: str, headers: dict | None = None) -> dict | str:
    """JSON as a dict; anything else (RSS) as text."""
    async with httpx.AsyncClient(timeout=10, headers={"User-Agent": "kip-companion"}, follow_redirects=True) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        try:
            return response.json()
        except ValueError:
            return response.text


async def answer(question: dict, now: datetime, get=None, cricket_key: str = "",
                 football_key: str = "") -> FeedAnswer | None:
    """question: the router's reading (domain, entities, focus, time, place) and their words (said)."""
    get = get or fetch
    if question.get("domain") == "f1":
        return await _f1(question, now, get)
    if question.get("domain") == "cricket":
        return await _cricket(question, now, get, cricket_key)
    if question.get("domain") == "weather":
        return await _weather(question, get)
    if question.get("domain") == "finance":
        return await _markets(question, now, get)
    if question.get("domain") == "news":
        return await _news(question, get)
    if question.get("domain") == "football":
        return await _football(question, now, get, football_key)
    if question.get("domain") == "basketball":
        return await _basketball(question, now, get)
    return None


# ── Formula 1 ───────────────────────────────────────────────────────────────────
_KINDS = {"qualifying position": ("qualifying", "QualifyingResults"), "sprint result": ("sprint", "SprintResults")}
_PAST_TIMES = (None, "latest", "latest completed", "today or latest")


async def _f1(q: dict, now: datetime, get) -> FeedAnswer | None:
    focus, time = q.get("focus"), q.get("time")
    who = q.get("entities") or []
    if focus in ("schedule", "start time") or time in ("next upcoming", "today or next"):
        url = f"{F1}/current/next.json"
        races = (await get(url))["MRData"]["RaceTable"]["Races"]
        if not races:
            return None
        race = races[0]
        when = _local(race["date"], race.get("time"), now)
        where = race["Circuit"]["Location"]
        return FeedAnswer(f"The next race is the {race['raceName']} in {where['locality']}, {where['country']}, "
                            f"on {when}.", url, f"Jolpica F1: {race['raceName']} schedule", "jolpica")
    if focus == "standings":
        url = f"{F1}/current/driverStandings.json"
        table = (await get(url))["MRData"]["StandingsTable"]["StandingsLists"]
        if not table:
            return None
        rows = table[0]["DriverStandings"]
        row = _driver(rows, who) if who else rows[0]
        if row is None:
            return None
        return FeedAnswer(f"{_name(row)} is {_ordinal(row['position'])} in the {table[0]['season']} championship "
                            f"with {row['points']} points.", url, "Jolpica F1: driver standings", "jolpica")
    if focus not in (None, "finishing position", "result", "starting position", *_KINDS) or (
            time not in _PAST_TIMES and not re.fullmatch(r"\d{4}", str(time))):
        return None  # practice times, pit stops, "yesterday" and the like: the web lookup
    year = time if re.fullmatch(r"\d{4}", str(time)) else str(now.year)
    kind, key = _KINDS.get(focus, ("results", "Results"))
    round_ = "last"
    if q.get("place"):
        round_ = await _round_at(q["place"], year, now, get)
        if round_ is None:
            return None
    url = f"{F1}/{year}/{round_}/{kind}.json"
    races = (await get(url))["MRData"]["RaceTable"]["Races"]
    if not races or not races[0].get(key):
        return None
    race, rows = races[0], races[0][key]
    on = f"on {_day(race['date'])}"
    event = race["raceName"] + {"qualifying": " qualifying", "sprint": " sprint"}.get(kind, "")
    title = f"Jolpica F1: {event} {race['season']}"
    if who:
        row = _driver(rows, who)
        if row is None:
            return None
        if focus == "starting position":
            return FeedAnswer(f"{_name(row)} started {_ordinal(row['grid'])} on the grid at the "
                                f"{race['raceName']} {on}.", url, title, "jolpica")
        if kind == "qualifying":
            return FeedAnswer(f"{_name(row)} qualified {_ordinal(row['position'])} for the {race['raceName']} {on}.",
                                url, title, "jolpica")
        return FeedAnswer(f"{_name(row)} finished {_ordinal(row['position'])} in the {event} {on}.",
                            url, title, "jolpica")
    first = rows[0]
    if kind == "qualifying":
        return FeedAnswer(f"{_name(first)} took pole in the {event} {on}, for {first['Constructor']['name']}.",
                            url, title, "jolpica")
    return FeedAnswer(f"{_name(first)} won the {event} {on}, for {first['Constructor']['name']}.",
                        url, title, "jolpica")


async def _round_at(place: str, year: str, now: datetime, get) -> str | None:
    """The round run at a place ("Mexico", "Interlagos") this year, if it has happened."""
    races = (await get(f"{F1}/{year}.json"))["MRData"]["RaceTable"]["Races"]
    wanted = place.casefold()
    for race in races:
        where = race["Circuit"]["Location"]
        names = (race["raceName"], race["Circuit"]["circuitName"], where["locality"], where["country"])
        if any(wanted in n.casefold() for n in names) and date.fromisoformat(race["date"]) <= now.date():
            return race["round"]
    return None


def _driver(rows: list[dict], who: list[str]) -> dict | None:
    """The row for the driver (or team) named: full name or surname, as the router writes it."""
    for name in who:
        wanted = name.casefold()
        for row in rows:
            driver = row["Driver"]
            full = f"{driver['givenName']} {driver['familyName']}".casefold()
            if wanted in (full, driver["familyName"].casefold()):
                return row
        for row in rows:  # a team: its best-placed driver
            team = row.get("Constructor") or (row.get("Constructors") or [{}])[0]
            if team.get("name", "").casefold() == wanted:
                return row
    return None


def _name(row: dict) -> str:
    return f"{row['Driver']['givenName']} {row['Driver']['familyName']}"


# ── cricket ─────────────────────────────────────────────────────────────────────
_NEXT = ("next upcoming", "today or next", "tomorrow")
_CRICAPI_STATES = {"fixture": "pre", "live": "in", "result": "post"}
_CACHE: dict[str, tuple[float, dict]] = {}


async def _cricket(q: dict, now: datetime, get, key: str) -> FeedAnswer | None:
    """CricketData's cricScore first (a week or more of results and fixtures, 100 requests a day),
    then ESPN's scoreboard (a few days, no key) when there is no key, the day's quota is used up,
    or the match is not in it."""
    if not q.get("entities") or q.get("focus") in ("broadcast", "standings"):
        return None
    if key:
        try:
            data = await _cached(f"{CRICAPI}?apikey={key}", get)
            if data.get("status") == "success":
                found = _pick(q, now, _cricapi_matches(data), "cricketdata")
                if found:
                    return found
            else:
                _log.info("CricketData not used: %s", str(data.get("reason", "no reason given"))[:120])
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            _log.info("CricketData not used: %s", type(exc).__name__)
    return _pick(q, now, _espn_matches(await get(ESPN)), "espn")


async def _cached(url: str, get, seconds: float = 60, headers: dict | None = None) -> dict:
    """One request a minute at most: a follow-up question costs none of a free plan's quota."""
    hit = _CACHE.get(url)
    if hit and time.monotonic() - hit[0] < seconds:
        return hit[1]
    data = await (get(url, headers) if headers else get(url))
    _CACHE[url] = (time.monotonic(), data)
    return data


def _cricapi_matches(data: dict) -> list[dict]:
    return [{"state": _CRICAPI_STATES.get(row.get("ms")), "date": row.get("dateTimeGMT", ""),
             "teams": [(_team(row.get("t1", "")), row.get("t1s", "")), (_team(row.get("t2", "")), row.get("t2s", ""))],
             "summary": row.get("status", ""), "where": "", "link": "https://cricketdata.org/"}
            for row in data.get("data") or []]


def _espn_matches(data: dict) -> list[dict]:
    events = [e for league in data["sports"][0].get("leagues", []) for e in league.get("events", [])]
    return [{"state": e["status"], "date": e["date"],
             "teams": [(c.get("displayName", ""), c.get("score", "")) for c in e.get("competitors", [])],
             "summary": (e.get("fullStatus") or {}).get("longSummary", ""), "where": e.get("location", ""),
             "link": e.get("link") or "https://www.espncricinfo.com/"} for e in events]


def _team(name: str) -> str:
    """"India [IND]" -> "India"."""
    return re.sub(r"\s*\[[^\]]*\]\s*$", "", name).strip()


def _pick(q: dict, now: datetime, matches: list[dict], provider: str) -> FeedAnswer | None:
    """The live match between the teams named, else the next one (when asked about what is coming)
    or the latest result; in fixed words."""
    teams = q["entities"]
    # Exact team names only: "India" is not "India Under-19s", "India U19" or "India Women".
    mine = [m for m in matches if all(any(name.casefold() == t.casefold() for name, _ in m["teams"]) for t in teams)]
    by_state = {s: sorted((m for m in mine if m["state"] == s), key=lambda m: m["date"]) for s in ("pre", "in", "post")}
    upcoming = q.get("time") in _NEXT or q.get("focus") in ("start time", "schedule", "teams", "venue")
    if by_state["in"]:
        match = by_state["in"][-1]
    elif upcoming and by_state["pre"]:
        match = by_state["pre"][0]
    elif by_state["post"] and not upcoming:
        match = by_state["post"][-1]
    else:
        return None
    names = " v ".join(name for name, _ in match["teams"])
    scores = ", ".join(f"{name} {score}" for name, score in match["teams"] if score)
    summary = match["summary"].rstrip(".")
    if match["state"] == "pre":
        text = f"{names} starts {_local(match['date'][:10], match['date'][11:], now)}"
        text += (f" at {match['where']}" if match["where"] else "") + "."
    elif match["state"] == "in":
        text = f"{names} is under way: {scores}." + (f" {summary}." if summary else "")
    else:
        moment = datetime.fromisoformat(f"{match['date'][:10]}T{match['date'][11:].rstrip('Z') or '00:00'}+00:00")
        on = _day(moment.astimezone(now.tzinfo).date().isoformat()).rsplit(" ", 1)[0]  # the date in IST
        text = (f"{summary} on {on}." if summary else f"{names} finished on {on}.") + (f" {scores}." if scores else "")
    source = {"cricketdata": "CricketData", "espn": "ESPN", "football-data": "football-data.org"}[provider]
    return FeedAnswer(text, match["link"], f"{source}: {names}", provider)


# ── football and NBA ────────────────────────────────────────────────────────────
_ESPN_FOOTBALL = ("eng.1", "esp.1", "ita.1", "ger.1", "fra.1", "uefa.champions", "ind.1")
_FD_STATES = {"SCHEDULED": "pre", "TIMED": "pre", "IN_PLAY": "in", "PAUSED": "in", "LIVE": "in",
              "FINISHED": "post", "AWARDED": "post"}


async def _football(q: dict, now: datetime, get, key: str) -> FeedAnswer | None:
    """football-data.org first (ten days of matches in the competitions the key covers, 10 requests a
    minute free), then ESPN's league scoreboards (this round only, no key)."""
    if not q.get("entities") or q.get("focus") in ("broadcast", "standings"):
        return None
    if key:
        # A request covers at most ten days: the last ten and the next ten, so an international
        # break still leaves the last and next match in view.
        today = now.date()
        urls = [f"{FOOTBALL}?dateFrom={today - timedelta(days=10)}&dateTo={today}",
                f"{FOOTBALL}?dateFrom={today}&dateTo={today + timedelta(days=10)}"]
        try:
            windows = await asyncio.gather(*(_cached(url, get, headers={"X-Auth-Token": key}) for url in urls))
            unique = {m.get("id") or (m["utcDate"], m["homeTeam"]["name"]): m for w in windows for m in w.get("matches") or []}
            found = _pick(q, now, _fd_matches({"matches": list(unique.values())}), "football-data")
            if found:
                return found
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            _log.info("football-data.org not used: %s", type(exc).__name__)
    boards = await asyncio.gather(*(get(f"{ESPN_SPORTS}/soccer/{league}/scoreboard") for league in _ESPN_FOOTBALL),
                                  return_exceptions=True)
    return _pick(q, now, [m for b in boards if isinstance(b, dict) for m in _espn_events(b)], "espn")


async def _basketball(q: dict, now: datetime, get) -> FeedAnswer | None:
    """NBA from ESPN: yesterday's, today's and tomorrow's games (no key)."""
    if not q.get("entities") or q.get("focus") in ("broadcast", "standings"):
        return None
    days = [now.date() + timedelta(days=d) for d in (-1, 0, 1)]
    boards = await asyncio.gather(*(get(f"{ESPN_SPORTS}/basketball/nba/scoreboard?dates={d:%Y%m%d}") for d in days),
                                  return_exceptions=True)
    return _pick(q, now, [m for b in boards if isinstance(b, dict) for m in _espn_events(b)], "espn")


def _club(name: str) -> str:
    """"Arsenal FC" -> "Arsenal", "Hull City AFC" -> "Hull City"."""
    return re.sub(r"\s+(?:A?FC|CF)$", "", name.strip())


def _outcome(home: str, away: str, winner: str | None) -> str:
    if winner is None:
        return f"{home} and {away} drew"
    return f"{winner} beat {away if winner == home else home}"


def _fd_matches(data: dict) -> list[dict]:
    matches = []
    for m in data.get("matches") or []:
        state = _FD_STATES.get(m.get("status"))
        if not state:
            continue  # postponed, cancelled
        home, away = _club(m["homeTeam"]["name"]), _club(m["awayTeam"]["name"])
        score = m["score"]["fullTime"]
        goals = ("" if score["home"] is None else str(score["home"]), "" if score["away"] is None else str(score["away"]))
        winner = {"HOME_TEAM": home, "AWAY_TEAM": away}.get(m["score"].get("winner"))
        summary = _outcome(home, away, winner) if state == "post" else ""
        matches.append({"state": state, "date": m["utcDate"], "teams": [(home, goals[0]), (away, goals[1])],
                        "summary": summary, "where": m.get("venue") or "", "link": "https://www.football-data.org/"})
    return matches


def _espn_events(board: dict) -> list[dict]:
    matches = []
    for event in board.get("events") or []:
        game, state = event["competitions"][0], event["status"]["type"]["state"]
        sides = sorted(game["competitors"], key=lambda side: side.get("homeAway") != "home")
        teams = [(side["team"]["displayName"], side.get("score", "") if state != "pre" else "") for side in sides]
        winner = next((side["team"]["displayName"] for side in sides if side.get("winner")), None)
        summary = (_outcome(teams[0][0], teams[1][0], winner) if state == "post"
                   else event["status"]["type"].get("detail", "") if state == "in" else "")
        matches.append({"state": state, "date": event["date"], "teams": teams, "summary": summary,
                        "where": (game.get("venue") or {}).get("fullName", ""), "link": "https://www.espn.com/"})
    return matches


# ── weather ─────────────────────────────────────────────────────────────────────
# WMO weather codes, as Open-Meteo reports them.
_WMO = {0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "freezing fog",
        51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
        61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain",
        71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains", 80: "light showers", 81: "showers",
        82: "heavy showers", 85: "snow showers", 86: "heavy snow showers", 95: "thunderstorms",
        96: "thunderstorms with hail", 99: "thunderstorms with hail"}
# Asked about something the forecast does not carry: the web lookup answers instead.
_NOT_WEATHER = re.compile(r"\b(?:uv|pollen|sunrise|sunset)\b", re.I)
_AIR = re.compile(r"\b(?:aqi|air quality|air pollution|pollution|polluted|smog|pm ?2\.?5|pm ?10)\b", re.I)
# The US EPA scale Open-Meteo reports (India's own NAQI bands differ).
_AQI_BANDS = ((50, "good"), (100, "moderate"), (150, "unhealthy for sensitive groups"), (200, "unhealthy"),
              (300, "very unhealthy"), (10_000, "hazardous"))


async def _weather(q: dict, get) -> FeedAnswer | None:
    """Now, today, tomorrow, yesterday or this weekend at a named place, in °C, from Open-Meteo;
    air quality now at a named place, from its air-quality service."""
    place, focus, time_ = q.get("place"), q.get("focus"), q.get("time")
    if not place or _NOT_WEATHER.search(q.get("said", "")):
        return None
    air = bool(_AIR.search(q.get("said", "")))
    days = {"yesterday": [-1], "tomorrow": [1], "this weekend": "weekend"}.get(
        time_, [0] if time_ in (None, "now", "today", "upcoming") else None)
    if days is None or (air and days != [0]):
        return None  # last week, this week, tomorrow's air and the like
    found = (await get(f"{GEOCODE}?{urlencode({'name': place, 'count': 1, 'language': 'en'})}")).get("results")
    if not found:
        return None
    spot = found[0]
    if air:
        return await _air(spot, get)
    forecast = await get(f"{FORECAST}?" + urlencode({
        "latitude": spot["latitude"], "longitude": spot["longitude"], "timezone": "auto", "past_days": 1,
        "forecast_days": 7,
        "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,precipitation_sum"}))
    daily, name = forecast["daily"], spot["name"]
    today = daily["time"].index(forecast["current"]["time"][:10])  # the place's own today
    if days == "weekend":
        weekdays = [date.fromisoformat(d).weekday() for d in daily["time"]]
        days = [i - today for i in range(today, len(daily["time"])) if weekdays[i] >= 5][:2]
    parts = []
    if days == [0] and focus != "forecast":
        now = forecast["current"]
        parts.append(f"{name} now: {_WMO.get(now['weather_code'], 'unsettled')}, {round(now['temperature_2m'])} °C "
                     f"(feels like {round(now['apparent_temperature'])} °C), humidity {now['relative_humidity_2m']}%, "
                     f"wind {round(now['wind_speed_10m'])} km/h.")
    for offset in days:
        i = today + offset
        label = {-1: "yesterday", 0: "today", 1: "tomorrow"}.get(offset, date.fromisoformat(daily["time"][i]).strftime("%A"))
        line = (f"{name} {label}: " if not parts else f"{label.capitalize()}: ")
        line += (f"{_WMO.get(daily['weather_code'][i], 'unsettled')}, {round(daily['temperature_2m_min'][i])} to "
                 f"{round(daily['temperature_2m_max'][i])} °C")
        if offset < 0 and daily["precipitation_sum"][i]:
            line += f", {daily['precipitation_sum'][i]} mm of rain"
        elif offset >= 0 and daily["precipitation_probability_max"][i]:
            line += f", {daily['precipitation_probability_max'][i]}% chance of rain"
        parts.append(line + ".")
    return FeedAnswer(" ".join(parts), "https://open-meteo.com/", f"Open-Meteo: {name}", "open-meteo")


async def _air(spot: dict, get) -> FeedAnswer | None:
    now = (await get(f"{AIR}?" + urlencode({"latitude": spot["latitude"], "longitude": spot["longitude"],
                                            "timezone": "auto", "current": "us_aqi,pm2_5,pm10"})))["current"]
    if now.get("us_aqi") is None:
        return None
    aqi = round(now["us_aqi"])
    band = next(name for top, name in _AQI_BANDS if aqi <= top)
    return FeedAnswer(f"{spot['name']} air now: AQI {aqi} (US scale), {band}. PM2.5 {round(now['pm2_5'])} µg/m³, "
                      f"PM10 {round(now['pm10'])} µg/m³.", "https://open-meteo.com/", f"Open-Meteo air: {spot['name']}",
                      "open-meteo")


# ── markets ─────────────────────────────────────────────────────────────────────
_COINS = {"bitcoin": "bitcoin", "btc": "bitcoin", "ethereum": "ethereum", "ether": "ethereum", "eth": "ethereum",
          "dogecoin": "dogecoin", "doge": "dogecoin", "solana": "solana", "xrp": "ripple", "ripple": "ripple",
          "cardano": "cardano"}
_INDICES = {"bank nifty": ("^NSEBANK", "Nifty Bank"), "nifty": ("^NSEI", "Nifty 50"), "sensex": ("^BSESN", "Sensex"),
            "nasdaq": ("^IXIC", "Nasdaq"), "dow": ("^DJI", "Dow Jones"), "s&p": ("^GSPC", "S&P 500")}
_CURRENCIES = (("us dollar", "USD"), ("dollar", "USD"), ("usd", "USD"), ("euro", "EUR"), ("eur", "EUR"),
               ("pound", "GBP"), ("sterling", "GBP"), ("gbp", "GBP"), ("yen", "JPY"), ("yuan", "CNY"),
               ("swiss franc", "CHF"), ("rupee", "INR"), ("inr", "INR"))
_CURRENCY_NAMES = {"USD": "US dollar", "EUR": "euro", "GBP": "British pound", "JPY": "Japanese yen",
                   "CNY": "Chinese yuan", "CHF": "Swiss franc", "INR": "rupee"}


async def _markets(q: dict, now: datetime, get) -> FeedAnswer | None:
    """Crypto from CoinGecko, Indian and US indices from Yahoo (unofficial), exchange rates from
    Frankfurter (European Central Bank). Gold, shares and past prices go to the web lookup."""
    if q.get("time") not in (None, "now", "today"):
        return None
    words = " ".join([q.get("said", ""), *(q.get("entities") or [])]).casefold()
    coin = next((cid for name, cid in _COINS.items() if re.search(rf"\b{name}\b", words)), None)
    if coin:
        price = (await get(f"{CRYPTO}?ids={coin}&vs_currencies=inr,usd&include_24hr_change=true")).get(coin)
        if not price:
            return None
        change = price.get("inr_24h_change")
        text = (f"{coin.capitalize()} is {_inr(price['inr'])} (${price['usd']:,.0f})"
                + (f", {'up' if change >= 0 else 'down'} {abs(change):.1f}% in 24 hours." if change is not None else "."))
        return FeedAnswer(text, "https://www.coingecko.com/", f"CoinGecko: {coin}", "coingecko")
    index = next((v for name, v in _INDICES.items() if re.search(rf"\b{re.escape(name)}\b", words)), None)
    if index:
        meta = (await get(f"{INDEX}/{quote(index[0])}?range=1d&interval=1d"))["chart"]["result"][0]["meta"]
        price, before = meta["regularMarketPrice"], meta.get("chartPreviousClose") or meta.get("previousClose")
        at = datetime.fromtimestamp(meta["regularMarketTime"], now.tzinfo)
        text = f"{index[1]} is at {price:,.2f}"
        if before:
            text += f", {'up' if price >= before else 'down'} {abs(price - before) / before * 100:.2f}% on the day"
        text += f" (as of {at.strftime('%A')} {at.day} {at.strftime('%B')}, {at.strftime('%I:%M %p').lstrip('0').lower()})."
        return FeedAnswer(text, "https://finance.yahoo.com/", f"Yahoo Finance: {index[1]}", "yahoo")
    if "gold" in words or "silver" in words:
        return None
    # Currencies in the order they were said; the one after "a"/"one"/"1" is being priced ("how many
    # dollars is a pound?"), else the first one that is not the rupee.
    found = sorted((m.start(), code, words[max(0, m.start() - 4):m.start()])
                   for name, code in _CURRENCIES for m in re.finditer(rf"\b{name}s?\b", words))
    named = list(dict.fromkeys(code for _, code, _ in found))
    priced = [code for _, code, before in found if re.search(r"\b(?:a|an|one|1)\s$", before)]
    base = next((c for c in priced if c != "INR"), None) or next((c for c in named if c != "INR"), None)
    if not base:
        return None
    target = next((c for c in named if c not in ("INR", base)), "INR")
    data = await get(f"{RATES}?from={base}&to={target}")
    rate = (data.get("rates") or {}).get(target)
    if rate is None:
        return None
    amount = _inr(rate, decimals=2) if target == "INR" else f"{rate:,.4f} {_CURRENCY_NAMES[target]}s"
    on = _day(data["date"]).rsplit(" ", 1)[0]
    return FeedAnswer(f"1 {_CURRENCY_NAMES[base]} is {amount} (European Central Bank rate, {on}).",
                      "https://www.frankfurter.app/", f"Frankfurter: {base}/{target}", "frankfurter")


# ── news ────────────────────────────────────────────────────────────────────────
_TOPIC = re.compile(r"\b(?:news|headlines?|latest)\s+(?:about|on|regarding|from|in)\s+(?:the\s+)?(.+?)[?.!]*$", re.I)


async def _news(q: dict, get) -> FeedAnswer | None:
    """Top three headlines from Google News (India), or on a topic they name ("news about the
    elections"). Read out as headlines: nothing is summarised or inferred."""
    if q.get("time") not in (None, "now", "today"):
        return None
    topic = _TOPIC.search(q.get("said", "").strip())
    where = "hl=en-IN&gl=IN&ceid=IN:en"
    url = f"{NEWS}/search?q={quote(topic.group(1))}&{where}" if topic else f"{NEWS}?{where}"
    feed = await get(url)
    titles = [item.findtext("title", "").strip() for item in ElementTree.fromstring(feed).iter("item")][:3]
    titles = [t for t in titles if t]
    if not titles:
        return None
    # "Headline - Source": the source in brackets reads better aloud.
    spoken = [re.sub(r"^(.*) - ([^-]+)$", r"\1 (\2)", t) for t in titles]
    lead = f"Headlines on {topic.group(1)}" if topic else "Top headlines"
    return FeedAnswer(f"{lead}: " + "; ".join(spoken) + ".", "https://news.google.com/", f"Google News: {lead}",
                      "google-news")


def _inr(amount: float, decimals: int = 0) -> str:
    """₹ with Indian digit grouping: ₹61,09,514."""
    whole, _, cents = f"{amount:.{decimals}f}".partition(".")
    head, tail = whole[:-3], whole[-3:]
    grouped = ",".join(filter(None, [*re.findall(r"\d{1,2}(?=(?:\d{2})*$)", head), tail]))
    return f"₹{grouped}" + (f".{cents}" if cents else "")


# ── wording ─────────────────────────────────────────────────────────────────────
def _ordinal(n) -> str:
    n = int(n)
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _day(iso: str) -> str:
    day = date.fromisoformat(iso)
    return f"{day.day} {day.strftime('%B %Y')}"


def _local(day: str, time: str | None, now: datetime) -> str:
    """A UTC date and time as the listener's local time ("Saturday 21 November, 9:30 am")."""
    if not time:
        return _day(day)
    moment = datetime.fromisoformat(f"{day}T{time.rstrip('Z')}+00:00").astimezone(now.tzinfo)
    return f"{moment.strftime('%A')} {moment.day} {moment.strftime('%B')}, {moment.strftime('%I:%M %p').lstrip('0').lower()}"
