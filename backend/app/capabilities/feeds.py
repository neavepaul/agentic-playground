"""Live facts from structured feeds: Formula 1 from Jolpica (the Ergast successor, free, no key)
and cricket from CricketData.org (with a key) or ESPN's scoreboard (unofficial, no key), weather from
Open-Meteo (free, no key, °C). One request, exact facts, an answer in
fixed words: no model reads pages or writes the answer. Anything outside what a feed covers gives
None, and the usual web lookup answers instead."""
import logging
import re
import time
from dataclasses import dataclass
from urllib.parse import urlencode
from datetime import date, datetime

import httpx

_log = logging.getLogger("agentic_friend.capabilities")
F1 = "https://api.jolpi.ca/ergast/f1"
ESPN = "https://site.web.api.espn.com/apis/v2/scoreboard/header?sport=cricket"
CRICAPI = "https://api.cricapi.com/v1/cricScore"
GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST = "https://api.open-meteo.com/v1/forecast"
FEEDS = {"f1": "jolpica", "cricket": "cricketdata, then espn", "weather": "open-meteo"}


@dataclass
class FeedAnswer:
    answer: str
    url: str
    title: str
    provider: str


async def fetch(url: str) -> dict:
    async with httpx.AsyncClient(timeout=10, headers={"User-Agent": "kip-companion"}) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()


async def answer(question: dict, now: datetime, get=None, cricket_key: str = "") -> FeedAnswer | None:
    """question: the router's reading (domain, entities, focus, time, place) and their words (said)."""
    get = get or fetch
    if question.get("domain") == "f1":
        return await _f1(question, now, get)
    if question.get("domain") == "cricket":
        return await _cricket(question, now, get, cricket_key)
    if question.get("domain") == "weather":
        return await _weather(question, get)
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


async def _cached(url: str, get, seconds: float = 60) -> dict:
    """One request a minute at most: a follow-up question costs none of the day's 100."""
    hit = _CACHE.get(url)
    if hit and time.monotonic() - hit[0] < seconds:
        return hit[1]
    data = await get(url)
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
        on = _day(match["date"][:10]).rsplit(" ", 1)[0]
        text = (f"{summary} on {on}." if summary else f"{names} finished on {on}.") + (f" {scores}." if scores else "")
    source = {"cricketdata": "CricketData", "espn": "ESPN"}[provider]
    return FeedAnswer(text, match["link"], f"{source}: {names}", provider)


# ── weather ─────────────────────────────────────────────────────────────────────
# WMO weather codes, as Open-Meteo reports them.
_WMO = {0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "freezing fog",
        51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
        61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain",
        71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains", 80: "light showers", 81: "showers",
        82: "heavy showers", 85: "snow showers", 86: "heavy snow showers", 95: "thunderstorms",
        96: "thunderstorms with hail", 99: "thunderstorms with hail"}
# Asked about something the forecast does not carry: the web lookup answers instead.
_NOT_WEATHER = re.compile(r"\b(?:aqi|air quality|pollution|smog|uv|pollen|sunrise|sunset)\b", re.I)


async def _weather(q: dict, get) -> FeedAnswer | None:
    """Now, today, tomorrow, yesterday or this weekend at a named place, in °C, from Open-Meteo."""
    place, focus, time_ = q.get("place"), q.get("focus"), q.get("time")
    if not place or _NOT_WEATHER.search(q.get("said", "")):
        return None
    days = {"yesterday": [-1], "tomorrow": [1], "this weekend": "weekend"}.get(
        time_, [0] if time_ in (None, "now", "today", "upcoming") else None)
    if days is None:
        return None  # last week, this week and the like
    found = (await get(f"{GEOCODE}?{urlencode({'name': place, 'count': 1, 'language': 'en'})}")).get("results")
    if not found:
        return None
    spot = found[0]
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
