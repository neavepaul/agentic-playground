"""Live facts from structured feeds: Formula 1 from Jolpica (the Ergast successor, free, no key)
and cricket from CricketData.org (with a key) or ESPN's scoreboard (unofficial, no key), weather from
Open-Meteo (free, no key, °C). One request, exact facts, an answer in
fixed words: no model reads pages or writes the answer. Anything outside what a feed covers gives
None, and the usual web lookup answers instead."""
import asyncio
import base64
import hashlib
import hmac
import logging
import re
import struct
import time
from dataclasses import dataclass
from urllib.parse import quote, urlencode
from xml.etree import ElementTree  # feeds with a DOCTYPE are refused before parsing  # nosec B405
from datetime import date, datetime, timedelta

import httpx

from app.capabilities.readers import Reader, Reading

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
SPARK = "https://query1.finance.yahoo.com/v7/finance/spark"
SEARCH = "https://query1.finance.yahoo.com/v1/finance/search"
GROWW = "https://api.groww.in/v1"
NEWS = "https://news.google.com/rss"
FOOTBALL = "https://api.football-data.org/v4/matches"
ESPN_SPORTS = "https://site.api.espn.com/apis/site/v2/sports"
FEEDS = {"f1": "jolpica", "cricket": "cricketdata, then espn", "weather": "open-meteo",
         "finance": "groww, coingecko, yahoo, frankfurter", "news": "google-news",
         "football": "football-data, then espn", "basketball": "espn"}


# What a feed that is down, changed shape or sent something else (an HTML bot check, JSON null, a cut-off
# RSS body) can raise. Any of them means: not answered from the feed; the web lookup takes over.
FEED_ERRORS = (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError, AttributeError, ArithmeticError,
               ElementTree.ParseError, TimeoutError)
# A whole feed request, however slowly the server sends it, and the most it may send. The largest
# real response (a football-data.org fortnight) is well under a megabyte.
FEED_SECONDS = 15
FEED_BYTES = 5_000_000


@dataclass
class FeedAnswer:
    answer: str
    url: str
    title: str
    provider: str


def text_encoding(name: str | None) -> str:
    """The charset a response names, if it is one for text; else utf-8 ("base64" names a codec, not text)."""
    import codecs
    try:
        return name if name and codecs.lookup(name)._is_text_encoding else "utf-8"
    except LookupError:
        return "utf-8"


async def fetch(url: str, headers: dict | None = None) -> dict | str:
    """JSON as a dict; anything else (RSS) as text. A request carrying a key in its headers
    (football-data.org, Groww) never follows a redirect, which would hand the key to wherever it
    points; the whole request is bounded in time and size."""
    import json
    async with asyncio.timeout(FEED_SECONDS):
        async with httpx.AsyncClient(timeout=10, headers={"User-Agent": "kip-companion"},
                                     follow_redirects=not headers) as client:
            async with client.stream("GET", url, headers=headers) as response:
                response.raise_for_status()  # an unfollowed redirect is an error too
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > FEED_BYTES:
                        raise ValueError("Feed response too large.")
                text = bytes(body).decode(text_encoding(response.encoding), errors="replace")
    try:
        return json.loads(text)
    except ValueError:
        return text


async def answer(question: dict, now: datetime, get=None, cricket_key: str = "",
                 football_key: str = "", groww: tuple[str, str] = ("", "")) -> FeedAnswer | None:
    """question: the router's reading (domain, entities, focus, time, place) and their words (said)."""
    get = get or fetch
    if question.get("domain") == "f1":
        return await _f1(question, now, get)
    if question.get("domain") == "cricket":
        return await _cricket(question, now, get, cricket_key)
    if question.get("domain") == "weather":
        return await _weather(question, get)
    if question.get("domain") == "finance":
        return await _markets(question, now, get, groww)
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


_F1_FOCUS = {"standings": "standings", "race result": "finishing position", "qualifying": "qualifying position",
             "sprint": "sprint result", "starting grid": "starting position", "next race": "schedule"}


def _f1_ask(q: dict) -> tuple[str | None, str] | None:
    """(focus, table) for an F1 question: from their words by the F1 reader, or the router's focus
    when there are no words to read. None when the reader is unsure or it is not a feed question
    ("what does DRS mean?"): the web lookup answers."""
    if not q.get("said") or q.get("followup"):
        # "What about Max?" says nothing of what is asked: the router carried that over from the
        # question it follows (a team named then is found in the constructors' table).
        return q.get("focus"), "drivers"
    reading = reader("f1").read(q["said"])
    if not reading.confident or reading.form["ask"] not in _F1_FOCUS:
        return None
    return _F1_FOCUS[reading.form["ask"]], reading.form["table"]


async def _f1(q: dict, now: datetime, get) -> FeedAnswer | None:
    asked = _f1_ask(q)
    if asked is None:
        return None
    (focus, table), time = asked, q.get("time")
    who = q.get("entities") or []
    if focus in ("schedule", "start time") or (time in ("next upcoming", "today or next") and not q.get("said")):
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
        # A driver, or the drivers' leader, from the drivers' table. A team ("how many points does
        # McLaren have?") or the constructors' championship from the constructors' table: never
        # the team's best-placed driver.
        # A season they name ("who won the 2025 championship?") is that season's final table.
        season = time if re.fullmatch(r"\d{4}", str(time)) and int(time) < now.year else "current"
        if table == "drivers":
            url = f"{F1}/{season}/driverStandings.json"
            table = (await get(url))["MRData"]["StandingsTable"]["StandingsLists"]
            if not table:
                return None
            row = _driver(table[0]["DriverStandings"], who, teams=False) if who else table[0]["DriverStandings"][0]
            if row is not None:
                return FeedAnswer(f"{_name(row)} {_placed(row, season, table[0]['season'])} championship with "
                                  f"{row['points']} points.", url, "Jolpica F1: driver standings", "jolpica")
            if not who:
                return None
        url = f"{F1}/{season}/constructorStandings.json"
        table = (await get(url))["MRData"]["StandingsTable"]["StandingsLists"]
        if not table:
            return None
        rows = table[0]["ConstructorStandings"]
        row = _constructor(rows, who) if who else rows[0]
        if row is None:
            return None
        placed = _placed(row, season, table[0]["season"]).replace("is ", "are ", 1)
        return FeedAnswer(f"{row['Constructor']['name']} {placed} constructors' championship with {row['points']} "
                          f"points.", url, "Jolpica F1: constructor standings", "jolpica")
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


def _driver(rows: list[dict], who: list[str], teams: bool = True) -> dict | None:
    """The row for the driver (or, with teams, the team) named: full name or surname, as the router writes it."""
    for name in who:
        wanted = name.casefold()
        for row in rows:
            driver = row["Driver"]
            full = f"{driver['givenName']} {driver['familyName']}".casefold()
            if wanted in (full, driver["familyName"].casefold()):
                return row
        if not teams:
            continue
        for row in rows:  # a team: its best-placed driver
            team = row.get("Constructor") or (row.get("Constructors") or [{}])[0]
            if team.get("name", "").casefold() == wanted:
                return row
    return None


def _placed(row: dict, season: str, year: str) -> str:
    """"is 2nd in the 2026", or for a season that is over "won the 2025" / "finished 2nd in the 2025"."""
    if season == "current":
        return f"is {_ordinal(row['position'])} in the {year}"
    return f"won the {year}" if row["position"] == "1" else f"finished {_ordinal(row['position'])} in the {year}"


def _constructor(rows: list[dict], who: list[str]) -> dict | None:
    """The constructors' table row for the team named: "McLaren", "Red Bull" or "Red Bull Racing"."""
    for name in who:
        wanted = _bare(name)
        for row in rows:
            team = _bare(row["Constructor"]["name"])
            if wanted and (wanted in team or team in wanted):
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
    if not _match_question(q):
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
        except FEED_ERRORS as exc:
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
    elif by_state["pre"] and q.get("time") in (None, "now", "today", "today or latest"):
        match = by_state["pre"][0]  # "lakers score" before tip-off, with no recent result in view: when it starts
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


# Leagues named in a question with no team: (ESPN code, football-data.org code or None, name).
_LEAGUES = [
    (re.compile(r"\b(?:premier league|epl)\b"), ("eng.1", "PL", "Premier League")),
    (re.compile(r"\b(?:la ?liga)\b"), ("esp.1", "PD", "La Liga")),
    (re.compile(r"\bserie a\b"), ("ita.1", "SA", "Serie A")),
    (re.compile(r"\bbundesliga\b"), ("ger.1", "BL1", "Bundesliga")),
    (re.compile(r"\bligue 1\b"), ("fra.1", "FL1", "Ligue 1")),
    (re.compile(r"\b(?:champions league|ucl)\b"), ("uefa.champions", "CL", "Champions League")),
    (re.compile(r"\b(?:isl|indian super league)\b"), ("ind.1", None, "Indian Super League")),
]
_ROUNDUP_GAMES = 8  # games read out before "and N more"


def _league(q: dict) -> tuple[str, str | None, str] | None:
    """The football league a question names ("premier league results"), if any."""
    said = q.get("said", "").casefold()
    return next((league for pattern, league in _LEAGUES if pattern.search(said)), None)


_UPCOMING = re.compile(r"\b(?:tonight|later today|upcoming|fixtures?|schedule|next)\b")


def _roundup(name: str, matches: list[dict], q: dict, now: datetime, provider: str,
             round_days: int = 0) -> FeedAnswer | None:
    """Every game of a league on the day asked about, by its date in the listener's zone: today,
    yesterday, the latest results or the next games. round_days > 0 (football, whose rounds span a
    weekend) widens the latest results and the next games to the whole round."""
    def local_day(match):
        moment = datetime.fromisoformat(match["date"].replace("Z", "+00:00"))
        return moment.astimezone(now.tzinfo).date()
    by_day: dict[date, list[dict]] = {}
    for match in sorted(matches, key=lambda m: m["date"]):
        by_day.setdefault(local_day(match), []).append(match)
    today, time_ = now.date(), q.get("time")
    span, wanted = 0, None  # wanted: only games still to come ("pre") or only finished ones ("post")
    if time_ == "yesterday":
        day = today - timedelta(days=1)
    elif time_ in _NEXT or _UPCOMING.search(q.get("said", "").casefold()):
        day = next((d for d in sorted(by_day) if any(m["state"] == "pre" for m in by_day[d]) and d >= today), None)
        span, wanted = round_days, "pre"
    elif time_ in ("latest", "latest completed") or q.get("focus") == "match result" and time_ in (None, "today or latest"):
        day = max((d for d, games in by_day.items() if any(m["state"] == "post" for m in games) and d <= today),
                  default=None)
        span, wanted = -round_days, "post"
    else:  # today, now, today or latest: today's games, else the latest results
        day = today if today in by_day else max((d for d in by_day if d < today), default=None)
    if day is None:
        return None
    first, last = (day + timedelta(days=span), day) if span < 0 else (day, day + timedelta(days=span))
    games = [m for d in sorted(by_day) if first <= d <= last for m in by_day[d] if wanted in (None, m["state"])]
    if not games:
        return None
    days = sorted({local_day(m) for m in games})
    first, last = days[0], days[-1]
    when = (f"{first.strftime('%A')} {first.day} {first.strftime('%B')}" if first == last
            else f"{first.day} to {last.day} {last.strftime('%B')}" if first.month == last.month
            else f"{first.day} {first.strftime('%B')} to {last.day} {last.strftime('%B')}")
    lines = []
    for match in games[:_ROUNDUP_GAMES]:
        (home, home_score), (away, away_score) = match["teams"]
        if match["state"] == "pre":
            clock = datetime.fromisoformat(match["date"].replace("Z", "+00:00")).astimezone(now.tzinfo)
            weekday = f"{clock.strftime('%a')} " if first != last else ""  # a round over several days
            lines.append(f"{home} v {away} at {weekday}{clock.strftime('%I:%M %p').lstrip('0').lower()}")
        else:
            lines.append(f"{home} {home_score}, {away} {away_score}" + (" (live)" if match["state"] == "in" else ""))
    more = len(games) - _ROUNDUP_GAMES
    text = f"{name}, {when}: " + "; ".join(lines) + (f"; and {more} more" if more > 0 else "") + "."
    source = {"espn": "ESPN", "football-data": "football-data.org"}[provider]
    return FeedAnswer(text, games[0]["link"], f"{source}: {name}", provider)


async def _football(q: dict, now: datetime, get, key: str) -> FeedAnswer | None:
    """football-data.org first (ten days of matches in the competitions the key covers, 10 requests a
    minute free), then ESPN's league scoreboards (this round only, no key). With no team named, a
    named league's games for the day ("premier league results")."""
    league = None if q.get("entities") else _league(q)
    if league:
        return await _football_roundup(q, now, get, key, league)
    if not _match_question(q):
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
        except FEED_ERRORS as exc:
            _log.info("football-data.org not used: %s", type(exc).__name__)
    boards = await asyncio.gather(*(get(f"{ESPN_SPORTS}/soccer/{league}/scoreboard") for league in _ESPN_FOOTBALL),
                                  return_exceptions=True)
    return _pick(q, now, [m for b in boards if isinstance(b, dict) for m in _espn_events(b)], "espn")


async def _football_roundup(q: dict, now: datetime, get, key: str, league) -> FeedAnswer | None:
    espn, code, name = league
    if q.get("focus") in ("broadcast", "standings"):
        return None
    if key and code:
        today = now.date()
        urls = [f"{FOOTBALL}?dateFrom={today - timedelta(days=10)}&dateTo={today}",
                f"{FOOTBALL}?dateFrom={today}&dateTo={today + timedelta(days=10)}"]
        try:
            windows = await asyncio.gather(*(_cached(url, get, headers={"X-Auth-Token": key}) for url in urls))
            unique = {m.get("id") or (m["utcDate"], m["homeTeam"]["name"]): m for w in windows for m in w.get("matches") or []}
            mine = [m for m in unique.values() if (m.get("competition") or {}).get("code") == code]
            found = _roundup(name, _fd_matches({"matches": mine}), q, now, "football-data", round_days=3)
            if found:
                return found
        except FEED_ERRORS as exc:
            _log.info("football-data.org not used: %s", type(exc).__name__)
    board = await get(f"{ESPN_SPORTS}/soccer/{espn}/scoreboard")
    return _roundup(name, _espn_events(board), q, now, "espn", round_days=3)


async def _basketball(q: dict, now: datetime, get) -> FeedAnswer | None:
    """NBA from ESPN: yesterday's, today's and tomorrow's games (no key). With no team named, the
    day's games across the league ("nba scores today")."""
    if q.get("focus") in ("broadcast", "standings"):
        return None
    days = [now.date() + timedelta(days=d) for d in (-2, -1, 0, 1)]
    boards = await asyncio.gather(*(get(f"{ESPN_SPORTS}/basketball/nba/scoreboard?dates={d:%Y%m%d}") for d in days),
                                  return_exceptions=True)
    matches = [m for b in boards if isinstance(b, dict) for m in _espn_events(b)]
    if not q.get("entities"):
        return _roundup("NBA", matches, q, now, "espn")
    return _pick(q, now, matches, "espn")


def _match_question(q: dict) -> bool:
    """Cricket, football and NBA feeds answer about a named side or player, not standings or broadcasters."""
    return bool(q.get("entities")) and q.get("focus") not in ("broadcast", "standings")


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


# Old names the geocoder does not know, or knows as somewhere else ("Bangalore" is a part of Karachi).
_RENAMED = {"bangalore": "Bengaluru", "bombay": "Mumbai", "madras": "Chennai", "calcutta": "Kolkata",
            "trivandrum": "Thiruvananthapuram", "mysore": "Mysuru", "mangalore": "Mangaluru", "cochin": "Kochi",
            "gurgaon": "Gurugram", "pondicherry": "Puducherry", "poona": "Pune", "baroda": "Vadodara",
            "calicut": "Kozhikode", "belgaum": "Belagavi", "allahabad": "Prayagraj", "benares": "Varanasi"}


def _weather_days(q: dict):
    """The days a weather question is about ([0], [-1], [1] or "weekend"), or None when Open-Meteo is
    not asked: no place, not about the weather, last week, this week, tomorrow's air and the like."""
    said, time_ = q.get("said", ""), q.get("time")
    if not q.get("place") or _NOT_WEATHER.search(said):
        return None
    days = {"yesterday": [-1], "tomorrow": [1], "this weekend": "weekend"}.get(
        time_, [0] if time_ in (None, "now", "today", "upcoming") else None)
    return None if _AIR.search(said) and days != [0] else days


async def _weather(q: dict, get) -> FeedAnswer | None:
    """Now, today, tomorrow, yesterday or this weekend at a named place, in °C, from Open-Meteo;
    air quality now at a named place, from its air-quality service."""
    place, focus = q.get("place"), q.get("focus")
    days = _weather_days(q)
    if days is None:
        return None
    air = bool(_AIR.search(q.get("said", "")))
    name = _RENAMED.get(place.casefold(), place)
    found = (await get(f"{GEOCODE}?{urlencode({'name': name, 'count': 5, 'language': 'en'})}")).get("results")
    if not found:
        return None
    spot = max(found, key=lambda r: r.get("population") or 0)  # Hyderabad in India, not Sindh
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


# Feeds that answer from the person's own accounts: their answers never leave the device.
PRIVATE_PROVIDERS = frozenset({"groww"})
_MARKET_SOURCES = {"portfolio": "Groww", "coin": "CoinGecko", "index": "Yahoo Finance", "share": "Yahoo Finance",
                   "currency": "Frankfurter"}


def source_for(q: dict, cricket_key: str = "", football_key: str = "", groww: tuple[str, str] = ("", "")) -> str | None:
    """The feed Kip is about to ask, by name, for "Let me check Jolpica for that"; None when no feed
    will try (the web lookup answers instead)."""
    domain = q.get("domain")
    if domain == "finance":
        kind = _market_kind(q)
        if kind is None or (kind[0] == "portfolio" and not all(groww)):
            return None
        return _MARKET_SOURCES[kind[0]]
    # The feed would not be asked: no "let me check" for the web lookup's answer.
    if domain == "f1" and _f1_ask(q) is None:
        return None
    league = domain == "football" and not q.get("entities") and _league(q)
    if domain == "cricket" and not _match_question(q):
        return None
    if domain == "football" and not (league and q.get("focus") not in ("broadcast", "standings")) \
            and not _match_question(q):
        return None
    if domain == "basketball" and q.get("focus") in ("broadcast", "standings"):
        return None  # a named team or the whole league's games are both answered
    if domain == "weather" and _weather_days(q) is None or domain == "news" and not _news_question(q):
        return None
    football = "football-data.org" if football_key and not (league and league[1] is None) else "ESPN"
    return {"f1": "Jolpica", "cricket": "CricketData" if cricket_key else "ESPN",
            "football": football, "basketball": "ESPN",
            "weather": "Open-Meteo", "news": "Google News"}.get(domain)


def is_portfolio(q: dict) -> bool:
    """A question about the owner's own holdings (Groww)."""
    return q.get("domain") == "finance" and (_market_kind(q) or ("",))[0] == "portfolio"


# Where Grands Prix are held, as people name them: the F1 reader's places.
_F1_PLACES = dict.fromkeys(
    "australia|australian|melbourne|albert park|china|chinese|shanghai|japan|japanese|suzuka|bahrain|sakhir|saudi|"
    "saudi arabia|jeddah|miami|imola|emilia romagna|monaco|monte carlo|spain|spanish|barcelona|catalunya|madrid|"
    "canada|canadian|montreal|austria|austrian|red bull ring|spielberg|britain|british|silverstone|belgium|belgian|"
    "spa|hungary|hungarian|budapest|hungaroring|netherlands|dutch|zandvoort|italy|italian|monza|azerbaijan|baku|"
    "singapore|marina bay|usa|united states|austin|cota|mexico|mexican|mexico city|brazil|brazilian|sao paulo|"
    "são paulo|interlagos|las vegas|vegas|qatar|lusail|abu dhabi|yas marina".split("|"), "")


_METALS = dict.fromkeys(("gold", "silver"), "")  # the web lookup prices these
_READERS: dict[str, Reader] = {}


def reader(domain: str) -> Reader:
    """The domain's form reader (reader_data/<domain>.tsv), trained once on first use. Names it
    knows by kind are masked so one example covers them all: a driver, a team, an index, a coin."""
    if domain not in _READERS:
        from app.chat.router import F1_DRIVERS, F1_TEAMS
        kinds = {"f1": ((F1_DRIVERS, "xdriver"), (F1_TEAMS, "xteam"), (_F1_PLACES, "xplace")),
                 "finance": ((_INDICES, "xindex"), (_COINS, "xcoin"), (_METALS, "xmetal"))}[domain]
        # A [bracketed] example name reads like a name the router knows: a company, a place. Any
        # company may be asked about; an F1 name is a driver, team or place on these lists.
        _READERS[domain] = Reader(domain, {alias: mask for names, mask in kinds for key in names
                                           for alias in key.split("|")},
                                  name_mask={"f1": "xplace", "finance": "xticker"}[domain],
                                  unknown_names=domain == "finance")
    return _READERS[domain]


def warm_readers() -> None:
    """Train every reader ahead of the first question (Kip's startup, in a worker thread)."""
    for domain in ("f1", "finance"):
        reader(domain)


def _holding_name(reading: Reading, q: dict) -> str | None:
    """The company asked about: the router's name for it when it knows one ("HDFC Bank"), else
    the words the reader found ("tata motors")."""
    named = [e for e in q.get("entities") or [] if e.casefold() not in _COINS and e.casefold() not in ("gold", "silver")
             and not any(e == v[1] for v in _INDICES.values())]
    if named:
        return named[0]
    if reading.name:
        return " ".join(w.upper() if len(w) <= 4 else w.capitalize() for w in reading.name.split())
    return None


def _market_kind(q: dict) -> tuple[str, object] | None:
    """What a markets question asks for, as the finance reader reads it, decided once so the
    callout and the answer agree: ("portfolio", None or a holding), ("coin", id), ("index", (symbol,
    name)), ("currency", (base, target)), ("share", name), or None for what goes to the web lookup
    (gold, past prices, anything the reader is unsure of)."""
    said = q.get("said", "")
    # A follow-up ("and TCS?") is read through the question it follows; its own words name the company.
    earlier = q.get("earlier") if q.get("followup") else ""
    reading = reader("finance").read(earlier or said)
    subject = reader("finance").read(said) if earlier else reading
    if not reading.confident or reading.form["when"] == "past" or q.get("time") not in (None, "now", "today"):
        return None
    kind = reading.form["kind"]
    words = " ".join([q.get("said", ""), *(q.get("entities") or [])]).casefold()
    if kind == "my portfolio":
        return "portfolio", None
    if kind == "my holding":
        return "portfolio", _holding_name(subject, q)
    if kind == "crypto":
        coin = next((_COINS[n] for n in subject.known if n in _COINS), None)
        return ("coin", coin) if coin else None
    if kind == "index":
        index = next((_INDICES[n] for n in subject.known if n in _INDICES), None)
        return ("index", index) if index else None
    if kind == "exchange rate":
        # Currencies in the order they were said; the one after "a"/"one"/"1" is being priced ("how many
        # dollars is a pound?"), else the first one that is not the rupee.
        found = sorted((m.start(), code, words[max(0, m.start() - 4):m.start()])
                       for name, code in _CURRENCIES for m in re.finditer(rf"\b{name}s?\b", words))
        named = list(dict.fromkeys(code for _, code, _ in found))
        priced = [code for _, code, before in found if re.search(r"\b(?:a|an|one|1)\s$", before)]
        base = next((c for c in priced if c != "INR"), None) or next((c for c in named if c != "INR"), None)
        return ("currency", (base, next((c for c in named if c not in ("INR", base)), "INR"))) if base else None
    if kind == "share price":
        name = _holding_name(subject, q)
        return ("share", name) if name else None
    return None


async def _markets(q: dict, now: datetime, get, groww: tuple[str, str] = ("", "")) -> FeedAnswer | None:
    """Their Groww holdings, crypto from CoinGecko, indices and shares from Yahoo (unofficial),
    exchange rates from Frankfurter (European Central Bank). Gold and past prices go to the web lookup."""
    kind = _market_kind(q)
    if kind is None:
        return None
    what, detail = kind
    if what == "portfolio":
        return await _portfolio(now, get, groww, detail) if all(groww) else None
    if what == "coin":
        price = (await get(f"{CRYPTO}?ids={detail}&vs_currencies=inr,usd&include_24hr_change=true")).get(detail)
        if not price:
            return None
        change = price.get("inr_24h_change")
        text = (f"{detail.capitalize()} is {_inr(price['inr'])} (${price['usd']:,.0f})"
                + (f", {'up' if change >= 0 else 'down'} {abs(change):.1f}% in 24 hours." if change is not None else "."))
        return FeedAnswer(text, "https://www.coingecko.com/", f"CoinGecko: {detail}", "coingecko")
    if what == "index":
        return await _quote(detail[0], detail[1], now, get)
    if what == "share":
        return await _share(detail, now, get)
    base, target = detail
    data = await get(f"{RATES}?from={base}&to={target}")
    rate = (data.get("rates") or {}).get(target)
    if rate is None:
        return None
    amount = _inr(rate, decimals=2) if target == "INR" else f"{rate:,.4f} {_CURRENCY_NAMES[target]}s"
    on = _day(data["date"]).rsplit(" ", 1)[0]
    return FeedAnswer(f"1 {_CURRENCY_NAMES[base]} is {amount} (European Central Bank rate, {on}).",
                      "https://www.frankfurter.app/", f"Frankfurter: {base}/{target}", "frankfurter")


async def _quote(symbol: str, name: str, now: datetime, get) -> FeedAnswer:
    """An index or share from Yahoo's chart: the price, the day's move and when, in IST."""
    meta = (await get(f"{INDEX}/{quote(symbol)}?range=1d&interval=1d"))["chart"]["result"][0]["meta"]
    price, before = meta["regularMarketPrice"], meta.get("chartPreviousClose") or meta.get("previousClose")
    at = datetime.fromtimestamp(meta["regularMarketTime"], now.tzinfo)
    rupees = meta.get("currency") == "INR" and not symbol.startswith("^")
    text = f"{name} is " + (f"{_inr(price, decimals=2)}" if rupees else f"at {price:,.2f}")
    if before:
        move = (price - before) / before * 100
        text += ", flat on the day" if abs(move) < 0.005 else f", {'up' if move > 0 else 'down'} {abs(move):.2f}% on the day"
    text += f" (as of {at.strftime('%A')} {at.day} {at.strftime('%B')}, {at.strftime('%I:%M %p').lstrip('0').lower()})."
    return FeedAnswer(text, "https://finance.yahoo.com/", f"Yahoo Finance: {name}", "yahoo")


async def _share(name: str, now: datetime, get) -> FeedAnswer | None:
    """A company they name ("Reliance Industries"), found by Yahoo's search: its NSE listing, else BSE."""
    found = (await get(f"{SEARCH}?{urlencode({'q': name, 'quotesCount': 5, 'newsCount': 0})}")).get("quotes") or []
    listing = next((x["symbol"] for suffix in (".NS", ".BO") for x in found
                    if x.get("symbol", "").endswith(suffix)), None)
    return await _quote(listing, name, now, get) if listing else None


# ── their Groww holdings (read-only) ────────────────────────────────────────────
# Kip calls exactly two Groww endpoints: the token and the holdings. Never orders, funds or positions.
GROWW_TOKEN_URL = f"{GROWW}/token/api/access"
GROWW_HOLDINGS_URL = f"{GROWW}/holdings/user"
_GROWW_TOKEN: dict[str, tuple[str, datetime]] = {}


def _totp(seed: str, at: float | None = None) -> str:
    """The six-digit code an authenticator app shows for this seed (RFC 6238: SHA-1, 30 s)."""
    raw = base64.b32decode(seed.strip().upper() + "=" * (-len(seed.strip()) % 8))
    digest = hmac.new(raw, struct.pack(">Q", int(time.time() if at is None else at) // 30), hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    return f"{(struct.unpack('>I', digest[offset:offset + 4])[0] & 0x7FFFFFFF) % 1_000_000:06d}"


async def _groww_token(groww: tuple[str, str], now: datetime) -> str:
    """A Groww access token from the TOTP key, kept until it expires (6 am): the token endpoint
    allows 150 requests a day."""
    key, seed = groww
    cached = _GROWW_TOKEN.get(key)
    if cached and now < cached[1]:
        return cached[0]
    reply = await post(GROWW_TOKEN_URL, {"key_type": "totp", "totp": _totp(seed)}, {"Authorization": f"Bearer {key}"})
    expiry = datetime.fromisoformat(reply["expiry"]).replace(tzinfo=now.tzinfo) - timedelta(minutes=5)
    _GROWW_TOKEN[key] = (reply["token"], expiry)
    return reply["token"]


async def _portfolio(now: datetime, get, groww: tuple[str, str], name: str | None = None) -> FeedAnswer | None:
    """Their holdings from Groww, priced from Yahoo (this Groww key is refused live market data):
    all of them, or the one they name ("how much HDFC Bank do I own?")."""
    token = await _groww_token(groww, now)
    data = await get(GROWW_HOLDINGS_URL, {"Authorization": f"Bearer {token}", "Accept": "application/json",
                                          "X-API-VERSION": "1.0"})
    holdings = [h for h in (data.get("payload") or {}).get("holdings") or [] if h.get("quantity")]
    if not holdings:
        return FeedAnswer("You hold no shares on Groww right now.", "https://groww.in/", "Groww holdings", "groww")
    if name:
        return await _holding(name, holdings, get)
    priced = await _priced(holdings, get)
    if not priced:
        return None
    value = sum(h["quantity"] * price for h, price, _ in priced)
    paid = sum(h["quantity"] * h["average_price"] for h, _, _ in priced)
    before = sum(h["quantity"] * (prev or price) for h, price, prev in priced)
    moves = sorted(((price / prev - 1) * 100, h["trading_symbol"]) for h, price, prev in priced if prev)
    # A zero cost (gifted or bonus shares, or a cost Groww has not recorded) has no percentage gain.
    text = (f"Your Groww portfolio: {len(priced)} holding{'s' if len(priced) != 1 else ''} worth {_inr(value)}, "
            f"{'up' if value >= paid else 'down'} {_inr(abs(value - paid))}{_percent(value, paid, 1)} on what you paid. "
            f"On the day {'up' if value >= before else 'down'} {_inr(abs(value - before))}{_percent(value, before, 2)}")
    if len(moves) > 1:
        text += f"; best {moves[-1][1]} {moves[-1][0]:+.1f}%, worst {moves[0][1]} {moves[0][0]:+.1f}%"
    text += "." + (f" {len(holdings) - len(priced)} could not be priced." if len(priced) < len(holdings) else "")
    return FeedAnswer(text, "https://groww.in/", "Groww holdings", "groww")


async def _priced(holdings: list[dict], get) -> list[tuple[dict, float, float | None]]:
    """(holding, price, previous close) for each holding Yahoo could price, NSE where it trades there."""
    listed = {h["trading_symbol"]: f"{h['trading_symbol']}{'.NS' if 'NSE' in (h.get('tradable_exchanges') or ['NSE']) else '.BO'}"
              for h in holdings}
    symbols = list(dict.fromkeys(listed.values()))
    prices = {}
    for start in range(0, len(symbols), 20):  # Yahoo prices up to 20 symbols a request
        spark = await get(f"{SPARK}?{urlencode({'symbols': ','.join(symbols[start:start + 20]), 'range': '1d', 'interval': '1d'})}")
        for item in (spark.get("spark") or {}).get("result") or []:
            meta = item["response"][0]["meta"]
            prices[item["symbol"]] = (meta["regularMarketPrice"], meta.get("chartPreviousClose") or meta.get("previousClose"))
    return [(h, *prices[listed[h["trading_symbol"]]]) for h in holdings if listed[h["trading_symbol"]] in prices]


def _bare(text: str) -> str:
    """"HDFC Bank" and "HDFCBANK" alike: lower case, letters and digits only."""
    return re.sub(r"[^a-z0-9]", "", text.casefold())


async def _holding(name: str, holdings: list[dict], get) -> FeedAnswer:
    """The holding they name: its symbol as said ("HDFCBANK", "HDFC Bank"), else the NSE or BSE
    listing Yahoo finds for the name ("Tata Consultancy Services" is TCS), else symbols that start
    with it ("HDFC": every HDFC company they hold)."""
    key = _bare(name)
    found = [h for h in holdings if _bare(h["trading_symbol"]) == key]
    if not found:
        quotes = (await get(f"{SEARCH}?{urlencode({'q': name, 'quotesCount': 5, 'newsCount': 0})}")).get("quotes") or []
        listing = next((x["symbol"].rsplit(".", 1)[0] for suffix in (".NS", ".BO") for x in quotes
                        if x.get("symbol", "").endswith(suffix)), None)
        found = [h for h in holdings if listing and _bare(h["trading_symbol"]) == _bare(listing)]
    if not found and len(key) >= 3:
        found = [h for h in holdings if _bare(h["trading_symbol"]).startswith(key)]
    if not found:
        return FeedAnswer(f"You don't hold any {name} on Groww.", "https://groww.in/", "Groww holdings", "groww")
    prices = {h["trading_symbol"]: (price, prev) for h, price, prev in await _priced(found, get)}
    lines = []
    for h in found:
        quantity, paid = h["quantity"], h["quantity"] * h["average_price"]
        line = f"You hold {quantity:g} {h['trading_symbol']} share{'s' if quantity != 1 else ''}"
        if h["trading_symbol"] not in prices:
            lines.append(f"{line} (bought at {_inr(h['average_price'], decimals=2)} each); I couldn't get today's price.")
            continue
        price, prev = prices[h["trading_symbol"]]
        value = quantity * price
        line += (f", worth {_inr(value)} at {_inr(price, decimals=2)}: "
                 f"{'up' if value >= paid else 'down'} {_inr(abs(value - paid))}{_percent(value, paid, 1)} "
                 f"on the {_inr(paid)} you paid")
        if prev:
            line += f", {'up' if price >= prev else 'down'} {abs(price / prev - 1) * 100:.2f}% today"
        lines.append(line + ".")
    return FeedAnswer(" ".join(lines), "https://groww.in/", "Groww holdings", "groww")


def _percent(now: float, then: float, places: int) -> str:
    """" (6.6%)", or nothing when there is no base to compare with."""
    return f" ({abs(now / then - 1) * 100:.{places}f}%)" if then > 0 else ""


async def post(url: str, body: dict, headers: dict) -> dict:
    """The Groww token request: no redirects (httpx's default) and bounded like fetch."""
    async with asyncio.timeout(FEED_SECONDS):
        async with httpx.AsyncClient(timeout=10, headers={"User-Agent": "kip-companion"}) as client:
            response = await client.post(url, json=body, headers=headers)
            response.raise_for_status()
            return response.json()


# ── news ────────────────────────────────────────────────────────────────────────
_TOPIC = re.compile(r"\b(?:news|headlines?|latest)\s+(?:about|on|regarding|from|in)\s+(?:the\s+)?(.+?)[?.!]*$", re.I)


def _news_question(q: dict) -> bool:
    return q.get("time") in (None, "now", "today")  # headlines are only ever today's


async def _news(q: dict, get) -> FeedAnswer | None:
    """Top three headlines from Google News (India), or on a topic they name ("news about the
    elections"). Read out as headlines: nothing is summarised or inferred."""
    if not _news_question(q):
        return None
    topic = _TOPIC.search(q.get("said", "").strip())
    where = "hl=en-IN&gl=IN&ceid=IN:en"
    url = f"{NEWS}/search?q={quote(topic.group(1))}&{where}" if topic else f"{NEWS}?{where}"
    feed = await get(url)
    if "<!DOCTYPE" in feed.upper():
        # RSS never needs one; refusing it rules out entity tricks (expat also limits them).
        raise ValueError("RSS with a DOCTYPE")
    items = ElementTree.fromstring(feed).iter("item")  # no DOCTYPE, so no entities  # nosec B314
    titles = [item.findtext("title", "").strip() for item in items][:3]
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
