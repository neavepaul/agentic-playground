"""Whether a looked-up answer is backed by what was actually read.

Groq's browser search reads pages and writes an answer in one call, so nothing stops it
writing a figure no page gave. The answer must therefore end with SOURCE: and words
copied from what it read. Code then checks that those words really are in the browsing
output, and that every clock time and number in the answer comes from them, converting
time zones itself (07:00 UTC is 12:30 IST). An answer that fails is never spoken.
"""
import re
from dataclasses import dataclass
from datetime import date, timedelta

# Fixed offsets in minutes. Daylight-saving names are listed separately (NZDT, BST).
ZONES = {"utc": 0, "gmt": 0, "z": 0, "ist": 330, "nzdt": 780, "nzst": 720, "bst": 60, "cet": 60, "cest": 120,
         "est": -300, "edt": -240, "cst": -360, "cdt": -300, "pst": -480, "pdt": -420, "aedt": 660, "aest": 600,
         "acdt": 630, "awst": 480, "jst": 540, "sgt": 480, "gst": 240, "pkt": 300, "sast": 120, "bdt": 360,
         "npt": 345}
_ZONE = "|".join(sorted(ZONES, key=len, reverse=True))
_TIME = re.compile(r"\b(\d{1,2})(?:[:.](\d{2}))?\s*(a\.?\s?m\.?|p\.?\s?m\.?)?(?=\W|$)(?:\s*\(?\s*(" + _ZONE + r")\b)?",
                   re.I)
_SOURCE = re.compile(r"\n?\s*\**source\**\s*:\s*(.*)\s*$", re.I | re.S)
_MARKUP = re.compile(r"\bl\d+:\s|【[^】]*】|[†*#|\\]")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


@dataclass
class Clock:
    minutes: int          # minutes after midnight, in its own zone
    zone: str | None      # None: whatever zone the page or listener uses


def normalize(text: str) -> str:
    return " ".join(_MARKUP.sub(" ", text.casefold().replace("’", "'")).split())


def split_source(text: str) -> tuple[str, str]:
    """(answer, quoted source) from a reply ending in "SOURCE: ..."."""
    match = _SOURCE.search(text)
    if not match:
        return text.strip(), ""
    quote = match.group(1).strip().strip('"“”\'').strip()
    return text[:match.start()].strip(), "" if quote.casefold() in ("none", "n/a", "") else quote


def clock_times(text: str) -> list[Clock]:
    """Clock times written as 12:30, 7.30 pm, 7 pm or 19:00, with a zone when one follows."""
    found = []
    for match in _TIME.finditer(text):
        hour, minute, half, zone = match.group(1), match.group(2), match.group(3), match.group(4)
        if minute is None and half is None:
            continue  # a bare number is not a time
        hour, minute = int(hour), int(minute or 0)
        if hour > 23 or minute > 59:
            continue
        if half:
            pm = half.casefold().startswith("p")
            hour = (hour % 12) + (12 if pm else 0)
        found.append(Clock(hour * 60 + minute, zone.casefold() if zone else None))
    return found


def _numbers(text: str) -> set[str]:
    without_times = _TIME.sub(lambda m: " " if (m.group(2) or m.group(3)) else m.group(0), text)
    return {n.replace(",", "") for n in _NUMBER.findall(without_times)}


def quote_found(quote: str, evidence: str) -> bool:
    """The quoted words are in what was read: verbatim, or nearly (every figure present, and
    nine in ten of the words), since pages wrap and the model trims punctuation."""
    quote, evidence = normalize(quote), normalize(evidence)
    if len(quote) < 8:
        return False
    if quote in evidence:
        return True
    words = re.findall(r"[a-z0-9]+", quote)
    present = set(re.findall(r"[a-z0-9]+", evidence))
    digits = [w for w in words if any(c.isdigit() for c in w)]
    return bool(words) and all(d in present for d in digits) and sum(w in present for w in words) >= 0.9 * len(words)


def _same(said: Clock, source: Clock, local_offset: int) -> bool:
    """Do these two clock times name the same moment? A time with no zone is the listener's
    when Kip says it, and the page's own when the page says it."""
    if said.zone is None and source.zone is None:
        return said.minutes == source.minutes
    said_zone = ZONES[said.zone] if said.zone else local_offset
    if source.zone is None:
        return said.minutes == source.minutes  # converted from nothing: must be the page's figure
    return (said.minutes - said_zone - (source.minutes - ZONES[source.zone])) % 1440 == 0


def unsupported(answer: str, quote: str, evidence: str, local_offset: int, today: date | None = None) -> list[str]:
    """What in the answer is not backed by the quote and the pages read; empty when all is."""
    if not quote:
        return ["no source quoted"]
    if not quote_found(quote, evidence):
        return [f"the quoted source is not in what was read: {quote[:120]!r}"]
    problems = []
    if today is not None:
        # "12:30 today, Saturday" on a Sunday: a page written in another zone, copied unconverted.
        lowered = answer.casefold()
        for word, day in (("today", today), ("tonight", today), ("tomorrow", today + timedelta(days=1))):
            named = [d for d in _WEEKDAYS if re.search(rf"\b{d}\b", lowered)]
            if re.search(rf"\b{word}\b", lowered) and named and day.strftime("%A").casefold() not in named:
                problems.append(f"it says {word} is {named[0].title()}, but {word} is {day.strftime('%A')}")
    sources = clock_times(quote)
    for said in clock_times(answer):
        if not any(_same(said, source, local_offset) for source in sources):
            problems.append(f"time {said.minutes // 60:02d}:{said.minutes % 60:02d}"
                            f"{' ' + said.zone.upper() if said.zone else ''} is not in the quoted source")
    known = _numbers(quote) | _numbers(evidence) | (_numbers(today.strftime("%Y-%m-%d %d")) if today else set())
    for number in sorted(_numbers(answer) - known):
        problems.append(f"{number} is in nothing that was read")
    return problems


# ── a time belongs to the match it is listed with ──────────────────────────────
# Fixture lists put many matches on one page ("08:05 am IST ... Sri Lanka vs Hong Kong",
# "12:30 pm IST ... New Zealand vs India"). A time is only the answer if the page lists it
# with the teams asked about, not under some other fixture.
_VS = re.compile(r"([a-z][a-z .'&-]{1,40}?)\s+(?:vs?\.?|versus)\s+([a-z][a-z .'&-]{1,40})", re.I)
_CODES = {"india": {"ind"}, "new zealand": {"nz"}, "australia": {"aus"}, "england": {"eng"}, "pakistan": {"pak"},
          "south africa": {"sa", "rsa"}, "sri lanka": {"sl"}, "bangladesh": {"ban"}, "afghanistan": {"afg"},
          "west indies": {"wi"}}


def _listed_times(text: str) -> list[tuple[Clock, str]]:
    """Every clock time in a page, with the text that follows it up to the next time: the
    record a fixture list attaches to that time."""
    spans = []
    for match in _TIME.finditer(text):
        found = clock_times(match.group(0))
        if found:
            spans.append((found[0], match.start(), match.end()))
    return [(clock, text[end:(spans[i + 1][1] if i + 1 < len(spans) else len(text))][:250])
            for i, (clock, _, end) in enumerate(spans)]


def _names(team: str) -> set[str]:
    team = team.casefold()
    return {team, *_CODES.get(team, set()), *team.split()[-1:]} if " " in team else {team, *_CODES.get(team, set())}


def wrong_event(answer: str, pages: list[str], teams: list[str], local_offset: int) -> list[str]:
    """Times in the answer that the pages give only for some other match; empty when every
    time is listed with one of `teams` somewhere, or no fixture is named near it."""
    if not teams:
        return []
    wanted = set().union(*(_names(t) for t in teams))
    mentions = lambda side: any(re.search(rf"\b{re.escape(name)}\b", side.casefold()) for name in wanted)
    listed = [entry for page in pages for entry in _listed_times(page)]
    problems = []
    for said in clock_times(answer):
        records = [record for clock, record in listed if _same(said, clock, local_offset)]
        label = f"{said.minutes // 60:02d}:{said.minutes % 60:02d}"
        if not records:
            problems.append(f"{label} is in none of the pages read")
            continue
        others = []
        for record in records:
            pairs = _VS.findall(record)
            if not pairs or any(mentions(a) or mentions(b) for a, b in pairs):
                break  # listed with the requested teams, or a time for the whole series
            first, second = pairs[0]
            others.append(f"{' '.join(first.split()[-2:])} vs {' '.join(second.split()[:2])}")
        else:
            problems.append(f"{label} is the time of {others[0]}, not of {' vs '.join(teams)}")
    return problems
