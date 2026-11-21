"""Reading reminders, timers and scheduled checks from the person's own words, deterministically.

"Remind me in 10 minutes to stretch", "set a 5 minute timer", "every weekday at 8 tell me the weather in
Pune", "what reminders do I have?", "cancel the stretch reminder". No model is involved, so they work when
Groq is down and never get a time wrong by guessing: a request that cannot be read is asked about, and
what was set is always read back with its time.

Times are in the timezone of the `now` handed in (Kip's own), so "9 pm" means 9 pm for the person.
"""
import re
from dataclasses import dataclass, field
from datetime import datetime, time as clock_time, timedelta

MIN_SECONDS = 5
MAX_DAYS = 400
MIN_REPEAT_MINUTES = 10        # a reminder may repeat this often at most
MIN_LOOKUP_REPEAT_MINUTES = 60  # a scheduled check, which spends model quota, less often

_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
          "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30,
          "forty": 40, "forty five": 45, "fifty": 50, "sixty": 60, "ninety": 90}
_NUMBER = r"\d+(?:\.\d+)?|" + "|".join(sorted((re.escape(w) for w in _WORDS), key=len, reverse=True))
_UNITS = r"seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?"
_DURATION = re.compile(rf"\b(?P<n>{_NUMBER})\s*(?P<u>{_UNITS})\b")
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
           "november", "december"]
_MONTH = "|".join(m[:3] + ("(?:" + m[3:] + ")?" if len(m) > 3 else "") for m in _MONTHS)
_DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_WEEKDAY = "|".join(d[:3] + "(?:" + d[3:] + ")?" for d in _DAYS)
_PARTS = {"morning": 8, "afternoon": 15, "evening": 18, "night": 21}

_SPECIAL_DURATIONS = [
    (re.compile(r"\ban hour and a half\b"), "90 minutes"), (re.compile(r"\bhalf an? hour\b"), "30 minutes"),
    (re.compile(r"\b(?:a )?quarter of an hour\b|\ba quarter hour\b"), "15 minutes"),
    (re.compile(r"\ba couple of (minutes?|hours?|days?)\b"), r"2 \1")]

_EVERY_UNITS = re.compile(rf"\bevery\s+(?P<n>{_NUMBER})?\s*(?P<u>minutes?|mins?|hours?|hrs?)\b")
_EVERY_DAY = re.compile(r"\b(?:every\s+(?:single\s+)?(?:day|morning|afternoon|evening|night)|daily|each\s+(?:day|morning|"
                        r"afternoon|evening|night)|everyday)\b")
_EVERY_WEEKDAY = re.compile(r"\b(?:every\s+(?:week\s?day|working day)|week\s?days|on week\s?days)\b")
_EVERY_DOW = re.compile(rf"\b(?:every|each)\s+(?P<d>{_WEEKDAY})s?\b|\bweekly\s+on\s+(?P<d2>{_WEEKDAY})s?\b")
_EVERY_WEEK = re.compile(r"\b(?:every|each)\s+week\b|\bweekly\b")
_EVERY_YEAR = re.compile(r"\b(?:every\s+year|yearly|annually|each\s+year)\b")

_CLOCK = re.compile(
    r"(?:\b(?:at|by|around|@)\s*)?\b(?P<h>\d{1,2})(?:[:.](?P<m>\d{2}))?\s*(?P<ap>a\.?m\.?|p\.?m\.?)\b"
    r"|\b(?:at|by|around|@)\s*(?P<h2>\d{1,2})(?:[:.](?P<m2>\d{2}))?\b(?!\s*(?:%|percent|km|kg|rupees|rs|dollars|people|"
    rf"items|apples|{_UNITS}))"
    r"|\b(?P<h3>\d{1,2}):(?P<m3>\d{2})\b|\b(?P<special>noon|midday|midnight)\b")
_PART_OF_DAY = re.compile(r"\b(?:in the |this )?(?P<p>morning|afternoon|evening|night)\b|\b(?P<tonight>tonight)\b")
_DAY_WORD = re.compile(r"\b(?P<w>today|tomorrow|day after tomorrow|tonight)\b")
_ON_WEEKDAY = re.compile(rf"\b(?:(?P<next>next|this|coming)\s+|on\s+)(?P<d>{_WEEKDAY})\b")
_BARE_WEEKDAY = re.compile(rf"\b(?P<d>{_WEEKDAY})\b")
_DATE_DM = re.compile(rf"\b(?:on\s+)?(?:the\s+)?(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?(?P<m>{_MONTH})\b(?:\s+(?P<y>\d{{4}}))?")
_DAY_OF_MONTH = re.compile(r"\b(?:on\s+)?the\s+(?P<d>\d{1,2})(?:st|nd|rd|th)\b")
_VAGUE = re.compile(r"\b(?:later|sometime|some time|soon|eventually)\b")
_DATE_MD = re.compile(rf"\b(?:on\s+)?(?P<m>{_MONTH})\s+(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\b(?:,?\s+(?P<y>\d{{4}}))?")

_REMIND = re.compile(r"\b(?:please\s+)?(?:(?:can|could|will|would) you\s+)?(?:remind|ping|alert|notify|nudge|wake)\s+me\b(?P<rest>.*)")
_REMIND_NOT = re.compile(r"^\s*(?:what|who|where|how|why|which|when|if|whether|of|again)\b")
_TIMER = re.compile(r"\b(?:timer|alarm|stopwatch)\b")
_INFO_VERB = re.compile(r"\b(?:tell me|give me|send me|show me|update me|brief me|let me know|check|look up|find out|"
                        r"what'?s|what is)\b")
_ONE_SHOT_INFO = re.compile(r"\b(?:tell me|let me know|message me|update me)\b")
_LIST = re.compile(r"\b(?:list|show|see|view|what|which|any|do i have|have i|how (?:long|much)|are there|check)\b")
_NOUN = re.compile(r"\b(?:reminders?|timers?|alarms?|scheduled (?:checks?|tasks?|jobs?))\b")
_CANCEL = re.compile(r"\b(?:cancel|delete|remove|clear|stop|turn off|switch off|forget|never ?mind|drop)\b")
_STOP_REMINDING = re.compile(r"\bstop reminding me\b(?:\s+(?:about|to|of))?\s*(?P<what>.*)")
_LEAD = re.compile(r"^\W*(?:(?:hey\s+)?kip\W+)?(?:please\s+)?(?:(?:can|could|will|would) you\s+)?(?:please\s+)?")


@dataclass
class Schedule:
    first: datetime
    repeat: str = ""       # "", "daily", "weekdays", "weekly:<0-6>", "every:<minutes>", "yearly"


@dataclass
class Request:
    action: str            # set | list | cancel | ask_when | ask_what
    kind: str = "reminder"  # reminder | timer | lookup
    what: str = ""
    schedule: Schedule | None = None
    match: str = ""        # cancel: words naming which one ("" with all=False: the only one)
    noun: str = ""         # cancel: what they called it ("timer", "reminder", "check"), so "the timer" is only a timer
    all: bool = False
    note: str = ""         # something to say when answering ("that time has passed today")
    extras: dict = field(default_factory=dict)


def _clean(text: str) -> str:
    return " ".join(text.casefold().replace("’", "'").split())


def _to_int(word: str) -> float:
    return float(word) if re.fullmatch(r"\d+(?:\.\d+)?", word) else float(_WORDS[word])


def _normalise(text: str) -> str:
    """"half an hour" -> "30 minutes", so one pattern reads every length of time."""
    for pattern, replacement in _SPECIAL_DURATIONS:
        text = pattern.sub(replacement, text)
    return text


def _duration(text: str) -> tuple[timedelta, tuple[int, int]] | None:
    """The first "N unit" in text, with where it sits."""
    found = _DURATION.search(text)
    if not found:
        return None
    seconds = _to_int(found.group("n")) * _UNIT_SECONDS[found.group("u")[0]]
    end = found.end()
    for extra in _DURATION.finditer(text, end):
        if text[end:extra.start()].strip() not in ('', 'and', ','):
            break
        seconds += _to_int(extra.group('n')) * _UNIT_SECONDS[extra.group('u')[0]]
        end = extra.end()
    return timedelta(seconds=seconds), (found.start(), end)


def _strip(text: str, spans: list[tuple[int, int]]) -> str:
    """text with the given spans (which may overlap) cut out."""
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    for start, end in reversed(merged):
        text = text[:start] + " " + text[end:]
    return " ".join(text.split())


def _month_number(name: str) -> int:
    return next(i for i, m in enumerate(_MONTHS, 1) if m.startswith(name[:3]))


def _weekday_number(name: str) -> int:
    return next(i for i, d in enumerate(_DAYS) if d.startswith(name[:3]))


def _at(day, hour: int, minute: int, now: datetime) -> datetime:
    return datetime.combine(day, clock_time(hour % 24, minute), tzinfo=now.tzinfo)


def _clock_of(text: str, spans: list, default_part: str | None) -> tuple[int, int, str] | None:
    """(hour, minute, how sure) for the first clock time in text. 'sure' is "yes" when am/pm, a part of day
    or 24-hour made it exact, else "guess" (a bare "at 6")."""
    found = _CLOCK.search(text)
    if not found:
        return None
    spans.append(found.span())
    if found.group("special"):
        return (12, 0, "yes") if found.group("special") != "midnight" else (0, 0, "yes")
    hour = int(found.group("h") or found.group("h2") or found.group("h3"))
    minute = int(found.group("m") or found.group("m2") or found.group("m3") or 0)
    if hour > 23 or minute > 59:
        return None
    meridiem = (found.group("ap") or "").replace(".", "")
    if meridiem:
        if not 1 <= hour <= 12:
            return None
        return (hour % 12 + (12 if meridiem.startswith("p") else 0), minute, "yes")
    part = _PART_OF_DAY.search(text)
    if part and (part.group("p") or part.group("tonight")):
        name = part.group("p") or "night"
        if name in ("afternoon", "evening", "night") and 1 <= hour <= 11:
            hour += 12
        if name == "night" and hour == 24:
            hour = 0
        spans.append(part.span())
        return hour % 24, minute, "yes"
    if hour >= 13 or hour == 0 or found.group("h3") and hour >= 13:
        return hour % 24, minute, "yes"
    return hour, minute, "guess"


def _resolve_clock(hour: int, minute: int, sure: str, day, now: datetime, explicit_day: bool) -> datetime:
    """A clock time on a day (or the next time it comes round, with no day given)."""
    if explicit_day:
        if sure == "guess":
            hour = hour + 12 if hour < 7 else hour       # "tomorrow at 5" is 5 pm, "tomorrow at 9" is 9 am
        return _at(day, hour, minute, now)
    today = _at(now.date(), hour, minute, now)
    if sure == "guess" and hour <= 12:
        candidates = sorted([_at(now.date(), hour % 12, minute, now), _at(now.date(), hour % 12 + 12, minute, now)])
        for candidate in candidates + [c + timedelta(days=1) for c in candidates]:
            if candidate > now + timedelta(seconds=MIN_SECONDS):
                return candidate
    if today <= now + timedelta(seconds=MIN_SECONDS):
        today += timedelta(days=1)
    return today


def _date_in_year(now: datetime, month: int, dom: int, year: int | None):
    """That day this year, or next year if it has gone by (or the given year)."""
    from datetime import date
    try:
        target = date(year or now.year, month, dom)
        if year is None and target < now.date():
            target = date(now.year + 1, month, dom)
        return target
    except ValueError:
        return None


def read_schedule(text: str, now: datetime) -> tuple[Schedule | None, str, str]:
    """(schedule, the text with the time words taken out, a problem to tell the person).

    schedule is None when no time was given (problem "" ) or when it could not be used (problem set)."""
    low = _normalise(_clean(text))
    if _CLOCK.search(low) and _clock_of(low, [], None) is None:
        return None, low, 'That clock time is invalid. What time did you mean?'
    spans: list[tuple[int, int]] = []
    problem = ""

    # ── repeating ───────────────────────────────────────────────────────────
    repeat, anchor_day = "", None
    found = _EVERY_UNITS.search(low)
    if found:
        number = _to_int(found.group("n")) if found.group("n") else 1
        minutes = int(number * (60 if found.group("u").startswith("h") else 1))
        spans.append(found.span())
        remainder = _strip(low, spans)
        return Schedule(now + timedelta(minutes=minutes), f"every:{minutes}"), remainder, ""
    for pattern, label in ((_EVERY_WEEKDAY, "weekdays"), (_EVERY_YEAR, "yearly")):
        found = pattern.search(low)
        if found:
            repeat = label
            spans.append(found.span())
            break
    if not repeat:
        found = _EVERY_DOW.search(low)
        if found:
            repeat, anchor_day = f"weekly:{_weekday_number(found.group('d') or found.group('d2'))}", None
            spans.append(found.span())
    if not repeat:
        found = _EVERY_DAY.search(low)
        if found:
            repeat = "daily"
            spans.append(found.span())
            part = re.search(r"morning|afternoon|evening|night", found.group(0))
            anchor_day = part.group(0) if part else None
    if not repeat:
        found = _EVERY_WEEK.search(low)
        if found:
            repeat = f"weekly:{now.weekday()}"
            spans.append(found.span())

    if repeat:
        clock = _clock_of(low, spans, None)
        date_found = None
        if repeat == "yearly":
            date_found = _DATE_DM.search(low) or _DATE_MD.search(low)
            if date_found:
                spans.append(date_found.span())
        if clock is None:
            part = anchor_day or (_PART_OF_DAY.search(low).group("p") if _PART_OF_DAY.search(low) else None)
            hour = _PARTS.get(part, 9)
            clock = (hour, 0, "yes")
            part_match = _PART_OF_DAY.search(low)
            if part_match:
                spans.append(part_match.span())
        hour, minute, sure = clock
        if repeat == "yearly":
            if not date_found:
                return None, _strip(low, spans), "Which date each year?"
            day = _date_in_year(now, _month_number(date_found.group("m")), int(date_found.group("d")), None)
            if day is None:
                return None, low, "That date does not exist."
            first = _at(day, hour, minute, now)
        elif repeat.startswith("weekly:"):
            dow = int(repeat.split(":")[1])
            first = _at(now.date() + timedelta(days=(dow - now.weekday()) % 7), hour, minute, now)
            if first <= now + timedelta(seconds=MIN_SECONDS):
                first += timedelta(days=7)
        elif repeat == "weekdays":
            first = _at(now.date(), hour, minute, now)
            while first <= now + timedelta(seconds=MIN_SECONDS) or first.weekday() > 4:
                first += timedelta(days=1)
        else:
            first = _resolve_clock(hour, minute, sure, now.date(), now, explicit_day=False)
        return Schedule(first, repeat), _strip(low, spans), ""

    # ── once ────────────────────────────────────────────────────────────────
    dur = _duration(low)
    if dur:
        delta, (begin, finish) = dur
        before = re.search(r"\b(?:in|after|for|within)\s*$", low[:begin])
        # "in 10 minutes", "after 2 hours", "for 5 minutes" and "a 5 minute timer": a length of time from now.
        if before or re.search(r"\b(?:timer|alarm)\b", low):
            if delta.total_seconds() < MIN_SECONDS:
                return None, low, "That is too short."
            if delta > timedelta(days=MAX_DAYS):
                return None, low, "That is too far away."
            return Schedule(now + delta), _strip(low, [(before.start() if before else begin, finish)]), ""

    spans = []
    day, explicit, tonight = now.date(), False, False
    found = _DAY_WORD.search(low)
    if found:
        word = found.group("w")
        spans.append(found.span())
        explicit = True
        if word == "tomorrow":
            day = now.date() + timedelta(days=1)
        elif word == "day after tomorrow":
            day = now.date() + timedelta(days=2)
        elif word == "tonight":
            tonight = True
    else:
        found = _DATE_DM.search(low) or _DATE_MD.search(low)
        if found:
            year = int(found.group("y")) if found.group("y") else None
            day = _date_in_year(now, _month_number(found.group("m")), int(found.group("d")), year)
            if day is None:
                return None, low, "That date does not exist."
            spans.append(found.span())
            explicit = True
        elif (found := _DAY_OF_MONTH.search(low)):
            day = _next_day_of_month(now, int(found.group("d")))
            if day is None:
                return None, low, "That date does not exist."
            spans.append(found.span())
            explicit = True
        elif (found := _ON_WEEKDAY.search(low)):
            dow = _weekday_number(found.group("d"))
            day = now.date() + timedelta(days=(dow - now.weekday()) % 7 or 7)
            spans.append(found.span())
            explicit = True
    clock = _clock_of(low, spans, None)
    part_match = _PART_OF_DAY.search(low)
    today_only = False
    if clock is None and part_match:
        name = part_match.group("p") or "night"
        spans.append(part_match.span())
        if tonight or part_match.group("tonight"):
            # Tonight: nine, or half past ten if nine has gone.
            options = [(21, 0), (22, 30)]
            hour, minute = next(((h, m) for h, m in options if _at(day, h, m, now) > now + timedelta(seconds=MIN_SECONDS)),
                                options[0])
        else:
            hour, minute = _PARTS[name], 0
        clock = (hour, minute, "yes")
        today_only = not explicit
    elif part_match:
        spans.append(part_match.span())
    if clock is None:
        if tonight:
            clock = (21, 0, "yes")
        elif explicit:
            clock = (9, 0, "yes")
        else:
            return None, _strip(low, [(m.start(), m.end()) for m in _VAGUE.finditer(low)]), ""
    hour, minute, sure = clock
    when = _resolve_clock(hour, minute, sure, day, now, explicit or today_only)
    if when <= now + timedelta(seconds=MIN_SECONDS):
        return None, _strip(low, spans), "That time has already passed. When should it be?"
    if when - now > timedelta(days=MAX_DAYS):
        return None, low, "That is too far away."
    return Schedule(when), _strip(low, spans), problem


def _next_day_of_month(now: datetime, dom: int):
    """The next time the calendar reaches this day of the month (today if it is today and not yet over)."""
    from datetime import date
    year, month = now.year, now.month
    for _ in range(14):
        try:
            candidate = date(year, month, dom)
            if candidate >= now.date():
                return candidate
        except ValueError:
            pass
        month, year = (1, year + 1) if month == 12 else (month + 1, year)
    return None


# ── what the person is asking ───────────────────────────────────────────────
def _what(rest: str) -> str:
    rest = re.sub(r"\b(?:please|thanks|thank you)\b", " ", rest)
    rest = _VAGUE.sub(" ", rest)
    rest = re.sub(r"^\W*(?:to|that|about|of)\b", " ", rest.strip())
    rest = re.sub(r"\b(?:remind|ping|alert|notify|nudge|wake)\s+me\b", " ", rest)
    rest = re.sub(r"^\W*up\b", " ", rest)
    rest = re.sub(r"\b(?:set|start|make)\s+(?:a|an|the|my)\s+(?:timer|alarm|reminder)\b", " ", rest)
    rest = re.sub(r"\b(?:a|an|the|my)?\s*(?:timer|alarm)\b", " ", rest)
    rest = " ".join(rest.split())
    for _ in range(4):  # "...at noon to pay rent" leaves "to pay rent": drop connectors left at either end
        tidy = re.sub(r"^\W*(?:to|that|about|of|for|me|on|at|in|and)\b", " ", rest)
        tidy = " ".join(re.sub(r"\b(?:to|that|about|for|and|then|at|on|in|by)\s*$", " ", tidy.strip(" .,;:!?-")).split())
        if tidy == rest:
            break
        rest = tidy
    return rest.strip(" .,;:!?-")


def read_request(text: str, now: datetime) -> Request | None:
    """What the message asks of reminders and timers, or None when it is not about them at all."""
    low = _clean(text)
    low = _LEAD.sub("", low, count=1)
    if re.search(r"\b(?:don't|do not|never)\s+(?:remind|ping|alert|notify|nudge|wake|cancel|delete|remove|clear|stop|set|start)\b", low):
        return Request('noop')
    noun = _NOUN.search(low)

    # cancelling: "cancel the stretch reminder", "stop reminding me about water", "clear all timers"
    stop = _STOP_REMINDING.search(low)
    if stop:
        return Request("cancel", match=_match_words(stop.group("what")), all=not stop.group("what").strip())
    if _CANCEL.search(low) and noun and not _REMIND.search(low) and not re.search(r"\b(?:every|in \d)", low):
        words = _CANCEL.sub(" ", low)
        every = bool(re.search(r"\b(?:all|every|everything)\b", words))
        called = noun.group(0)
        kind = "timer" if called.startswith(("timer", "alarm")) else "lookup" if called.startswith("scheduled") else "reminder"
        return Request("cancel", match=_match_words(_NOUN.sub(" ", words)), noun=kind, all=every)

    # listing: "what reminders do I have?", "list my timers", "how long is left on my timer?"
    if noun and _LIST.search(low) and not _REMIND.search(low) and not re.search(
            r"^(?:set|start|make|create)\b|\b(?:set|start|make|create)\s+(?:a|an|the|my|me|another|one)\b", low):
        return Request("list")

    remind = _REMIND.search(low)
    timer = _TIMER.search(low) and re.search(r"\b(?:set|start|make|create|put|run)\b|\btimer (?:for|of)\b|"
                                              rf"\b(?:{_NUMBER})[ -]*(?:{_UNITS})[ -]*(?:timer|alarm)\b|"
                                              r"\b(?:timer|alarm)\s+(?:for|at|in|on)\b", low)
    info = None
    schedule, rest, problem = read_schedule(low, now)
    recurring = bool(schedule and schedule.repeat)
    if not remind and not timer:
        # "every morning at 8 tell me the weather in Pune": a repeating check; "tell me in 10 minutes how it
        # went": a one-off. A plain question that mentions tomorrow ("what's the weather tomorrow?") is neither.
        verb = _INFO_VERB.search(low)
        if schedule is not None and verb and (recurring or _ONE_SHOT_INFO.search(low)) and not low.rstrip().endswith("?"):
            info = _what_to_check(rest)
        if info is None:
            return None
    if remind and not timer:
        tail = remind.group("rest").strip()
        if _REMIND_NOT.match(tail):
            return None                               # "remind me what we decided" is a question
        about_only = re.match(r"^(?:about|of)\b", tail) is not None
        if about_only and schedule is None:
            return None                               # "remind me about the budget": a recall, not a reminder
        if low.rstrip().endswith("?") and schedule is None:
            return None

    kind = "timer" if timer and not remind else ("lookup" if info else "reminder")
    what = _recase(info if info else _what(rest), text)
    if re.search(r"\bwake me\b", low):
        what = what or "wake up"
    if schedule is None:
        if problem:
            return Request("ask_when", kind=kind, what=what, note=problem)
        if kind == "timer" and not what:
            return Request("ask_when", kind=kind, note="How long?")
        return Request("ask_when", kind=kind, what=what)
    if kind == "reminder" and not what and not re.search(r"\b(?:ping|alert|notify|nudge)\b", low):
        return Request("ask_what", kind=kind, schedule=schedule)
    if schedule.repeat:
        step = int(schedule.repeat.split(":")[1]) if schedule.repeat.startswith("every:") else None
        floor = MIN_LOOKUP_REPEAT_MINUTES if kind == "lookup" else MIN_REPEAT_MINUTES
        if step is not None and step < floor:
            noun_ = "checks" if kind == "lookup" else "reminders"
            return Request("ask_when", kind=kind, what=what, note=f"{noun_.capitalize()} can repeat every {floor} minutes at most.")
    return Request("set", kind=kind, what=what, schedule=schedule)


def _recase(what: str, original: str) -> str:
    """`what` (lower-cased words taken from the message) with the person's own capitals back: "call Rohan"."""
    tokens = original.replace("’", "'").split()
    out, index = [], 0
    for word in what.split():
        key = word.strip(" .,;:!?-\"")
        for position in range(index, len(tokens)):
            if tokens[position].lower().strip(" .,;:!?-\"") == key:
                out.append(word.replace(key, tokens[position].strip(" .,;:!?-\"")) if key else word)
                index = position + 1
                break
        else:
            out.append(word)
    return " ".join(out)


def _what_to_check(rest: str) -> str | None:
    """"tell me the weather in Pune" -> "the weather in Pune"."""
    rest = re.sub(_INFO_VERB, " ", rest, count=1)
    rest = re.sub(r"\b(?:please|then|and)\b", " ", rest)
    rest = " ".join(rest.split()).strip(" .,;:!?-")
    rest = re.sub(r"^(?:me|about|how|what|if|whether)\b\s*", "", rest)
    return rest if len(rest) >= 3 else None


_STOP_WORDS = {"the", "my", "a", "an", "all", "every", "everything", "please", "that", "this", "it", "one", "about", "to",
               "of", "for", "me", "reminding", "set", "i", "you"}


def _match_words(text: str) -> str:
    return " ".join(w for w in re.findall(r"[\w']+", text.casefold()) if w not in _STOP_WORDS)


# ── saying a time back ──────────────────────────────────────────────────────
def say_time(moment: datetime, now: datetime) -> str:
    """"9:15 pm today", "9:00 am tomorrow", "9:00 am Mon 3 Mar"."""
    clock = f"{moment.hour % 12 or 12}:{moment.minute:02d} {'am' if moment.hour < 12 else 'pm'}"
    days = (moment.date() - now.date()).days
    if days == 0:
        return f"{clock} today"
    if days == 1:
        return f"{clock} tomorrow"
    if days < 7:
        return f"{clock} {moment.strftime('%A')}"
    return f"{clock} {moment.strftime('%a')} {moment.day} {moment.strftime('%b')}"


def say_repeat(repeat: str) -> str:
    if repeat == "daily":
        return "every day"
    if repeat == "weekdays":
        return "every weekday"
    if repeat == "yearly":
        return "every year"
    if repeat.startswith("weekly:"):
        return f"every {_DAYS[int(repeat.split(':')[1])].title()}"
    if repeat.startswith("every:"):
        minutes = int(repeat.split(":")[1])
        return f"every {minutes // 60} hour{'s' if minutes // 60 != 1 else ''}" if minutes % 60 == 0 and minutes >= 60 \
            else f"every {minutes} minutes"
    return ""


def say_length(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 90:
        return f"{seconds} seconds"
    minutes = round(seconds / 60)
    if minutes < 120:
        return f"{minutes} minutes"
    hours = seconds / 3600
    return f"{hours:g} hours" if hours < 36 else f"{round(hours / 24)} days"
