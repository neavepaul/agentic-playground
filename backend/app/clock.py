"""The real clock, in Kip's timezone (TIMEZONE, India Standard Time by default) whatever the device's
own clock is set to. Everything that needs the time asks here."""
from datetime import datetime
from zoneinfo import ZoneInfo

from app.config import get_settings


def now() -> datetime:
    return datetime.now(ZoneInfo(get_settings().timezone))


def date_str(moment: datetime | None = None) -> str:
    """'Sunday 1 November 2026, 17:29 India Standard Time (UTC+05:30)'."""
    moment = moment or now()
    offset = moment.strftime("%z")
    return (f"{moment.strftime('%A')} {moment.day} {moment.strftime('%B %Y')}, {moment.strftime('%H:%M')} "
            f"{moment.tzname()} (UTC{offset[:3]}:{offset[3:]})")


def spoken_time(moment: datetime | None = None) -> str:
    """The time as Kip says it: "5:29 evening. Sunday." """
    moment = moment or now()
    hour, minute = moment.hour, moment.minute
    part = ("night" if hour < 5 else "morning" if hour < 12 else "afternoon" if hour < 17
            else "evening" if hour < 21 else "night")
    return f"{hour % 12 or 12}:{minute:02d} {part}. {moment.strftime('%A')}."
