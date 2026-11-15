"""The real device clock: how Kip states the date and says the time."""
from datetime import datetime, timedelta, timezone

import pytest

from app.clock import date_str, now, spoken_time

IST = timezone(timedelta(hours=5, minutes=30), "India Standard Time")
NEWFOUNDLAND = timezone(timedelta(hours=-3, minutes=-30), "NDT")


def test_now_is_timezone_aware():
    assert now().tzinfo is not None


def test_date_str_in_a_non_utc_timezone():
    moment = datetime(2026, 11, 1, 17, 29, tzinfo=IST)
    assert date_str(moment) == "Sunday 1 November 2026, 17:29 India Standard Time (UTC+05:30)"


def test_date_str_with_a_negative_offset():
    moment = datetime(2026, 7, 4, 9, 5, tzinfo=NEWFOUNDLAND)
    assert date_str(moment) == "Saturday 4 July 2026, 09:05 NDT (UTC-03:30)"


def test_date_str_in_utc():
    moment = datetime(2026, 1, 2, 0, 0, tzinfo=timezone.utc)
    assert date_str(moment) == "Friday 2 January 2026, 00:00 UTC (UTC+00:00)"


def test_date_str_defaults_to_now():
    assert str(now().year) in date_str()


def test_spoken_time_uses_twelve_hour_clock_and_weekday():
    assert spoken_time(datetime(2026, 11, 1, 17, 29, tzinfo=IST)) == "5:29 evening. Sunday."
    assert spoken_time(datetime(2026, 11, 1, 6, 7, tzinfo=IST)) == "6:07 morning. Sunday."
    assert spoken_time(datetime(2026, 11, 2, 0, 0, tzinfo=IST)) == "12:00 night. Monday."
    assert spoken_time(datetime(2026, 11, 2, 12, 0, tzinfo=IST)) == "12:00 afternoon. Monday."


@pytest.mark.parametrize("hour, minute, part", [
    (0, 0, "night"),
    (4, 59, "night"),
    (5, 0, "morning"),
    (11, 59, "morning"),
    (12, 0, "afternoon"),
    (16, 59, "afternoon"),
    (17, 0, "evening"),
    (20, 59, "evening"),
    (21, 0, "night"),
    (23, 59, "night"),
])
def test_spoken_time_part_of_day_boundaries(hour, minute, part):
    moment = datetime(2026, 11, 1, hour, minute, tzinfo=IST)
    assert spoken_time(moment) == f"{hour % 12 or 12}:{minute:02d} {part}. Sunday."


def test_spoken_time_defaults_to_now():
    assert spoken_time().endswith(f"{now().strftime('%A')}.")
