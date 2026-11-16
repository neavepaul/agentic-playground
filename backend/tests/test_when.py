"""Reading reminders and timers from plain words. "Now" is Monday 16 November 2026, 8:45 pm in India."""
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from app.chat.when import read_request, say_length, say_repeat, say_time

NOW = datetime(2026, 11, 16, 20, 45, tzinfo=ZoneInfo("Asia/Kolkata"))


def ask(text):
    return read_request(text, NOW)


def when(text):
    request = ask(text)
    assert request is not None and request.action == "set", (text, request)
    return request.schedule.first, request.schedule.repeat, request.what, request.kind


def at(day, hour, minute=0):
    return datetime(2026, 11, day, hour, minute, tzinfo=ZoneInfo("Asia/Kolkata"))


# ── once ────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("text, due, what", [
    ("remind me in 10 minutes to stretch", at(16, 20, 55), "stretch"),
    ("Remind me to call Rohan at 6pm", at(17, 18), "call Rohan"),               # 6 pm has gone: tomorrow
    ("remind me tomorrow at 9 to call the dentist", at(17, 9), "call the dentist"),
    ("remind me at 9 to sleep", at(16, 21), "sleep"),                           # the next 9 is tonight
    ("remind me at 8am to run", at(17, 8), "run"),
    ("remind me in half an hour to check the oven", at(16, 21, 15), "check the oven"),
    ("remind me in an hour and a half to leave", at(16, 22, 15), "leave"),
    ("remind me in two hours to eat", at(16, 22, 45), "eat"),
    ("remind me tonight to lock the door", at(16, 21), "lock the door"),
    ("remind me tomorrow morning to take the bins out", at(17, 8), "take the bins out"),
    ("remind me next friday at 3pm to call Rohan", at(20, 15), "call Rohan"),
    ("remind me on the 15th at noon to pay rent", datetime(2026, 12, 15, 12, tzinfo=ZoneInfo("Asia/Kolkata")), "pay rent"),
    ("remind me on 3 March about Asha's birthday", datetime(2027, 3, 3, 9, tzinfo=ZoneInfo("Asia/Kolkata")),
     "Asha's birthday"),
    ("remind me at 7:30 pm to start dinner", at(16, 19, 30) .replace(day=17), "start dinner"),
    ("wake me at 6am", at(17, 6), "wake up"),
    ("remind me to buy milk in 20 minutes", at(16, 21, 5), "buy milk"),
])
def test_a_reminder_is_read_with_its_time_and_its_words(text, due, what):
    first, repeat, got, kind = when(text)
    assert (first, repeat, got, kind) == (due, "", what, "reminder")


def test_a_ping_with_no_words_is_a_reminder_of_nothing_in_particular():
    first, _, what, kind = when("ping me in 20 minutes")
    assert first == at(16, 21, 5) and what == "" and kind == "reminder"


@pytest.mark.parametrize("text, seconds", [
    ("set a 5 minute timer", 300), ("set a timer for 10 minutes", 600), ("timer for 90 seconds", 90),
    ("10 minute timer", 600), ("start a timer for an hour", 3600), ("set an alarm for 7am", None)])
def test_a_timer_is_a_length_from_now(text, seconds):
    first, repeat, _, kind = when(text)
    assert kind == "timer" and repeat == ""
    assert (first.timestamp() - NOW.timestamp() == seconds) if seconds else first == at(17, 7)


# ── repeating ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize("text, first, repeat, what", [
    ("remind me every day at 8am to take vitamins", at(17, 8), "daily", "take vitamins"),
    ("remind me every monday at 9 to submit the report", at(23, 9), "weekly:0", "submit the report"),
    ("remind me every weekday at 8 to check email", at(17, 8), "weekdays", "check email"),
    ("remind me every 2 hours to drink water", at(16, 22, 45), "every:120", "drink water"),
    ("remind me every 30 minutes to stand up", at(16, 21, 15), "every:30", "stand up"),
    ("remind me every evening to water the plants", at(17, 18), "daily", "water the plants"),
    ("remind me every year on 3 March about Asha's birthday",
     datetime(2027, 3, 3, 9, tzinfo=ZoneInfo("Asia/Kolkata")), "yearly", "Asha's birthday"),
])
def test_a_repeating_reminder_is_read(text, first, repeat, what):
    assert when(text) == (first, repeat, what, "reminder")


def test_a_check_can_repeat_and_a_reminder_can_repeat_faster_than_a_check():
    first, repeat, what, kind = when("every weekday at 8 tell me the weather in Pune")
    assert (first, repeat, what, kind) == (at(17, 8), "weekdays", "the weather in Pune", "lookup")
    first, repeat, what, kind = when("every morning at 7:30 give me the weather in Pune")
    assert (first, repeat, kind) == (at(17, 7, 30), "daily", "lookup") and what == "the weather in Pune"
    assert ask("remind me every 5 minutes to blink").action == "ask_when"
    assert "10 minutes" in ask("remind me every 5 minutes to blink").note
    assert "60 minutes" in ask("every 20 minutes tell me the cricket score").note


# ── asking again ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("text, action", [
    ("remind me to call mom", "ask_when"), ("remind me later", "ask_when"), ("remind me tomorrow", "ask_what"),
    ("remind me in 2 hours", "ask_what"), ("set a timer", "ask_when"), ("remind me tonight at 8", "ask_when")])
def test_what_cannot_be_read_is_asked_about(text, action):
    assert ask(text).action == action


def test_a_time_that_has_gone_is_said_so_and_never_moved_silently():
    request = ask("remind me this evening to water the plants")     # 6 pm has gone
    assert request.action == "ask_when" and "already passed" in request.note


def test_what_is_kept_when_only_the_time_is_missing():
    assert ask("remind me to buy 3 apples").what == "buy 3 apples"


# ── not about reminders ─────────────────────────────────────────────────────
@pytest.mark.parametrize("text", [
    "remind me what we decided about Goa", "can you remind me about the budget?", "what's the weather tomorrow?",
    "tell me the weather in Pune", "i need to be there in 10 minutes", "I'll be there at 6pm", "hello",
    "what time is it?", "Who is playing tonight?", "remind me of the plan", "tomorrow is my birthday"])
def test_ordinary_talk_is_left_to_the_model(text):
    assert ask(text) is None


# ── managing ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("text", ["what reminders do I have?", "list my timers", "show my reminders",
                                  "do I have any reminders?", "how long is left on my timer?", "which alarms are set"])
def test_listing_is_read(text):
    assert ask(text).action == "list"


@pytest.mark.parametrize("text, match, everything", [
    ("cancel the stretch reminder", "stretch", False), ("cancel all reminders", "", True),
    ("stop reminding me about water", "water", False), ("cancel the timer", "", False),
    ("delete my dentist reminder", "dentist", False), ("clear all my timers", "", True),
    ("stop reminding me", "", True)])
def test_cancelling_is_read(text, match, everything):
    request = ask(text)
    assert request.action == "cancel" and request.match == match and request.all is everything


def test_setting_one_is_never_mistaken_for_cancelling():
    assert ask("remind me to cancel the subscription tomorrow at 9").action == "set"


# ── saying times back ───────────────────────────────────────────────────────
def test_times_are_read_back_plainly():
    assert say_time(at(16, 21, 15), NOW) == "9:15 pm today"
    assert say_time(at(17, 9), NOW) == "9:00 am tomorrow"
    assert say_time(at(19, 12), NOW) == "12:00 pm Thursday"
    assert say_time(datetime(2027, 3, 3, 9, tzinfo=NOW.tzinfo), NOW) == "9:00 am Wed 3 Mar"
    assert say_repeat("weekdays") == "every weekday" and say_repeat("weekly:0") == "every Monday"
    assert say_repeat("every:120") == "every 2 hours" and say_repeat("every:45") == "every 45 minutes"
    assert say_length(30) == "30 seconds" and say_length(300) == "5 minutes" and say_length(7200) == "2 hours"
