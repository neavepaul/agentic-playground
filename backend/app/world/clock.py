import time
from datetime import datetime
from typing import Literal

Mode = Literal["device", "realtime", "action_driven"]


class WorldClock:
    """The house clock, in one of three modes.

    device: the machine's own local time, at normal speed. This is the real Kip: what he
    says the time is, and what people's routines follow, is the time on the device.

    realtime: simulated, advancing at `speed` × real time from `start_hour`. speed=60
    means 1 real second → 1 simulated minute, so a full day runs in 24 real minutes.

    action_driven: simulated, advancing only when `advance()` is called, by
    `action_seconds` each time. Model inference latency then cannot move the world, which
    makes runs reproducible and comparable across machines and model speeds.
    """

    def __init__(self, speed: float = 60.0, start_hour: float = 8.0,
                 mode: Mode = "realtime", action_seconds: float = 60.0) -> None:
        self._speed = 1.0 if mode == "device" else speed
        self._mode = mode
        self._action_seconds = action_seconds
        self._ticks = 0
        self._origin_real = time.monotonic()
        self._origin_sim = _seconds_since_midnight() if mode == "device" else start_hour * 3600.0

    @property
    def mode(self) -> Mode:
        return self._mode

    def advance(self, steps: int = 1) -> None:
        """Move the world forward by one agent turn. Ignored unless action-driven."""
        if self._mode == "action_driven":
            self._ticks += steps

    def simulated_seconds(self) -> float:
        """Seconds since midnight of the day the clock started. Never wraps, so durations
        measured with it stay correct across midnight."""
        if self._mode == "action_driven":
            return self._origin_sim + self._ticks * self._action_seconds
        return self._origin_sim + (time.monotonic() - self._origin_real) * self._speed

    def reset(self) -> None:
        self._origin_real = time.monotonic()
        self._ticks = 0
        if self._mode == "device":
            self._origin_sim = _seconds_since_midnight()

    def hour(self) -> float:
        """Hour of day as a float in [0, 24)."""
        return (self.simulated_seconds() % 86400.0) / 3600.0

    def time_str(self) -> str:
        """Current time as HH:MM."""
        if self._mode == "device":
            return datetime.now().strftime("%H:%M")
        total_minutes = int(self.simulated_seconds() % 86400.0) // 60
        return f"{total_minutes // 60:02d}:{total_minutes % 60:02d}"

    def now(self) -> datetime | None:
        """The device's local date and time, in device mode; a simulated clock has no date."""
        return datetime.now().astimezone() if self._mode == "device" else None

    def date_str(self) -> str | None:
        """'Sunday 1 November 2026, India Standard Time (UTC+05:30)' in device mode."""
        now = self.now()
        if now is None:
            return None
        offset = now.strftime("%z")
        return f"{now.strftime('%A')} {now.day} {now.strftime('%B %Y')}, {now.tzname()} (UTC{offset[:3]}:{offset[3:]})"


def _seconds_since_midnight() -> float:
    now = datetime.now()
    return now.hour * 3600.0 + now.minute * 60.0 + now.second + now.microsecond / 1e6


def spoken_time(clock: WorldClock) -> str:
    """The time as Kip says it: "12:52 afternoon. Sunday." (no weekday on a simulated clock)."""
    hour, minute = (int(part) for part in clock.time_str().split(":"))
    part = ("night" if hour < 5 else "morning" if hour < 12 else "afternoon" if hour < 17
            else "evening" if hour < 21 else "night")
    now = clock.now()
    return f"{hour % 12 or 12}:{minute:02d} {part}." + (f" {now.strftime('%A')}." if now else "")
