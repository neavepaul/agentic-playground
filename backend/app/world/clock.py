import time
from typing import Literal

Mode = Literal["realtime", "action_driven"]


class WorldClock:
    """Simulated wall clock with two progression modes.

    realtime (default): advances at `speed` × real time. speed=60 means 1 real
    second → 1 simulated minute, so a full day runs in 24 real minutes.

    action_driven: advances only when `advance()` is called, by `action_seconds`
    each time. Model inference latency then cannot move the world, which makes
    runs reproducible and comparable across machines and model speeds.
    """

    def __init__(self, speed: float = 60.0, start_hour: float = 8.0,
                 mode: Mode = "realtime", action_seconds: float = 60.0) -> None:
        self._speed = speed
        self._mode = mode
        self._action_seconds = action_seconds
        self._ticks = 0
        self._origin_real = time.monotonic()
        self._origin_sim = start_hour * 3600.0

    @property
    def mode(self) -> Mode:
        return self._mode

    def advance(self, steps: int = 1) -> None:
        """Move the world forward by one agent turn. Ignored in realtime mode."""
        if self._mode == "action_driven":
            self._ticks += steps

    def simulated_seconds(self) -> float:
        if self._mode == "action_driven":
            return self._origin_sim + self._ticks * self._action_seconds
        return self._origin_sim + (time.monotonic() - self._origin_real) * self._speed

    def reset(self) -> None:
        self._origin_real = time.monotonic()
        self._ticks = 0

    def hour(self) -> float:
        """Hour of day as a float in [0, 24)."""
        return (self.simulated_seconds() % 86400.0) / 3600.0

    def time_str(self) -> str:
        """Current simulated time as HH:MM."""
        total_minutes = int(self.simulated_seconds() % 86400.0) // 60
        return f"{total_minutes // 60:02d}:{total_minutes % 60:02d}"
