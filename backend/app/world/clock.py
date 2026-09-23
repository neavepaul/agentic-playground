import time


class WorldClock:
    """Simulated wall clock advancing at `speed` × real time.

    Default speed=60: 1 real second → 1 simulated minute; a full day runs in 24 real minutes.
    start_hour sets the simulated hour when the clock is created.
    """

    def __init__(self, speed: float = 60.0, start_hour: float = 8.0) -> None:
        self._speed = speed
        self._origin_real = time.monotonic()
        self._origin_sim = start_hour * 3600.0

    def simulated_seconds(self) -> float:
        return self._origin_sim + (time.monotonic() - self._origin_real) * self._speed

    def hour(self) -> float:
        """Hour of day as a float in [0, 24)."""
        return (self.simulated_seconds() % 86400.0) / 3600.0

    def time_str(self) -> str:
        """Current simulated time as HH:MM."""
        total_minutes = int(self.simulated_seconds() % 86400.0) // 60
        return f"{total_minutes // 60:02d}:{total_minutes % 60:02d}"
