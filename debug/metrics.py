"""Driving performance metrics for the virtual lane bot."""

from dataclasses import dataclass, field


@dataclass
class DrivingMetrics:
    """Track lane-following quality during a simulation run."""

    departure_threshold_px: float = 80.0
    elapsed: float = 0.0
    frames: int = 0
    lane_departures: int = 0
    total_abs_error: float = 0.0
    max_abs_error: float = 0.0
    total_speed: float = 0.0
    _was_departed: bool = field(default=False, init=False)

    def update(self, lane_error: float, speed: float, dt: float) -> None:
        """Record one simulation frame."""
        abs_error = abs(float(lane_error))
        self.elapsed += max(0.0, dt)
        self.frames += 1
        self.total_abs_error += abs_error
        self.max_abs_error = max(self.max_abs_error, abs_error)
        self.total_speed += float(speed)

        departed = abs_error >= self.departure_threshold_px
        if departed and not self._was_departed:
            self.lane_departures += 1
        self._was_departed = departed

    @property
    def average_abs_error(self) -> float:
        """Average absolute lane-center error in pixels."""
        return self.total_abs_error / self.frames if self.frames else 0.0

    @property
    def average_speed(self) -> float:
        """Average simulated speed."""
        return self.total_speed / self.frames if self.frames else 0.0

    def summary_lines(self) -> list[str]:
        """Return printable metric summary lines."""
        return [
            "Driving metrics",
            f"time: {self.elapsed:.1f}s",
            f"frames: {self.frames}",
            f"avg lane error: {self.average_abs_error:.1f}px",
            f"max lane error: {self.max_abs_error:.1f}px",
            f"lane departures: {self.lane_departures}",
            f"avg speed: {self.average_speed:.2f}",
        ]
