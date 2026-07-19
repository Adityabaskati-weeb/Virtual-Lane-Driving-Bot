"""Driving performance metrics for the virtual lane bot."""

import csv
from dataclasses import dataclass, field
from pathlib import Path


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

    def to_row(self, detector: str, road: str, obstacles: bool = False) -> dict[str, float | int | str | bool]:
        """Return one CSV row for this run."""
        return {
            "detector": detector,
            "road": road,
            "obstacles": obstacles,
            "duration_s": round(self.elapsed, 2),
            "frames": self.frames,
            "avg_lane_error_px": round(self.average_abs_error, 2),
            "max_lane_error_px": round(self.max_abs_error, 2),
            "lane_departures": self.lane_departures,
            "avg_speed": round(self.average_speed, 2),
            "departure_threshold_px": self.departure_threshold_px,
        }


def append_metrics_csv(
    path: str,
    metrics: DrivingMetrics,
    detector: str,
    road: str,
    obstacles: bool = False,
) -> None:
    """Append one benchmark row to a CSV file, creating headers as needed."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True) if output_path.parent != Path(".") else None
    row = metrics.to_row(detector=detector, road=road, obstacles=obstacles)
    write_header = not output_path.exists() or output_path.stat().st_size == 0

    with output_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
