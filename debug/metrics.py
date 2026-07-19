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
    obstacle_detections: int = 0
    total_obstacle_closeness: float = 0.0
    max_obstacle_closeness: float = 0.0
    braking_frames: int = 0
    min_throttle: float | None = None
    _was_departed: bool = field(default=False, init=False)

    def update(
        self,
        lane_error: float,
        speed: float,
        dt: float,
        throttle: float | None = None,
        obstacle: dict | None = None,
    ) -> None:
        """Record one simulation frame."""
        abs_error = abs(float(lane_error))
        self.elapsed += max(0.0, dt)
        self.frames += 1
        self.total_abs_error += abs_error
        self.max_abs_error = max(self.max_abs_error, abs_error)
        self.total_speed += float(speed)

        if throttle is not None:
            throttle_value = float(throttle)
            self.min_throttle = throttle_value if self.min_throttle is None else min(self.min_throttle, throttle_value)
            if throttle_value < 0.50:
                self.braking_frames += 1

        if obstacle and obstacle.get("detected"):
            closeness = float(obstacle.get("closeness", 0.0))
            self.obstacle_detections += 1
            self.total_obstacle_closeness += closeness
            self.max_obstacle_closeness = max(self.max_obstacle_closeness, closeness)

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

    @property
    def average_obstacle_closeness(self) -> float:
        """Average closeness across frames with a detected obstacle."""
        if not self.obstacle_detections:
            return 0.0
        return self.total_obstacle_closeness / self.obstacle_detections

    @property
    def braking_time_ratio(self) -> float:
        """Fraction of frames where the controller reduced speed."""
        return self.braking_frames / self.frames if self.frames else 0.0

    @property
    def minimum_throttle(self) -> float:
        """Lowest throttle command seen during the run."""
        return self.min_throttle if self.min_throttle is not None else 0.0

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
            f"obstacle detections: {self.obstacle_detections}",
            f"avg obstacle closeness: {self.average_obstacle_closeness:.2f}",
            f"max obstacle closeness: {self.max_obstacle_closeness:.2f}",
            f"braking ratio: {self.braking_time_ratio:.2f}",
            f"min throttle: {self.minimum_throttle:.2f}",
        ]

    def to_row(
        self,
        detector: str,
        road: str,
        obstacles: bool = False,
        obstacle_mode: str = "none",
    ) -> dict[str, float | int | str | bool]:
        """Return one CSV row for this run."""
        return {
            "detector": detector,
            "road": road,
            "obstacles": obstacles,
            "obstacle_mode": obstacle_mode,
            "duration_s": round(self.elapsed, 2),
            "frames": self.frames,
            "avg_lane_error_px": round(self.average_abs_error, 2),
            "max_lane_error_px": round(self.max_abs_error, 2),
            "lane_departures": self.lane_departures,
            "avg_speed": round(self.average_speed, 2),
            "obstacle_detections": self.obstacle_detections,
            "avg_obstacle_closeness": round(self.average_obstacle_closeness, 3),
            "max_obstacle_closeness": round(self.max_obstacle_closeness, 3),
            "braking_ratio": round(self.braking_time_ratio, 3),
            "min_throttle": round(self.minimum_throttle, 3),
            "departure_threshold_px": self.departure_threshold_px,
        }


def append_metrics_csv(
    path: str,
    metrics: DrivingMetrics,
    detector: str,
    road: str,
    obstacles: bool = False,
    obstacle_mode: str = "none",
) -> None:
    """Append one benchmark row to a CSV file, creating headers as needed."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True) if output_path.parent != Path(".") else None
    row = metrics.to_row(detector=detector, road=road, obstacles=obstacles, obstacle_mode=obstacle_mode)
    write_header = not output_path.exists() or output_path.stat().st_size == 0

    with output_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
