"""Road geometry and lane markings for the simulator."""

from dataclasses import dataclass

import cv2
import numpy as np


ROAD_PROFILES = ("straight", "left-curve", "right-curve", "s-curve", "lane-shift")
OBSTACLE_MODES = ("none", "single", "frequent", "side")
ROAD_CONDITIONS = ("normal", "faded", "noisy", "night", "missing-lanes")


@dataclass(frozen=True)
class RoadConfig:
    """Pixel-space settings for the simulated camera road."""

    lane_width_bottom: int = 260
    lane_width_top: int = 72
    horizon_y_ratio: float = 0.42
    bottom_y_ratio: float = 0.98
    curve_strength: float = 42.0


class VirtualRoad:
    """Generate camera frames with lane markings for OpenCV detection."""

    def __init__(
        self,
        profile: str = "s-curve",
        obstacles: bool = False,
        obstacle_mode: str = "none",
        condition: str = "normal",
        config: RoadConfig | None = None,
    ) -> None:
        if profile not in ROAD_PROFILES:
            raise ValueError(f"Unknown road profile: {profile}")
        if obstacles and obstacle_mode == "none":
            obstacle_mode = "single"
        if obstacle_mode not in OBSTACLE_MODES:
            raise ValueError(f"Unknown obstacle mode: {obstacle_mode}")
        if condition not in ROAD_CONDITIONS:
            raise ValueError(f"Unknown road condition: {condition}")
        self.profile = profile
        self.obstacle_mode = obstacle_mode
        self.condition = condition
        self.obstacles = obstacle_mode != "none"
        self.config = config or RoadConfig()

    def render_camera_frame(self, width: int, height: int, car) -> np.ndarray:
        """Return a BGR frame from the bot camera perspective."""
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        background_color, road_color, lane_color = self._condition_colors()
        frame[:, :] = background_color

        horizon_y = int(height * self.config.horizon_y_ratio)
        bottom_y = int(height * self.config.bottom_y_ratio)
        center_x = width // 2

        offset_px = int(car.lateral_offset * 78)
        heading_px = int(car.heading * 120)
        curve_px, shift_px = self._profile_offsets(car.distance)

        road_top_center = center_x + curve_px + shift_px - heading_px // 2
        road_bottom_center = center_x + shift_px - offset_px - heading_px

        left_bottom = road_bottom_center - self.config.lane_width_bottom // 2
        right_bottom = road_bottom_center + self.config.lane_width_bottom // 2
        left_top = road_top_center - self.config.lane_width_top // 2
        right_top = road_top_center + self.config.lane_width_top // 2

        road_polygon = np.array(
            [[left_bottom, bottom_y], [left_top, horizon_y], [right_top, horizon_y], [right_bottom, bottom_y]],
            dtype=np.int32,
        )
        cv2.fillPoly(frame, [road_polygon], road_color)

        center_color = (50, 210, 255) if self.condition != "night" else (35, 130, 160)
        thickness = 6 if self.condition in ("faded", "night") else 8
        self._draw_lane_boundary(frame, left_bottom, bottom_y, left_top, horizon_y, lane_color, thickness, car.distance, 0)
        self._draw_lane_boundary(frame, right_bottom, bottom_y, right_top, horizon_y, lane_color, thickness, car.distance, 1)
        cv2.line(
            frame,
            (road_bottom_center, bottom_y),
            (road_top_center, horizon_y),
            center_color,
            2,
        )

        marker_color = (210, 210, 210) if self.condition != "night" else (125, 125, 125)
        for y in range(horizon_y + 18, bottom_y, 48):
            t = (y - horizon_y) / max(1, bottom_y - horizon_y)
            marker_x = int(road_top_center * (1 - t) + road_bottom_center * t)
            marker_half = int(4 + 10 * t)
            cv2.line(frame, (marker_x, y), (marker_x, min(bottom_y, y + 22)), marker_color, marker_half)

        if self.obstacles:
            self._draw_obstacle(frame, car.distance, road_top_center, road_bottom_center, horizon_y, bottom_y)

        if self.condition == "noisy":
            frame = self._add_camera_noise(frame, car.distance)

        return frame

    def _profile_offsets(self, distance: float) -> tuple[int, int]:
        """Return top-road curve and whole-road shift offsets in pixels."""
        strength = self.config.curve_strength
        if self.profile == "straight":
            return 0, 0
        if self.profile == "left-curve":
            return int(-strength * 1.20), 0
        if self.profile == "right-curve":
            return int(strength * 1.20), 0
        if self.profile == "lane-shift":
            shift = int(np.sin(distance * 0.035) * 58)
            curve = int(np.sin(distance * 0.022) * strength * 0.35)
            return curve, shift
        curve = int(np.sin(distance * 0.045) * strength)
        return curve, 0

    def _condition_colors(self) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
        if self.condition == "night":
            return (8, 9, 10), (24, 25, 25), (155, 155, 155)
        if self.condition == "faded":
            return (24, 25, 25), (48, 48, 48), (185, 185, 185)
        return (24, 25, 25), (48, 48, 48), (235, 235, 235)

    def _draw_lane_boundary(
        self,
        frame: np.ndarray,
        x_bottom: int,
        y_bottom: int,
        x_top: int,
        y_top: int,
        color: tuple[int, int, int],
        thickness: int,
        distance: float,
        side_index: int,
    ) -> None:
        if self.condition != "missing-lanes":
            cv2.line(frame, (x_bottom, y_bottom), (x_top, y_top), color, thickness)
            return

        segments = 9
        phase = int(distance / 18.0) + side_index * 2
        for index in range(segments):
            if (index + phase) % 5 in (0, 1):
                continue
            t0 = index / segments
            t1 = (index + 0.80) / segments
            start = (int(x_bottom * (1 - t0) + x_top * t0), int(y_bottom * (1 - t0) + y_top * t0))
            end = (int(x_bottom * (1 - t1) + x_top * t1), int(y_bottom * (1 - t1) + y_top * t1))
            cv2.line(frame, start, end, color, thickness)

    def _add_camera_noise(self, frame: np.ndarray, distance: float) -> np.ndarray:
        seed = int(distance * 10) % 100000
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, 12, frame.shape).astype(np.int16)
        noisy = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return cv2.GaussianBlur(noisy, (3, 3), 0)

    def _draw_obstacle(
        self,
        frame: np.ndarray,
        distance: float,
        road_top_center: int,
        road_bottom_center: int,
        horizon_y: int,
        bottom_y: int,
    ) -> None:
        cycle = self._obstacle_cycle()
        progress = (distance % cycle) / cycle
        if progress < self._obstacle_start_progress():
            return

        visible_progress = (progress - self._obstacle_start_progress()) / (1.0 - self._obstacle_start_progress())
        t = 0.18 + 0.72 * visible_progress
        y = int(horizon_y * (1 - t) + bottom_y * t)
        lane_center_x = int(road_top_center * (1 - t) + road_bottom_center * t)
        lane_width = int(self.config.lane_width_top * (1 - t) + self.config.lane_width_bottom * t)
        x = lane_center_x + self._obstacle_lateral_offset(distance, lane_width)
        size = int(14 + 34 * t)

        top_left = (x - size, y - size)
        bottom_right = (x + size, y + size)
        cv2.rectangle(frame, top_left, bottom_right, (30, 30, 220), -1)
        cv2.rectangle(frame, top_left, bottom_right, (255, 255, 255), 2)
        cv2.line(frame, (x - size, y), (x + size, y), (255, 255, 255), 2)

    def _obstacle_cycle(self) -> float:
        if self.obstacle_mode == "frequent":
            return 70.0
        if self.obstacle_mode == "side":
            return 95.0
        return 120.0

    def _obstacle_start_progress(self) -> float:
        if self.obstacle_mode == "frequent":
            return 0.28
        if self.obstacle_mode == "side":
            return 0.35
        return 0.45

    def _obstacle_lateral_offset(self, distance: float, lane_width: int) -> int:
        if self.obstacle_mode != "side":
            return 0
        side = 1.0 if np.sin(distance * 0.045) >= 0.0 else -1.0
        return int(side * lane_width * 0.28)
