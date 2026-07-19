"""Road geometry and lane markings for the simulator."""

from dataclasses import dataclass

import cv2
import numpy as np


ROAD_PROFILES = ("straight", "left-curve", "right-curve", "s-curve", "lane-shift")


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

    def __init__(self, profile: str = "s-curve", obstacles: bool = False, config: RoadConfig | None = None) -> None:
        if profile not in ROAD_PROFILES:
            raise ValueError(f"Unknown road profile: {profile}")
        self.profile = profile
        self.obstacles = obstacles
        self.config = config or RoadConfig()

    def render_camera_frame(self, width: int, height: int, car) -> np.ndarray:
        """Return a BGR frame from the bot camera perspective."""
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (24, 25, 25)

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
        cv2.fillPoly(frame, [road_polygon], (48, 48, 48))

        lane_color = (235, 235, 235)
        center_color = (50, 210, 255)
        cv2.line(frame, (left_bottom, bottom_y), (left_top, horizon_y), lane_color, 8)
        cv2.line(frame, (right_bottom, bottom_y), (right_top, horizon_y), lane_color, 8)
        cv2.line(
            frame,
            (road_bottom_center, bottom_y),
            (road_top_center, horizon_y),
            center_color,
            2,
        )

        for y in range(horizon_y + 18, bottom_y, 48):
            t = (y - horizon_y) / max(1, bottom_y - horizon_y)
            marker_x = int(road_top_center * (1 - t) + road_bottom_center * t)
            marker_half = int(4 + 10 * t)
            cv2.line(frame, (marker_x, y), (marker_x, min(bottom_y, y + 22)), (210, 210, 210), marker_half)

        if self.obstacles:
            self._draw_obstacle(frame, car.distance, road_top_center, road_bottom_center, horizon_y, bottom_y)

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

    def _draw_obstacle(
        self,
        frame: np.ndarray,
        distance: float,
        road_top_center: int,
        road_bottom_center: int,
        horizon_y: int,
        bottom_y: int,
    ) -> None:
        cycle = 120.0
        progress = (distance % cycle) / cycle
        t = 0.18 + 0.72 * progress
        y = int(horizon_y * (1 - t) + bottom_y * t)
        x = int(road_top_center * (1 - t) + road_bottom_center * t)
        size = int(14 + 34 * t)

        top_left = (x - size, y - size)
        bottom_right = (x + size, y + size)
        cv2.rectangle(frame, top_left, bottom_right, (30, 30, 220), -1)
        cv2.rectangle(frame, top_left, bottom_right, (255, 255, 255), 2)
        cv2.line(frame, (x - size, y), (x + size, y), (255, 255, 255), 2)
