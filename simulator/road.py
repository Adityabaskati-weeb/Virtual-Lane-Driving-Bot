"""Road geometry and lane markings for the simulator."""

from dataclasses import dataclass

import cv2
import numpy as np


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

    def __init__(self, config: RoadConfig | None = None) -> None:
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
        curve_px = int(np.sin(car.distance * 0.045) * self.config.curve_strength)

        road_top_center = center_x + curve_px - heading_px // 2
        road_bottom_center = center_x - offset_px - heading_px

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

        return frame
