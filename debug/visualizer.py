"""Draw lane detection, steering, and telemetry debug views."""

import cv2
import numpy as np


class DebugVisualizer:
    """Render perception and control hints over camera frames."""

    def draw(self, frame_bgr, lane_info: dict, steering: float, throttle: float):
        """Return a copy of the frame with lane-following overlays."""
        output = lane_info.get("lane_overlay")
        if output is None:
            output = frame_bgr.copy()
        else:
            output = output.copy()

        height = output.shape[0]
        y_bottom = int(height * 0.86)
        y_top = int(height * 0.56)

        left_line = lane_info.get("left_line")
        right_line = lane_info.get("right_line")
        if left_line:
            cv2.line(output, left_line[:2], left_line[2:], (0, 230, 80), 3)
        if right_line:
            cv2.line(output, right_line[:2], right_line[2:], (0, 230, 80), 3)

        car_center = int(lane_info.get("car_center_x", output.shape[1] // 2))
        lane_center = int(
            lane_info.get(
                "display_lane_center_x",
                lane_info.get("lane_center_x", car_center),
            )
        )

        self._draw_dashed_line(output, car_center, y_top, y_bottom, (255, 90, 0), 2)
        cv2.line(output, (lane_center, y_bottom), (lane_center, y_top), (255, 255, 0), 3)
        cv2.arrowedLine(
            output,
            (car_center, y_bottom - 26),
            (lane_center, y_bottom - 26),
            (0, 180, 255),
            2,
            tipLength=0.18,
        )

        self._draw_obstacle(output, lane_info)
        self._draw_ego_car(output, car_center, steering)

        cv2.putText(
            output,
            f"steer={steering:+.2f} throttle={throttle:.2f}",
            (18, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return output

    def _draw_dashed_line(self, image, x: int, y_top: int, y_bottom: int, color, thickness: int) -> None:
        dash = 16
        gap = 10
        y = y_top
        while y < y_bottom:
            cv2.line(image, (x, y), (x, min(y + dash, y_bottom)), color, thickness)
            y += dash + gap

    def _draw_obstacle(self, image, lane_info: dict) -> None:
        obstacle = lane_info.get("obstacle")
        if not obstacle or not obstacle.get("detected") or obstacle.get("bbox") is None:
            return
        x, y, width, height = obstacle["bbox"]
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 255), 2)
        effective = lane_info.get("effective_obstacle_closeness", 0.0)
        relevance = lane_info.get("obstacle_relevance", 0.0)
        cv2.putText(
            image,
            f"obs {obstacle.get('closeness', 0.0):.2f} eff {effective:.2f} r{relevance:.2f}",
            (x, max(24, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    def _draw_ego_car(self, image, center_x: int, steering: float) -> None:
        height, width = image.shape[:2]
        base_y = height - 8
        hood_top = int(height * 0.76)
        car_half_width = int(width * 0.115)
        nose_half_width = int(width * 0.045)

        hood = np.array(
            [[
                (center_x - car_half_width, base_y),
                (center_x - nose_half_width, hood_top),
                (center_x + nose_half_width, hood_top),
                (center_x + car_half_width, base_y),
            ]],
            dtype=np.int32,
        )
        overlay = image.copy()
        cv2.fillPoly(overlay, hood, (32, 74, 138))
        cv2.addWeighted(overlay, 0.70, image, 0.30, 0, image)
        cv2.polylines(image, hood, True, (235, 235, 235), 2)

        windshield = np.array(
            [[
                (center_x - int(car_half_width * 0.42), hood_top + 8),
                (center_x + int(car_half_width * 0.42), hood_top + 8),
                (center_x + int(car_half_width * 0.24), hood_top + 38),
                (center_x - int(car_half_width * 0.24), hood_top + 38),
            ]],
            dtype=np.int32,
        )
        cv2.fillPoly(image, windshield, (30, 38, 52))
        cv2.polylines(image, windshield, True, (120, 160, 210), 1)

        wheel_y = int(height * 0.93)
        wheel_offset = int(car_half_width * 0.78)
        wheel_angle = int(steering * 22)
        for side in (-1, 1):
            wheel_center = (center_x + side * wheel_offset, wheel_y)
            cv2.ellipse(image, wheel_center, (12, 24), wheel_angle, 0, 360, (18, 18, 18), -1)
            cv2.ellipse(image, wheel_center, (12, 24), wheel_angle, 0, 360, (90, 90, 90), 2)

        cv2.circle(image, (center_x - int(car_half_width * 0.55), base_y - 22), 7, (70, 210, 255), -1)
        cv2.circle(image, (center_x + int(car_half_width * 0.55), base_y - 22), 7, (70, 210, 255), -1)
