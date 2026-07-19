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
        car_width = int(width * 0.13)
        car_height = int(height * 0.18)
        base_y = height - 30
        top_y = base_y - car_height
        left_x = center_x - car_width // 2
        right_x = center_x + car_width // 2

        shadow = np.array(
            [[
                (left_x - 10, base_y + 8),
                (right_x + 10, base_y + 8),
                (right_x - 6, top_y + 30),
                (left_x + 6, top_y + 30),
            ]],
            dtype=np.int32,
        )
        shadow_layer = image.copy()
        cv2.fillPoly(shadow_layer, shadow, (0, 0, 0))
        cv2.addWeighted(shadow_layer, 0.25, image, 0.75, 0, image)

        body = np.array(
            [[
                (left_x, base_y),
                (left_x + int(car_width * 0.18), top_y + int(car_height * 0.22)),
                (center_x - int(car_width * 0.20), top_y),
                (center_x + int(car_width * 0.20), top_y),
                (right_x - int(car_width * 0.18), top_y + int(car_height * 0.22)),
                (right_x, base_y),
            ]],
            dtype=np.int32,
        )
        overlay = image.copy()
        cv2.fillPoly(overlay, body, (38, 92, 165))
        cv2.addWeighted(overlay, 0.86, image, 0.14, 0, image)
        cv2.polylines(image, body, True, (230, 238, 245), 2)

        cabin = np.array(
            [[
                (center_x - int(car_width * 0.28), top_y + int(car_height * 0.25)),
                (center_x + int(car_width * 0.28), top_y + int(car_height * 0.25)),
                (center_x + int(car_width * 0.20), top_y + int(car_height * 0.62)),
                (center_x - int(car_width * 0.20), top_y + int(car_height * 0.62)),
            ]],
            dtype=np.int32,
        )
        cv2.fillPoly(image, cabin, (28, 38, 54))
        cv2.polylines(image, cabin, True, (125, 170, 220), 1)

        hood_line_y = top_y + int(car_height * 0.72)
        cv2.line(image, (left_x + 12, hood_line_y), (right_x - 12, hood_line_y), (74, 130, 205), 2)
        cv2.circle(image, (left_x + 18, base_y - 16), 5, (70, 215, 255), -1)
        cv2.circle(image, (right_x - 18, base_y - 16), 5, (70, 215, 255), -1)

        wheel_y = base_y - int(car_height * 0.18)
        wheel_angle = int(steering * 24)
        for x in (left_x + 6, right_x - 6):
            cv2.ellipse(image, (x, wheel_y), (8, 18), wheel_angle, 0, 360, (18, 18, 18), -1)
            cv2.ellipse(image, (x, wheel_y), (8, 18), wheel_angle, 0, 360, (95, 95, 95), 1)
