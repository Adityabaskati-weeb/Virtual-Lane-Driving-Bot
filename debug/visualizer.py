"""Draw lane detection, steering, and telemetry debug views."""

import cv2


class DebugVisualizer:
    """Render perception and control hints over camera frames."""

    def draw(self, frame_bgr, lane_info: dict, steering: float, throttle: float):
        """Return a copy of the frame with lane-following overlays."""
        output = frame_bgr.copy()
        height = output.shape[0]
        y_bottom = int(height * 0.92)
        y_top = int(height * 0.55)

        left_line = lane_info.get("left_line")
        right_line = lane_info.get("right_line")
        if left_line:
            cv2.line(output, left_line[:2], left_line[2:], (0, 255, 0), 5)
        if right_line:
            cv2.line(output, right_line[:2], right_line[2:], (0, 255, 0), 5)

        car_center = int(lane_info.get("car_center_x", output.shape[1] // 2))
        lane_center = int(lane_info.get("lane_center_x", car_center))
        cv2.line(output, (car_center, y_bottom), (car_center, y_top), (255, 0, 0), 2)
        cv2.line(output, (lane_center, y_bottom), (lane_center, y_top), (0, 255, 255), 2)
        cv2.arrowedLine(
            output,
            (car_center, y_bottom - 28),
            (lane_center, y_bottom - 28),
            (0, 180, 255),
            2,
            tipLength=0.18,
        )

        cv2.putText(
            output,
            f"steer={steering:+.2f} throttle={throttle:.2f}",
            (18, height - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return output
