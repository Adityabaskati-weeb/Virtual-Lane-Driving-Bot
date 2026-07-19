"""Basic lane detector based on Canny edges and Hough lines."""

import cv2
import numpy as np


class BasicLaneDetector:
    """Detect lane position using Canny edges and Hough line segments."""

    def __init__(self, canny_low: int = 60, canny_high: int = 160) -> None:
        self.canny_low = canny_low
        self.canny_high = canny_high

    def detect(self, frame_bgr: np.ndarray) -> dict:
        """Return lane lines, lane center, and steering error in pixels."""
        height, width = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, self.canny_low, self.canny_high)
        masked = self._region_of_interest(edges)

        segments = cv2.HoughLinesP(
            masked,
            rho=1,
            theta=np.pi / 180,
            threshold=35,
            minLineLength=35,
            maxLineGap=80,
        )

        left_points: list[tuple[int, int]] = []
        right_points: list[tuple[int, int]] = []
        if segments is not None:
            for segment in segments[:, 0]:
                x1, y1, x2, y2 = map(int, segment)
                if x2 == x1:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < 0.35:
                    continue
                if slope < 0:
                    left_points.extend([(x1, y1), (x2, y2)])
                else:
                    right_points.extend([(x1, y1), (x2, y2)])

        y_bottom = int(height * 0.92)
        y_top = int(height * 0.55)
        left_line = self._fit_line(left_points, y_bottom, y_top)
        right_line = self._fit_line(right_points, y_bottom, y_top)

        car_center_x = width // 2
        lane_center_x = car_center_x
        if left_line and right_line:
            lane_center_x = (left_line[0] + right_line[0]) // 2
        elif left_line:
            lane_center_x = left_line[0] + width // 4
        elif right_line:
            lane_center_x = right_line[0] - width // 4

        return {
            "left_line": left_line,
            "right_line": right_line,
            "lane_center_x": lane_center_x,
            "car_center_x": car_center_x,
            "error": lane_center_x - car_center_x,
            "edges": edges,
            "mask": masked,
        }

    def _region_of_interest(self, edges: np.ndarray) -> np.ndarray:
        height, width = edges.shape[:2]
        polygon = np.array(
            [[
                (int(width * 0.08), height),
                (int(width * 0.42), int(height * 0.42)),
                (int(width * 0.58), int(height * 0.42)),
                (int(width * 0.92), height),
            ]],
            dtype=np.int32,
        )
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, polygon, 255)
        return cv2.bitwise_and(edges, mask)

    def _fit_line(
        self,
        points: list[tuple[int, int]],
        y_bottom: int,
        y_top: int,
    ) -> tuple[int, int, int, int] | None:
        if len(points) < 2:
            return None
        xs = np.array([point[0] for point in points])
        ys = np.array([point[1] for point in points])
        slope, intercept = np.polyfit(ys, xs, 1)
        x_bottom = int(slope * y_bottom + intercept)
        x_top = int(slope * y_top + intercept)
        return x_bottom, y_bottom, x_top, y_top
