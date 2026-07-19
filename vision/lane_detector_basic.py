"""Basic lane detector based on Canny edges and Hough lines."""

import cv2
import numpy as np


class BasicLaneDetector:
    """Detect the outer white lane boundaries from the virtual camera frame."""

    def __init__(self, canny_low: int = 50, canny_high: int = 140) -> None:
        self.canny_low = canny_low
        self.canny_high = canny_high

    def detect(self, frame_bgr: np.ndarray) -> dict:
        """Return lane lines, lane center, and steering error in pixels."""
        height, width = frame_bgr.shape[:2]
        white_mask = self._white_lane_mask(frame_bgr)
        edges = cv2.Canny(white_mask, self.canny_low, self.canny_high)
        masked = self._region_of_interest(edges)

        segments = cv2.HoughLinesP(
            masked,
            rho=1,
            theta=np.pi / 180,
            threshold=28,
            minLineLength=45,
            maxLineGap=70,
        )

        y_bottom = int(height * 0.92)
        y_top = int(height * 0.52)
        left_candidates: list[tuple[tuple[int, int], tuple[int, int], float]] = []
        right_candidates: list[tuple[tuple[int, int], tuple[int, int], float]] = []

        if segments is not None:
            for segment in segments[:, 0]:
                x1, y1, x2, y2 = map(int, segment)
                if x2 == x1:
                    continue

                slope_xy = (y2 - y1) / (x2 - x1)
                if abs(slope_xy) < 0.45:
                    continue

                line = self._line_from_segment(x1, y1, x2, y2, y_bottom, y_top)
                if line is None:
                    continue

                x_bottom, _, x_top, _ = line
                length = float(np.hypot(x2 - x1, y2 - y1))

                if slope_xy < 0 and x_bottom < width * 0.50 and x_top < width * 0.52:
                    left_candidates.append(((x_bottom, y_bottom), (x_top, y_top), length))
                elif slope_xy > 0 and x_bottom > width * 0.50 and x_top > width * 0.48:
                    right_candidates.append(((x_bottom, y_bottom), (x_top, y_top), length))

        left_line = self._weighted_average_line(left_candidates)
        right_line = self._weighted_average_line(right_candidates)

        car_center_x = width // 2
        lane_center_x = car_center_x
        if left_line and right_line:
            lane_center_x = (left_line[0] + right_line[0]) // 2
        elif left_line:
            lane_center_x = left_line[0] + int(width * 0.40)
        elif right_line:
            lane_center_x = right_line[0] - int(width * 0.40)

        return {
            "left_line": left_line,
            "right_line": right_line,
            "lane_center_x": lane_center_x,
            "car_center_x": car_center_x,
            "error": lane_center_x - car_center_x,
            "edges": edges,
            "mask": masked,
            "white_mask": white_mask,
        }

    def _white_lane_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Keep bright white lane paint and reject yellow center markings."""
        hls = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HLS)
        lower_white = np.array([0, 185, 0], dtype=np.uint8)
        upper_white = np.array([180, 255, 80], dtype=np.uint8)
        white_mask = cv2.inRange(hls, lower_white, upper_white)
        kernel = np.ones((5, 5), dtype=np.uint8)
        return cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

    def _region_of_interest(self, edges: np.ndarray) -> np.ndarray:
        height, width = edges.shape[:2]
        polygon = np.array(
            [[
                (int(width * 0.06), height),
                (int(width * 0.36), int(height * 0.40)),
                (int(width * 0.64), int(height * 0.40)),
                (int(width * 0.94), height),
            ]],
            dtype=np.int32,
        )
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, polygon, 255)
        return cv2.bitwise_and(edges, mask)

    def _line_from_segment(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        y_bottom: int,
        y_top: int,
    ) -> tuple[int, int, int, int] | None:
        points = np.array([[y1, x1], [y2, x2]], dtype=np.float32)
        try:
            slope, intercept = np.polyfit(points[:, 0], points[:, 1], 1)
        except np.linalg.LinAlgError:
            return None
        x_bottom = int(slope * y_bottom + intercept)
        x_top = int(slope * y_top + intercept)
        return x_bottom, y_bottom, x_top, y_top

    def _weighted_average_line(
        self,
        candidates: list[tuple[tuple[int, int], tuple[int, int], float]],
    ) -> tuple[int, int, int, int] | None:
        if not candidates:
            return None
        total_weight = sum(candidate[2] for candidate in candidates)
        if total_weight <= 0:
            return None
        x_bottom = int(sum(candidate[0][0] * candidate[2] for candidate in candidates) / total_weight)
        y_bottom = int(sum(candidate[0][1] * candidate[2] for candidate in candidates) / total_weight)
        x_top = int(sum(candidate[1][0] * candidate[2] for candidate in candidates) / total_weight)
        y_top = int(sum(candidate[1][1] * candidate[2] for candidate in candidates) / total_weight)
        return x_bottom, y_bottom, x_top, y_top
