"""Advanced lane detector based on perspective transform and robust boundary fitting."""

import cv2
import numpy as np

from vision.perspective import warp_birds_eye


class AdvancedLaneDetector:
    """Detect the active lane and reject center-line artifacts."""

    def __init__(self, windows: int = 9, margin: int = 70, min_pixels: int = 35) -> None:
        self.windows = windows
        self.margin = margin
        self.min_pixels = min_pixels
        self.previous_left_line = None
        self.previous_right_line = None
        self.previous_lane_center_x = None

    def detect(self, frame_bgr: np.ndarray) -> dict:
        """Return lane boundaries, lane center, and debug masks."""
        height, width = frame_bgr.shape[:2]
        warped, _, inverse = warp_birds_eye(frame_bgr)
        binary = self._binary_lane_mask(warped)
        camera_mask = self._camera_white_lane_mask(frame_bgr)

        left_line, right_line = self._detect_outer_camera_boundaries(camera_mask)
        if left_line is None:
            left_line = self.previous_left_line
        if right_line is None:
            right_line = self.previous_right_line
        if left_line is not None:
            self.previous_left_line = left_line
        if right_line is not None:
            self.previous_right_line = right_line

        car_center_x = width // 2
        lane_center_x = self.previous_lane_center_x or car_center_x
        if left_line is not None and right_line is not None:
            lane_center_x = (left_line[0] + right_line[0]) // 2
        elif left_line is not None:
            lane_center_x = left_line[0] + int(width * 0.40)
        elif right_line is not None:
            lane_center_x = right_line[0] - int(width * 0.40)
        self.previous_lane_center_x = lane_center_x

        lane_overlay = self._camera_lane_overlay(frame_bgr, left_line, right_line)

        return {
            "left_line": left_line,
            "right_line": right_line,
            "left_fit": None,
            "right_fit": None,
            "lane_center_x": lane_center_x,
            "display_lane_center_x": lane_center_x,
            "car_center_x": car_center_x,
            "error": lane_center_x - car_center_x,
            "binary": binary,
            "camera_mask": camera_mask,
            "warped": warped,
            "inverse_matrix": inverse,
            "lane_polygon": None,
            "lane_overlay": lane_overlay,
            "debug_windows": [],
        }

    def _binary_lane_mask(self, warped_bgr: np.ndarray) -> np.ndarray:
        """Create a bird's-eye binary mask for lane-paint debugging."""
        hls = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2HLS)
        white = cv2.inRange(hls, np.array([0, 185, 0]), np.array([180, 255, 70]))
        kernel = np.ones((5, 5), dtype=np.uint8)
        return cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel)

    def _camera_white_lane_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Keep only bright white lane paint in the original camera view."""
        hls = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HLS)
        white = cv2.inRange(hls, np.array([0, 190, 0]), np.array([180, 255, 65]))
        roi = self._camera_roi(white)
        kernel = np.ones((5, 5), dtype=np.uint8)
        return cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)

    def _camera_roi(self, mask: np.ndarray) -> np.ndarray:
        height, width = mask.shape[:2]
        polygon = np.array(
            [[
                (int(width * 0.06), height),
                (int(width * 0.36), int(height * 0.40)),
                (int(width * 0.64), int(height * 0.40)),
                (int(width * 0.94), height),
            ]],
            dtype=np.int32,
        )
        roi = np.zeros_like(mask)
        cv2.fillPoly(roi, polygon, 255)
        return cv2.bitwise_and(mask, roi)

    def _detect_outer_camera_boundaries(self, mask: np.ndarray):
        height, width = mask.shape[:2]
        edges = cv2.Canny(mask, 40, 120)
        segments = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=24,
            minLineLength=45,
            maxLineGap=80,
        )
        if segments is None:
            return None, None

        y_bottom = int(height * 0.92)
        y_top = int(height * 0.52)
        left_candidates = []
        right_candidates = []

        for segment in segments[:, 0]:
            x1, y1, x2, y2 = map(int, segment)
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.45:
                continue

            line = self._line_from_segment(x1, y1, x2, y2, y_bottom, y_top)
            if line is None:
                continue
            x_bottom, _, x_top, _ = line
            length = float(np.hypot(x2 - x1, y2 - y1))

            if slope < 0 and x_bottom < width * 0.55 and x_top < width * 0.58:
                # Choose the outer left boundary, not any center marker near the vehicle.
                score = length + max(0.0, width * 0.50 - x_bottom)
                left_candidates.append((line, score))
            elif slope > 0 and x_bottom > width * 0.45 and x_top > width * 0.42:
                # Choose the outer right boundary, not any center marker near the vehicle.
                score = length + max(0.0, x_bottom - width * 0.50)
                right_candidates.append((line, score))

        left_line = max(left_candidates, key=lambda item: item[1])[0] if left_candidates else None
        right_line = max(right_candidates, key=lambda item: item[1])[0] if right_candidates else None
        return left_line, right_line

    def _line_from_segment(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        y_bottom: int,
        y_top: int,
    ) -> tuple[int, int, int, int] | None:
        try:
            slope, intercept = np.polyfit(np.array([y1, y2]), np.array([x1, x2]), 1)
        except np.linalg.LinAlgError:
            return None
        x_bottom = int(slope * y_bottom + intercept)
        x_top = int(slope * y_top + intercept)
        return x_bottom, y_bottom, x_top, y_top

    def _camera_lane_overlay(self, frame_bgr, left_line, right_line):
        if left_line is None or right_line is None:
            return frame_bgr.copy()
        height = frame_bgr.shape[0]
        overlay_bottom = int(height * 0.86)
        left_bottom_x = self._x_at_y(left_line, overlay_bottom)
        right_bottom_x = self._x_at_y(right_line, overlay_bottom)

        overlay = frame_bgr.copy()
        polygon = np.array(
            [[
                (left_bottom_x, overlay_bottom),
                (left_line[2], left_line[3]),
                (right_line[2], right_line[3]),
                (right_bottom_x, overlay_bottom),
            ]],
            dtype=np.int32,
        )
        fill = np.zeros_like(frame_bgr)
        cv2.fillPoly(fill, polygon, (0, 120, 0))
        return cv2.addWeighted(overlay, 1.0, fill, 0.30, 0)

    def _x_at_y(self, line: tuple[int, int, int, int], y: int) -> int:
        x1, y1, x2, y2 = line
        if y2 == y1:
            return x1
        t = (y - y1) / (y2 - y1)
        return int(x1 + t * (x2 - x1))
