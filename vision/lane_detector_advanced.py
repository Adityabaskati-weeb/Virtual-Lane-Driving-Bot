"""Advanced lane detector based on perspective transform and sliding windows."""

import cv2
import numpy as np

from vision.perspective import unwarp_birds_eye, warp_birds_eye


class AdvancedLaneDetector:
    """Detect lane boundaries with a bird's-eye sliding-window pipeline."""

    def __init__(self, windows: int = 9, margin: int = 70, min_pixels: int = 35) -> None:
        self.windows = windows
        self.margin = margin
        self.min_pixels = min_pixels
        self.previous_left_fit = None
        self.previous_right_fit = None

    def detect(self, frame_bgr: np.ndarray) -> dict:
        """Return lane center, polynomial fits, and projected lane lines."""
        height, width = frame_bgr.shape[:2]
        warped, _, inverse = warp_birds_eye(frame_bgr)
        binary = self._binary_lane_mask(warped)

        left_fit, right_fit, debug_windows = self._fit_lane_pixels(binary)
        if left_fit is None:
            left_fit = self.previous_left_fit
        if right_fit is None:
            right_fit = self.previous_right_fit
        if left_fit is not None:
            self.previous_left_fit = left_fit
        if right_fit is not None:
            self.previous_right_fit = right_fit

        y_eval = height - 1
        car_center_x = width // 2
        lane_center_x = car_center_x
        left_line = None
        right_line = None
        lane_polygon = None
        lane_overlay = None

        if left_fit is not None and right_fit is not None:
            left_bottom = int(np.polyval(left_fit, y_eval))
            right_bottom = int(np.polyval(right_fit, y_eval))
            lane_center_x = (left_bottom + right_bottom) // 2
            left_line = self._line_from_fit(left_fit, height)
            right_line = self._line_from_fit(right_fit, height)
            lane_polygon, lane_overlay = self._project_lane_area(
                frame_bgr,
                left_fit,
                right_fit,
                inverse,
            )
        elif left_fit is not None:
            left_bottom = int(np.polyval(left_fit, y_eval))
            lane_center_x = left_bottom + int(width * 0.40)
            left_line = self._line_from_fit(left_fit, height)
        elif right_fit is not None:
            right_bottom = int(np.polyval(right_fit, y_eval))
            lane_center_x = right_bottom - int(width * 0.40)
            right_line = self._line_from_fit(right_fit, height)

        return {
            "left_line": left_line,
            "right_line": right_line,
            "left_fit": left_fit,
            "right_fit": right_fit,
            "lane_center_x": lane_center_x,
            "car_center_x": car_center_x,
            "error": lane_center_x - car_center_x,
            "binary": binary,
            "warped": warped,
            "inverse_matrix": inverse,
            "lane_polygon": lane_polygon,
            "lane_overlay": lane_overlay,
            "debug_windows": debug_windows,
        }

    def _binary_lane_mask(self, warped_bgr: np.ndarray) -> np.ndarray:
        """Create a clean binary mask for bright white lane paint."""
        hls = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2HLS)
        white = cv2.inRange(hls, np.array([0, 175, 0]), np.array([180, 255, 90]))

        gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobel = np.absolute(sobel_x)
        scaled = np.uint8(255 * abs_sobel / max(1.0, np.max(abs_sobel)))
        sobel_binary = np.zeros_like(scaled)
        sobel_binary[(scaled >= 35) & (scaled <= 255)] = 255

        binary = cv2.bitwise_or(white, cv2.bitwise_and(sobel_binary, white))
        kernel = np.ones((5, 5), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return binary

    def _fit_lane_pixels(self, binary: np.ndarray):
        height, width = binary.shape[:2]
        histogram = np.sum(binary[height // 2 :, :], axis=0)
        midpoint = width // 2

        left_base = self._histogram_peak(histogram[:midpoint], fallback=midpoint - width // 4)
        right_base = self._histogram_peak(histogram[midpoint:], fallback=width // 4) + midpoint

        window_height = height // self.windows
        nonzero_y, nonzero_x = binary.nonzero()
        left_current = left_base
        right_current = right_base
        left_indices: list[np.ndarray] = []
        right_indices: list[np.ndarray] = []
        debug_windows = []

        for window in range(self.windows):
            win_y_low = height - (window + 1) * window_height
            win_y_high = height - window * window_height
            left_x_low = left_current - self.margin
            left_x_high = left_current + self.margin
            right_x_low = right_current - self.margin
            right_x_high = right_current + self.margin

            debug_windows.append((left_x_low, win_y_low, left_x_high, win_y_high))
            debug_windows.append((right_x_low, win_y_low, right_x_high, win_y_high))

            good_left = (
                (nonzero_y >= win_y_low)
                & (nonzero_y < win_y_high)
                & (nonzero_x >= left_x_low)
                & (nonzero_x < left_x_high)
            ).nonzero()[0]
            good_right = (
                (nonzero_y >= win_y_low)
                & (nonzero_y < win_y_high)
                & (nonzero_x >= right_x_low)
                & (nonzero_x < right_x_high)
            ).nonzero()[0]

            left_indices.append(good_left)
            right_indices.append(good_right)

            if len(good_left) > self.min_pixels:
                left_current = int(np.mean(nonzero_x[good_left]))
            if len(good_right) > self.min_pixels:
                right_current = int(np.mean(nonzero_x[good_right]))

        left_indices_flat = np.concatenate(left_indices) if left_indices else np.array([], dtype=np.int64)
        right_indices_flat = np.concatenate(right_indices) if right_indices else np.array([], dtype=np.int64)

        left_fit = self._polyfit_or_none(nonzero_y[left_indices_flat], nonzero_x[left_indices_flat])
        right_fit = self._polyfit_or_none(nonzero_y[right_indices_flat], nonzero_x[right_indices_flat])
        return left_fit, right_fit, debug_windows

    def _histogram_peak(self, values: np.ndarray, fallback: int) -> int:
        if values.size == 0 or np.max(values) <= 0:
            return int(fallback)
        return int(np.argmax(values))

    def _polyfit_or_none(self, y: np.ndarray, x: np.ndarray):
        if len(x) < 80:
            return None
        return np.polyfit(y, x, 2)

    def _line_from_fit(self, fit: np.ndarray, height: int) -> tuple[int, int, int, int]:
        y_bottom = int(height * 0.92)
        y_top = int(height * 0.52)
        x_bottom = int(np.polyval(fit, y_bottom))
        x_top = int(np.polyval(fit, y_top))
        return x_bottom, y_bottom, x_top, y_top

    def _project_lane_area(self, frame_bgr, left_fit, right_fit, inverse_matrix):
        height, width = frame_bgr.shape[:2]
        plot_y = np.linspace(0, height - 1, height)
        left_x = np.polyval(left_fit, plot_y)
        right_x = np.polyval(right_fit, plot_y)

        left_points = np.array([np.transpose(np.vstack([left_x, plot_y]))])
        right_points = np.array([np.flipud(np.transpose(np.vstack([right_x, plot_y])))])
        polygon = np.hstack((left_points, right_points)).astype(np.int32)

        lane_fill = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.fillPoly(lane_fill, [polygon], (0, 120, 0))
        unwarped_fill = unwarp_birds_eye(lane_fill, inverse_matrix)
        overlay = cv2.addWeighted(frame_bgr, 1.0, unwarped_fill, 0.35, 0)
        return polygon, overlay
