"""Advanced lane detector based on perspective transform and sliding windows."""

import cv2
import numpy as np

from vision.perspective import warp_birds_eye


class AdvancedLaneDetector:
    """Bird's-eye histogram detector for curved-lane experiments."""

    def detect(self, frame_bgr: np.ndarray) -> dict:
        """Return lane center from a warped binary road image."""
        warped, _, inverse = warp_birds_eye(frame_bgr)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        scaled = np.uint8(255 * np.absolute(sobel_x) / max(1.0, np.max(np.absolute(sobel_x))))
        binary = np.zeros_like(scaled)
        binary[(scaled >= 30) & (scaled <= 255)] = 255

        height, width = binary.shape[:2]
        histogram = np.sum(binary[height // 2 :, :], axis=0)
        midpoint = width // 2
        left_base = int(np.argmax(histogram[:midpoint])) if np.any(histogram[:midpoint]) else midpoint - width // 4
        right_base = int(np.argmax(histogram[midpoint:]) + midpoint) if np.any(histogram[midpoint:]) else midpoint + width // 4
        lane_center_x = (left_base + right_base) // 2

        return {
            "left_line": None,
            "right_line": None,
            "lane_center_x": lane_center_x,
            "car_center_x": midpoint,
            "error": lane_center_x - midpoint,
            "binary": binary,
            "warped": warped,
            "inverse_matrix": inverse,
        }
