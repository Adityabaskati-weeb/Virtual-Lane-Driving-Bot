"""Perspective transform helpers for bird's-eye lane detection."""

import cv2
import numpy as np


def default_perspective_points(width: int, height: int):
    """Return source and destination points for a road-like trapezoid."""
    src = np.float32(
        [
            [width * 0.42, height * 0.45],
            [width * 0.58, height * 0.45],
            [width * 0.90, height * 0.95],
            [width * 0.10, height * 0.95],
        ]
    )
    dst = np.float32(
        [
            [width * 0.25, 0],
            [width * 0.75, 0],
            [width * 0.75, height],
            [width * 0.25, height],
        ]
    )
    return src, dst


def warp_birds_eye(frame_bgr: np.ndarray):
    """Warp a camera frame into a simple bird's-eye view."""
    height, width = frame_bgr.shape[:2]
    src, dst = default_perspective_points(width, height)
    matrix = cv2.getPerspectiveTransform(src, dst)
    inverse = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(frame_bgr, matrix, (width, height), flags=cv2.INTER_LINEAR)
    return warped, matrix, inverse


def unwarp_birds_eye(frame_bgr: np.ndarray, inverse_matrix: np.ndarray):
    """Project a bird's-eye frame back into the camera view."""
    height, width = frame_bgr.shape[:2]
    return cv2.warpPerspective(frame_bgr, inverse_matrix, (width, height), flags=cv2.INTER_LINEAR)
