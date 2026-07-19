"""Detect simple red obstacles in the virtual camera frame."""

import cv2
import numpy as np


class ObstacleDetector:
    """Detect red block obstacles rendered in the lane."""

    def detect(self, frame_bgr: np.ndarray) -> dict:
        """Return obstacle bounding box and normalized closeness."""
        height, width = frame_bgr.shape[:2]
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        lower_red_1 = np.array([0, 80, 80], dtype=np.uint8)
        upper_red_1 = np.array([12, 255, 255], dtype=np.uint8)
        lower_red_2 = np.array([170, 80, 80], dtype=np.uint8)
        upper_red_2 = np.array([180, 255, 255], dtype=np.uint8)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, lower_red_1, upper_red_1),
            cv2.inRange(hsv, lower_red_2, upper_red_2),
        )

        roi = np.zeros_like(mask)
        polygon = np.array(
            [[
                (int(width * 0.15), height),
                (int(width * 0.40), int(height * 0.40)),
                (int(width * 0.60), int(height * 0.40)),
                (int(width * 0.85), height),
            ]],
            dtype=np.int32,
        )
        cv2.fillPoly(roi, polygon, 255)
        mask = cv2.bitwise_and(mask, roi)

        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {"detected": False, "bbox": None, "closeness": 0.0, "mask": mask}

        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < 80:
            return {"detected": False, "bbox": None, "closeness": 0.0, "mask": mask}

        x, y, box_w, box_h = cv2.boundingRect(contour)
        bottom = y + box_h
        vertical_closeness = max(0.0, min(1.0, (bottom - height * 0.45) / max(1.0, height * 0.45)))
        size_closeness = max(0.0, min(1.0, box_h / max(1.0, height * 0.16)))
        closeness = max(vertical_closeness, size_closeness)

        return {
            "detected": True,
            "bbox": (x, y, box_w, box_h),
            "closeness": closeness,
            "area": area,
            "mask": mask,
        }
