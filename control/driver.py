"""Convert lane detection output into steering and throttle commands."""

from control.pid import PIDController


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class Driver:
    """Lane-following driver using smoothed pixel error and PID steering."""

    def __init__(self, smoothing_alpha: float = 0.28) -> None:
        self.steering_pid = PIDController(kp=0.007, ki=0.00002, kd=0.0015)
        self.smoothing_alpha = smoothing_alpha
        self.smoothed_error: float | None = None

    def drive(self, lane_info: dict, dt: float) -> tuple[float, float]:
        """Return normalized steering and throttle commands."""
        raw_error = float(lane_info.get("error", 0.0))
        obstacle = lane_info.get("obstacle")
        effective_closeness = self._effective_obstacle_closeness(lane_info, obstacle)
        avoidance_error = self._avoidance_error(lane_info, obstacle, effective_closeness)

        lane_info["avoidance_error"] = avoidance_error
        lane_info["effective_obstacle_closeness"] = effective_closeness

        error = raw_error + avoidance_error
        smoothed_error = self._smooth_error(error)
        lane_info["smoothed_error"] = smoothed_error

        steering = self.steering_pid.update(smoothed_error, dt)
        base_throttle = max(0.28, 0.62 - abs(steering) * 0.30)
        throttle = self._apply_obstacle_speed_limit(base_throttle, effective_closeness)
        return steering, throttle

    def reset(self) -> None:
        """Reset controller memory."""
        self.steering_pid.reset()
        self.smoothed_error = None

    def _smooth_error(self, error: float) -> float:
        if self.smoothed_error is None:
            self.smoothed_error = error
        else:
            alpha = self.smoothing_alpha
            self.smoothed_error = alpha * error + (1.0 - alpha) * self.smoothed_error
        return self.smoothed_error

    def _effective_obstacle_closeness(self, lane_info: dict, obstacle: dict | None) -> float:
        if not obstacle or not obstacle.get("detected"):
            lane_info["obstacle_relevance"] = 0.0
            return 0.0

        closeness = float(obstacle.get("closeness", 0.0))
        lane_center_x = float(lane_info.get("lane_center_x", lane_info.get("display_lane_center_x", 0.0)))
        obstacle_center_x = float(obstacle.get("center_x", lane_center_x))
        lateral_distance = abs(obstacle_center_x - lane_center_x)

        relevance = _clamp(1.0 - lateral_distance / 170.0, 0.0, 1.0)
        lane_info["obstacle_relevance"] = relevance
        return closeness * relevance

    def _avoidance_error(self, lane_info: dict, obstacle: dict | None, effective_closeness: float) -> float:
        if effective_closeness < 0.35 or not obstacle or not obstacle.get("detected"):
            return 0.0

        lane_center_x = float(lane_info.get("lane_center_x", lane_info.get("display_lane_center_x", 0.0)))
        obstacle_center_x = float(obstacle.get("center_x", lane_center_x))
        direction = -1.0 if obstacle_center_x >= lane_center_x else 1.0
        return direction * min(45.0, 75.0 * effective_closeness)

    def _apply_obstacle_speed_limit(self, throttle: float, effective_closeness: float) -> float:
        if effective_closeness <= 0.15:
            return throttle

        braking = 0.78 * effective_closeness
        return max(0.12, throttle * (1.0 - braking))
