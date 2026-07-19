"""Convert lane detection output into steering and throttle commands."""

from control.pid import PIDController


class Driver:
    """Lane-following driver using smoothed pixel error and PID steering."""

    def __init__(self, smoothing_alpha: float = 0.28) -> None:
        self.steering_pid = PIDController(kp=0.007, ki=0.00002, kd=0.0015)
        self.smoothing_alpha = smoothing_alpha
        self.smoothed_error: float | None = None

    def drive(self, lane_info: dict, dt: float) -> tuple[float, float]:
        """Return normalized steering and throttle commands."""
        error = float(lane_info.get("error", 0.0))
        smoothed_error = self._smooth_error(error)
        lane_info["smoothed_error"] = smoothed_error

        steering = self.steering_pid.update(smoothed_error, dt)
        throttle = max(0.28, 0.62 - abs(steering) * 0.30)
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
