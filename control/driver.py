"""Convert lane detection output into steering and throttle commands."""

from control.pid import PIDController


class Driver:
    """Lane-following driver using pixel error and PID steering."""

    def __init__(self) -> None:
        self.steering_pid = PIDController(kp=0.007, ki=0.00002, kd=0.0015)

    def drive(self, lane_info: dict, dt: float) -> tuple[float, float]:
        """Return normalized steering and throttle commands."""
        error = float(lane_info.get("error", 0.0))
        steering = self.steering_pid.update(error, dt)
        throttle = max(0.28, 0.62 - abs(steering) * 0.30)
        return steering, throttle

    def reset(self) -> None:
        """Reset controller memory."""
        self.steering_pid.reset()
