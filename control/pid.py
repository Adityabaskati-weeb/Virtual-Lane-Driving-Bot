"""PID controller for steering and speed control."""

from dataclasses import dataclass


@dataclass
class PIDController:
    """Small PID controller with output clamping."""

    kp: float
    ki: float = 0.0
    kd: float = 0.0
    output_min: float = -1.0
    output_max: float = 1.0

    def __post_init__(self) -> None:
        self.integral = 0.0
        self.previous_error: float | None = None

    def update(self, error: float, dt: float) -> float:
        """Compute the next controller output."""
        dt = max(dt, 1e-6)
        self.integral += error * dt
        derivative = 0.0 if self.previous_error is None else (error - self.previous_error) / dt
        self.previous_error = error

        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return max(self.output_min, min(self.output_max, output))

    def reset(self) -> None:
        """Clear accumulated state."""
        self.integral = 0.0
        self.previous_error = None
