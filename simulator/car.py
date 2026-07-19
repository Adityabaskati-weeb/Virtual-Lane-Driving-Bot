"""Bot vehicle state and motion model."""

from dataclasses import dataclass


@dataclass
class Car:
    """Simple lane-following vehicle model used by the simulator."""

    lateral_offset: float = 0.0
    heading: float = 0.0
    speed: float = 0.0
    distance: float = 0.0
    max_speed: float = 18.0
    max_lateral_offset: float = 2.2

    def update(self, steering: float, throttle: float, dt: float) -> None:
        """Advance the vehicle state from normalized steering and throttle."""
        steering = max(-1.0, min(1.0, steering))
        throttle = max(0.0, min(1.0, throttle))

        target_speed = throttle * self.max_speed
        self.speed += (target_speed - self.speed) * min(1.0, dt * 2.5)
        self.heading += (steering * 0.55 - self.heading) * min(1.0, dt * 4.0)
        self.lateral_offset += self.heading * self.speed * dt * 0.45
        self.lateral_offset = max(
            -self.max_lateral_offset,
            min(self.max_lateral_offset, self.lateral_offset),
        )
        self.distance += self.speed * dt

    def reset(self) -> None:
        """Place the bot back near the center of the lane."""
        self.lateral_offset = 0.0
        self.heading = 0.0
        self.speed = 0.0
        self.distance = 0.0
