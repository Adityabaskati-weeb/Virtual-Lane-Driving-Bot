"""Virtual camera frame capture from the bot perspective."""

from dataclasses import dataclass


@dataclass(frozen=True)
class VirtualCamera:
    """Camera configuration for synthetic road frames."""

    width: int = 640
    height: int = 360

    def capture(self, road, car):
        """Capture a BGR image from the simulated road."""
        return road.render_camera_frame(self.width, self.height, car)
