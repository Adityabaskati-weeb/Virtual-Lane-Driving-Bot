"""Virtual world loop and rendering surface."""

import cv2
import pygame


class World:
    """Small pygame window that displays the camera and bot telemetry."""

    def __init__(self, width: int = 960, height: int = 540, fps: int = 30) -> None:
        pygame.init()
        pygame.display.set_caption("Virtual Lane Driving Bot")
        self.width = width
        self.height = height
        self.fps = fps
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18)

    def tick(self) -> float:
        """Limit FPS and return elapsed seconds."""
        return self.clock.tick(self.fps) / 1000.0

    def should_quit(self) -> bool:
        """Process window events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                return True
        return False

    def render(self, frame_bgr, car, steering: float, throttle: float, lane_info: dict) -> None:
        """Draw the camera frame and telemetry."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        surface = pygame.image.frombuffer(frame_rgb.tobytes(), frame_rgb.shape[1::-1], "RGB")
        surface = pygame.transform.smoothscale(surface, (self.width, self.height))
        self.screen.blit(surface, (0, 0))

        obstacle = lane_info.get("obstacle") or {}
        obstacle_status = "none"
        if obstacle.get("detected"):
            obstacle_status = f"{obstacle.get('closeness', 0.0):.2f}"

        telemetry = [
            f"road: {lane_info.get('road', 'unknown')}",
            f"speed: {car.speed:05.2f}",
            f"offset: {car.lateral_offset:+.2f}",
            f"steering: {steering:+.2f}",
            f"throttle: {throttle:.2f}",
            f"obstacle: {obstacle_status}",
            f"lane error: {lane_info.get('error', 0):+06.1f}px",
            f"smooth err: {lane_info.get('smoothed_error', lane_info.get('error', 0)):+06.1f}px",
        ]
        metrics = lane_info.get("metrics")
        if metrics is not None:
            telemetry.extend(
                [
                    f"time: {metrics.elapsed:05.1f}s",
                    f"avg err: {metrics.average_abs_error:05.1f}px",
                    f"max err: {metrics.max_abs_error:05.1f}px",
                    f"departures: {metrics.lane_departures}",
                ]
            )
        telemetry.append("ESC/Q: quit")
        self._draw_panel(telemetry)
        pygame.display.flip()

    def close(self) -> None:
        """Close pygame resources."""
        pygame.quit()

    def _draw_panel(self, lines: list[str]) -> None:
        panel_height = 22 + len(lines) * 22
        panel = pygame.Surface((270, panel_height), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 150))
        self.screen.blit(panel, (16, 16))
        for index, line in enumerate(lines):
            text = self.font.render(line, True, (245, 245, 245))
            self.screen.blit(text, (28, 28 + index * 22))


class HeadlessWorld:
    """No-window world used for fast automated benchmark runs."""

    def __init__(self, fps: int = 30) -> None:
        self.fps = fps

    def tick(self) -> float:
        """Return a fixed simulation timestep without sleeping."""
        return 1.0 / self.fps

    def should_quit(self) -> bool:
        """Headless benchmarks only stop by duration."""
        return False

    def render(self, frame_bgr, car, steering: float, throttle: float, lane_info: dict) -> None:
        """Skip rendering during benchmark runs."""

    def close(self) -> None:
        """No resources to release."""
