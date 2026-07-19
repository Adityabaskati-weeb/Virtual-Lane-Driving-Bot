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

        telemetry = [
            f"speed: {car.speed:05.2f}",
            f"offset: {car.lateral_offset:+.2f}",
            f"steering: {steering:+.2f}",
            f"throttle: {throttle:.2f}",
            f"lane error: {lane_info.get('error', 0):+06.1f}px",
            "ESC/Q: quit",
        ]
        self._draw_panel(telemetry)
        pygame.display.flip()

    def close(self) -> None:
        """Close pygame resources."""
        pygame.quit()

    def _draw_panel(self, lines: list[str]) -> None:
        panel = pygame.Surface((230, 150), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 150))
        self.screen.blit(panel, (16, 16))
        for index, line in enumerate(lines):
            text = self.font.render(line, True, (245, 245, 245))
            self.screen.blit(text, (28, 28 + index * 22))
