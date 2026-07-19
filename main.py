"""Entry point for the virtual lane driving bot."""

from control.driver import Driver
from debug.visualizer import DebugVisualizer
from simulator.camera import VirtualCamera
from simulator.car import Car
from simulator.road import VirtualRoad
from simulator.world import World
from vision.lane_detector_basic import BasicLaneDetector


def main() -> None:
    """Run the virtual lane-following bot."""
    world = World()
    road = VirtualRoad()
    car = Car()
    camera = VirtualCamera()
    detector = BasicLaneDetector()
    driver = Driver()
    visualizer = DebugVisualizer()

    steering = 0.0
    throttle = 0.0

    try:
        while not world.should_quit():
            dt = world.tick()
            frame = camera.capture(road, car)
            lane_info = detector.detect(frame)
            steering, throttle = driver.drive(lane_info, dt)
            car.update(steering, throttle, dt)
            debug_frame = visualizer.draw(frame, lane_info, steering, throttle)
            world.render(debug_frame, car, steering, throttle, lane_info)
    finally:
        world.close()


if __name__ == "__main__":
    main()
