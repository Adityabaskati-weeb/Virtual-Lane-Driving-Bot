"""Entry point for the virtual lane driving bot."""

import argparse

from control.driver import Driver
from debug.visualizer import DebugVisualizer
from simulator.camera import VirtualCamera
from simulator.car import Car
from simulator.road import ROAD_PROFILES, VirtualRoad
from simulator.world import World
from vision.lane_detector_advanced import AdvancedLaneDetector
from vision.lane_detector_basic import BasicLaneDetector


def build_detector(name: str):
    """Create the selected lane detector."""
    if name == "basic":
        return BasicLaneDetector()
    return AdvancedLaneDetector()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the virtual lane driving bot.")
    parser.add_argument(
        "--detector",
        choices=("advanced", "basic"),
        default="advanced",
        help="Lane detector pipeline to use.",
    )
    parser.add_argument(
        "--road",
        choices=ROAD_PROFILES,
        default="s-curve",
        help="Virtual road scenario to run.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the virtual lane-following bot."""
    args = parse_args()
    world = World()
    road = VirtualRoad(profile=args.road)
    car = Car()
    camera = VirtualCamera()
    detector = build_detector(args.detector)
    driver = Driver()
    visualizer = DebugVisualizer()

    steering = 0.0
    throttle = 0.0

    try:
        while not world.should_quit():
            dt = world.tick()
            frame = camera.capture(road, car)
            lane_info = detector.detect(frame)
            lane_info["road"] = road.profile
            steering, throttle = driver.drive(lane_info, dt)
            car.update(steering, throttle, dt)
            debug_frame = visualizer.draw(frame, lane_info, steering, throttle)
            world.render(debug_frame, car, steering, throttle, lane_info)
    finally:
        world.close()


if __name__ == "__main__":
    main()
