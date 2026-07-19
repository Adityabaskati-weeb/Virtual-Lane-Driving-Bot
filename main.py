"""Entry point for the virtual lane driving bot."""

import argparse

from control.driver import Driver
from debug.metrics import DrivingMetrics, append_metrics_csv
from debug.visualizer import DebugVisualizer
from simulator.camera import VirtualCamera
from simulator.car import Car
from simulator.road import ROAD_PROFILES, VirtualRoad
from simulator.world import World
from vision.lane_detector_advanced import AdvancedLaneDetector
from vision.lane_detector_basic import BasicLaneDetector
from vision.obstacle_detector import ObstacleDetector


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
    parser.add_argument(
        "--obstacles",
        action="store_true",
        help="Render and detect red lane obstacles with speed control.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Optional run duration in seconds. Use 0 for no time limit.",
    )
    parser.add_argument(
        "--departure-threshold",
        type=float,
        default=80.0,
        help="Lane error in pixels counted as a lane departure.",
    )
    parser.add_argument(
        "--save-metrics",
        default="",
        help="Optional CSV path to append benchmark metrics after the run.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the virtual lane-following bot."""
    args = parse_args()
    world = World()
    road = VirtualRoad(profile=args.road, obstacles=args.obstacles)
    car = Car()
    camera = VirtualCamera()
    detector = build_detector(args.detector)
    obstacle_detector = ObstacleDetector() if args.obstacles else None
    driver = Driver()
    visualizer = DebugVisualizer()
    metrics = DrivingMetrics(departure_threshold_px=args.departure_threshold)

    steering = 0.0
    throttle = 0.0

    try:
        while not world.should_quit():
            dt = world.tick()
            frame = camera.capture(road, car)
            lane_info = detector.detect(frame)
            lane_info["road"] = road.profile
            if obstacle_detector is not None:
                lane_info["obstacle"] = obstacle_detector.detect(frame)
            steering, throttle = driver.drive(lane_info, dt)
            car.update(steering, throttle, dt)
            metrics.update(lane_info.get("error", 0.0), car.speed, dt)
            lane_info["metrics"] = metrics
            debug_frame = visualizer.draw(frame, lane_info, steering, throttle)
            world.render(debug_frame, car, steering, throttle, lane_info)

            if args.duration > 0 and metrics.elapsed >= args.duration:
                break
    finally:
        world.close()
        print()
        for line in metrics.summary_lines():
            print(line)
        if args.save_metrics:
            append_metrics_csv(args.save_metrics, metrics, detector=args.detector, road=args.road)
            print(f"metrics saved: {args.save_metrics}")


if __name__ == "__main__":
    main()
