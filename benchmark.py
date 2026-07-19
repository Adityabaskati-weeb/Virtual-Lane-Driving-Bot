"""Run automated benchmarks across virtual road scenarios."""

import argparse
from pathlib import Path

from control.driver import Driver
from debug.metrics import DrivingMetrics, append_metrics_csv
from simulator.camera import VirtualCamera
from simulator.car import Car
from simulator.road import OBSTACLE_MODES, ROAD_CONDITIONS, ROAD_PROFILES, VirtualRoad
from simulator.world import HeadlessWorld
from vision.lane_detector_advanced import AdvancedLaneDetector
from vision.lane_detector_basic import BasicLaneDetector
from vision.obstacle_detector import ObstacleDetector


def build_detector(name: str):
    """Create the selected lane detector."""
    if name == "basic":
        return BasicLaneDetector()
    return AdvancedLaneDetector()


def is_collision_risk(lane_info: dict, throttle: float) -> bool:
    """Estimate whether the bot failed to slow for a close obstacle in its path."""
    obstacle = lane_info.get("obstacle") or {}
    if not obstacle.get("detected"):
        return False
    closeness = float(obstacle.get("closeness", 0.0))
    lateral_distance = float(lane_info.get("obstacle_lateral_distance_px", 999.0))
    effective_closeness = float(lane_info.get("effective_obstacle_closeness", 0.0))
    return closeness >= 0.92 and effective_closeness >= 0.80 and lateral_distance <= 32.0 and throttle > 0.24


def run_one(
    detector_name: str,
    road_name: str,
    condition: str,
    duration: float,
    departure_threshold: float,
    obstacle_mode: str,
) -> DrivingMetrics:
    """Run one headless benchmark scenario."""
    world = HeadlessWorld()
    road = VirtualRoad(profile=road_name, obstacle_mode=obstacle_mode, condition=condition)
    car = Car()
    camera = VirtualCamera()
    detector = build_detector(detector_name)
    obstacle_detector = ObstacleDetector() if obstacle_mode != "none" else None
    driver = Driver()
    metrics = DrivingMetrics(departure_threshold_px=departure_threshold)

    while metrics.elapsed < duration:
        dt = world.tick()
        frame = camera.capture(road, car)
        lane_info = detector.detect(frame)
        if obstacle_detector is not None:
            lane_info["obstacle"] = obstacle_detector.detect(frame)
        steering, throttle = driver.drive(lane_info, dt)
        lane_info["collision_risk"] = is_collision_risk(lane_info, throttle)
        car.update(steering, throttle, dt)
        metrics.update(
            lane_info.get("error", 0.0),
            car.speed,
            dt,
            throttle=throttle,
            obstacle=lane_info.get("obstacle"),
            braking_pressure=lane_info.get("obstacle_braking_pressure", 0.0),
            collision=lane_info.get("collision_risk", False),
        )

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the virtual lane driving bot.")
    parser.add_argument(
        "--detector",
        choices=("advanced", "basic"),
        default="advanced",
        help="Lane detector pipeline to benchmark.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Seconds to simulate for each road scenario.",
    )
    parser.add_argument(
        "--departure-threshold",
        type=float,
        default=80.0,
        help="Lane error in pixels counted as a lane departure.",
    )
    parser.add_argument(
        "--output",
        default="benchmark_results.csv",
        help="CSV path for benchmark results.",
    )
    parser.add_argument(
        "--roads",
        nargs="+",
        choices=ROAD_PROFILES,
        default=list(ROAD_PROFILES),
        help="Road scenarios to benchmark.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=ROAD_CONDITIONS,
        default=("normal",),
        help="Road visual conditions to benchmark.",
    )
    parser.add_argument(
        "--obstacles",
        action="store_true",
        help="Shortcut for --obstacle-mode single.",
    )
    parser.add_argument(
        "--obstacle-mode",
        choices=OBSTACLE_MODES,
        default="none",
        help="Obstacle scenario to benchmark.",
    )
    return parser.parse_args()


def resolve_obstacle_mode(args: argparse.Namespace) -> str:
    """Resolve legacy obstacle flag and explicit obstacle mode into one value."""
    if args.obstacles and args.obstacle_mode == "none":
        return "single"
    return args.obstacle_mode


def main() -> None:
    args = parse_args()
    obstacle_mode = resolve_obstacle_mode(args)
    obstacles_enabled = obstacle_mode != "none"
    output_path = Path(args.output)
    if output_path.exists():
        output_path.unlink()

    print(
        f"benchmark detector={args.detector} duration={args.duration:.1f}s "
        f"obstacle_mode={obstacle_mode} conditions={','.join(args.conditions)} output={output_path}"
    )
    print("condition,road,avg_error,max_error,departures,collisions,avg_speed,obstacle_detections,braking_ratio")

    for condition in args.conditions:
        for road_name in args.roads:
            metrics = run_one(
                detector_name=args.detector,
                road_name=road_name,
                condition=condition,
                duration=args.duration,
                departure_threshold=args.departure_threshold,
                obstacle_mode=obstacle_mode,
            )
            append_metrics_csv(
                str(output_path),
                metrics,
                detector=args.detector,
                road=road_name,
                obstacles=obstacles_enabled,
                obstacle_mode=obstacle_mode,
                condition=condition,
            )
            print(
                f"{condition},"
                f"{road_name},"
                f"{metrics.average_abs_error:.1f},"
                f"{metrics.max_abs_error:.1f},"
                f"{metrics.lane_departures},"
                f"{metrics.collisions},"
                f"{metrics.average_speed:.2f},"
                f"{metrics.obstacle_detections},"
                f"{metrics.braking_time_ratio:.2f}"
            )

    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
