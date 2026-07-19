# Virtual-Lane-Driving-Bot

A virtual lane-driving bot that combines:

- OpenCV lane detection
- obstacle detection
- a simulated driving world
- steering and throttle control
- driving performance metrics

The project runs in a lightweight `pygame` window. The bot sees a synthetic road through a virtual camera, detects lane position with OpenCV, detects red lane obstacles when enabled, smooths the lane error, and uses PID steering plus obstacle-aware throttle control.

## Run Locally

```bash
pip install -r requirements.txt
python main.py
```

The advanced detector and `s-curve` road are used by default. You can compare detector pipelines:

```bash
python main.py --detector advanced
python main.py --detector basic
```

You can run different road scenarios:

```bash
python main.py --road straight
python main.py --road left-curve
python main.py --road right-curve
python main.py --road s-curve
python main.py --road lane-shift
```

Enable red obstacle detection and speed control:

```bash
python main.py --road lane-shift --obstacles
```

Detector and road options can be combined:

```bash
python main.py --detector advanced --road lane-shift --obstacles
```

Run a timed benchmark and print metrics when it ends:

```bash
python main.py --detector advanced --road lane-shift --duration 60
```

Save benchmark metrics to CSV:

```bash
python main.py --detector advanced --road lane-shift --duration 30 --save-metrics results.csv
```

Run all road scenarios headlessly and save one CSV:

```bash
python benchmark.py --duration 30 --output benchmark_results.csv
```

Benchmark with obstacles enabled:

```bash
python benchmark.py --duration 30 --obstacles --output obstacle_benchmark.csv
```

Run selected roads only:

```bash
python benchmark.py --roads straight lane-shift --duration 30
```

Use a custom lane-departure threshold in pixels:

```bash
python main.py --road s-curve --duration 30 --departure-threshold 70
```

Controls:

- `ESC` or `Q`: quit the simulation

## Project Structure

```text
Virtual-Lane-Driving-Bot/
  benchmark.py
  main.py
  requirements.txt
  README.md

  simulator/
    __init__.py
    world.py
    road.py
    car.py
    camera.py

  vision/
    __init__.py
    lane_detector_basic.py
    lane_detector_advanced.py
    obstacle_detector.py
    perspective.py

  control/
    __init__.py
    pid.py
    driver.py

  debug/
    __init__.py
    metrics.py
    visualizer.py

  assets/
    .gitkeep
    roads/
      .gitkeep
```

## Module Responsibilities

- `benchmark.py`: runs headless benchmarks across road scenarios and exports CSV results.
- `main.py`: connects simulator, camera, lane detector, obstacle detector, driver, metrics, and debug overlay.
- `simulator/`: virtual road generation, obstacle rendering, car motion, window rendering, and camera capture.
- `vision/`: OpenCV lane and obstacle detection modules inspired by the referenced lane detection repos.
- `control/`: smoothed lane-error steering, PID control, obstacle-aware speed control, and throttle decisions.
- `debug/`: metrics and visual overlays for lane lines, lane center, obstacles, steering, and telemetry.
- `assets/roads/`: road maps, lane textures, or generated driving scenes.

## Current Driving Loop

```text
Virtual road -> camera frame -> lane/obstacle detection -> smoothed lane error -> PID steering + speed control -> car update -> metrics
```

## Road Scenarios

- `straight`: simple baseline road.
- `left-curve`: constant left bend.
- `right-curve`: constant right bend.
- `s-curve`: alternating left/right curve.
- `lane-shift`: whole lane shifts sideways over time.

## Metrics

The app tracks and can export:

- elapsed time
- frame count
- average absolute lane error
- maximum lane error
- lane departures
- average speed
- detector and road scenario
- obstacle mode

## Next Improvements

- Save screenshots or demo video for the README.
- Add harder road conditions like missing lanes, noisy paint, and night mode.
- Add steering around obstacles instead of speed-only obstacle response.
- Later, port the same control loop to CARLA or another 3D simulator.
