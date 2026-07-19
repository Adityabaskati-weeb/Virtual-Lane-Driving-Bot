# Virtual-Lane-Driving-Bot

A virtual lane-driving bot that combines:

- OpenCV lane detection
- obstacle detection
- a simulated driving world
- steering and throttle control
- driving performance metrics
- optional demo video recording

The project runs in a lightweight `pygame` window. The bot sees a synthetic road through a virtual camera, detects lane position with OpenCV, detects red vehicle obstacles when enabled, smooths the lane error, and uses PID steering plus obstacle-aware throttle control. The debug view renders the ego car, lane overlays, and obstacle vehicles so demos look like an actual driving scene.

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

Test harder visual road conditions:

```bash
python main.py --road lane-shift --condition faded
python main.py --road lane-shift --condition noisy
python main.py --road lane-shift --condition night
python main.py --road lane-shift --condition missing-lanes
```

Road conditions:

- `normal`: clear lane paint and normal brightness.
- `faded`: lower-contrast lane paint.
- `noisy`: camera noise and slight blur.
- `night`: darker scene with dimmer lane paint.
- `missing-lanes`: broken outer lane boundaries.

Enable red vehicle obstacle detection and obstacle-aware speed/avoidance control:

```bash
python main.py --road lane-shift --obstacles
```

`--obstacles` is a shortcut for the single-obstacle scenario. You can also choose an obstacle mode directly:

```bash
python main.py --road lane-shift --obstacle-mode single
python main.py --road lane-shift --obstacle-mode frequent
python main.py --road lane-shift --obstacle-mode side
```

Obstacle modes:

- `none`: no obstacles.
- `single`: one centered vehicle obstacle cycles toward the bot.
- `frequent`: centered vehicle obstacles appear more often.
- `side`: vehicle obstacles appear offset toward one side of the lane so the bot can test relevance and avoidance.

Detector, road, obstacle, and condition options can be combined:

```bash
python main.py --detector advanced --road lane-shift --condition noisy --obstacle-mode side
```

Record the debug camera view to an AVI demo file:

```bash
python main.py --road lane-shift --obstacle-mode side --condition noisy --duration 20 --record demo.avi
```

Run a timed benchmark and print metrics when it ends:

```bash
python main.py --detector advanced --road lane-shift --duration 60
```

Save benchmark metrics to CSV:

```bash
python main.py --detector advanced --road lane-shift --duration 30 --save-metrics results.csv
python main.py --detector advanced --road lane-shift --obstacle-mode frequent --duration 30 --save-metrics obstacle_results.csv
```

Run all road scenarios headlessly and save one CSV:

```bash
python benchmark.py --duration 30 --output benchmark_results.csv
```

Benchmark with obstacles enabled:

```bash
python benchmark.py --duration 30 --obstacles --output obstacle_benchmark.csv
python benchmark.py --duration 30 --obstacle-mode frequent --output frequent_obstacle_benchmark.csv
python benchmark.py --duration 30 --obstacle-mode side --output side_obstacle_benchmark.csv
```

Benchmark harder visual conditions:

```bash
python benchmark.py --duration 30 --conditions faded noisy night missing-lanes --output condition_benchmark.csv
python benchmark.py --duration 30 --obstacle-mode side --conditions normal noisy missing-lanes --output robust_obstacle_benchmark.csv
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

- `benchmark.py`: runs headless benchmarks across road scenarios, road conditions, and obstacle modes, then exports CSV results.
- `main.py`: connects simulator, camera, lane detector, obstacle detector, driver, metrics, debug overlay, and optional AVI recording.
- `simulator/`: virtual road generation, harder visual road conditions, vehicle obstacle rendering, car motion, window rendering, and camera capture.
- `vision/`: OpenCV lane and obstacle detection modules inspired by the referenced lane detection repos.
- `control/`: smoothed lane-error steering, PID control, obstacle-aware speed control, avoidance bias, and throttle decisions.
- `debug/`: metrics and visual overlays for lane lines, lane center, ego car, obstacle vehicles, steering, collisions, and telemetry.
- `assets/roads/`: road maps, lane textures, or generated driving scenes.

## Current Driving Loop

```text
Virtual road -> camera frame -> lane/obstacle detection -> smoothed lane error -> PID steering + speed/avoidance control -> car update -> metrics
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
- collision count
- average speed
- detector and road scenario
- road condition
- obstacle mode
- obstacle detection count
- average and maximum obstacle closeness
- braking ratio
- minimum throttle

## Next Improvements

- Add detector comparison reports for `basic` versus `advanced`.
- Add screenshots and benchmark tables to the README.
- Later, port the same control loop to CARLA or another 3D simulator.
