# Virtual-Lane-Driving-Bot

A virtual lane-driving bot that combines:

- OpenCV lane detection
- a simulated driving world
- steering and throttle control

The project runs in a lightweight `pygame` window. The bot sees a synthetic road through a virtual camera, detects lane position with OpenCV, and uses PID steering to stay near the lane center.

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

Detector and road options can be combined:

```bash
python main.py --detector advanced --road lane-shift
```

Controls:

- `ESC` or `Q`: quit the simulation

## Project Structure

```text
Virtual-Lane-Driving-Bot/
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
    perspective.py

  control/
    __init__.py
    pid.py
    driver.py

  debug/
    __init__.py
    visualizer.py

  assets/
    .gitkeep
    roads/
      .gitkeep
```

## Module Responsibilities

- `main.py`: connects simulator, camera, lane detector, driver, and debug overlay.
- `simulator/`: virtual road generation, car motion, window rendering, and camera capture.
- `vision/`: OpenCV lane detection modules inspired by the referenced lane detection repos.
- `control/`: PID steering and throttle decisions.
- `debug/`: visual overlays for lane lines, lane center, steering, and telemetry.
- `assets/roads/`: road maps, lane textures, or generated driving scenes.

## Current Driving Loop

```text
Virtual road -> camera frame -> lane detection -> lane error -> PID steering -> car update
```

## Road Scenarios

- `straight`: simple baseline road.
- `left-curve`: constant left bend.
- `right-curve`: constant right bend.
- `s-curve`: alternating left/right curve.
- `lane-shift`: whole lane shifts sideways over time.

## Next Improvements

- Add driving metrics: average lane error, max lane error, lane departures, and time survived.
- Tune steering smoothness and lane-error filtering.
- Add obstacle detection and speed control.
- Later, port the same control loop to CARLA or another 3D simulator.
