# Virtual-Lane-Driving-Bot

A virtual lane-driving bot project that will combine:

- OpenCV lane detection
- a simulated driving world
- steering and throttle control

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

- `main.py`: entry point that will connect simulator, vision, and control.
- `simulator/`: virtual road, car movement, world rendering, and camera capture.
- `vision/`: OpenCV lane detection modules based on the referenced lane detection repos.
- `control/`: steering, throttle, and PID driving logic.
- `debug/`: visual overlays for lane lines, lane center, steering, and telemetry.
- `assets/roads/`: road maps, lane textures, or generated driving scenes.
