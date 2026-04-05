# Problem 1 — Forearm Length Fault

**Fault class:** Kinematic geometry error
**Status:** Complete
**Video:** Video1_CantReach.mp4
**Part of:** [SimCorrect](https://github.com/caid-technologies/SimCorrect)

---

## Overview

First of three fault demonstrations in SimCorrect. A structural geometric mismatch between the robot CAD model and physical hardware. The controller is correct. The motors are correct. The encoders are correct. The robot still fails because the forearm is 37% shorter than the CAD model specifies.

---

## The Fault

    Controller assumes:   reach = base + 0.38 m
    Physical reality:     reach = base + 0.24 m
                          constant error = 0.14 m at every pose

Same joint command. Same Cartesian miss. Every single time.

---

## What You See

- Left arm (silver) — ground truth. Reaches can, grasps, places. Every time.
- Right arm (red) — faulty. Identical commands. Stops 301mm short. Cannot grasp.
- Right arm (teal) — corrected. Reaches can, grasps, places. Task restored.

---

## Fault Signature

| Signal | Ground Truth | Faulty Arm |
|---|---|---|
| Joint commands | Identical | Identical |
| Joint execution | Correct | Correct |
| Joint RMSE | 0 rad | 0 rad |
| EE error at pick | 1.5 mm | 301 mm |
| Grasp success | Yes | No |

Joint space looks perfect. Task space reveals the fault.

---

## Why This Matters

| Cause | Example |
|---|---|
| Wrong part installed | Replacement forearm from different supplier |
| Manufacturing tolerance | Link machined 37% short of spec |
| CAD model drift | Hardware revised, CAD not updated |
| Field substitution | Non-identical component swapped in maintenance |

---

## Detection

    Detection signal:   Cartesian EE divergence
    Miss at pick pose:  301 mm
    Joint RMSE:         0 rad — confirms fault is geometric not dynamic

Zero joint RMSE rules out control errors, actuator faults, and sensor drift in one observation.

---

## Correction

SimCorrect identifies the forearm as the faulty parameter and corrects it through the OpenCAD API.

    from opencad import Part, Sketch
    Part('forearm').extrude(Sketch().circle(r=0.028), depth=0.38).export('forearm.stl')
    sim.reload('forearm.stl')

No human writes the correction. No human touches a file.

---

## Results

| | Reference Arm | Faulty Arm | Corrected Arm |
|---|---|---|---|
| Forearm length | 0.38 m | 0.24 m (-37%) | 0.38 m |
| EE error at pick | 1.5 mm | 301 mm | 1.5 mm |
| Grasp success | Yes | No | Yes |
| Correction time | — | — | 0.28 s |

---

## Verified Joint Configurations

| Config | j1 | j2 | j3 | j4 | EE position |
|---|---|---|---|---|---|
| HOME | 0.000 | 0.800 | -1.800 | -1.079 | (0.313, y, 0.620) |
| HOVER | 0.000 | 0.695 | -0.753 | -1.385 | (0.750, y, 0.400) |
| PICK | 0.000 | 1.075 | -0.943 | -1.231 | (0.749, y, 0.189) |
| LIFT | 0.000 | 0.330 | -0.563 | -1.385 | (0.749, y, 0.601) |
| PLACE | 3.14159 | -0.589 | -0.715 | 1.962 | (-0.750, y, 0.638) |

---

## Technical Specification

| Parameter | Value |
|---|---|
| Ground truth forearm | 0.38 m |
| Faulty forearm | 0.24 m (-37%) |
| EE miss at pick | 301 mm |
| Corrected EE error | 1.5 mm |
| Correction time | 0.28 s |
| Physics engine | MuJoCo 3.x |
| Timestep | 0.002 s |
| Resolution | 1920 x 1080 |
| Duration | 90 s at 30 fps |

---

## Files

| File | Purpose |
|---|---|
| render_demo.py | Main renderer — dual sim, correction, verification |
| sim_pair.py | Dual simulation with trajectory logging |
| divergence_detector.py | Real-time EE divergence detection |
| parameter_identifier.py | Sensitivity analysis and fault isolation |
| correction_and_validation.py | OpenCAD correction and verification |

---

## Run

    cd Problem1_ForearmLength
    python3 render_demo.py

    pip install mujoco numpy pillow imageio[ffmpeg]

---

## Part of SimCorrect

| Problem | Fault | Class | Status |
|---|---|---|---|
| 1 — Forearm Length | Link 37% too short | Kinematic geometry | Complete |
| 2 — Wrist Offset | Lateral offset +7mm | Hidden misalignment | In progress |
| 3 — Joint Friction | Friction 2x nominal | Dynamics error | In progress |

---

## Citation

    @misc{priya2026simcorrect,
      title     = {SimCorrect: Autonomous Geometric Fault Detection and CAD Correction},
      author    = {Priya, Shreya},
      year      = {2026},
      publisher = {CAID Technologies},
      url       = {https://github.com/caid-technologies/SimCorrect}
    }

---

SimCorrect — CAID Technologies
Shreya Priya — Robotics Engineer
