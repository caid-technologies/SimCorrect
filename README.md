# SimCorrect

**Autonomous Fault Detection and Correction for Robot Simulation Models**

*Shreya Priya · Dean (Dien) Hu · CAID Technologies*

[[OpenCAD]](https://github.com/caid-technologies/OpenCAD)

---

## What is SimCorrect?

SimCorrect answers a deceptively simple question: **what happens when your simulation model is wrong?**

Not wrong in a noisy, probabilistic way — wrong in a precise, physical way. A forearm link machined 80mm too long. A wrist bracket mounted 150mm off-center. A base encoder with an 8-degree zero offset. A gripper 60 grams heavier than its model says. A joint running with twice its specified friction.

In each case, the robot controller sees nothing wrong. Joint encoders report exactly what they expect. No alarms. No errors. And yet the robot fails its task — every single cycle — because the simulation it was built on does not match the machine it is running on.

SimCorrect detects these faults, identifies the exact parameter responsible, corrects the CAD geometry autonomously, and verifies the fix. No human writes the correction. No hardware is touched. The entire loop — detect, identify, correct, verify — closes in under 0.3 seconds.

---

## How It Works

Two simulation instances run side by side under identical joint commands. The left arm is ground truth — correctly modelled, correctly calibrated. The right arm carries an injected fault. SimCorrect watches the divergence between them.

When the faulty arm fails a task the ground truth arm completes, the pipeline begins. Sensitivity analysis traces the end-effector error back to its source parameter. The OpenCAD API rebuilds the corrected geometry from first principles, exports it, and reloads the simulation. The corrected arm re-executes the task. Success is verified programmatically.

---

## Five Fault Scenarios

### Problem 1 — Forearm Length Error
The right arm forearm link is 80mm longer than its CAD specification. The arm overshoots its target vertically on every pick attempt, closing the gripper above the can. Joint RMSE is zero. SimCorrect detects the vertical end-effector offset, identifies the forearm length as the fault source, and corrects it.

### Problem 2 — Wrist Lateral Offset
The right arm wrist is physically mounted 150mm off-center in the Y axis. The arm executes every joint command with perfect precision — and lands its gripper 250mm from the can on every attempt. Joint RMSE is zero. The fault exists entirely in Cartesian space. SimCorrect detects the lateral drift, corrects the wrist offset parameter, and the arm grasps correctly.

### Problem 3 — Joint Friction Fault
Joint friction is running at more than double its specified value. The arm stalls mid-trajectory, losing positional accuracy under load. Unlike the geometric faults, this one produces non-zero joint RMSE — the joints physically cannot reach their commanded positions. SimCorrect identifies the friction coefficient as the fault source and corrects it.

### Problem 4 — Base Encoder Offset
The base rotation joint has its encoder mounted 8 degrees off its correct zero position. Every trajectory inherits this rotational error. The arm moves with absolute precision relative to what it believes is forward — and misses its target by 103mm because forward is wrong. Joint RMSE is zero. The miss scales with reach: double the extension, double the error. SimCorrect identifies the encoder zero offset and corrects it in 0.28 seconds.

### Problem 5 — Tool Mass Mismatch
The gripper physically weighs 0.16kg but the model records it as 0.10kg. The controller calculates gravity compensation torques based on the wrong mass, sending insufficient torque to hold the arm at height. The arm droops 55mm below its commanded position on every pick attempt. No encoder error. No controller alarm. SimCorrect detects the non-zero joint RMSE, estimates the mass delta, corrects the model, and restores full accuracy.

---

## Fault Coverage

| Problem | Fault | Detection Signal | Joint RMSE |
|---------|-------|-----------------|------------|
| 1 | Forearm +80mm too long | Vertical EE overshoot | 0 |
| 2 | Wrist +150mm lateral offset | Lateral EE drift | 0 |
| 3 | Joint friction 2x specification | Trajectory stall | > 0 |
| 4 | Encoder zero offset +8 degrees | Rotational EE miss | 0 |
| 5 | Tool mass +0.06kg underestimated | Vertical EE droop | > 0 |

Three faults are invisible in joint space. Two are detectable but unidentifiable without SimCorrect. All five are fully corrected autonomously.

---

## Installation

    git clone https://github.com/caid-technologies/SimCorrect.git
    cd SimCorrect
    conda create -n simcorrect python=3.10
    conda activate simcorrect
    pip install mujoco numpy pillow imageio[ffmpeg]
    pip install opencad

---

## Quickstart

    cd Problem2_WristOffset
    python render_demo.py

---

## Repository Structure

    SimCorrect/
    ├── README.md
    ├── Problem1_ForearmLength/
    │   ├── render_demo.py
    │   ├── sim_pair.py
    │   ├── divergence_detector.py
    │   ├── parameter_identifier.py
    │   ├── correction_and_validation.py
    │   ├── README.md
    │   └── output/
    ├── Problem2_WristOffset/
    │   ├── render_demo.py
    │   ├── sim_pair.py
    │   ├── divergence_detector.py
    │   ├── parameter_identifier.py
    │   ├── correction_and_validation.py
    │   ├── README.md
    │   └── output/
    ├── Problem3_JointFriction/
    ├── Problem4_EncoderOffset/
    └── Problem5_ToolMass/

---

## Technical Stack

| Component | Technology |
|-----------|------------|
| Physics engine | MuJoCo 3.x |
| CAD correction | OpenCAD API — CAID Technologies |
| Divergence detection | End-effector trajectory analysis |
| Parameter identification | Sensitivity-based geometric analysis |
| Rendering | MuJoCo offscreen renderer + PIL + ffmpeg |
| Language | Python 3.10+ |

---

## Authors

**Shreya Priya** — Robotics & Autonomy Engineer
Divergence detection, parameter identification, correction pipeline, simulation rendering

**Dean (Dien) Hu** — Founder, CAID Technologies
OpenCAD parametric CAD engine, geometry rebuild, STL/MJCF export

---

## License

MIT License
