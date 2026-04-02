# SimCorrect: CAD Geometry Fault Detection and Correction for Sim-to-Real Gap Closure

**Shreya Priya, Dean Hu**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![MuJoCo](https://img.shields.io/badge/MuJoCo-3.x-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Active%20Development-orange.svg)

[[Video — Problem 1]](#) &nbsp;|&nbsp; [[OpenCAD]](https://github.com/caid-technologies/OpenCAD)

---

## Overview

SimCorrect is a fully autonomous pipeline for detecting, isolating, and correcting geometric faults in robot CAD models that cause sim-to-real performance gaps. Two simulation instances run side by side under identical joint commands — one with ground-truth geometry, one with an injected geometric fault. SimCorrect identifies the faulty CAD parameter, corrects it programmatically via the OpenCAD API, reloads the simulation, and verifies task success. No human intervention. No real-world hardware. No additional data collection.

---

## Background and Motivation

Simulation is central to modern robot development. Training and validating robot policies in simulation before real-world deployment reduces cost, accelerates iteration, and improves safety. However, the sim-to-real gap — the performance difference between a simulated and physically deployed robot — remains one of the most studied challenges in robotics.

Existing approaches to closing this gap address dynamics discrepancies (friction, mass, contact forces) through domain randomization and system identification, and perceptual discrepancies (visual appearance, lighting, sensor response) through domain adaptation and real-to-sim rendering pipelines.

A third class of gap has received considerably less systematic attention: **the geometric gap** — discrepancies between a robot's CAD model and its physical geometry, arising from manufacturing tolerances, part substitutions, assembly errors, and model drift over time.

The 2025 Annual Review of Control, Robotics, and Autonomous Systems (Aljalbout et al.) identifies this directly: robot simulations based on CAD files "simplify or omit important physical details," and real-world factors including manufacturing tolerances and mechanical backlash "are rarely modeled" and "can cause self-collisions, unstable motions, or failed task execution."

Unlike dynamics gaps, geometric gaps are deterministic. A link that is 37% shorter than its CAD specification produces the same end-effector error at every execution of the same joint command. This determinism makes geometric faults both highly tractable and, without a dedicated correction mechanism, highly damaging — a policy trained against a geometrically incorrect simulation will fail in the real world in a consistent, reproducible way.

SimCorrect addresses this gap directly.

---

## How SimCorrect Works

The pipeline operates in four stages:

### Stage 1 — Behavioral Divergence Detection

Two simulation instances run concurrently under identical joint commands. The reference instance uses ground-truth CAD geometry. The faulty instance carries an injected geometric fault. SimCorrect monitors end-effector trajectories and task outcomes in real time. When the faulty arm fails a task that the reference arm completes, divergence is flagged and the correction pipeline begins.

### Stage 2 — Geometric Parameter Identification

Sensitivity analysis traces the detected divergence to its source CAD parameter. Because geometric faults produce deterministic, reproducible end-effector errors, SimCorrect isolates the exact dimension responsible — link length, joint offset, structural profile — without requiring real-world observations. The fault is a specific incorrect value in a specific CAD file.

### Stage 3 — Autonomous Correction via OpenCAD

The identified parameter is corrected programmatically through the **OpenCAD API** — a modular, service-oriented parametric CAD engine developed by CAID Technologies. OpenCAD rebuilds the affected geometry from first principles, exports the corrected file, and reloads the simulation:
```python
from opencad import Part, Sketch

# Correcting a forearm link 37% too short
Part('forearm').extrude(
    Sketch().circle(r=0.028),
    depth=0.38
).export('forearm.stl')

sim.reload('forearm.stl')
```

No human writes the correction. No human touches a file.

### Stage 4 — Closed-Loop Verification

The corrected simulation re-executes the task under the same joint commands. Success is verified programmatically. The geometric gap is closed.

---

## Key Properties

**Simulation-only operation.** SimCorrect detects and corrects geometric faults entirely within simulation, using behavioral divergence between two simulation instances as the detection signal. Hardware is not required until deployment.

**Deterministic fault isolation.** Geometric faults produce exact, reproducible end-effector errors. SimCorrect exploits this determinism to isolate faults with precision.

**Source-level correction.** Correcting the CAD model propagates automatically to every derived artifact — URDF, MJCF, collision meshes, inertial tensors, motion planning models. One correction fixes the entire simulation stack.

**Zero human intervention.** The full detect-identify-correct-verify loop runs without human input at any stage.

---

## System Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                         SimCorrect                              │
│                                                                 │
│  ┌─────────────────┐   ┌──────────────────┐   ┌─────────────┐  │
│  │   Divergence    │   │   Parameter      │   │  Closed-    │  │
│  │   Detector      │──▶│   Identifier     │──▶│  Loop       │  │
│  │                 │   │                  │   │  Verifier   │  │
│  │ • EE trajectory │   │ • Sensitivity    │   │             │  │
│  │   monitoring    │   │   analysis       │   │ • Task re-  │  │
│  │ • Task outcome  │   │ • Fault          │   │   execution │  │
│  │   classification│   │   isolation      │   │ • Success   │  │
│  │                 │   │ • Parameter ID   │   │   assertion │  │
│  └─────────────────┘   └──────────────────┘   └─────────────┘  │
│                                │                                │
└────────────────────────────────┼────────────────────────────────┘
                                 │
                                 ▼
             ┌────────────────────────────────────┐
             │           OpenCAD API              │
             │       (CAID Technologies)          │
             │                                    │
             │  • Parametric geometry rebuild     │
             │  • STL / MJCF export               │
             │  • Simulation reload               │
             └────────────────────────────────────┘
```

**SimCorrect owns:** fault detection, parameter identification, correction orchestration, verification.
**OpenCAD owns:** geometry representation, parametric rebuild, file export.

---

## Installation
```bash
git clone https://github.com/caid-technologies/SimCorrect.git
cd SimCorrect
conda create -n simcorrect python=3.10
conda activate simcorrect
pip install mujoco numpy pillow imageio[ffmpeg]
pip install opencad
```

---

## Quickstart — Problem 1
```bash
cd Problem1_ForearmLength
python render_demo.py
# Output: ~/Desktop/Video1_CantReach.mp4
```

---

## Demonstration Scenarios

### Problem 1 — Forearm Length Fault ✓ Complete

| | Reference Arm | Faulty Arm | Corrected Arm |
|---|---|---|---|
| Elbow link length | 0.38 m | 0.24 m (−37%) | 0.38 m |
| EE error at PICK config | 1.5 mm | 140 mm | 1.5 mm |
| Grasp success | ✓ | ✗ | ✓ |

Both arms execute identical joint commands. The faulty arm's end-effector falls 14 cm short of the target due to the shortened forearm. SimCorrect detects the grasp failure, isolates the elbow link as the fault source, corrects the geometry via OpenCAD, reloads the simulation. The corrected arm succeeds.

### Problem 2 — Wrist Offset Fault *(in progress)*

| | Reference Arm | Faulty Arm |
|---|---|---|
| Wrist lateral offset | 0.000 m | 0.007 m (+7.7%) |
| Lateral EE error | ~0 mm | ~7 mm |
| Joint RMSE | ~0 rad | ~0 rad (fault invisible in joint space) |
| Grasp success | ✓ | ✗ |

### Problem 3 — Joint Friction Fault *(in progress)*

| | Reference Arm | Faulty Arm |
|---|---|---|
| Joint friction coefficients | Nominal | +112% above specification |
| Trajectory completion | Full | Stall before pick |

---

## Limitations

**Single-fault assumption.** The current pipeline assumes one geometric fault is present at a time. Multi-fault scenarios require extension of the sensitivity analysis to handle coupled parameter interactions.

**Simulation-internal detection.** Divergence detection currently operates between two simulation instances. Extending to real-world sensor data is a necessary step toward full deployment.

**Manipulation-specific demonstrations.** The three demonstration scenarios focus on robot arm pick-and-place tasks. Generalization to other robot morphologies and task types has not yet been validated.

**Fault type coverage.** The current framework targets geometric parameter faults. Other sources of the reality gap — sensor noise models, actuator dynamics, environmental contact properties — are outside the current scope.

---

## Technical Stack

| Component | Technology |
|---|---|
| Physics engine | MuJoCo 3.x |
| CAD correction engine | OpenCAD API — CAID Technologies |
| Divergence detection | End-effector trajectory analysis |
| Parameter identification | Sensitivity-based geometric analysis |
| Visualization | MuJoCo offscreen renderer |
| Rendering pipeline | PIL, imageio / ffmpeg |
| Language | Python 3.10+ |

---

## Repository Structure
```
SimCorrect/
├── README.md
├── Problem1_ForearmLength/
│   ├── sim_pair.py
│   ├── divergence_detector.py
│   ├── parameter_identifier.py
│   ├── correction_and_validation.py
│   ├── render_demo.py
│   └── README.md
├── Problem2_WristOffset/
│   ├── sim_pair.py
│   ├── divergence_detector.py
│   ├── parameter_identifier.py
│   ├── correction_and_validation.py
│   ├── render_demo.py
│   └── README.md
└── Problem3_JointFriction/
    └── README.md
```

---

## Citation
```bibtex
@misc{priya2026simcorrect,
  title     = {SimCorrect: Autonomous Geometric Fault Detection and CAD Correction
               for Sim-to-Real Gap Closure in Robot Manipulation},
  author    = {Priya, Shreya and Hu, Dean},
  year      = {2026},
  publisher = {CAID Technologies},
  url       = {https://github.com/caid-technologies/SimCorrect}
}
```

---

## Authors

**Shreya Priya** — Robotics & Autonomy Engineer
Divergence detection, parameter identification, correction loop, simulation pipeline

**Dean Hu** — Founder, CAID Technologies
OpenCAD parametric CAD engine, geometry rebuild, STL export pipeline

---

## License

This project is licensed under the MIT License.
