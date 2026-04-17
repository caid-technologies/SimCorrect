# SimCorrect

## Autonomous Fault Detection and Correction for Robot Simulation Models

Shreya Priya · Dean (Dien) Hu

---

## What is SimCorrect?

SimCorrect answers a deceptively simple question: what happens when your simulation model is wrong?

Not wrong in a noisy, probabilistic way. Wrong in a precise, physical way. A forearm link machined 80mm too long. A wrist bracket mounted 150mm off-center. A base encoder with an 8 degree zero offset. A gripper 60 grams heavier than its model says. A joint running with twice its specified friction.

In each case, the robot controller sees nothing wrong. Joint encoders report exactly what they expect. No alarms. No errors. And yet the robot fails its task every single cycle, because the simulation it was built on does not match the machine it is running on.

SimCorrect detects these faults, identifies the exact parameter responsible, corrects the model autonomously, and verifies the fix before the robot is deployed. No human writes the correction. No hardware is touched. The entire loop from detection through correction through validation closes in under 0.3 seconds.

---

## How It Works

Two simulation instances run side by side under identical joint commands. The left arm is ground truth, correctly modelled and correctly calibrated. The right arm carries an injected fault. SimCorrect monitors the divergence between them continuously.

When the faulty arm fails a task that the ground truth arm completes, the diagnostic pipeline begins. It measures the end-effector error, computes joint RMSE, and traces the divergence back to its source parameter through sensitivity analysis. The OpenCAD API then rebuilds the corrected geometry from first principles, exports the corrected MJCF, and reloads the simulation. The corrected arm re-executes the task. Success is verified programmatically before the result is accepted.

The same pipeline handles both geometric faults, where joints execute correctly but the geometry is wrong, and dynamics faults, where joints physically cannot reach their commanded positions because the physics model is wrong.

---

## The Five Fault Scenarios

### Problem 1 — Forearm Length Error

The right arm forearm link is 80mm longer than its CAD specification. The arm overshoots its target on every pick attempt, closing the gripper above the can rather than around it. The joint encoders report normal values because the joints are doing exactly what they are told. The fault lives in geometry, not in motion. SimCorrect detects the vertical end-effector overshoot, isolates the forearm length as the responsible parameter, and corrects the link geometry via OpenCAD.

### Problem 2 — Wrist Lateral Offset

The right arm wrist bracket is physically mounted 150mm off-center in the lateral axis. The arm executes every joint command with perfect precision and lands its gripper 250mm away from the can on every attempt. Joint RMSE is zero throughout. The fault exists entirely in Cartesian space and is invisible to any joint-level diagnostic. SimCorrect detects the consistent lateral drift, identifies the wrist offset parameter, corrects it, and the arm grasps correctly on the next attempt.

### Problem 3 — Joint Friction Fault

Joint friction is running at more than double its specified value. The arm stalls mid-trajectory and loses positional accuracy under load. Unlike the geometric faults, this one produces non-zero joint RMSE. The joints physically cannot reach their commanded positions because friction is consuming torque that was meant for motion. SimCorrect detects the velocity-dependent joint lag, identifies the friction coefficient as the fault source, and corrects it.

### Problem 4 — Base Encoder Offset

The base rotation joint has its encoder mounted 8 degrees off its correct zero position. Every trajectory the arm executes inherits this rotational error at the root. The arm moves with absolute precision relative to what it believes is forward, and misses its target by 103mm because its definition of forward is wrong. Joint RMSE is zero. The positional error scales linearly with reach distance. SimCorrect detects the rotational signature, identifies the encoder zero offset, and corrects the joint reference in 0.28 seconds.

### Problem 5 — Tool Mass Mismatch

The gripper physically weighs 0.160kg but the simulation model records it as 0.100kg. The controller calculates gravity compensation torques based on the modelled mass, sending insufficient torque to hold the arm against the real gravitational load. At full horizontal extension, the uncompensated torque is 0.44 Nm. The arm droops 55mm below its commanded position on every pick attempt. No encoder error is generated. No controller alarm fires. The robot just quietly misses, every cycle. SimCorrect detects the non-zero joint RMSE at extended poses, confirms the gravity-dependent signature that distinguishes mass mismatch from friction, estimates the mass delta analytically, and corrects the model via OpenCAD.

---

## Fault Coverage

Problem 1 detects forearm length error through vertical end-effector overshoot with zero joint RMSE and corrects in 0.28 seconds. Problem 2 detects wrist lateral offset through lateral end-effector drift with zero joint RMSE and corrects in 0.28 seconds. Problem 3 detects joint friction excess through velocity-dependent joint lag with non-zero joint RMSE and corrects in 0.28 seconds. Problem 4 detects encoder zero offset through rotational end-effector miss with zero joint RMSE and corrects in 0.28 seconds. Problem 5 detects tool mass mismatch through gravity-dependent joint droop with non-zero joint RMSE and corrects in 0.28 seconds.

Three of the five faults are completely invisible in joint space and undetectable by any onboard diagnostic. Two are detectable as joint errors but unidentifiable without the paired simulation approach. All five are fully corrected autonomously with no human intervention.

---

## OpenCAD

OpenCAD is the correction engine at the core of SimCorrect. It provides a programmatic interface for modifying simulation model parameters directly from fault identification results.

```python
from opencad import Part
Part('grip').set_mass(0.160).export('grip_corrected.xml')
Part('joint1').set_ref(0.0000).export('joint1_corrected.xml')
```

The correction record is written to an XML file, the simulation reloads the corrected model, and the validation pipeline confirms the fault is eliminated before the result is accepted.

---

## Installation

```bash
git clone https://github.com/caid-technologies/SimCorrect.git
cd SimCorrect
conda create -n simcorrect python=3.10
conda activate simcorrect
pip install mujoco numpy pillow imageio[ffmpeg]
```

---

## Quickstart

```bash
cd Problem5_ToolMassMismatch
python step0.py
python demo.py
python render_demo.py
```

---

## Repository Structure

Each problem folder follows the same structure. The opencad.py module lives at the repository root and is shared across all five problems.
SimCorrect/
├── opencad.py
├── Problem1_ForearmLength/
│   ├── render_demo.py
│   ├── sim_pair.py
│   ├── divergence_detector.py
│   ├── parameter_identifier.py
│   ├── correction_and_validation.py
│   ├── demo.py
│   ├── step0.py
│   ├── README.md
│   └── output/
├── Problem2_WristOffset/
├── Problem3_JointFriction/
├── Problem4_JointZeroOffset/
└── Problem5_ToolMassMismatch/
The render_demo.py file in each folder produces an 88-second video demonstrating the fault, the divergence, the correction, and the validation. The correction_and_validation.py file runs the full automated pipeline with assertions and prints a pass or fail result.

---

## Technical Stack

The physics simulation uses MuJoCo 3.x. CAD correction uses the OpenCAD API. Divergence detection uses end-effector trajectory analysis combined with joint RMSE monitoring. Parameter identification uses sensitivity-based geometric and dynamic analysis. Rendering uses the MuJoCo offscreen renderer with PIL and ffmpeg. The codebase is Python 3.10.

---

## Authors

Shreya Priya is a Robotics and Autonomy Engineer. Her contributions include the divergence detection architecture, parameter identification algorithms, the correction and validation pipeline, and the simulation rendering system across all five problems.

Dean (Dien) Hu is the founder of the project. His contributions include the OpenCAD parametric correction engine, the geometry rebuild system, and the MJCF export pipeline.

---

## License

MIT License
