# Problem 3 — Joint Friction Fault

> *The arm looks perfect at rest. The fault only appears when it moves.*

## What Goes Wrong

The robot's geometry is completely correct. Both arms are identical in shape.
The fault is purely dynamic: joint friction and damping have degraded to
**+112% above nominal** due to six real-world causes accumulating over time.

| Parameter | Nominal | Faulty |
|---|---|---|
| `frictionloss` | 0.5 Nm | 1.06 Nm (+112%) |
| `damping` | 8.0 Ns/m | 16.96 Ns/m (+112%) |

The controller sends the same torques to both arms. The degraded joints burn
most of that torque overcoming resistance instead of producing motion. The
result: the arm starts normally, visibly slows, and eventually stalls
completely — unable to reach the can.

## The Six Degradation Causes

| Cause | Rate | Real-World Mechanism |
|---|---|---|
| Wear | 100% | Joint surfaces degrade with use — metal-on-metal contact |
| Lubrication | 85% | Grease dries out, oil breaks down, seals age |
| Contamination | 70% | Dust, debris, metal shavings enter joints |
| Corrosion | 55% | Moisture causes rust in joint surfaces |
| Thermal | 75% | Grease thickens in cold, seals swell in heat |
| Seal Aging | 90% | O-rings and lip seals harden, increasing drag |

All six build progressively. The live Joint Degradation Monitor in the
video shows each factor filling in real time, with the combined friction
value updating live.

## Why It Is Hard to Catch

Unlike Problems 1 and 2 (geometry faults visible on day 1), friction
degradation is:
- Invisible at rest — arm looks normal and passes static checks
- Gradual — performance degrades slowly over months
- Often misdiagnosed — engineers blame the controller or path planner
- Only visible dynamically — under fast or loaded trajectories

## Three-Stage Visual Degradation

| Stage | Degradation | Arm Behaviour |
|---|---|---|
| 1 | 0-30% | Moves normally — bars filling on monitor |
| 2 | 30-60% | Visibly slowing — arm falls behind reference |
| 3 | 60%+ | Completely frozen — stalled mid-reach |

## Detection

Signal: sliding-window joint-velocity RMSE.

Velocity error appears immediately when the arm starts lagging.
Position error accumulates slowly. Velocity is the right early-warning signal.

    E(t) = sqrt( mean_joints( (qd_nom - qd_flt)^2 ) )
    alarm when sliding_window_mean(E) > 0.08 rad/s

Detection latency: < 0.15 s from motion start.

## Identification

Sensitivity analysis with cosine similarity. Four candidates perturbed +10%:

| Candidate | Cosine Similarity |
|---|---|
| friction_loss | ~0.97 — winner |
| damping | ~0.91 |
| joint_stiffness | ~0.43 |
| link_length | ~0.18 |

link_length scores low because geometry faults produce spatially-structured
divergence patterns, not velocity-scaled ones. This rules out misdiagnosis.

## Correction

Reset frictionloss and damping to nominal. Reload simulation.
The corrected arm tracking alpha returns to 1.0 — full speed, full reach.

This correction operates on behavioral model parameters, not geometry —
proving SimCorrect handles both structural and dynamic fault classes.

## Files

| File | Role |
|---|---|
| sim_pair.py | Nominal vs faulty arm; progressive degradation model |
| divergence_detector.py | Sliding-window velocity RMSE alarm |
| parameter_identifier.py | Cosine similarity; rules out geometry faults |
| correction_and_validation.py | Reset parameters; validate tracking restored |
| render_demo.py | Full 88s video with live degradation monitor |

## Run

    cd Problem3_JointFriction
    python3 render_demo.py
    python3 sim_pair.py
    python3 divergence_detector.py
    python3 parameter_identifier.py
    python3 correction_and_validation.py

## Relation to Problems 1 and 2

| | Problem 1 | Problem 2 | Problem 3 |
|---|---|---|---|
| Fault type | Link length -37% | Wrist offset +150mm | Friction +112% |
| Visible at rest | Yes | Subtle | No |
| Detection signal | Position | Position | Velocity |
| Correction target | CAD geometry | CAD geometry | Dynamic params |
| Failure mode | Cannot reach | Lateral miss | Stall |
