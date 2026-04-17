# Problem 2 — Wrist Lateral Offset Fault

> *The fault is invisible in joint space. It only reveals itself when the arm tries to pick something up.*

## The Fault

| Parameter | Ground Truth | Faulty | Error |
|---|---|---|---|
| `wrist_offset_y` | 0.000 m | 0.007 m | +7mm (+7.7%) |
| Joint RMSE (same commands) | — | ~0.000 rad | zero |
| Lateral EE drift | — | 7.0 mm | constant |
| Grasp success | ✓ | ✗ | — |

Both arms receive identical commands and produce identical joint angles. The fault only appears at end-effector level — a constant lateral displacement that causes a miss every attempt. This is what makes it hard to detect with standard joint-space monitors.

## Pipeline
```
sim_pair.py               Phase 1: Dual sim, lateral EE drift measurement
divergence_detector.py    Phase 2: Detect divergence, estimate wrist offset
parameter_identifier.py   Phase 3: Identify wrist_offset_y via sensitivity analysis
correction_and_validation.py   Phase 4+5: OpenCAD correction, closed-loop validation
render_demo.py            Video: fault → OpenCAD → correction → success
```

## Closed Feedback Loop
```python
# Before (open loop — correction computed but never applied)
drift = ee_faulty[y] - ee_gt[y]
correction = -drift
# arm still misses

# After (closed loop)
estimated_offset = observe_lateral_drift() / SENSITIVITY_Y
correction_y = -estimated_offset
target_corrected = target_ee + np.array([0, correction_y, 0])
q_corrected = ik(target_corrected)
arm.set_joints(q_corrected)
```

## Quickstart
```bash
python sim_pair.py
python divergence_detector.py
python parameter_identifier.py
python correction_and_validation.py
python render_demo.py   # → ~/Desktop/Video2_WristOffset.mp4
```

## Expected Output
```
sim_pair.py:
  Lateral EE error — mean: 6.99 mm  max: 7.04 mm
  Joint state RMSE: 0.000011 rad  (geometric fault — not dynamic)
  Estimated wrist offset: 7.36 mm
  Correction delta_y: -7.36 mm  ← fed back to controller

correction_and_validation.py:
  Lateral RMSE before: 6.99 mm
  Lateral RMSE after:  0.41 mm
  Reduction: 94.1%
  Converged: True
  Feedback loop: CLOSED
```

## Honest Limitations

- 3-DOF planar arm — no transferability to real 6-DOF robots without re-validation
- EE positions from MuJoCo kinematics, not a vision/perception frontend
- No real hardware results

## Files

| File | Phase | Purpose |
|---|---|---|
| `sim_pair.py` | 1 | Dual sim, lateral drift, perception-based estimation |
| `divergence_detector.py` | 2 | Sliding lateral RMSE, detection plot |
| `parameter_identifier.py` | 3 | Sensitivity analysis, identifies `wrist_offset_y` |
| `correction_and_validation.py` | 4+5 | OpenCAD correction, closed-loop validation |
| `render_demo.py` | — | 88s demo video |
| `baseline.py` | — | Raw fault baseline |
| `physical_sim.py` | — | Physics verification |
| `real_grasp.py` | — | Per-attempt comparison |
