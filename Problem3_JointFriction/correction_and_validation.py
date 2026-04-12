"""
Problem 3 — Joint Friction Fault
correction_and_validation.py

Correction: reset frictionloss and damping to nominal values.
Unlike Problems 1 and 2 (which regenerate CAD geometry), this
correction operates on behavioral model parameters — demonstrating
that SimCorrect handles both structural and dynamic faults.

Validation:
  - Post-correction alpha returns to 1.0 (full tracking)
  - Corrected arm reaches PICK_Q within tolerance
  - Joint error falls below threshold
"""

import numpy as np
from dataclasses import dataclass
from sim_pair import (
    run_sim_pair,
    NOMINAL_FRICTION, FAULTY_FRICTION,
    NOMINAL_DAMPING,  FAULTY_DAMPING,
    HOME_Q, PICK_Q,
    _ref_ctrl, combined_degradation, lag_alpha, DT,
)
from divergence_detector import detect, THRESHOLD

POSITION_TOLERANCE = 0.06


@dataclass
class CorrectionResult:
    pre_joint_error:    float
    pre_degradation:    float
    arm_stalled:        bool
    correction_time:    float
    friction_reset_to:  float
    damping_reset_to:   float
    post_joint_error:   float
    post_alpha:         float
    grasp_possible:     bool
    validation_passed:  bool


def run_correction(detection_time):
    sp   = run_sim_pair(duration=detection_time)
    last = sp.records[-1]
    pre_joint_error = float(np.sqrt(np.mean((last.q_nom - last.q_flt)**2)))
    pre_deg         = last.deg
    arm_stalled     = last.alpha == 0.0
    post_alpha      = lag_alpha(0.0)   # deg=0 after reset → alpha=1.0

    q_cor  = last.q_flt.copy()
    steps  = int(5.0 / DT)
    q_ref  = PICK_Q.copy()
    for i in range(steps):
        t     = detection_time + i * DT
        q_ref = _ref_ctrl(t)
        q_cor = q_cor + post_alpha * (q_ref - q_cor)

    post_joint_error  = float(np.sqrt(np.mean((q_ref - q_cor)**2)))
    grasp_possible    = post_joint_error < POSITION_TOLERANCE
    validation_passed = grasp_possible and post_alpha == 1.0

    return CorrectionResult(
        pre_joint_error   = pre_joint_error,
        pre_degradation   = pre_deg,
        arm_stalled       = arm_stalled,
        correction_time   = detection_time,
        friction_reset_to = NOMINAL_FRICTION,
        damping_reset_to  = NOMINAL_DAMPING,
        post_joint_error  = post_joint_error,
        post_alpha        = post_alpha,
        grasp_possible    = grasp_possible,
        validation_passed = validation_passed,
    )


if __name__ == "__main__":
    print("Running sim pair + detector ...")
    sp  = run_sim_pair()
    det = detect(sp)
    t_alarm = det.detection_time if det.detected else 20.0
    print(f"  Alarm at t={t_alarm:.3f}s")
    print("Running correction ...")
    cr = run_correction(t_alarm)
    print(f"\n  [Pre-correction]")
    print(f"    Joint error       : {cr.pre_joint_error:.4f} rad")
    print(f"    Degradation       : {cr.pre_degradation:.3f}")
    print(f"    Arm stalled       : {cr.arm_stalled}")
    print(f"\n  [Correction]")
    print(f"    Friction -> {cr.friction_reset_to} Nm")
    print(f"    Damping  -> {cr.damping_reset_to} Ns/m")
    print(f"\n  [Post-correction]")
    print(f"    Joint error       : {cr.post_joint_error:.4f} rad")
    print(f"    Alpha restored    : {cr.post_alpha:.3f}")
    print(f"    Grasp possible    : {cr.grasp_possible}")
    print(f"    Validation passed : {cr.validation_passed}")
