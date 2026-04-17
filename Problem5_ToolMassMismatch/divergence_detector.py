"""
Divergence detector for Problem 5 — Tool Mass Mismatch.

Classification logic:
  1. Large EE divergence + joint RMSE ~ 0  ->  geometric fault
  2. Large EE divergence + joint RMSE > 0  ->  dynamics fault
     2a. RMSE scales with velocity          ->  friction (Problem 3)
     2b. RMSE present at rest, horizontal   ->  mass mismatch (Problem 5)
"""
import numpy as np

RMSE_DYNAMICS_THRESHOLD = 0.005   # rad
EE_FAULT_THRESHOLD      = 0.040   # m


def detect(dist_gt, dist_faulty, j_rmse, sag_z_mm,
           vel_rmse_fast=None, vel_rmse_slow=None,
           threshold_ee=EE_FAULT_THRESHOLD):
    """
    Classify the fault from paired simulation measurements.

    Parameters
    ----------
    dist_gt      : float  GT arm EE distance to target (m)
    dist_faulty  : float  Faulty arm EE distance to target (m)
    j_rmse       : float  Joint RMSE between commanded and actual (rad)
    sag_z_mm     : float  Vertical sag of faulty arm (mm, positive = lower)
    vel_rmse_fast: float  Joint RMSE measured during fast motion (optional)
    vel_rmse_slow: float  Joint RMSE measured during slow motion (optional)

    Returns
    -------
    fault_detected  : bool
    is_dynamics     : bool
    is_gravity_dep  : bool
    fault_class     : str
    """
    print("=" * 50)
    print("SimCorrect — Divergence Detector")
    print("=" * 50)
    print(f"GT  EE error:       {dist_gt*1000:.1f} mm")
    print(f"Faulty EE error:    {dist_faulty*1000:.1f} mm")
    print(f"Vertical sag:       {sag_z_mm:.1f} mm")
    print(f"Joint RMSE:         {j_rmse:.4f} rad")

    fault_detected = dist_faulty > threshold_ee

    if not fault_detected:
        print("STATUS: No fault detected — EE error within tolerance")
        return False, False, False, "NOMINAL"

    print(f"FAULT DETECTED: EE error {dist_faulty*1000:.1f}mm > threshold {threshold_ee*1000:.0f}mm")

    is_dynamics = j_rmse > RMSE_DYNAMICS_THRESHOLD

    if not is_dynamics:
        print("CLASSIFICATION: GEOMETRIC")
        print("  Large EE divergence + zero joint RMSE")
        print("  Candidates: link length, joint zero offset, wrist offset")
        return fault_detected, False, False, "GEOMETRIC"

    print("CLASSIFICATION: DYNAMICS")
    print("  Large EE divergence + non-zero joint RMSE")
    print("  Joints cannot reach commanded angles")

    # Distinguish mass from friction via velocity dependence
    is_gravity_dep = True
    if vel_rmse_fast is not None and vel_rmse_slow is not None:
        vel_ratio = vel_rmse_fast / (vel_rmse_slow + 1e-9)
        print(f"  Velocity ratio (fast/slow RMSE): {vel_ratio:.2f}")
        if vel_ratio > 2.5:
            is_gravity_dep = False
            print("  HIGH velocity dependence -> FRICTION (Problem 3 pattern)")
        else:
            print("  LOW velocity dependence -> GRAVITY-DEPENDENT (mass mismatch)")
    else:
        print("  Velocity data not provided — inferring from sag pattern")
        is_gravity_dep = sag_z_mm > 5.0

    if is_gravity_dep:
        print("SUB-CLASS: GRAVITY-DEPENDENT DYNAMICS")
        print("  Error present at rest at horizontal pose")
        print("  Error scales with horizontal extension")
        print("  FAULT: Tool mass mismatch")
        fault_class = "DYNAMICS_MASS_MISMATCH"
    else:
        print("SUB-CLASS: VELOCITY-DEPENDENT DYNAMICS")
        print("  FAULT: Joint friction excess (Problem 3)")
        fault_class = "DYNAMICS_FRICTION"

    return fault_detected, is_dynamics, is_gravity_dep, fault_class


if __name__ == "__main__":
    detect(dist_gt=0.012, dist_faulty=0.078,
           j_rmse=0.0082, sag_z_mm=19.4,
           vel_rmse_fast=0.009, vel_rmse_slow=0.008)
