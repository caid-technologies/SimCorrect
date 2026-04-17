"""
Parameter identifier for Problem 5 — Tool Mass Mismatch.

Estimates actual tool mass from gravitational sag measured at two
arm extensions. Uses linear scaling analysis to confirm mass signature
and solve for delta_mass via the gravity compensation equation.

Physics:
    extra_torque = delta_mass x g x lever_arm
    joint_lag    = extra_torque / kp          (linearised)
    delta_mass   = joint_lag x kp / (g x reach)

The 2:1 sag ratio between full and half reach is the unique mathematical
signature of a pure mass error and distinguishes this from all other
dynamics faults.
"""
import sys, os
sys.path.insert(0, os.path.expanduser("~/simcorrect"))
import numpy as np
from opencad import Part

G          = 9.81
KP_J4      = 400.0
REACH_FULL = 0.75
REACH_HALF = 0.375


def identify(sag_full_mm, sag_half_mm, model_mass,
             reach_full=REACH_FULL, reach_half=REACH_HALF,
             export_path="/tmp/grip_corrected.xml"):
    """
    Estimate actual tool mass and apply OpenCAD correction.

    Parameters
    ----------
    sag_full_mm  : float  Vertical sag at full reach (mm)
    sag_half_mm  : float  Vertical sag at half reach (mm)
    model_mass   : float  Mass currently in the model (kg)
    export_path  : str    Where to write the corrected XML

    Returns
    -------
    actual_mass  : float  Estimated physical tool mass (kg)
    delta_mass   : float  Mass delta (kg)
    record       : CorrectionRecord
    """
    print("=" * 50)
    print("SimCorrect — Parameter Identifier")
    print("=" * 50)
    print(f"Sag at {reach_full:.3f}m reach:  {sag_full_mm:.1f} mm")
    print(f"Sag at {reach_half:.3f}m reach:  {sag_half_mm:.1f} mm")

    sag_ratio = sag_full_mm / (sag_half_mm + 1e-9)
    print(f"Sag ratio:          {sag_ratio:.2f}  (expected 2.0 for pure mass error)")

    if abs(sag_ratio - 2.0) < 0.3:
        print("Scaling confirmed: MASS MISMATCH signature")
    else:
        print(f"Warning: ratio {sag_ratio:.2f} deviates from 2.0 — mixed fault possible")

    # Estimate delta_mass from both measurements and average
    dm_full = (sag_full_mm / 1000.0) * KP_J4 / (G * reach_full)
    dm_half = (sag_half_mm / 1000.0) * KP_J4 / (G * reach_half)
    delta_mass  = (dm_full + dm_half) / 2.0
    actual_mass = model_mass + delta_mass

    extra_torque = delta_mass * G * reach_full

    print(f"Delta mass (full):  {dm_full*1000:.1f} g")
    print(f"Delta mass (half):  {dm_half*1000:.1f} g")
    print(f"IDENTIFIED delta:   +{delta_mass*1000:.1f} g")
    print(f"Modelled mass:      {model_mass:.3f} kg")
    print(f"ESTIMATED actual:   {actual_mass:.3f} kg")
    print(f"Extra torque:       {extra_torque:.3f} Nm (was uncompensated)")
    print()

    # Apply correction via OpenCAD
    print("Applying OpenCAD correction...")
    part = Part("grip").set_mass(actual_mass)
    part.export(export_path)
    print(part.report())
    print(f"Corrected XML written to: {export_path}")

    return actual_mass, delta_mass, part.corrections


if __name__ == "__main__":
    identify(sag_full_mm=19.4, sag_half_mm=9.7, model_mass=0.100)
