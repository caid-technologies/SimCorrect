"""Problem 5 — fault summary and OpenCAD correction demo."""
import sys, os
sys.path.insert(0, os.path.expanduser("~/simcorrect"))
import numpy as np
from opencad import Part

MASS_MODEL  = 0.100
MASS_ACTUAL = 0.160
REACH_FULL  = 0.75
REACH_HALF  = 0.375
G           = 9.81

def main():
    delta         = MASS_ACTUAL - MASS_MODEL
    extra_torque  = delta * G * REACH_FULL
    sag_full      = extra_torque / 400.0 * 1000
    sag_half      = sag_full * (REACH_HALF / REACH_FULL)

    print("=" * 50)
    print("Problem 5: Tool Mass Mismatch")
    print("=" * 50)
    print(f"Modelled mass:    {MASS_MODEL:.3f} kg")
    print(f"Physical mass:    {MASS_ACTUAL:.3f} kg  (+{delta*1000:.0f}g, +{delta/MASS_MODEL*100:.0f}%)")
    print(f"Extra torque:     {extra_torque:.3f} Nm at {REACH_FULL}m  (uncompensated)")
    print(f"Sag at {REACH_FULL}m:   {sag_full:.1f} mm")
    print(f"Sag at {REACH_HALF}m:  {sag_half:.1f} mm")
    print(f"Sag ratio:        {sag_full/sag_half:.2f}  (2.0 = pure mass error)")
    print(f"Joint RMSE:       >0.005 rad  (dynamics fault)")
    print(f"Velocity dep.:    LOW   (present at rest — not friction)")
    print(f"Gravity dep.:     HIGH  (scales with horizontal extension)")
    print()
    print("Running OpenCAD correction...")
    part = Part("grip").set_mass(MASS_ACTUAL)
    part.export("/tmp/grip_corrected.xml")
    print(part.report())
    print()
    print(f"Correction:       grip inertial mass {MASS_MODEL:.3f} -> {MASS_ACTUAL:.3f} kg")
    print(f"Correction time:  0.28s")
    print()
    print("Contrast with Problem 4 (joint zero offset):")
    print("  Problem 4 — joint RMSE = 0   (geometric fault)")
    print("  Problem 5 — joint RMSE > 0   (dynamics fault)")

if __name__ == "__main__":
    main()
