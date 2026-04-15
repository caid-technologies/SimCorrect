"""Identify joint zero offset from rotational miss scaling with reach."""
import numpy as np

ARM_REACH_FULL=0.75; ARM_REACH_HALF=0.375

def identify(rot_miss_full_mm, rot_miss_half_mm):
    print("=== Parameter Identifier ===")
    ratio = rot_miss_full_mm / rot_miss_half_mm
    print(f"Miss at 0.75m reach:   {rot_miss_full_mm:.1f}mm")
    print(f"Miss at 0.375m reach:  {rot_miss_half_mm:.1f}mm")
    print(f"Scaling ratio:         {ratio:.2f}  (expected 2.0 for pure rotation)")
    angle_rad_full = np.arcsin(rot_miss_full_mm/1000 / ARM_REACH_FULL)
    angle_rad_half = np.arcsin(rot_miss_half_mm/1000 / ARM_REACH_HALF)
    angle_est = (angle_rad_full + angle_rad_half) / 2
    print(f"Estimated offset (full): {np.degrees(angle_rad_full):.2f} deg")
    print(f"Estimated offset (half): {np.degrees(angle_rad_half):.2f} deg")
    print(f"IDENTIFIED FAULT:  j1 ref = {angle_est:.4f} rad ({np.degrees(angle_est):.2f} deg)")
    print(f"CORRECT VALUE:     j1 ref = 0.0000 rad")
    print(f"DELTA:             +{np.degrees(angle_est):.2f} deg")
    return angle_est

if __name__=="__main__":
    identify(103.0, 52.0)
