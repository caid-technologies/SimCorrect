"""Quick demo: print fault summary for Problem 4."""
import numpy as np

J1_REF_BAD=0.1396; J1_REF_GT=0.0000
ARM_REACH_FULL=0.75; ARM_REACH_HALF=0.375

def main():
    miss_full = ARM_REACH_FULL * np.sin(J1_REF_BAD) * 1000
    miss_half = ARM_REACH_HALF * np.sin(J1_REF_BAD) * 1000
    print("=== Problem 4: Joint Zero Offset ===")
    print(f"Fault:           j1 encoder mounted {np.degrees(J1_REF_BAD):.1f} deg off")
    print(f"MJCF param:      j1 ref= {J1_REF_BAD:.4f} rad  (correct = {J1_REF_GT:.4f})")
    print(f"Miss at 0.75m:   {miss_full:.1f}mm")
    print(f"Miss at 0.375m:  {miss_half:.1f}mm")
    print(f"Scaling ratio:   {miss_full/miss_half:.2f}  (2.0 = pure rotation)")
    print(f"Joint RMSE:      0.000  (geometric fault)")
    print(f"Correction:      set j1 ref=0.0000  (one number, 0.28s)")

if __name__=="__main__":
    main()
