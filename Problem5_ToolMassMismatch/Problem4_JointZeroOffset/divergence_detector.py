"""Detect EE divergence with zero joint RMSE -> geometric fault classification."""
import numpy as np

def detect(dist_l, dist_r, j_rmse, rot_miss, threshold_ee=0.04):
    print("=== Divergence Detector ===")
    print(f"GT  EE->can:      {dist_l*1000:.1f}mm")
    print(f"Faulty EE->can:   {dist_r*1000:.1f}mm")
    print(f"Rotational miss:  {rot_miss:.1f}mm")
    print(f"Joint RMSE:       {j_rmse:.4f}")
    fault_detected = dist_r > threshold_ee
    is_geometric   = j_rmse < 0.001 and dist_r > threshold_ee
    if fault_detected:
        print(f"FAULT DETECTED: EE miss {dist_r*1000:.1f}mm > threshold {threshold_ee*1000:.0f}mm")
    if is_geometric:
        print("CLASSIFICATION: GEOMETRIC (large Cartesian miss + zero joint RMSE)")
        print("Candidate: joint zero offset or link length error")
    return fault_detected, is_geometric

if __name__=="__main__":
    detect(0.012, 0.110, 0.000, 103.0)
