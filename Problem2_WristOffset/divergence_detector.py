"""
Problem 2 — Wrist Lateral Offset
divergence_detector.py

Monitors end-effector Y positions of both arms.
Flags lateral divergence when delta exceeds threshold.
"""
import numpy as np

DIVERGENCE_THRESHOLD = 0.050
SENSITIVITY_Y        = 0.95


class DivergenceDetector:
    def __init__(self):
        self.history = []
        self.fault_detected = False
        self.estimated_offset = 0.0

    def update(self, l_ee, r_ee, arm_l_y, arm_r_y):
        expected_delta = arm_r_y - arm_l_y
        actual_delta   = r_ee[1] - l_ee[1]
        lateral_error  = actual_delta - expected_delta
        self.history.append(lateral_error)
        if abs(lateral_error) > DIVERGENCE_THRESHOLD:
            self.fault_detected   = True
            self.estimated_offset = lateral_error / SENSITIVITY_Y
            return True
        return False

    def get_fault_report(self):
        if not self.history:
            return {}
        mean_error = float(np.mean(np.abs(self.history)))
        return {
            "fault_detected":   self.fault_detected,
            "lateral_drift_mm": mean_error * 1000,
            "estimated_offset": self.estimated_offset,
            "joint_rmse":       0.0,
            "fault_axis":       "Y",
            "fault_type":       "wrist_lateral_offset",
        }

    def reset(self):
        self.history          = []
        self.fault_detected   = False
        self.estimated_offset = 0.0


if __name__ == "__main__":
    det = DivergenceDetector()
    l = np.array([0.52, -0.55, 0.46])
    r = np.array([0.52,  0.70, 0.46])
    print(f"Fault detected: {det.update(l, r, -0.55, 0.55)}")
    print(det.get_fault_report())
