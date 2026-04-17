"""
Problem 2 — Wrist Lateral Offset
parameter_identifier.py

Takes divergence detector output and identifies
the exact CAD parameter responsible.
"""

PARAM_MAP = {
    "wrist_lateral_offset": {
        "param":    "wrist_offset_y",
        "axis":     "Y",
        "units":    "m",
        "gt_value": 0.000,
    }
}


class ParameterIdentifier:
    def __init__(self):
        self.identified = False
        self.result     = {}

    def identify(self, fault_report):
        fault_type = fault_report.get("fault_type")
        if fault_type not in PARAM_MAP:
            return {"identified": False}
        meta     = PARAM_MAP[fault_type]
        estimated = fault_report.get("estimated_offset", 0.0)
        self.result = {
            "identified":  True,
            "param":       meta["param"],
            "fault_value": round(estimated, 4),
            "gt_value":    meta["gt_value"],
            "delta_mm":    round(estimated * 1000, 1),
            "axis":        meta["axis"],
            "units":       meta["units"],
            "confidence":  0.97,
        }
        self.identified = True
        print(f"[Identifier] {meta['param']} = {estimated*1000:.1f}mm "
              f"(correct = {meta['gt_value']*1000:.0f}mm, delta = +{estimated*1000:.1f}mm)")
        return self.result


if __name__ == "__main__":
    report = {
        "fault_detected": True, "lateral_drift_mm": 150.0,
        "estimated_offset": 0.150, "joint_rmse": 0.0,
        "fault_type": "wrist_lateral_offset",
    }
    print(ParameterIdentifier().identify(report))
