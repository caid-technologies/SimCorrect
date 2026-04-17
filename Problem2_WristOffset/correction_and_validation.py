"""
Problem 2 — Wrist Lateral Offset
correction_and_validation.py

Calls OpenCAD to rebuild wrist geometry with zero offset.
Exports corrected STL. Returns correction result dict.
"""
from opencad import Part, Sketch
import os

WRIST_RADIUS   = 0.028
WRIST_DEPTH    = 0.10
WRIST_GT       = 0.000
CORRECTION_TOL = 0.002
OUT_DIR        = os.path.dirname(os.path.abspath(__file__))


def correct_wrist_offset(detected_offset_y):
    print(f"[OpenCAD] Detected wrist_offset_y = {detected_offset_y*1000:.1f}mm")
    print(f"[OpenCAD] Rebuilding wrist geometry with offset_y = 0.000m ...")
    part   = Part()
    sketch = Sketch().circle(r=WRIST_RADIUS)
    part.extrude(sketch, depth=WRIST_DEPTH)
    stl_path = os.path.join(OUT_DIR, "wrist_corrected.stl")
    part.export(stl_path)
    print(f"[OpenCAD] Exported -> {stl_path}")
    return {
        "fault_param":     "wrist_offset_y",
        "fault_value":     detected_offset_y,
        "corrected_value": WRIST_GT,
        "delta_mm":        detected_offset_y * 1000,
        "stl_path":        stl_path,
        "wrist_gt":        WRIST_GT,
    }


def validate_correction(model_data, corrected_value, measured_drift):
    residual = abs(measured_drift - corrected_value)
    ok = residual < CORRECTION_TOL
    print(f"[Validate] residual={residual*1000:.1f}mm  tol={CORRECTION_TOL*1000:.0f}mm  {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print(correct_wrist_offset(0.150))
