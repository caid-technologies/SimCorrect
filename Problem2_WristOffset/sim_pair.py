"""
Problem 2 — Wrist Lateral Offset
sim_pair.py

Runs reference and faulty sim instances side by side.
Feeds EE positions to divergence detector.
Triggers correction pipeline when fault detected.
"""
import mujoco
import numpy as np
from divergence_detector import DivergenceDetector
from parameter_identifier import ParameterIdentifier
from correction_and_validation import correct_wrist_offset, validate_correction

ARM_L_Y = -0.55
ARM_R_Y =  0.55


class SimPair:
    def __init__(self, model_ref, model_faulty):
        self.m_ref    = model_ref
        self.m_flt    = model_faulty
        self.d_ref    = mujoco.MjData(model_ref)
        self.d_flt    = mujoco.MjData(model_faulty)
        self.detector = DivergenceDetector()
        self.ident    = ParameterIdentifier()
        self.corrected = False
        self.correction_result = {}
        self.lee_ref = mujoco.mj_name2id(model_ref,   mujoco.mjtObj.mjOBJ_SITE, "l_ee")
        self.ree_flt = mujoco.mj_name2id(model_faulty, mujoco.mjtObj.mjOBJ_SITE, "r_ee")

    def step(self, ctrl_ref, ctrl_flt):
        self.d_ref.ctrl[:] = ctrl_ref
        self.d_flt.ctrl[:] = ctrl_flt
        mujoco.mj_step(self.m_ref, self.d_ref)
        mujoco.mj_step(self.m_flt, self.d_flt)
        l_ee = self.d_ref.site_xpos[self.lee_ref].copy()
        r_ee = self.d_flt.site_xpos[self.ree_flt].copy()
        fault = self.detector.update(l_ee, r_ee, ARM_L_Y, ARM_R_Y)
        if fault and not self.corrected:
            self._run_correction()
        return l_ee, r_ee, self.corrected

    def _run_correction(self):
        report = self.detector.get_fault_report()
        ident  = self.ident.identify(report)
        if not ident.get("identified"):
            return
        result = correct_wrist_offset(ident["fault_value"])
        valid  = validate_correction(None, result["corrected_value"], ident["fault_value"])
        if valid:
            self.corrected         = True
            self.correction_result = result
            print(f"[SimPair] Correction validated. Reloading corrected arm.")
        else:
            print(f"[SimPair] Correction failed validation.")

    def get_correction_result(self):
        return self.correction_result
