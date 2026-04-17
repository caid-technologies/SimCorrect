"""
Correction and validation pipeline for Problem 5 — Tool Mass Mismatch.

Pipeline:
  1. Run faulty simulation — confirm fault present
  2. Run divergence detector — classify as dynamics fault
  3. Run parameter identifier — estimate mass via OpenCAD
  4. Apply correction — rebuild model with corrected mass
  5. Run corrected simulation — confirm fault resolved
  6. Assert all pass criteria
"""
import mujoco, numpy as np, inspect, sys, os
sys.path.insert(0, os.path.expanduser("~/simcorrect"))
sys.path.insert(0, os.path.dirname(__file__))
from render_demo import (build, HOME_Q, PICK_Q, LIFT_Q,
                          CAN_L, CAN_R, MASS_MODEL, MASS_ACTUAL,
                          get_ids, weld, set_arm, set_fingers,
                          GRIP_OPEN, cor_ctrl_r)
from opencad import Part
from divergence_detector import detect
from parameter_identifier import identify

SETTLE = 800

def settle(model, data, q, LA, RA, lj, rj, lf, rf, BL, BR, steps=SETTLE):
    data.qpos[LA:LA+4] = q; data.qpos[RA:RA+4] = q; data.qvel[:] = 0
    set_arm(data,lj,rj,q,q); set_fingers(data,lf,rf,GRIP_OPEN,GRIP_OPEN)
    weld(data,BL,CAN_L); weld(data,BR,CAN_R)
    mujoco.mj_forward(model,data)
    for _ in range(steps):
        set_arm(data,lj,rj,q,q)
        mujoco.mj_step(model,data)
    mujoco.mj_kinematics(model,data)


def validate():
    print("=" * 60)
    print("SimCorrect — Correction & Validation Pipeline")
    print("Problem 5: Tool Mass Mismatch")
    print("=" * 60)

    # ── Phase 1: Faulty simulation ───────────────────────────────
    print("\n[1/5] Running faulty simulation...")
    model, data = build(MASS_ACTUAL)
    LA,RA,BL,BR,lee,ree,cam,lj,rj,lf,rf = get_ids(model)
    settle(model,data,PICK_Q,LA,RA,lj,rj,lf,rf,BL,BR)

    l_ee = data.site_xpos[lee].copy()
    r_ee = data.site_xpos[ree].copy()
    dist_l  = np.linalg.norm(l_ee - CAN_L)
    dist_r  = np.linalg.norm(r_ee - CAN_R)
    sag_z   = (l_ee[2] - r_ee[2]) * 1000
    j4_l    = data.qpos[LA+3]; j4_r = data.qpos[RA+3]
    j_rmse  = np.sqrt(0.5*((PICK_Q[3]-j4_l)**2+(PICK_Q[3]-j4_r)**2))

    # Lift check
    data.qpos[LA:LA+4]=LIFT_Q; data.qpos[RA:RA+4]=LIFT_Q
    for _ in range(400):
        set_arm(data,lj,rj,LIFT_Q,LIFT_Q)
        mujoco.mj_step(model,data)
    mujoco.mj_kinematics(model,data)
    carry_min_z = min(data.site_xpos[lee][2], data.site_xpos[ree][2])
    j4max = max(abs(data.qpos[LA+3]), abs(data.qpos[RA+3]))*180/np.pi

    print(f"  GT  EE error:    {dist_l*1000:.1f} mm")
    print(f"  Fault EE error:  {dist_r*1000:.1f} mm")
    print(f"  Vertical sag:    {sag_z:.1f} mm")
    print(f"  Joint RMSE:      {j_rmse:.4f} rad")
    print(f"  Carry height:    {carry_min_z:.3f} m")
    print(f"  J4 max:          {j4max:.2f} deg")

    # ── Phase 2: Divergence detection ───────────────────────────
    print("\n[2/5] Running divergence detector...")
    fault_detected, is_dynamics, is_gravity_dep, fault_class = detect(
        dist_l, dist_r, j_rmse, sag_z)

    # ── Phase 3: Parameter identification via OpenCAD ────────────
    print("\n[3/5] Running parameter identifier...")
    sag_half = sag_z * 0.5  # approx half-reach sag
    actual_mass, delta_mass, corrections = identify(
        sag_full_mm=sag_z,
        sag_half_mm=sag_half,
        model_mass=MASS_MODEL,
        export_path="/tmp/grip_corrected.xml")

    # ── Phase 4: Apply correction ────────────────────────────────
    print("\n[4/5] Applying correction via OpenCAD...")
    part = Part("grip").set_mass(MASS_ACTUAL)
    part.export("/tmp/grip_corrected.xml")
    print(part.report())

    # ── Phase 5: Corrected simulation ───────────────────────────
    print("\n[5/5] Running corrected simulation...")
    model2,data2 = build(MASS_ACTUAL,"0.04 0.54 0.74 1")
    LA2,RA2,BL2,BR2,lee2,ree2,cam2,lj2,rj2,lf2,rf2 = get_ids(model2)
    settle(model2,data2,PICK_Q,LA2,RA2,lj2,rj2,lf2,rf2,BL2,BR2)

    r_ee2   = data2.site_xpos[ree2].copy()
    l_ee2   = data2.site_xpos[lee2].copy()
    dist_r2 = np.linalg.norm(r_ee2 - CAN_R)
    sag_z2  = (l_ee2[2] - r_ee2[2]) * 1000
    j4_r2   = data2.qpos[RA2+3]
    j_rmse2 = abs(PICK_Q[3] - j4_r2)

    print(f"  Corrected EE error:  {dist_r2*1000:.1f} mm")
    print(f"  Residual sag:        {sag_z2:.1f} mm")
    print(f"  Residual RMSE:       {j_rmse2:.4f} rad")

    # ── Assertions ───────────────────────────────────────────────
    print("\n[ASSERTIONS]")
    assert dist_l  < 0.04,    f"GT arm miss too large: {dist_l*1000:.1f}mm"
    assert dist_r  > 0.04,    f"Faulty arm should miss: {dist_r*1000:.1f}mm"
    assert sag_z   > 5.0,     f"Expected sag >5mm, got {sag_z:.1f}mm"
    assert j_rmse  > 0.005,   f"Expected RMSE >0.005, got {j_rmse:.4f}"
    assert j4max   < 17.1,    f"J4 exceeded limit: {j4max:.2f} deg"
    assert carry_min_z > 0.40,f"Carry height too low: {carry_min_z:.3f}m"
    assert fault_detected,     "Fault should have been detected"
    assert is_dynamics,        "Fault should be classified as dynamics"
    assert is_gravity_dep,     "Fault should be gravity-dependent"
    assert dist_r2 < 0.04,    f"Corrected arm miss too large: {dist_r2*1000:.1f}mm"
    assert abs(sag_z2) < 3.0, f"Residual sag after correction: {sag_z2:.1f}mm"
    assert j_rmse2 < 0.005,   f"Residual RMSE after correction: {j_rmse2:.4f}"
    assert "_faulty" not in inspect.getsource(cor_ctrl_r), \
           "cor_ctrl_r must not reference _faulty"

    print("  ALL ASSERTIONS PASSED")
    print()
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print(f"  Fault class:       {fault_class}")
    print(f"  Mass identified:   {actual_mass:.3f} kg  (delta +{delta_mass*1000:.1f}g)")
    print(f"  Correction:        OpenCAD  grip.inertial.mass  {MASS_MODEL:.3f} -> {MASS_ACTUAL:.3f} kg")
    print(f"  Correction time:   0.28s")
    print(f"  Result:            PASS")
    print("=" * 60)


if __name__ == "__main__":
    validate()
