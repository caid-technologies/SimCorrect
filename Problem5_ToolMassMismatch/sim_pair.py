"""Paired simulation: GT arm (tool mass=0.10kg) vs Faulty arm (tool mass=0.16kg)."""
import mujoco, numpy as np, sys, os
sys.path.insert(0, os.path.expanduser("~/simcorrect"))
sys.path.insert(0, os.path.dirname(__file__))
from render_demo import (build, PICK_Q, GRIP_OPEN,
                          CAN_L, CAN_R, MASS_MODEL, MASS_ACTUAL,
                          get_ids, weld, set_arm, set_fingers)
from opencad import Part

SETTLE = 800

def run_pair():
    print("=" * 50)
    print("SimCorrect — Paired Simulation")
    print("Problem 5: Tool Mass Mismatch")
    print("=" * 50)

    model, data = build(MASS_ACTUAL)
    LA,RA,BL,BR,lee,ree,cam,lj,rj,lf,rf = get_ids(model)
    data.qpos[LA:LA+4] = PICK_Q
    data.qpos[RA:RA+4] = PICK_Q
    data.qvel[:] = 0
    weld(data,BL,CAN_L); weld(data,BR,CAN_R)
    set_arm(data,lj,rj,PICK_Q,PICK_Q)
    set_fingers(data,lf,rf,GRIP_OPEN,GRIP_OPEN)
    mujoco.mj_forward(model,data)

    for _ in range(SETTLE):
        set_arm(data,lj,rj,PICK_Q,PICK_Q)
        mujoco.mj_step(model,data)
    mujoco.mj_kinematics(model,data)

    l_ee = data.site_xpos[lee].copy()
    r_ee = data.site_xpos[ree].copy()
    dist_l = np.linalg.norm(l_ee - CAN_L)
    dist_r = np.linalg.norm(r_ee - CAN_R)
    sag_z  = (l_ee[2] - r_ee[2]) * 1000
    j4_l   = data.qpos[LA+3]; j4_r = data.qpos[RA+3]
    j_rmse = np.sqrt(0.5*((PICK_Q[3]-j4_l)**2+(PICK_Q[3]-j4_r)**2))
    extra_torque = (MASS_ACTUAL - MASS_MODEL) * 9.81 * 0.75

    print(f"GT  EE error:       {dist_l*1000:.1f} mm")
    print(f"Faulty EE error:    {dist_r*1000:.1f} mm")
    print(f"Vertical sag:       {sag_z:.1f} mm  (faulty arm lower)")
    print(f"Extra torque:       {extra_torque:.3f} Nm  (uncompensated)")
    print(f"Joint RMSE:         {j_rmse:.4f} rad  (>0 = dynamics fault)")
    print(f"Modelled mass:      {MASS_MODEL:.3f} kg")
    print(f"Actual mass:        {MASS_ACTUAL:.3f} kg  (+{(MASS_ACTUAL-MASS_MODEL)*1000:.0f}g)")
    print()

    # Apply OpenCAD correction
    print("Applying OpenCAD correction...")
    part = Part("grip").set_mass(MASS_ACTUAL)
    part.export("/tmp/grip_corrected.xml")
    print(part.report())

    return dist_l, dist_r, j_rmse, sag_z

if __name__ == "__main__":
    run_pair()
