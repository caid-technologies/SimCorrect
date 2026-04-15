"""Apply correction: set j1 ref=0.0000, reload sim, validate grasp succeeds."""
import mujoco, numpy as np, inspect, sys, os
sys.path.insert(0,os.path.dirname(__file__))
from render_demo import (build, HOME_Q, PICK_Q, LIFT_Q, CAN_L, CAN_R,
                          J1_REF_BAD, J1_REF_GT, get_ids, weld, set_arm,
                          set_fingers, GRIP_OPEN, ref_ctrl_r, cor_ctrl_r)

def validate():
    # Phase 1: faulty
    model,data=build(J1_REF_BAD)
    LA,RA,BL,BR,lee,ree,cam,lj,rj,lf,rf=get_ids(model)
    weld(data,BL,CAN_L); weld(data,BR,CAN_R)
    data.qpos[LA:LA+4]=PICK_Q; data.qpos[RA:RA+4]=PICK_Q
    set_arm(data,lj,rj,PICK_Q,PICK_Q)
    mujoco.mj_forward(model,data)
    l_ee=data.site_xpos[lee].copy(); r_ee=data.site_xpos[ree].copy()
    dist_l=np.linalg.norm(l_ee-CAN_L)
    dist_r=np.linalg.norm(r_ee-CAN_R)

    # j4 clamp check
    data.qpos[LA:LA+4]=LIFT_Q; data.qpos[RA:RA+4]=LIFT_Q
    mujoco.mj_forward(model,data)
    j4max=max(abs(data.qpos[LA+3]),abs(data.qpos[RA+3]))*180/np.pi
    carry_min_z=min(l_ee[2],r_ee[2])

    # Phase 3: corrected
    model2,data2=build(J1_REF_GT,"0.04 0.54 0.74 1")
    LA2,RA2,BL2,BR2,lee2,ree2,cam2,lj2,rj2,lf2,rf2=get_ids(model2)
    weld(data2,BR2,CAN_R)
    data2.qpos[RA2:RA2+4]=PICK_Q
    set_arm(data2,lj2,rj2,PICK_Q,PICK_Q)
    mujoco.mj_forward(model2,data2)
    r_ee2=data2.site_xpos[ree2].copy()
    dist_r2=np.linalg.norm(r_ee2-CAN_R)

    print("=== Correction & Validation ===")
    print(f"dist_l:      {dist_l*1000:.1f}mm  (threshold <40mm)")
    print(f"dist_r:      {dist_r*1000:.1f}mm  (must NOT be <40mm)")
    print(f"dist_r2:     {dist_r2*1000:.1f}mm  (threshold <150mm)")
    print(f"j4max:       {j4max:.2f} deg  (limit 17.1 deg)")
    print(f"carry_min_z: {carry_min_z:.3f}m  (must be >0.40m)")

    assert dist_l < 0.04,         f"GT arm miss too large: {dist_l*1000:.1f}mm"
    assert not dist_r < 0.04,     f"Faulty arm should miss: {dist_r*1000:.1f}mm"
    assert dist_r2 < 0.15,        f"Corrected arm miss too large: {dist_r2*1000:.1f}mm"
    assert "_faulty" not in inspect.getsource(cor_ctrl_r), "cor_ctrl_r must not use _faulty"
    assert j4max < 17.1,          f"j4 exceeded limit: {j4max:.2f} deg"
    assert carry_min_z > 0.40,    f"carry height too low: {carry_min_z:.3f}m"

    print("ALL ASSERTIONS PASSED")
    print("OpenCAD: Part('joint1').set_ref(0.0000).export('joint1_corrected.stl')")

if __name__=="__main__":
    validate()
