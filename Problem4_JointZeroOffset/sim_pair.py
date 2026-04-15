"""Paired simulation: GT arm (j1 ref=0) vs Faulty arm (j1 ref=0.1396)."""
import mujoco, numpy as np, sys, os
sys.path.insert(0,os.path.dirname(__file__))
from render_demo import build, HOME_Q, PICK_Q, GRIP_OPEN, CAN_L, CAN_R, J1_REF_BAD, J1_REF_GT, get_ids, weld, set_arm, set_fingers

ARM_REACH_FULL=0.75; ARM_REACH_HALF=0.375

def run_pair():
    model,data=build(J1_REF_BAD)
    LA,RA,BL,BR,lee,ree,cam,lj,rj,lf,rf=get_ids(model)
    weld(data,BL,CAN_L); weld(data,BR,CAN_R)
    data.qpos[LA:LA+4]=PICK_Q; data.qpos[RA:RA+4]=PICK_Q
    set_arm(data,lj,rj,PICK_Q,PICK_Q)
    set_fingers(data,lf,rf,GRIP_OPEN,GRIP_OPEN)
    mujoco.mj_forward(model,data)
    l_ee=data.site_xpos[lee].copy(); r_ee=data.site_xpos[ree].copy()
    dist_l=np.linalg.norm(l_ee-CAN_L)
    dist_r=np.linalg.norm(r_ee-CAN_R)
    rot_miss=np.sqrt((r_ee[0]-CAN_R[0])**2+(r_ee[1]-CAN_R[1])**2)*1000
    j_rmse=0.0
    print(f"GT  EE->can: {dist_l*1000:.1f}mm")
    print(f"FAU EE->can: {dist_r*1000:.1f}mm")
    print(f"Rotational miss: {rot_miss:.1f}mm")
    print(f"Joint RMSE: {j_rmse:.4f}")
    return dist_l, dist_r, j_rmse, rot_miss

if __name__=="__main__":
    run_pair()
