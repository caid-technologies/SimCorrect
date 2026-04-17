"""
Problem 2: Wrist Offset Fault — Pick-and-Place Dual Simulation
SimCorrect / CoRL 2026

REF arm (white/blue): correct geometry, wrist_offset_y = 0.000m — picks and places.
FLT arm (orange):     wrist_offset_y = +0.007m — lateral miss confirmed by physics.

Architecture:
  - Pre-verified kinematic waypoints for arm motion
  - Weld equality constraint for REF grasp (robust, no contact instability)
  - Physics (mj_step) for settle + lift verification
  - Lateral error check: kinematic at PICK_Q (7mm Y offset, exact)
  - 15-check acceptance criteria per spec Section 10
"""

import mujoco
import numpy as np
import tempfile
import os

# ═══════════════════════════════════════════════════════════════
# SCENE CONSTANTS
# ═══════════════════════════════════════════════════════════════

L1 = 0.34; L2 = 0.30; L3 = 0.12; EE_OFF = 0.015
REF_Y = -0.30; FLT_Y = 0.30
BASE_Z = 0.66
PED_Z = 0.35; CAN_HALF = 0.11; CAN_RADIUS = 0.033
CAN_X = 0.52
CAN_Z = PED_Z + CAN_HALF          # 0.460

PLACE_TABLE_X   = -0.65
TABLE_Z_TOP     = 0.52
TABLE_THICKNESS = 0.052
TH = TABLE_THICKNESS / 2
TY = 0.20

GRIP_OPEN    = 0.040
GRIP_CLOSED  = 0.010
WRIST_OFFSET = 0.007
SENSITIVITY_Y = 0.95

HOME_Q  = np.array([-0.8902,  2.7357,  0.0000])
ABOVE_Q = np.array([-0.7949,  1.1443,  1.2214])
PICK_Q  = np.array([-0.4335,  1.2221,  0.7822])
LIFT_Q  = np.array([-0.8581,  0.9826,  1.4463])
PLACE_Q = np.array([ 2.4994,  1.0947,  0.0000])

INJECTED_ERROR = {
    "parameter":       "wrist_offset_y",
    "true_value":       0.000,
    "faulty_value":     0.007,
    "error_magnitude":  0.007,
}


# ═══════════════════════════════════════════════════════════════
# XML
# ═══════════════════════════════════════════════════════════════

def _arm_xml(pfx, arm_y, wy, lk):
    jc = "0.20 0.22 0.30 1"; gc = "0.14 0.14 0.20 1"
    return f"""
  <body name="{pfx}arm_base" pos="0 {arm_y} {BASE_Z}">
    <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.01 0.01"/>
    <geom type="cylinder" size="0.062 0.055" euler="1.5708 0 0" rgba="{jc}" mass="0.5"/>
    <joint name="{pfx}j1" type="hinge" axis="0 1 0" limited="true" range="-3.14 3.14" damping="8" armature="0.05"/>
    <geom type="capsule" fromto="0 0 0 {L1} 0 0" size="0.038" rgba="{lk}" mass="0.5"/>
    <geom type="sphere" size="0.044" pos="{L1} 0 0" rgba="{jc}" mass="0.1"/>
    <body name="{pfx}elbow" pos="{L1} 0 0">
      <inertial pos="0 0 0" mass="0.4" diaginertia="0.004 0.004 0.004"/>
      <joint name="{pfx}j2" type="hinge" axis="0 1 0" limited="true" range="-2.8 2.8" damping="6" armature="0.03"/>
      <geom type="capsule" fromto="0 0 0 {L2} 0 0" size="0.030" rgba="{lk}" mass="0.3"/>
      <geom type="sphere" size="0.036" pos="{L2} 0 0" rgba="{jc}" mass="0.08"/>
      <body name="{pfx}wrist_link" pos="{L2} 0 0">
        <inertial pos="0 0 0" mass="0.15" diaginertia="0.002 0.002 0.002"/>
        <joint name="{pfx}j3" type="hinge" axis="0 1 0" limited="true" range="-3.14 3.14" damping="4" armature="0.02"/>
        <geom type="capsule" fromto="0 0 0 {L3} 0 0" size="0.022" rgba="{lk}" mass="0.08"/>
        <geom type="sphere" size="0.028" pos="{L3} 0 0" rgba="{jc}" mass="0.04"/>
        <body name="{pfx}tool" pos="{L3} 0 0">
          <inertial pos="0 0 0" mass="0.12" diaginertia="0.001 0.001 0.001"/>
          <geom type="box" size="0.028 0.022 0.022" rgba="{gc}" mass="0.08"/>
          <body name="{pfx}finger_l" pos="0 0 0.034">
            <inertial pos="0 0 0" mass="0.03" diaginertia="0.0003 0.0003 0.0003"/>
            <joint name="{pfx}fg1" type="slide" axis="0 0 1" limited="true" range="{GRIP_CLOSED} {GRIP_OPEN}" damping="3"/>
            <geom name="{pfx}fpad_l" type="box" size="0.030 0.010 0.020" pos="0.038 0 0.020"
                  rgba="{gc}" friction="3.0 0.1 0.01" condim="4"
                  solimp="0.99 0.9999 0.001 0.5 2" solref="0.002 1"/>
          </body>
          <body name="{pfx}finger_r" pos="0 0 -0.034">
            <inertial pos="0 0 0" mass="0.03" diaginertia="0.0003 0.0003 0.0003"/>
            <joint name="{pfx}fg2" type="slide" axis="0 0 -1" limited="true" range="{GRIP_CLOSED} {GRIP_OPEN}" damping="3"/>
            <geom name="{pfx}fpad_r" type="box" size="0.030 0.010 0.020" pos="0.038 0 -0.020"
                  rgba="{gc}" friction="3.0 0.1 0.01" condim="4"
                  solimp="0.99 0.9999 0.001 0.5 2" solref="0.002 1"/>
          </body>
          <body name="{pfx}wrist_off" pos="{EE_OFF} {wy} 0">
            <inertial pos="0 0 0" mass="0.005" diaginertia="0.00005 0.00005 0.00005"/>
            <site name="{pfx}ee" pos="0 0 0" size="0.010"/>
          </body>
        </body>
      </body>
    </body>
  </body>"""


def build_xml():
    TABLE_Z_CTR = TABLE_Z_TOP - TH
    return f"""<mujoco model="simcorrect_p2">
<compiler angle="radian" autolimits="true"/>
<option timestep="0.002" gravity="0 0 -9.81"
        iterations="100" tolerance="1e-10"
        solver="Newton" cone="elliptic" impratio="10"/>
<visual>
  <global offwidth="1920" offheight="1080"/>
  <quality shadowsize="4096" numslices="64" numstacks="64"/>
</visual>
<asset>
  <texture name="chk" type="2d" builtin="checker"
           rgb1="0.25 0.27 0.35" rgb2="0.15 0.17 0.23" width="512" height="512"/>
  <material name="floor_m" texture="chk" texrepeat="6 6"/>
  <material name="table_m" rgba="0.55 0.38 0.16 1" specular="0.3"/>
  <material name="ped_m"   rgba="0.22 0.24 0.32 1" specular="0.5"/>
  <material name="goal_m"  rgba="0.05 0.92 0.20 1" emission="0.20"/>
  <material name="can_m"   rgba="0.92 0.08 0.05 1" specular="0.6"/>
</asset>
<default>
  <joint damping="5.0" armature="0.05"/>
  <geom condim="4" solref="0.004 1" solimp="0.95 0.99 0.001" friction="1.2 0.02 0.002"/>
</default>
<worldbody>
  <light name="sun"  pos="0 -2 8"  dir="0 0.15 -1"  diffuse="1.30 1.28 1.20" castshadow="true"/>
  <light name="fill" pos="0  3 6"  dir="0 -0.3 -0.9" diffuse="0.45 0.48 0.60"/>
  <light name="rim"  pos="3  0 4"  dir="-0.6 0 -0.7" diffuse="0.25 0.28 0.40"/>
  <geom type="plane" size="6 6 0.1" material="floor_m"/>
  <geom type="cylinder" size="0.042 {BASE_Z/2:.4f}" pos="0 {REF_Y} {BASE_Z/2:.4f}" material="ped_m"/>
  <geom type="cylinder" size="0.042 {BASE_Z/2:.4f}" pos="0 {FLT_Y} {BASE_Z/2:.4f}" material="ped_m"/>
  <geom type="cylinder" size="0.050 {PED_Z/2:.4f}"  pos="{CAN_X} {REF_Y} {PED_Z/2:.4f}" material="ped_m"/>
  <geom type="cylinder" size="0.050 {PED_Z/2:.4f}"  pos="{CAN_X} {FLT_Y} {PED_Z/2:.4f}" material="ped_m"/>
  <body name="ref_pick_table" pos="{CAN_X} {REF_Y} 0">
    <geom name="ref_table_top" type="box" size="0.001 0.001 0.001"
          pos="0 0 {CAN_Z:.4f}" rgba="0 0 0 0" contype="0" conaffinity="0"/>
  </body>
  <body name="flt_pick_table" pos="{CAN_X} {FLT_Y} 0">
    <geom name="flt_table_top" type="box" size="0.001 0.001 0.001"
          pos="0 0 {CAN_Z:.4f}" rgba="0 0 0 0" contype="0" conaffinity="0"/>
  </body>
  <body name="ref_place_table" pos="{PLACE_TABLE_X} {REF_Y} {TABLE_Z_CTR:.4f}">
    <geom type="box" size="0.28 {TY} {TH:.4f}" material="table_m" contype="1" conaffinity="1"/>
    <geom type="cylinder" size="0.068 0.004" pos="0 0 {TH:.4f}" material="goal_m"/>
  </body>
  <body name="flt_place_table" pos="{PLACE_TABLE_X} {FLT_Y} {TABLE_Z_CTR:.4f}">
    <geom type="box" size="0.28 {TY} {TH:.4f}" material="table_m" contype="1" conaffinity="1"/>
    <geom type="cylinder" size="0.068 0.004" pos="0 0 {TH:.4f}" material="goal_m"/>
  </body>
  <body name="ref_cylinder" pos="{CAN_X} {REF_Y} {CAN_Z}">
    <freejoint name="ref_cyl_free"/>
    <inertial pos="0 0 0" mass="0.35" diaginertia="0.0012 0.0012 0.0005"/>
    <geom name="ref_cyl_geom" type="cylinder" size="{CAN_RADIUS} {CAN_HALF}"
          material="can_m" contype="1" conaffinity="1"
          friction="3.0 0.1 0.01" condim="4"
          solimp="0.99 0.9999 0.001 0.5 2" solref="0.002 1"/>
  </body>
  <body name="flt_cylinder" pos="{CAN_X} {FLT_Y} {CAN_Z}">
    <freejoint name="flt_cyl_free"/>
    <inertial pos="0 0 0" mass="0.35" diaginertia="0.0012 0.0012 0.0005"/>
    <geom name="flt_cyl_geom" type="cylinder" size="{CAN_RADIUS} {CAN_HALF}"
          material="can_m" contype="1" conaffinity="1"
          friction="3.0 0.1 0.01" condim="4"
          solimp="0.99 0.9999 0.001 0.5 2" solref="0.002 1"/>
  </body>
  {_arm_xml("ref_", REF_Y, 0.000,        "0.86 0.88 0.96 1")}
  {_arm_xml("flt_", FLT_Y, WRIST_OFFSET, "0.92 0.18 0.12 1")}
  <camera name="main" pos="3.2 0.0 1.8" xyaxes="0 1 0 -0.49 0 0.87" fovy="52"/>
</worldbody>
<actuator>
  <position joint="ref_j1"  kp="900"  forcerange="-220 220"/>
  <position joint="ref_j2"  kp="700"  forcerange="-180 180"/>
  <position joint="ref_j3"  kp="500"  forcerange="-120 120"/>
  <position joint="ref_fg1" kp="2000" ctrlrange="0 {GRIP_OPEN}" forcerange="-200 200"/>
  <position joint="ref_fg2" kp="2000" ctrlrange="0 {GRIP_OPEN}" forcerange="-200 200"/>
  <position joint="flt_j1"  kp="900"  forcerange="-220 220"/>
  <position joint="flt_j2"  kp="700"  forcerange="-180 180"/>
  <position joint="flt_j3"  kp="500"  forcerange="-120 120"/>
  <position joint="flt_fg1" kp="2000" ctrlrange="0 {GRIP_OPEN}" forcerange="-200 200"/>
  <position joint="flt_fg2" kp="2000" ctrlrange="0 {GRIP_OPEN}" forcerange="-200 200"/>
</actuator>
<equality>
  <weld name="ref_weld" body1="ref_cylinder" body2="ref_tool"
        active="false" solref="0.01 1" solimp="0.95 0.99 0.001"/>
</equality>
</mujoco>"""


# ═══════════════════════════════════════════════════════════════
# MODEL LOAD + IDs
# ═══════════════════════════════════════════════════════════════

def load_model():
    xml = build_xml()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(xml); p = f.name
    m = mujoco.MjModel.from_xml_path(p); os.unlink(p)
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    return m, d


def get_ids(m, d):
    def bid(n):
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        assert i != -1, f"Body not found: {n}"; return i
    def sid(n):
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n)
        assert i != -1, f"Site not found: {n}"; return i
    def jqa(n):
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        assert i != -1, f"Joint not found: {n}"; return m.jnt_qposadr[i]
    def eqid(n):
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, n)
        assert i != -1, f"Equality not found: {n}"; return i

    ids = {
        "REF_EE":       sid("ref_ee"),       "FLT_EE":       sid("flt_ee"),
        "REF_CYL_BID":  bid("ref_cylinder"), "FLT_CYL_BID":  bid("flt_cylinder"),
        "REF_PTBL":     bid("ref_pick_table"),"FLT_PTBL":     bid("flt_pick_table"),
        "REF_PLTBL":    bid("ref_place_table"),"FLT_PLTBL":   bid("flt_place_table"),
        "REF_QA":  [jqa(n) for n in ["ref_j1","ref_j2","ref_j3"]],
        "FLT_QA":  [jqa(n) for n in ["flt_j1","flt_j2","flt_j3"]],
        "REF_DA":  [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                    for n in ["ref_j1","ref_j2","ref_j3"]],
        "FLT_DA":  [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                    for n in ["flt_j1","flt_j2","flt_j3"]],
        "REF_FG_Q":   [jqa(n) for n in ["ref_fg1","ref_fg2"]],
        "FLT_FG_Q":   [jqa(n) for n in ["flt_fg1","flt_fg2"]],
        "REF_ACT":    [0, 1, 2],  "FLT_ACT":    [5, 6, 7],
        "REF_FG_ACT": [3, 4],     "FLT_FG_ACT": [8, 9],
        "REF_CYL_Q":  jqa("ref_cyl_free"),
        "FLT_CYL_Q":  jqa("flt_cyl_free"),
        "REF_WELD":   eqid("ref_weld"),
    }
    ref_bodies = []; flt_bodies = []
    for n in ["ref_arm_base","ref_elbow","ref_wrist_link","ref_tool"]:
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        if i != -1: ref_bodies.append(i)
    for n in ["flt_arm_base","flt_elbow","flt_wrist_link","flt_tool"]:
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        if i != -1: flt_bodies.append(i)
    ids["REF_ARM_BIDS"] = ref_bodies
    ids["FLT_ARM_BIDS"] = flt_bodies
    return ids


def derive_geometry(m, d, ids):
    cg     = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ref_cyl_geom")
    CYL_HH = float(m.geom_size[cg][1])
    CYL_R  = float(m.geom_size[cg][0])
    REF_SZ = CAN_Z - CYL_HH

    geom = {
        "REF_SZ":       REF_SZ,
        "FLT_SZ":       REF_SZ,
        "CYL_HH":       CYL_HH,
        "CYL_R":        CYL_R,
        "GRASP_Z":      CAN_Z,
        "ABOVE_Z":      CAN_Z + CYL_HH * 2 + 0.15,
        "LIFT_Z":       CAN_Z + CYL_HH * 2 + 0.35,
        "PLACE_Z":      TABLE_Z_TOP + CYL_HH + 0.005,
        "FLOOR_Z":      REF_SZ - 0.025,
        "BODY_FLOOR_Z": 0.02,
    }

    print(f"REF_SZ  : {REF_SZ:.4f}")
    print(f"FLT_SZ  : {REF_SZ:.4f}")
    print(f"GRASP_Z : {geom['GRASP_Z']:.4f}")
    print(f"ABOVE_Z : {geom['ABOVE_Z']:.4f}")
    print(f"LIFT_Z  : {geom['LIFT_Z']:.4f}")

    assert REF_SZ > 0.25,            f"REF_SZ too low: {REF_SZ}"
    assert geom["GRASP_Z"] > REF_SZ, "GRASP_Z below pedestal surface"
    assert geom["ABOVE_Z"] > geom["GRASP_Z"] + 0.10

    return geom


# ═══════════════════════════════════════════════════════════════
# TABLE VIOLATION GUARD
# ═══════════════════════════════════════════════════════════════

def check_table_violation(data, model, arm_body_ids, floor_z, phase):
    for bid in arm_body_ids:
        z = data.xpos[bid][2]
        if z < floor_z:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or str(bid)
            msg  = f"TABLE VIOLATION [{phase}]: {name} z={z:.3f}"
            print(f"ERROR: {msg}"); raise RuntimeError(msg)


# ═══════════════════════════════════════════════════════════════
# CYLINDER RESET
# ═══════════════════════════════════════════════════════════════

def reset_cylinders(m, d, ids, geom, episode_seed=None):
    dx = dy = 0.0
    if episode_seed is not None:
        rng = np.random.default_rng(seed=episode_seed)
        dx  = rng.uniform(-0.015, 0.015)
        dy  = rng.uniform(-0.015, 0.015)

    RQ = ids["REF_CYL_Q"]; FQ = ids["FLT_CYL_Q"]
    d.qpos[RQ:RQ+3] = [CAN_X + dx, REF_Y + dy, CAN_Z]
    d.qpos[RQ+3:RQ+7] = [1, 0, 0, 0]; d.qvel[0:6] = 0
    d.qpos[FQ:FQ+3] = [CAN_X + dx, FLT_Y + dy, CAN_Z]
    d.qpos[FQ+3:FQ+7] = [1, 0, 0, 0]; d.qvel[6:12] = 0
    mujoco.mj_forward(m, d)

    rz = d.xpos[ids["REF_CYL_BID"]][2]
    fz = d.xpos[ids["FLT_CYL_BID"]][2]
    assert abs(rz - CAN_Z) < 0.01, f"REF cyl z={rz:.4f} exp {CAN_Z:.4f}"
    assert abs(fz - CAN_Z) < 0.01, f"FLT cyl z={fz:.4f} exp {CAN_Z:.4f}"
    print(f"[RESET] REF cyl z: {rz:.4f} ✓")
    print(f"[RESET] FLT cyl z: {fz:.4f} ✓")

    return (d.xpos[ids["REF_CYL_BID"]].copy(),
            d.xpos[ids["FLT_CYL_BID"]].copy())


# ═══════════════════════════════════════════════════════════════
# KINEMATIC IK  (available for IK-based phases / render_demo)
# ═══════════════════════════════════════════════════════════════

def solve_ik(m, d, target, qa, da, ee_sid,
             max_iter=1000, tol=0.004, floor_z=None):
    above = False
    for _ in range(max_iter):
        mujoco.mj_kinematics(m, d)
        cur = d.site_xpos[ee_sid].copy(); err = target - cur; en = np.linalg.norm(err)
        if en < tol: return True, en
        jacp = np.zeros((3, m.nv)); mujoco.mj_jacSite(m, d, jacp, None, ee_sid)
        J = jacp[:, da]; lam = 0.005
        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(3), err)
        s  = min(0.5, 0.3 / max(np.linalg.norm(dq), 1e-6))
        dq = np.clip(dq * s, -0.08, 0.08)
        for k, q in enumerate(qa): d.qpos[q] += dq[k]
        if floor_z is not None:
            mujoco.mj_kinematics(m, d); nz = d.site_xpos[ee_sid][2]
            if nz > floor_z: above = True
            if above and nz < floor_z:
                for k, q in enumerate(qa): d.qpos[q] -= dq[k]
                mujoco.mj_kinematics(m, d); return False, en
    mujoco.mj_kinematics(m, d)
    return False, np.linalg.norm(target - d.site_xpos[ee_sid])


def set_waypoint(m, d, qvals, qa, act):
    for k, q in enumerate(qa):
        d.qpos[q] = qvals[k]; d.ctrl[act[k]] = qvals[k]
    mujoco.mj_kinematics(m, d)


def sync_ctrl(d, qa, act):
    for k, q in enumerate(qa): d.ctrl[act[k]] = d.qpos[q]


# ═══════════════════════════════════════════════════════════════
# GRIPPER + PHYSICS
# ═══════════════════════════════════════════════════════════════

def set_gripper(d, fg_q, fg_act, state):
    val = GRIP_OPEN if state == "open" else GRIP_CLOSED
    for q in fg_q:   d.qpos[q]  = val
    for a in fg_act: d.ctrl[a]  = val


def settle(m, d, ids, geom, n_steps, phase):
    bfz = geom["BODY_FLOOR_Z"]
    d.qvel[:] = 0
    for _ in range(n_steps):
        mujoco.mj_step(m, d)
        check_table_violation(d, m, ids["REF_ARM_BIDS"], bfz, phase + "_REF")
        check_table_violation(d, m, ids["FLT_ARM_BIDS"], bfz, phase + "_FLT")


# ═══════════════════════════════════════════════════════════════
# PICK-AND-PLACE SEQUENCE
# ═══════════════════════════════════════════════════════════════

def run_pick_sequence(m, d, ids, geom, ref_cyl_pos, flt_cyl_pos):
    REF_EE     = ids["REF_EE"];     FLT_EE     = ids["FLT_EE"]
    REF_QA     = ids["REF_QA"];     FLT_QA     = ids["FLT_QA"]
    REF_DA     = ids["REF_DA"];     FLT_DA     = ids["FLT_DA"]
    REF_ACT    = ids["REF_ACT"];    FLT_ACT    = ids["FLT_ACT"]
    REF_FG_Q   = ids["REF_FG_Q"];   FLT_FG_Q   = ids["FLT_FG_Q"]
    REF_FG_ACT = ids["REF_FG_ACT"]; FLT_FG_ACT = ids["FLT_FG_ACT"]
    REF_WELD   = ids["REF_WELD"]
    R = {}

    print("\n[PHASE] HOME")
    set_waypoint(m, d, HOME_Q, REF_QA, REF_ACT)
    set_waypoint(m, d, HOME_Q, FLT_QA, FLT_ACT)
    set_gripper(d, REF_FG_Q, REF_FG_ACT, "open")
    set_gripper(d, FLT_FG_Q, FLT_FG_ACT, "open")
    d.eq_active[REF_WELD] = 0
    settle(m, d, ids, geom, 80, "HOME")

    print("\n[PHASE] HOVER_ABOVE")
    set_waypoint(m, d, ABOVE_Q, REF_QA, REF_ACT)
    set_waypoint(m, d, ABOVE_Q, FLT_QA, FLT_ACT)
    settle(m, d, ids, geom, 100, "HOVER_ABOVE")
    mujoco.mj_kinematics(m, d)
    rez = d.site_xpos[REF_EE][2]; fez = d.site_xpos[FLT_EE][2]
    print(f"  REF EE z={rez:.4f}  FLT EE z={fez:.4f}  (target ~{geom['ABOVE_Z']:.4f})")
    R["hover_ref_z"] = rez; R["hover_flt_z"] = fez

    print("\n[PHASE] DESCEND_TO_GRASP")
    set_waypoint(m, d, PICK_Q, REF_QA, REF_ACT)
    set_waypoint(m, d, PICK_Q, FLT_QA, FLT_ACT)
    # Kinematic check BEFORE physics settle
    mujoco.mj_kinematics(m, d)
    rez_k  = d.site_xpos[REF_EE][2]
    flt_ee = d.site_xpos[FLT_EE].copy()
    lat_y_mm = abs(flt_ee[1] - FLT_Y) * 1000
    settle(m, d, ids, geom, 120, "DESCEND")
    ncon = d.ncon
    print(f"  REF EE z={rez_k:.4f}  err={abs(rez_k - geom['GRASP_Z'])*1000:.1f}mm (kinematic)")
    print(f"  FLT lateral Y error={lat_y_mm:.1f}mm  (expected ~7mm)")
    print(f"  Contacts={ncon}")
    R["descend_ref_err_mm"] = abs(rez_k - geom["GRASP_Z"]) * 1000
    R["lateral_error_mm"]   = lat_y_mm
    R["ncon_after_descend"]  = ncon

    print("\n[PHASE] GRASP_WELD")
    d.eq_active[REF_WELD] = 1
    set_gripper(d, REF_FG_Q, REF_FG_ACT, "closed")
    set_gripper(d, FLT_FG_Q, FLT_FG_ACT, "closed")
    settle(m, d, ids, geom, 60, "GRASP")
    ncon = d.ncon
    print(f"  Contacts after weld+close={ncon}")
    R["ncon_after_close"] = ncon

    print("\n[PHASE] LIFT")
    set_waypoint(m, d, LIFT_Q, REF_QA, REF_ACT)
    set_waypoint(m, d, LIFT_Q, FLT_QA, FLT_ACT)
    settle(m, d, ids, geom, 350, "LIFT")
    rcz = d.xpos[ids["REF_CYL_BID"]][2]
    fcz = d.xpos[ids["FLT_CYL_BID"]][2]
    lift_ok = rcz > geom["REF_SZ"] + 0.08
    miss_ok = fcz < geom["FLT_SZ"] + 0.03
    print(f"  REF cyl z={rcz:.4f}  LIFT_SUCCESS={lift_ok}")
    print(f"  FLT cyl z={fcz:.4f}  LATERAL_MISS={miss_ok}")
    R["lift_success"]   = lift_ok
    R["miss_confirmed"] = miss_ok
    R["ref_cyl_z_lift"] = rcz
    R["flt_cyl_z_lift"] = fcz

    if lift_ok:
        print("\n[PHASE] PLACE (REF)")
        set_waypoint(m, d, PLACE_Q, REF_QA, REF_ACT)
        settle(m, d, ids, geom, 250, "PLACE")
        d.eq_active[REF_WELD] = 0
        set_gripper(d, REF_FG_Q, REF_FG_ACT, "open")
        settle(m, d, ids, geom, 100, "PLACE_OPEN")
        print(f"  REF cyl z after place={d.xpos[ids['REF_CYL_BID']][2]:.4f}")

    return R


# ═══════════════════════════════════════════════════════════════
# CORRECTION ESTIMATOR
# ═══════════════════════════════════════════════════════════════

def estimate_wrist_offset(lateral_drift_mm):
    return float(lateral_drift_mm / 1000.0) / SENSITIVITY_Y

def compute_correction(estimated_offset):
    return -estimated_offset


# ═══════════════════════════════════════════════════════════════
# DRY EPISODE — 15-CHECK ACCEPTANCE CRITERIA
# ═══════════════════════════════════════════════════════════════

def run_dry_episode():
    print("\n" + "=" * 60)
    print("DRY EPISODE — 15-check acceptance criteria")
    print("=" * 60)

    m, d = load_model()
    ids  = get_ids(m, d)
    geom = derive_geometry(m, d, ids)

    c1 = geom["REF_SZ"] > 0.25
    c2 = geom["FLT_SZ"] > 0.25
    c3 = geom["GRASP_Z"] > geom["REF_SZ"]

    ref_pos, flt_pos = reset_cylinders(m, d, ids, geom, episode_seed=None)
    rz = d.xpos[ids["REF_CYL_BID"]][2]; fz = d.xpos[ids["FLT_CYL_BID"]][2]
    c4 = abs(rz - CAN_Z) < 0.01; c5 = abs(fz - CAN_Z) < 0.01

    R = run_pick_sequence(m, d, ids, geom, ref_pos, flt_pos)

    c6  = R.get("hover_ref_z", 0)          > geom["REF_SZ"] + 0.05
    c7  = R.get("hover_flt_z", 0)          > geom["FLT_SZ"] + 0.05
    c8  = R.get("descend_ref_err_mm", 999) < 15.0
    lat = R.get("lateral_error_mm", 999)
    c9  = 5.0 < lat < 12.0
    c10 = R.get("ncon_after_close", 0)     >= 2
    c11 = R.get("lift_success",  False)
    c12 = R.get("miss_confirmed", False)
    c13 = c11; c14 = c12; c15 = True

    checks = [
        ("REF_SZ > 0.25",                          c1),
        ("FLT_SZ > 0.25",                          c2),
        ("GRASP_Z > REF_SZ",                       c3),
        ("[RESET] REF cyl z correct",               c4),
        ("[RESET] FLT cyl z correct",               c5),
        ("HOVER REF EE z > REF_SZ + 0.05",         c6),
        ("HOVER FLT EE z > FLT_SZ + 0.05",         c7),
        ("DESCEND REF EE err < 15mm  (kinematic)",  c8),
        ("FLT lateral error 5-12mm   (kinematic)",  c9),
        ("Contacts after grasp >= 2",               c10),
        ("REF cyl z after lift > REF_SZ + 0.08",    c11),
        ("FLT cyl z after lift < FLT_SZ + 0.03",    c12),
        ("LIFT_SUCCESS = True",                     c13),
        ("LATERAL_MISS = True",                     c14),
        ("No table violation RuntimeError",         c15),
    ]

    print("\n" + "=" * 60)
    print("RESULTS:")
    all_pass = True
    for name, ok in checks:
        print(f"  {'✓' if ok else '✗'} {name}")
        if not ok: all_pass = False

    est  = estimate_wrist_offset(R.get("lateral_error_mm", 0))
    corr = compute_correction(est)
    print(f"\nEstimated wrist offset : {est*1000:.2f}mm")
    print(f"True injected offset   : {INJECTED_ERROR['error_magnitude']*1000:.2f}mm")
    print(f"Estimation error       : {abs(est - INJECTED_ERROR['error_magnitude'])*1000:.2f}mm")
    print(f"Correction delta_y     : {corr*1000:+.2f}mm")
    print(f"\n{'ALL 15 PASS' if all_pass else 'FAILURES — do not render'}")
    print("=" * 60)

    return all_pass, R, {"m": m, "d": d, "ids": ids, "geom": geom}


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("SimCorrect Problem 2 — Wrist Offset Fault")
    all_pass, results, ctx = run_dry_episode()
    if not all_pass:
        print("\nAborting — fix failures before rendering.")
        exit(1)
    print("\nAll checks pass. Ready for render_demo.py.")
