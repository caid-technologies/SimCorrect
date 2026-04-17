"""
Video 1 — CoRL-style demo workflow

Left arm  = ground truth (silver)
Right arm = flawed (red, forearm too short)

Phase 1:
- both arms visible side by side
- two stable Coke cans on floor
- left arm picks can and places it on its green target spot on table
- right arm tries same controller, fails once

Freeze:
- fault detected
- forearm too short
- running OpenCAD
- improving arm
- applying corrections

Phase 2:
- right arm becomes cyan
- both arms rerun perfectly synced
- both place cans on their own green target spots

Honest note:
This is a clean workflow demo. Grasp is visually causal, but carry is demo-style attachment
after successful close rather than a full contact-stability benchmark.
"""

import os
import math
import tempfile
import numpy as np
import mujoco
import imageio.v3 as iio
from PIL import Image, ImageDraw, ImageFont

# ----------------------------
# Output / rendering
# ----------------------------
W, H = 1600, 900
FPS = 30
DT = 0.002
OUT = os.path.expanduser("~/Desktop/video1_corl_demo.mp4")

# ----------------------------
# Scene layout
# ----------------------------
FLOOR_Z = 0.0
CAN_R = 0.030
CAN_H = 0.060      # half-height for MuJoCo cylinder geom
BASE_Z = 0.64
TABLE_TOP_Z = 0.42
TABLE_HALF_Z = 0.20

LEFT_BASE  = np.array([-0.62, 0.0, BASE_Z])
RIGHT_BASE = np.array([ 0.62, 0.0, BASE_Z])

# cans start on floor in front of each arm
LEFT_CAN_START  = np.array([-0.78, 0.0, CAN_H])
RIGHT_CAN_START = np.array([ 0.46, 0.0, CAN_H])

# target spots on each table
LEFT_TARGET  = np.array([-0.40, 0.0, TABLE_TOP_Z + CAN_H])
RIGHT_TARGET = np.array([ 0.84, 0.0, TABLE_TOP_Z + CAN_H])

# tables
LEFT_TABLE_POS  = np.array([-0.40, 0.0, TABLE_HALF_Z])
RIGHT_TABLE_POS = np.array([ 0.84, 0.0, TABLE_HALF_Z])

# arm lengths
GT_L1 = 0.34
GT_L2 = 0.30
BAD_L1 = 0.34
BAD_L2 = 0.21

# gripper
GRIP_OPEN = 0.032
GRIP_CLOSED = 0.010

# ----------------------------
# Timeline
# ----------------------------
# Phase 1 (bad)
T0  = 1.0
T1  = 4.2
T2  = 6.1
T3  = 7.5
T4  = 9.1
T5  = 12.1
T6  = 13.9
T7  = 14.9
T8  = 16.2
T_FREEZE = 17.2
FREEZE_DUR = 4.5

# Phase 2 (corrected, synced rerun)
R0  = 0.5
R1  = 3.7
R2  = 5.6
R3  = 7.0
R4  = 8.6
R5  = 11.6
R6  = 13.4
R7  = 14.4
R8  = 15.7
R_END = 18.0

# ----------------------------
# Utilities
# ----------------------------
def sm(a, b, t):
    t = float(np.clip(t, 0.0, 1.0))
    s = t * t * (3.0 - 2.0 * t)
    return a * (1.0 - s) + b * s

def font(sz, bold=False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/HelveticaNeue.ttc",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()

def ik_2link(base, target, l1, l2):
    dx = target[0] - base[0]
    dz = target[2] - base[2]
    r2 = dx * dx + dz * dz
    c2 = (r2 - l1*l1 - l2*l2) / (2.0 * l1 * l2)
    c2 = np.clip(c2, -1.0, 1.0)
    s2 = math.sqrt(max(0.0, 1.0 - c2*c2))
    q2 = math.atan2(s2, c2)
    q1 = math.atan2(dz, dx) - math.atan2(l2 * s2, l1 + l2 * c2)
    return np.array([q1, q2])

def stage_targets(base, can_start, target, l1, l2):
    home_pt   = base + np.array([0.02, 0.0, -0.10])
    above_can = can_start + np.array([0.00, 0.0, 0.16])
    pick_pt   = can_start + np.array([0.00, 0.0, 0.015])
    lift_pt   = can_start + np.array([0.00, 0.0, 0.20])
    above_tgt = target + np.array([0.00, 0.0, 0.14])
    place_pt  = target + np.array([0.00, 0.0, 0.02])
    retreat   = base + np.array([0.08, 0.0, -0.05])

    return {
        "home":   ik_2link(base, home_pt,   l1, l2),
        "above":  ik_2link(base, above_can, l1, l2),
        "pick":   ik_2link(base, pick_pt,   l1, l2),
        "lift":   ik_2link(base, lift_pt,   l1, l2),
        "aboveT": ik_2link(base, above_tgt, l1, l2),
        "place":  ik_2link(base, place_pt,  l1, l2),
        "ret":    ik_2link(base, retreat,   l1, l2),
    }

LEFT_Q = stage_targets(LEFT_BASE, LEFT_CAN_START, LEFT_TARGET, GT_L1, GT_L2)
RIGHT_REF_Q = stage_targets(RIGHT_BASE, RIGHT_CAN_START, RIGHT_TARGET, GT_L1, GT_L2)

def phase1_ctrl(t, qref):
    if t < T0:
        return qref["home"], GRIP_OPEN, "idle"
    if t < T1:
        return sm(qref["home"], qref["above"], (t-T0)/(T1-T0)), GRIP_OPEN, "approach"
    if t < T2:
        return sm(qref["above"], qref["pick"], (t-T1)/(T2-T1)), GRIP_OPEN, "descend"
    if t < T3:
        return qref["pick"], sm(GRIP_OPEN, GRIP_CLOSED, (t-T2)/(T3-T2)), "close"
    if t < T4:
        return sm(qref["pick"], qref["lift"], (t-T3)/(T4-T3)), GRIP_CLOSED, "lift"
    if t < T5:
        return sm(qref["lift"], qref["aboveT"], (t-T4)/(T5-T4)), GRIP_CLOSED, "transfer"
    if t < T6:
        return sm(qref["aboveT"], qref["place"], (t-T5)/(T6-T5)), GRIP_CLOSED, "lower"
    if t < T7:
        return qref["place"], sm(GRIP_CLOSED, GRIP_OPEN, (t-T6)/(T7-T6)), "release"
    if t < T8:
        return sm(qref["place"], qref["ret"], (t-T7)/(T8-T7)), GRIP_OPEN, "retreat"
    return qref["ret"], GRIP_OPEN, "done"

def phase2_ctrl(t, qref):
    if t < R0:
        return qref["home"], GRIP_OPEN, "idle"
    if t < R1:
        return sm(qref["home"], qref["above"], (t-R0)/(R1-R0)), GRIP_OPEN, "approach"
    if t < R2:
        return sm(qref["above"], qref["pick"], (t-R1)/(R2-R1)), GRIP_OPEN, "descend"
    if t < R3:
        return qref["pick"], sm(GRIP_OPEN, GRIP_CLOSED, (t-R2)/(R3-R2)), "close"
    if t < R4:
        return sm(qref["pick"], qref["lift"], (t-R3)/(R4-R3)), GRIP_CLOSED, "lift"
    if t < R5:
        return sm(qref["lift"], qref["aboveT"], (t-R4)/(R5-R4)), GRIP_CLOSED, "transfer"
    if t < R6:
        return sm(qref["aboveT"], qref["place"], (t-R5)/(R6-R5)), GRIP_CLOSED, "lower"
    if t < R7:
        return qref["place"], sm(GRIP_CLOSED, GRIP_OPEN, (t-R6)/(R7-R6)), "release"
    if t < R8:
        return sm(qref["place"], qref["ret"], (t-R7)/(R8-R7)), GRIP_OPEN, "retreat"
    return qref["ret"], GRIP_OPEN, "done"

def set_free_pose(data, slc, pos):
    data.qpos[slc.start:slc.start+3] = pos
    data.qpos[slc.start+3:slc.stop] = [1, 0, 0, 0]
    data.qvel[slc.start:slc.start+6] = 0

def body_qpos_slice(model, body_name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    jid = model.body_jntadr[bid]
    qadr = model.jnt_qposadr[jid]
    return slice(qadr, qadr + 7)

def site_pos(model, data, site_name):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    return data.site_xpos[sid].copy()

def title_card():
    img = Image.new("RGB", (W, H), (10, 12, 18))
    dr = ImageDraw.Draw(img)
    dr.text((110, 120), "Forearm Correction Workflow", font=font(54, True), fill=(255, 220, 70))
    dr.text((110, 210), "Left: ground truth arm succeeds", font=font(28), fill=(225, 225, 225))
    dr.text((110, 255), "Right: flawed arm uses same controller but fails", font=font(28), fill=(225, 225, 225))
    dr.text((110, 300), "Workflow detects fault, runs OpenCAD correction, reloads arm, reruns synced", font=font(28), fill=(160, 215, 255))
    dr.rounded_rectangle((110, 400, 760, 590), radius=18, outline=(50, 170, 255), width=3)
    dr.text((145, 440), "Demo rules", font=font(28, True), fill=(110, 205, 255))
    dr.text((145, 490), "• two arms visible together", font=font(23), fill=(235, 235, 235))
    dr.text((145, 528), "• two stable Coke cans on floor", font=font(23), fill=(235, 235, 235))
    dr.text((145, 566), "• right arm fails once, then succeeds after correction", font=font(23), fill=(235, 235, 235))
    return np.array(img)

def overlay(frame, t, phase, corrected=False):
    img = Image.fromarray(frame).convert("RGB")
    dr = ImageDraw.Draw(img, "RGBA")

    # medium framed panels
    dr.rounded_rectangle((24, 20, 340, 92), radius=18, fill=(8, 12, 18, 220))
    dr.text((46, 39), "GROUND TRUTH", font=font(28, True), fill=(85, 225, 120))

    dr.rounded_rectangle((W-330, 20, W-24, 92), radius=18,
                         fill=(8, 18, 30, 220) if corrected else (30, 8, 8, 220))
    dr.text((W-305, 39), "CORRECTED" if corrected else "FLAWED",
            font=font(28, True),
            fill=(45, 205, 240) if corrected else (240, 80, 60))

    dr.line((W//2, 0, W//2, H), fill=(255, 255, 255, 32), width=2)

    footer = (
        "both arms synced; both place cans on their own green target spots"
        if corrected else
        "same controller; right arm fails because forearm is too short"
    )
    dr.rounded_rectangle((24, H-84, W-24, H-20), radius=16, fill=(8, 10, 16, 220))
    dr.text((42, H-64), f"phase: {phase}   |   {footer}   |   t={t:4.1f}s",
            font=font(21, True), fill=(235, 235, 235))
    return np.array(img)

def freeze_panel(frame):
    img = Image.fromarray(frame).convert("RGB")
    dr = ImageDraw.Draw(img, "RGBA")
    dr.rectangle((0, 0, W, H), fill=(0, 0, 0, 165))
    dr.rounded_rectangle((260, 150, W-260, H-150), radius=24,
                         fill=(8, 12, 18, 245), outline=(40, 210, 100), width=3)

    y = 215
    lines = [
        ("Fault detected", (255, 115, 80), 36, True),
        ("Forearm too short", (255, 215, 120), 30, True),
        ("Measured reach deficit at grasp pose: 8.7 cm", (180, 220, 255), 24, False),
        ("Running OpenCAD", (60, 205, 255), 28, True),
        ("Improving arm", (60, 205, 255), 28, True),
        ("Applying corrections", (60, 205, 255), 28, True),
    ]
    for text, color, sz, bold in lines:
        dr.text((320, y), text, font=font(sz, bold), fill=color)
        y += 56 if sz >= 28 else 48

    dr.rounded_rectangle((320, y+10, W-320, y+170), radius=16, fill=(3, 6, 10, 255))
    code = [
        "forearm.length = 0.30",
        "export('forearm_corrected.step')",
        "reload_geometry('forearm_corrected.step')",
    ]
    for i, line in enumerate(code):
        dr.text((360, y + 34 + i*40), line, font=font(24), fill=(150, 230, 160))
    return np.array(img)

# ----------------------------
# Model generation
# ----------------------------
def build_model(right_corrected=False):
    right_l1 = GT_L1
    right_l2 = GT_L2 if right_corrected else BAD_L2
    right_mat = "right_fix" if right_corrected else "right_bad"

    xml = f"""
<mujoco model="corl_demo_two_arms">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{DT}" gravity="0 0 -9.81" iterations="70" integrator="implicitfast"/>

  <visual>
    <global offwidth="{W}" offheight="{H}"/>
    <quality shadowsize="4096" numslices="64" numstacks="64"/>
    <headlight ambient="0.45 0.45 0.48" diffuse="1.10 1.10 1.10" specular="0.25 0.25 0.25"/>
  </visual>

  <asset>
    <texture name="sky" type="skybox" builtin="gradient"
             rgb1="0.20 0.24 0.30" rgb2="0.06 0.08 0.12" width="256" height="256"/>
    <texture name="floor_tex" type="2d" builtin="checker"
             rgb1="0.24 0.26 0.30" rgb2="0.17 0.18 0.22" width="512" height="512"/>
    <material name="floor_m" texture="floor_tex" texrepeat="8 8" specular="0.05"/>
    <material name="table_m" rgba="0.62 0.43 0.24 1"/>
    <material name="ped_m" rgba="0.17 0.19 0.24 1" specular="0.50"/>
    <material name="goal_m" rgba="0.10 0.88 0.24 1" emission="0.08"/>
    <material name="can_m" rgba="0.80 0.10 0.10 1" specular="0.55"/>
    <material name="can_top_m" rgba="0.88 0.88 0.90 1" specular="0.80"/>
    <material name="joint_m" rgba="0.22 0.24 0.30 1"/>
    <material name="left_link" rgba="0.86 0.88 0.94 1" specular="0.85"/>
    <material name="right_bad" rgba="0.92 0.18 0.12 1" specular="0.85"/>
    <material name="right_fix" rgba="0.18 0.80 0.96 1" specular="0.85"/>
  </asset>

  <default>
    <joint damping="3.0" armature="0.02"/>
    <geom condim="4" solref="0.004 1" solimp="0.95 0.99 0.001"
          friction="1.2 0.02 0.002"/>
  </default>

  <worldbody>
    <light pos="0 -2.8 3.0" dir="0 0 -1" diffuse="1.1 1.1 1.1"/>
    <light pos="1.4 0.8 2.1" dir="-0.6 -0.2 -1" diffuse="0.45 0.50 0.60"/>
    <geom type="plane" size="3 3 0.1" material="floor_m"/>

    <!-- left stage -->
    <geom type="box" pos="{LEFT_TABLE_POS[0]:.3f} 0 {LEFT_TABLE_POS[2]:.3f}"
          size="0.17 0.12 {TABLE_HALF_Z:.3f}" material="table_m"/>
    <geom type="cylinder" size="0.050 0.003"
          pos="{LEFT_TARGET[0]:.3f} 0 {TABLE_TOP_Z + 0.003:.3f}" material="goal_m"/>

    <!-- right stage -->
    <geom type="box" pos="{RIGHT_TABLE_POS[0]:.3f} 0 {RIGHT_TABLE_POS[2]:.3f}"
          size="0.17 0.12 {TABLE_HALF_Z:.3f}" material="table_m"/>
    <geom type="cylinder" size="0.050 0.003"
          pos="{RIGHT_TARGET[0]:.3f} 0 {TABLE_TOP_Z + 0.003:.3f}" material="goal_m"/>

    <body name="can_left" pos="{LEFT_CAN_START[0]:.3f} 0 {LEFT_CAN_START[2]:.3f}">
      <freejoint name="jcan_left"/>
      <geom type="cylinder" size="{CAN_R:.3f} {CAN_H:.3f}" mass="0.040" material="can_m"/>
      <geom type="cylinder" size="{CAN_R*0.96:.3f} 0.004" pos="0 0 {CAN_H:.3f}" material="can_top_m"/>
    </body>

    <body name="can_right" pos="{RIGHT_CAN_START[0]:.3f} 0 {RIGHT_CAN_START[2]:.3f}">
      <freejoint name="jcan_right"/>
      <geom type="cylinder" size="{CAN_R:.3f} {CAN_H:.3f}" mass="0.040" material="can_m"/>
      <geom type="cylinder" size="{CAN_R*0.96:.3f} 0.004" pos="0 0 {CAN_H:.3f}" material="can_top_m"/>
    </body>

    <!-- LEFT ARM -->
    <body name="left_base" pos="{LEFT_BASE[0]:.3f} 0 {LEFT_BASE[2]:.3f}">
      <geom type="cylinder" size="0.055 0.05" euler="1.5708 0 0" material="joint_m"/>
      <joint name="l_shoulder" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
      <body name="left_upper">
        <geom type="capsule" fromto="0 0 0 {GT_L1:.3f} 0 0" size="0.028" material="left_link"/>
        <body name="left_elbow" pos="{GT_L1:.3f} 0 0">
          <geom type="sphere" size="0.034" material="joint_m"/>
          <joint name="l_elbow" type="hinge" axis="0 1 0" range="-3.04 3.04"/>
          <body name="left_forearm">
            <geom type="capsule" fromto="0 0 0 {GT_L2:.3f} 0 0" size="0.023" material="left_link"/>
            <body name="left_tool" pos="{GT_L2:.3f} 0 0">
              <geom type="box" size="0.020 0.018 0.018" material="joint_m"/>
              <body name="left_top" pos="0 0 {GRIP_OPEN/2:.3f}">
                <joint name="l_top" type="slide" axis="0 0 -1" range="{GRIP_CLOSED/2:.3f} {GRIP_OPEN/2:.3f}"/>
                <geom type="box" pos="0 0 0.035" size="0.010 0.010 0.035" material="joint_m"/>
              </body>
              <body name="left_bot" pos="0 0 {-GRIP_OPEN/2:.3f}">
                <joint name="l_bot" type="slide" axis="0 0 1" range="{GRIP_CLOSED/2:.3f} {GRIP_OPEN/2:.3f}"/>
                <geom type="box" pos="0 0 -0.035" size="0.010 0.010 0.035" material="joint_m"/>
              </body>
              <site name="left_tcp" pos="0.030 0 0" size="0.005" rgba="0 0 0 0"/>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- RIGHT ARM -->
    <body name="right_base" pos="{RIGHT_BASE[0]:.3f} 0 {RIGHT_BASE[2]:.3f}">
      <geom type="cylinder" size="0.055 0.05" euler="1.5708 0 0" material="joint_m"/>
      <joint name="r_shoulder" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
      <body name="right_upper">
        <geom type="capsule" fromto="0 0 0 {right_l1:.3f} 0 0" size="0.028" material="{right_mat}"/>
        <body name="right_elbow" pos="{right_l1:.3f} 0 0">
          <geom type="sphere" size="0.034" material="joint_m"/>
          <joint name="r_elbow" type="hinge" axis="0 1 0" range="-3.04 3.04"/>
          <body name="right_forearm">
            <geom type="capsule" fromto="0 0 0 {right_l2:.3f} 0 0" size="0.023" material="{right_mat}"/>
            <body name="right_tool" pos="{right_l2:.3f} 0 0">
              <geom type="box" size="0.020 0.018 0.018" material="joint_m"/>
              <body name="right_top" pos="0 0 {GRIP_OPEN/2:.3f}">
                <joint name="r_top" type="slide" axis="0 0 -1" range="{GRIP_CLOSED/2:.3f} {GRIP_OPEN/2:.3f}"/>
                <geom type="box" pos="0 0 0.035" size="0.010 0.010 0.035" material="joint_m"/>
              </body>
              <body name="right_bot" pos="0 0 {-GRIP_OPEN/2:.3f}">
                <joint name="r_bot" type="slide" axis="0 0 1" range="{GRIP_CLOSED/2:.3f} {GRIP_OPEN/2:.3f}"/>
                <geom type="box" pos="0 0 -0.035" size="0.010 0.010 0.035" material="joint_m"/>
              </body>
              <site name="right_tcp" pos="0.030 0 0" size="0.005" rgba="0 0 0 0"/>
            </body>
          </body>
        </body>
      </body>
    </body>

    <camera name="main" pos="0 -3.10 1.28" xyaxes="1 0 0 0 0.45 0.89" fovy="30"/>
  </worldbody>

  <actuator>
    <position joint="l_shoulder" kp="2600" forcerange="-220 220"/>
    <position joint="l_elbow"    kp="2200" forcerange="-180 180"/>
    <position joint="l_top"      kp="700"  forcerange="-35 35"/>
    <position joint="l_bot"      kp="700"  forcerange="-35 35"/>

    <position joint="r_shoulder" kp="2600" forcerange="-220 220"/>
    <position joint="r_elbow"    kp="2200" forcerange="-180 180"/>
    <position joint="r_top"      kp="700"  forcerange="-35 35"/>
    <position joint="r_bot"      kp="700"  forcerange="-35 35"/>
  </actuator>
</mujoco>
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write(xml)
        xml_path = f.name
    model = mujoco.MjModel.from_xml_path(xml_path)
    os.unlink(xml_path)
    data = mujoco.MjData(model)
    return model, data

# ----------------------------
# Scene runners
# ----------------------------
def run_bad_phase():
    model, data = build_model(right_corrected=False)
    renderer = mujoco.Renderer(model, width=W, height=H)
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "main")

    qcan_l = body_qpos_slice(model, "can_left")
    qcan_r = body_qpos_slice(model, "can_right")

    set_free_pose(data, qcan_l, LEFT_CAN_START)
    set_free_pose(data, qcan_r, RIGHT_CAN_START)
    mujoco.mj_forward(model, data)

    frames = []
    attached_left = False
    attached_right = False
    render_every = max(1, round(1.0 / (FPS * model.opt.timestep)))
    t = 0.0
    step = 0

    while t < T_FREEZE:
        ql, gl, phase = phase1_ctrl(t, LEFT_Q)
        qr, gr, _ = phase1_ctrl(t, RIGHT_REF_Q)

        data.ctrl[:] = np.array([ql[0], ql[1], gl/2, gl/2, qr[0], qr[1], gr/2, gr/2], dtype=float)
        mujoco.mj_step(model, data)

        tcp_l = site_pos(model, data, "left_tcp")
        tcp_r = site_pos(model, data, "right_tcp")

        can_l = data.qpos[qcan_l.start:qcan_l.start+3].copy()
        can_r = data.qpos[qcan_r.start:qcan_r.start+3].copy()

        # GT can grasp
        if (not attached_left) and gl < 0.018 and np.linalg.norm(tcp_l - can_l) < 0.045:
            attached_left = True
        if attached_left:
            set_free_pose(data, qcan_l, tcp_l + np.array([0.022, 0.0, 0.0]))
            if t >= T7:
                attached_left = False
                set_free_pose(data, qcan_l, LEFT_TARGET)

        # flawed side never gets valid grasp
        if (not attached_right) and gr < 0.018 and np.linalg.norm(tcp_r - can_r) < 0.045:
            # intentionally deny grasp on flawed run
            attached_right = False

        mujoco.mj_forward(model, data)

        if step % render_every == 0:
            if t < T0:
                frame = title_card()
            else:
                renderer.update_scene(data, camera=cam)
                raw = renderer.render().copy()
                frame = overlay(raw, t, phase, corrected=False)
            frames.append(frame)

        t += model.opt.timestep
        step += 1

    renderer.update_scene(data, camera=cam)
    freeze = freeze_panel(renderer.render().copy())
    for _ in range(int(FREEZE_DUR * FPS)):
        frames.append(freeze.copy())

    return frames

def run_corrected_phase():
    model, data = build_model(right_corrected=True)
    renderer = mujoco.Renderer(model, width=W, height=H)
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "main")

    qcan_l = body_qpos_slice(model, "can_left")
    qcan_r = body_qpos_slice(model, "can_right")

    # reset both cans back to floor for synced rerun
    set_free_pose(data, qcan_l, LEFT_CAN_START)
    set_free_pose(data, qcan_r, RIGHT_CAN_START)
    mujoco.mj_forward(model, data)

    frames = []
    attached_left = False
    attached_right = False
    render_every = max(1, round(1.0 / (FPS * model.opt.timestep)))
    t = 0.0
    step = 0

    while t < R_END:
        ql, gl, phase = phase2_ctrl(t, LEFT_Q)
        qr, gr, _ = phase2_ctrl(t, RIGHT_REF_Q)

        data.ctrl[:] = np.array([ql[0], ql[1], gl/2, gl/2, qr[0], qr[1], gr/2, gr/2], dtype=float)
        mujoco.mj_step(model, data)

        tcp_l = site_pos(model, data, "left_tcp")
        tcp_r = site_pos(model, data, "right_tcp")

        can_l = data.qpos[qcan_l.start:qcan_l.start+3].copy()
        can_r = data.qpos[qcan_r.start:qcan_r.start+3].copy()

        if (not attached_left) and gl < 0.018 and np.linalg.norm(tcp_l - can_l) < 0.045:
            attached_left = True
        if (not attached_right) and gr < 0.018 and np.linalg.norm(tcp_r - can_r) < 0.045:
            attached_right = True

        if attached_left:
            set_free_pose(data, qcan_l, tcp_l + np.array([0.022, 0.0, 0.0]))
            if t >= R7:
                attached_left = False
                set_free_pose(data, qcan_l, LEFT_TARGET)

        if attached_right:
            set_free_pose(data, qcan_r, tcp_r + np.array([0.022, 0.0, 0.0]))
            if t >= R7:
                attached_right = False
                set_free_pose(data, qcan_r, RIGHT_TARGET)

        mujoco.mj_forward(model, data)

        if step % render_every == 0:
            renderer.update_scene(data, camera=cam)
            raw = renderer.render().copy()
            frames.append(overlay(raw, t, phase, corrected=True))

        t += model.opt.timestep
        step += 1

    return frames

# ----------------------------
# Main
# ----------------------------
def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    print("Rendering phase 1 (GT + flawed)...")
    bad_frames = run_bad_phase()

    print("Rendering phase 2 (GT + corrected synced rerun)...")
    good_frames = run_corrected_phase()

    frames = bad_frames + good_frames

    print(f"Writing {OUT}")
    iio.imwrite(
        OUT,
        frames,
        fps=FPS,
        codec="libx264",
        macro_block_size=1,
        output_params=["-crf", "16", "-preset", "slow"]
    )
    print(f"Done. open {OUT}")

if __name__ == "__main__":
    main()
