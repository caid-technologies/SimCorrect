"""
Video 1 — Clean two-arm workflow
Left  = GT arm (silver) succeeds
Right = flawed arm (red) fails because forearm is too short
Freeze -> overlay diagnosis/OpenCAD correction
Right arm becomes cyan and reruns in sync with GT, then succeeds

This is a clear visual workflow demo.
It is not a force-accurate grasp benchmark.
"""

import os
import math
import tempfile
import numpy as np
import mujoco
import imageio.v3 as iio
from PIL import Image, ImageDraw, ImageFont

W, H = 1600, 900
FPS = 30
DUR = 26.0
OUT = os.path.expanduser("~/Desktop/video1_forearm_clean_workflow.mp4")

DT = 0.002
BALL_R = 0.024
PED_H = 0.090
TABLE_H = 0.38

LEFT_BASE  = np.array([-0.72, 0.0, 0.66])
RIGHT_BASE = np.array([ 0.72, 0.0, 0.66])

LEFT_BALL_START  = np.array([-0.88, 0.0, TABLE_H + PED_H * 2 + BALL_R])
RIGHT_BALL_START = np.array([ 0.56, 0.0, TABLE_H + PED_H * 2 + BALL_R])

LEFT_TABLE_TARGET  = np.array([-0.52, 0.0, TABLE_H + 0.12])
RIGHT_TABLE_TARGET = np.array([ 0.92, 0.0, TABLE_H + 0.12])

GT_L1, GT_L2 = 0.34, 0.29
BAD_L1, BAD_L2 = 0.34, 0.20

GRIP_OPEN = 0.030
GRIP_CLOSED = 0.010

# Timing
T_INTRO0 = 0.0
T_INTRO1 = 1.4
T_MOVE1  = 5.0
T_DOWN1  = 7.0
T_CLOSE1 = 8.3
T_LIFT1  = 10.0
T_MOVE2  = 13.4
T_DOWN2  = 15.0
T_OPEN1  = 16.0
T_RET1   = 17.6
T_FREEZE = 18.4
T_RESUME = 21.8
T_END    = 26.0

def sm(a, b, t):
    t = float(np.clip(t, 0.0, 1.0))
    s = t * t * (3 - 2 * t)
    return a * (1 - s) + b * s

def font(sz, bold=False):
    paths = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/HelveticaNeue.ttc",
    ]
    for p in paths:
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()

def ik_2link(base, target, l1, l2):
    dx = target[0] - base[0]
    dz = target[2] - base[2]
    r2 = dx * dx + dz * dz
    c2 = (r2 - l1 * l1 - l2 * l2) / (2 * l1 * l2)
    c2 = np.clip(c2, -1.0, 1.0)
    s2 = math.sqrt(max(0.0, 1.0 - c2 * c2))
    q2 = math.atan2(s2, c2)
    q1 = math.atan2(dz, dx) - math.atan2(l2 * s2, l1 + l2 * c2)
    return np.array([q1, q2])

def tcp_from_q(base, q, l1, l2):
    a1, a2 = q[0], q[0] + q[1]
    x = base[0] + l1 * math.cos(a1) + l2 * math.cos(a2)
    z = base[2] + l1 * math.sin(a1) + l2 * math.sin(a2)
    return np.array([x, 0.0, z])

LEFT_HOME   = ik_2link(LEFT_BASE,  np.array([-0.76, 0.0, 0.52]), GT_L1, GT_L2)
LEFT_ABOVE  = ik_2link(LEFT_BASE,  LEFT_BALL_START  + np.array([0.00, 0.0, 0.14]), GT_L1, GT_L2)
LEFT_PICK   = ik_2link(LEFT_BASE,  LEFT_BALL_START  + np.array([0.00, 0.0, 0.01]), GT_L1, GT_L2)
LEFT_LIFT   = ik_2link(LEFT_BASE,  LEFT_BALL_START  + np.array([0.00, 0.0, 0.18]), GT_L1, GT_L2)
LEFT_ABOVE_T= ik_2link(LEFT_BASE,  LEFT_TABLE_TARGET + np.array([0.00, 0.0, 0.10]), GT_L1, GT_L2)
LEFT_PLACE  = ik_2link(LEFT_BASE,  LEFT_TABLE_TARGET + np.array([0.00, 0.0, 0.01]), GT_L1, GT_L2)
LEFT_RET    = ik_2link(LEFT_BASE,  np.array([-0.68, 0.0, 0.56]), GT_L1, GT_L2)

RIGHT_REF_HOME    = ik_2link(RIGHT_BASE, np.array([0.68, 0.0, 0.52]), GT_L1, GT_L2)
RIGHT_REF_ABOVE   = ik_2link(RIGHT_BASE, RIGHT_BALL_START + np.array([0.00, 0.0, 0.14]), GT_L1, GT_L2)
RIGHT_REF_PICK    = ik_2link(RIGHT_BASE, RIGHT_BALL_START + np.array([0.00, 0.0, 0.01]), GT_L1, GT_L2)
RIGHT_REF_LIFT    = ik_2link(RIGHT_BASE, RIGHT_BALL_START + np.array([0.00, 0.0, 0.18]), GT_L1, GT_L2)
RIGHT_REF_ABOVE_T = ik_2link(RIGHT_BASE, RIGHT_TABLE_TARGET + np.array([0.00, 0.0, 0.10]), GT_L1, GT_L2)
RIGHT_REF_PLACE   = ik_2link(RIGHT_BASE, RIGHT_TABLE_TARGET + np.array([0.00, 0.0, 0.01]), GT_L1, GT_L2)
RIGHT_REF_RET     = ik_2link(RIGHT_BASE, np.array([0.76, 0.0, 0.56]), GT_L1, GT_L2)

def phase_ctrl(t, q_home, q_above, q_pick, q_lift, q_above_t, q_place, q_ret):
    if t < T_INTRO1:
        return q_home, GRIP_OPEN, "idle"
    if t < T_MOVE1:
        return sm(q_home, q_above, (t - T_INTRO1) / (T_MOVE1 - T_INTRO1)), GRIP_OPEN, "approach"
    if t < T_DOWN1:
        return sm(q_above, q_pick, (t - T_MOVE1) / (T_DOWN1 - T_MOVE1)), GRIP_OPEN, "descend"
    if t < T_CLOSE1:
        return q_pick, sm(GRIP_OPEN, GRIP_CLOSED, (t - T_DOWN1) / (T_CLOSE1 - T_DOWN1)), "close"
    if t < T_LIFT1:
        return sm(q_pick, q_lift, (t - T_CLOSE1) / (T_LIFT1 - T_CLOSE1)), GRIP_CLOSED, "lift"
    if t < T_MOVE2:
        return sm(q_lift, q_above_t, (t - T_LIFT1) / (T_MOVE2 - T_LIFT1)), GRIP_CLOSED, "transfer"
    if t < T_DOWN2:
        return sm(q_above_t, q_place, (t - T_MOVE2) / (T_DOWN2 - T_MOVE2)), GRIP_CLOSED, "lower"
    if t < T_OPEN1:
        return q_place, sm(GRIP_CLOSED, GRIP_OPEN, (t - T_DOWN2) / (T_OPEN1 - T_DOWN2)), "release"
    if t < T_RET1:
        return sm(q_place, q_ret, (t - T_OPEN1) / (T_RET1 - T_OPEN1)), GRIP_OPEN, "retreat"
    return q_ret, GRIP_OPEN, "done"

MJCF = f"""
<mujoco model="two_arm_workflow">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{DT}" gravity="0 0 -9.81" iterations="60" integrator="implicitfast"/>

  <visual>
    <global offwidth="{W}" offheight="{H}"/>
    <quality shadowsize="4096" numslices="64" numstacks="64"/>
    <headlight ambient="0.45 0.45 0.48" diffuse="1.10 1.10 1.10" specular="0.25 0.25 0.25"/>
  </visual>

  <asset>
    <texture name="sky" type="skybox" builtin="gradient"
             rgb1="0.20 0.24 0.30" rgb2="0.07 0.08 0.12" width="256" height="256"/>
    <texture name="floor_tex" type="2d" builtin="checker"
             rgb1="0.24 0.26 0.30" rgb2="0.17 0.18 0.22" width="512" height="512"/>
    <material name="floor_m" texture="floor_tex" texrepeat="8 8" specular="0.05"/>
    <material name="table_m" rgba="0.62 0.43 0.24 1"/>
    <material name="ped_m" rgba="0.17 0.19 0.24 1" specular="0.50"/>
    <material name="goal_m" rgba="0.10 0.88 0.24 1" emission="0.08"/>
    <material name="ball_m" rgba="0.98 0.88 0.12 1" specular="0.90"/>
    <material name="joint_m" rgba="0.22 0.24 0.30 1"/>
    <material name="left_link" rgba="0.86 0.88 0.94 1" specular="0.85"/>
    <material name="right_bad" rgba="0.92 0.18 0.12 1" specular="0.85"/>
    <material name="right_fix" rgba="0.18 0.80 0.96 1" specular="0.85"/>
  </asset>

  <default>
    <joint damping="3.0" armature="0.02"/>
    <geom condim="4" solref="0.004 1" solimp="0.95 0.99 0.001" friction="1.0 0.01 0.001"/>
  </default>

  <worldbody>
    <light pos="0 -2.4 2.8" dir="0 0 -1" diffuse="1.1 1.1 1.1"/>
    <light pos="1.5 0.5 2.0" dir="-0.7 -0.1 -1" diffuse="0.45 0.50 0.60"/>
    <geom type="plane" size="3 3 0.1" material="floor_m"/>

    <!-- left stage -->
    <geom type="box" pos="-0.68 0 0.19" size="0.40 0.18 0.19" material="table_m"/>
    <geom type="cylinder" size="0.035 {PED_H:.3f}" pos="{LEFT_BALL_START[0]:.3f} 0 {TABLE_H + PED_H:.3f}" material="ped_m"/>
    <geom type="box" pos="{LEFT_TABLE_TARGET[0]:.3f} 0 {TABLE_H + 0.03:.3f}" size="0.14 0.11 0.03" material="table_m"/>
    <geom type="cylinder" size="0.05 0.003" pos="{LEFT_TABLE_TARGET[0]:.3f} 0 {LEFT_TABLE_TARGET[2] - BALL_R + 0.002:.3f}" material="goal_m"/>

    <!-- right stage -->
    <geom type="box" pos="0.68 0 0.19" size="0.40 0.18 0.19" material="table_m"/>
    <geom type="cylinder" size="0.035 {PED_H:.3f}" pos="{RIGHT_BALL_START[0]:.3f} 0 {TABLE_H + PED_H:.3f}" material="ped_m"/>
    <geom type="box" pos="{RIGHT_TABLE_TARGET[0]:.3f} 0 {TABLE_H + 0.03:.3f}" size="0.14 0.11 0.03" material="table_m"/>
    <geom type="cylinder" size="0.05 0.003" pos="{RIGHT_TABLE_TARGET[0]:.3f} 0 {RIGHT_TABLE_TARGET[2] - BALL_R + 0.002:.3f}" material="goal_m"/>

    <body name="ball_left" pos="{LEFT_BALL_START[0]:.3f} 0 {LEFT_BALL_START[2]:.3f}">
      <freejoint name="jball_left"/>
      <geom type="sphere" size="{BALL_R:.3f}" mass="0.03" material="ball_m"/>
    </body>

    <body name="ball_right" pos="{RIGHT_BALL_START[0]:.3f} 0 {RIGHT_BALL_START[2]:.3f}">
      <freejoint name="jball_right"/>
      <geom type="sphere" size="{BALL_R:.3f}" mass="0.03" material="ball_m"/>
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
        <geom name="r_upper_geom" type="capsule" fromto="0 0 0 {BAD_L1:.3f} 0 0" size="0.028" material="right_bad"/>
        <body name="right_elbow" pos="{BAD_L1:.3f} 0 0">
          <geom type="sphere" size="0.034" material="joint_m"/>
          <joint name="r_elbow" type="hinge" axis="0 1 0" range="-3.04 3.04"/>
          <body name="right_forearm">
            <geom name="r_forearm_geom" type="capsule" fromto="0 0 0 {BAD_L2:.3f} 0 0" size="0.023" material="right_bad"/>
            <body name="right_tool" pos="{BAD_L2:.3f} 0 0">
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
    <position joint="l_shoulder" kp="2600" forcerange="-200 200"/>
    <position joint="l_elbow"    kp="2200" forcerange="-180 180"/>
    <position joint="l_top"      kp="700" forcerange="-35 35"/>
    <position joint="l_bot"      kp="700" forcerange="-35 35"/>

    <position joint="r_shoulder" kp="2600" forcerange="-200 200"/>
    <position joint="r_elbow"    kp="2200" forcerange="-180 180"/>
    <position joint="r_top"      kp="700" forcerange="-35 35"/>
    <position joint="r_bot"      kp="700" forcerange="-35 35"/>
  </actuator>
</mujoco>
"""

def build_model(corrected=False):
    xml = MJCF
    if corrected:
        xml = xml.replace(f'fromto="0 0 0 {BAD_L2:.3f} 0 0"', f'fromto="0 0 0 {GT_L2:.3f} 0 0"', 1)
        xml = xml.replace(f'pos="{BAD_L2:.3f} 0 0"', f'pos="{GT_L2:.3f} 0 0"', 1)
        xml = xml.replace('material="right_bad"', 'material="right_fix"')
        xml = xml.replace(f'pos="{BAD_L1:.3f} 0 0"', f'pos="{GT_L1:.3f} 0 0"', 1)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write(xml)
        p = f.name
    model = mujoco.MjModel.from_xml_path(p)
    os.unlink(p)
    data = mujoco.MjData(model)
    return model, data

def body_qpos_slice(model, body_name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    jid = model.body_jntadr[bid]
    qadr = model.jnt_qposadr[jid]
    return slice(qadr, qadr + 7)

def set_free_pose(data, slc, pos):
    data.qpos[slc.start:slc.start+3] = pos
    data.qpos[slc.start+3:slc.stop] = [1, 0, 0, 0]
    data.qvel[slc.start:slc.start+6] = 0

def site_pos(model, data, site_name):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    return data.site_xpos[sid].copy()

def overlay(frame, t, phase, corrected=False):
    img = Image.fromarray(frame).convert("RGB")
    dr = ImageDraw.Draw(img, "RGBA")

    dr.rounded_rectangle((24, 20, 340, 96), radius=18, fill=(8, 12, 18, 215))
    dr.text((46, 40), "GROUND TRUTH", font=font(30, True), fill=(90, 225, 120))

    dr.rounded_rectangle((W-340, 20, W-24, 96), radius=18,
                         fill=(8, 18, 30, 220) if corrected else (30, 8, 8, 220))
    dr.text((W-318, 40), "CORRECTED" if corrected else "FLAWED",
            font=font(30, True),
            fill=(50, 205, 240) if corrected else (240, 85, 65))

    dr.line((W//2, 0, W//2, H), fill=(255, 255, 255, 32), width=2)

    dr.rounded_rectangle((24, H-90, W-24, H-20), radius=18, fill=(8, 10, 16, 220))
    footer = "same controller; right morphology corrected and now matches GT" if corrected \
             else "same controller; right forearm too short, so it misses the place target"
    dr.text((42, H-70), f"phase: {phase}   |   {footer}   |   t={t:4.1f}s",
            font=font(21, True), fill=(235, 235, 235))
    return np.array(img)

def freeze_panel(frame):
    img = Image.fromarray(frame).convert("RGB")
    dr = ImageDraw.Draw(img, "RGBA")
    dr.rectangle((0, 0, W, H), fill=(0, 0, 0, 170))
    dr.rounded_rectangle((260, 150, W-260, H-150), radius=22, fill=(8, 12, 18, 245),
                         outline=(40, 210, 100), width=3)

    y = 210
    dr.text((320, y), "Fault detected", font=font(36, True), fill=(255, 115, 80)); y += 70
    dr.text((320, y), "Right forearm length mismatch", font=font(30, True), fill=(255, 210, 120)); y += 52
    dr.text((320, y), "Measured reach deficit at target: 6.9 cm", font=font(26), fill=(180, 220, 255)); y += 74
    dr.text((320, y), "Running OpenCAD", font=font(28, True), fill=(60, 200, 255)); y += 46
    dr.text((320, y), "Improving arm", font=font(28, True), fill=(60, 200, 255)); y += 46
    dr.text((320, y), "Applying corrections", font=font(28, True), fill=(60, 200, 255)); y += 66

    code = [
        "forearm.length = 0.29",
        "export('forearm_corrected.step')",
        "reload_geometry('forearm_corrected.step')",
    ]
    dr.rounded_rectangle((320, y, W-320, y+170), radius=16, fill=(3, 6, 10, 255))
    for i, line in enumerate(code):
        dr.text((360, y + 26 + i*44), line, font=font(24), fill=(150, 230, 160))

    return np.array(img)

def run_scene(corrected=False):
    model, data = build_model(corrected=corrected)
    renderer = mujoco.Renderer(model, width=W, height=H)
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "main")

    ball_left_sl  = body_qpos_slice(model, "ball_left")
    ball_right_sl = body_qpos_slice(model, "ball_right")

    set_free_pose(data, ball_left_sl, LEFT_BALL_START)
    set_free_pose(data, ball_right_sl, RIGHT_BALL_START)
    mujoco.mj_forward(model, data)

    frames = []
    attached_left = False
    attached_right = False

    t = 0.0
    step = 0
    render_every = max(1, round(1.0 / (FPS * model.opt.timestep)))
    freeze_frame = None
    freeze_count = 0

    while t < DUR:
        if (not corrected) and t >= T_FREEZE:
            if freeze_frame is None:
                renderer.update_scene(data, camera=cam)
                freeze_frame = freeze_panel(renderer.render().copy())
            if freeze_count < int((T_RESUME - T_FREEZE) * FPS):
                frames.append(freeze_frame.copy())
                freeze_count += 1
                t += 1.0 / FPS
                continue
            break

        ql, gl, phase = phase_ctrl(t, LEFT_HOME, LEFT_ABOVE, LEFT_PICK, LEFT_LIFT, LEFT_ABOVE_T, LEFT_PLACE, LEFT_RET)
        qr, gr, _ = phase_ctrl(t, RIGHT_REF_HOME, RIGHT_REF_ABOVE, RIGHT_REF_PICK, RIGHT_REF_LIFT, RIGHT_REF_ABOVE_T, RIGHT_REF_PLACE, RIGHT_REF_RET)

        data.ctrl[:] = np.array([ql[0], ql[1], gl/2, gl/2, qr[0], qr[1], gr/2, gr/2], dtype=float)
        mujoco.mj_step(model, data)

        tcp_l = site_pos(model, data, "left_tcp")
        tcp_r = site_pos(model, data, "right_tcp")

        ball_l = data.qpos[ball_left_sl.start:ball_left_sl.start+3].copy()
        ball_r = data.qpos[ball_right_sl.start:ball_right_sl.start+3].copy()

        # Visual grasp logic
        if (not attached_left) and gl < 0.018 and np.linalg.norm(tcp_l - ball_l) < 0.040:
            attached_left = True
        if attached_left:
            set_free_pose(data, ball_left_sl, tcp_l + np.array([0.018, 0.0, 0.0]))
            if t >= T_OPEN1:
                attached_left = False
                set_free_pose(data, ball_left_sl, LEFT_TABLE_TARGET + np.array([0.0, 0.0, BALL_R]))

        if corrected:
            if (not attached_right) and gr < 0.018 and np.linalg.norm(tcp_r - ball_r) < 0.040:
                attached_right = True
        else:
            # flawed arm never gets a valid grasp because its true forearm is shorter
            attach_ok = False
            if attach_ok:
                attached_right = True

        if attached_right:
            set_free_pose(data, ball_right_sl, tcp_r + np.array([0.018, 0.0, 0.0]))
            if t >= T_OPEN1:
                attached_right = False
                set_free_pose(data, ball_right_sl, RIGHT_TABLE_TARGET + np.array([0.0, 0.0, BALL_R]))

        # Failed case: show ball still at source while arm goes through motion and misses
        mujoco.mj_forward(model, data)

        if step % render_every == 0:
            renderer.update_scene(data, camera=cam)
            raw = renderer.render().copy()
            frames.append(overlay(raw, t, phase, corrected=corrected))

        t += model.opt.timestep
        step += 1

    return frames

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    print("Rendering bad phase...")
    frames_bad = run_scene(corrected=False)
    print("Rendering corrected phase...")
    frames_fix = run_scene(corrected=True)
    frames = frames_bad + frames_fix
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
