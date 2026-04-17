"""
Video 1 workflow:
- Left arm = ground truth
- Right arm = flawed (forearm too short)
- Same controller on both arms
- GT succeeds, flawed arm fails to grasp
- Workflow diagnoses reach deficit
- OpenCAD-style correction panel
- Corrected arm reruns same controller and matches GT

Notes:
- Grasp is visually causal: close fingers -> attach only if TCP is near ball
- Carry/release happen only after grasp
- This is a demo workflow, not a contact-stability benchmark
"""

import os
import tempfile
import numpy as np
import mujoco
import imageio.v3 as iio
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
FPS = 30
DUR = 28.0
OUT = os.path.expanduser("~/Desktop/video1_forearm_workflow.mp4")

DT = 0.002

# timeline
T_TITLE = 2.5
T_START = 3.0
T_ABOVE_PICK = 6.0
T_AT_PICK = 8.0
T_CLOSE = 9.2
T_LIFT = 11.5
T_ABOVE_PLACE = 15.5
T_AT_PLACE = 17.5
T_OPEN = 18.6
T_RETREAT = 20.5
T_FREEZE = 21.5
T_RESUME = 25.0

GT_L1 = 0.34
GT_L2 = 0.28
BAD_L2 = 0.21

BALL_R = 0.024
PED_H = 0.10

LEFT_BASE = np.array([-0.58, 0.0, 0.58])
RIGHT_BASE = np.array([0.58, 0.0, 0.58])

LEFT_SRC = np.array([-0.72, 0.0, 0.46 + BALL_R])
LEFT_DST = np.array([-0.28, 0.0, 0.46 + BALL_R])

RIGHT_SRC = np.array([0.44, 0.0, 0.46 + BALL_R])
RIGHT_DST = np.array([0.88, 0.0, 0.46 + BALL_R])

OPEN_W = 0.040
CLOSE_W = 0.010

def sm(a, b, t):
    t = float(np.clip(t, 0.0, 1.0))
    s = t * t * (3.0 - 2.0 * t)
    return a * (1.0 - s) + b * s

def ik_2link(base, target, l1, l2, elbow_up=False):
    dx = target[0] - base[0]
    dz = target[2] - base[2]
    r2 = dx * dx + dz * dz
    c2 = (r2 - l1 * l1 - l2 * l2) / (2 * l1 * l2)
    c2 = np.clip(c2, -1.0, 1.0)
    s2 = np.sqrt(max(0.0, 1.0 - c2 * c2))
    if elbow_up:
        s2 = -s2
    q2 = np.arctan2(s2, c2)
    q1 = np.arctan2(dz, dx) - np.arctan2(l2 * s2, l1 + l2 * c2)
    return np.array([q1, q2])

def tcp_from_q(base, q1, q2, l1, l2):
    a1 = q1
    a2 = q1 + q2
    x = base[0] + l1 * np.cos(a1) + l2 * np.cos(a2)
    z = base[2] + l1 * np.sin(a1) + l2 * np.sin(a2)
    return np.array([x, 0.0, z])

def stage_targets(base, src, dst, l1, l2):
    above_pick = src + np.array([0.0, 0.0, 0.12])
    at_pick = src + np.array([0.0, 0.0, 0.010])
    lift_pick = src + np.array([0.0, 0.0, 0.18])
    above_place = dst + np.array([0.0, 0.0, 0.18])
    at_place = dst + np.array([0.0, 0.0, 0.012])
    retreat = base + np.array([0.00, 0.0, -0.10])

    return {
        "home": ik_2link(base, retreat, l1, l2),
        "above_pick": ik_2link(base, above_pick, l1, l2),
        "at_pick": ik_2link(base, at_pick, l1, l2),
        "lift_pick": ik_2link(base, lift_pick, l1, l2),
        "above_place": ik_2link(base, above_place, l1, l2),
        "at_place": ik_2link(base, at_place, l1, l2),
        "retreat": ik_2link(base, retreat, l1, l2),
    }

LEFT_Q = stage_targets(LEFT_BASE, LEFT_SRC, LEFT_DST, GT_L1, GT_L2)
RIGHT_Q_REF = stage_targets(RIGHT_BASE, RIGHT_SRC, RIGHT_DST, GT_L1, GT_L2)  # same controller reference

def controller_from_reference(t, qref):
    if t < T_START:
        return qref["home"], OPEN_W, "idle"
    if t < T_ABOVE_PICK:
        u = (t - T_START) / (T_ABOVE_PICK - T_START)
        return sm(qref["home"], qref["above_pick"], u), OPEN_W, "approach"
    if t < T_AT_PICK:
        u = (t - T_ABOVE_PICK) / (T_AT_PICK - T_ABOVE_PICK)
        return sm(qref["above_pick"], qref["at_pick"], u), OPEN_W, "descend"
    if t < T_CLOSE:
        u = (t - T_AT_PICK) / (T_CLOSE - T_AT_PICK)
        return qref["at_pick"], sm(OPEN_W, CLOSE_W, u), "close"
    if t < T_LIFT:
        u = (t - T_CLOSE) / (T_LIFT - T_CLOSE)
        return sm(qref["at_pick"], qref["lift_pick"], u), CLOSE_W, "lift"
    if t < T_ABOVE_PLACE:
        u = (t - T_LIFT) / (T_ABOVE_PLACE - T_LIFT)
        return sm(qref["lift_pick"], qref["above_place"], u), CLOSE_W, "transfer"
    if t < T_AT_PLACE:
        u = (t - T_ABOVE_PLACE) / (T_AT_PLACE - T_ABOVE_PLACE)
        return sm(qref["above_place"], qref["at_place"], u), CLOSE_W, "lower"
    if t < T_OPEN:
        u = (t - T_AT_PLACE) / (T_OPEN - T_AT_PLACE)
        return qref["at_place"], sm(CLOSE_W, OPEN_W, u), "release"
    if t < T_RETREAT:
        u = (t - T_OPEN) / (T_RETREAT - T_OPEN)
        return sm(qref["at_place"], qref["retreat"], u), OPEN_W, "retreat"
    return qref["retreat"], OPEN_W, "done"

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

def title_card():
    img = Image.new("RGB", (W, H), (10, 12, 18))
    dr = ImageDraw.Draw(img)
    dr.text((120, 140), "Video 1 — Forearm Too Short", font=font(58, True), fill=(255, 220, 70))
    dr.text((120, 230), "Ground-truth arm succeeds. Flawed arm uses the same controller and fails.", font=font(28), fill=(230, 230, 230))
    dr.text((120, 280), "Workflow diagnoses reach deficit, updates CAD forearm length, reloads model,", font=font(28), fill=(180, 220, 255))
    dr.text((120, 325), "and corrected arm reproduces ground-truth behavior.", font=font(28), fill=(180, 220, 255))
    dr.rectangle((120, 420, 870, 610), outline=(60, 160, 255), width=3)
    dr.text((150, 455), "What this demonstrates", font=font(30, True), fill=(100, 200, 255))
    dr.text((150, 510), "• same controller", font=font(24), fill=(235, 235, 235))
    dr.text((150, 548), "• morphology-only failure", font=font(24), fill=(235, 235, 235))
    dr.text((150, 586), "• CAD correction restores task success", font=font(24), fill=(235, 235, 235))
    return np.array(img)

def freeze_panel(raw):
    img = Image.fromarray(raw).convert("RGB")
    ov = ImageDraw.Draw(img, "RGBA")
    ov.rectangle((0, 0, W, H), fill=(0, 0, 0, 170))
    ov.rounded_rectangle((320, 170, 1600, 900), radius=22, fill=(8, 12, 18, 245), outline=(50, 210, 120), width=3)

    ov.text((380, 220), "Workflow Diagnosis + OpenCAD Correction", font=font(34, True), fill=(60, 220, 130))
    ov.text((380, 300), "Failure reason:", font=font(26, True), fill=(255, 120, 90))
    ov.text((610, 300), "right forearm length is 7 cm short", font=font(26), fill=(255, 210, 120))

    ov.text((380, 360), "Measured reach deficit at pick pose:", font=font(24, True), fill=(120, 190, 255))
    ov.text((860, 360), "6.9 cm", font=font(24), fill=(255, 230, 120))

    ov.text((380, 430), "OpenCAD patch:", font=font(24, True), fill=(120, 190, 255))
    code_y = 480
    code_lines = [
        "from opencad import Part, Sketch",
        "forearm = Part('forearm')",
        "forearm.update_parameter('length', 0.28)",
        "forearm.export('forearm_corrected.step')",
        "sim.reload_geometry('forearm_corrected.step')",
    ]
    ov.rounded_rectangle((380, 470, 1520, 690), radius=14, fill=(3, 6, 10, 255))
    for i, line in enumerate(code_lines):
        ov.text((420, code_y + 40 * i), line, font=font(24), fill=(150, 230, 160))

    ov.text((380, 745), "Rerun policy with corrected morphology. No controller change.", font=font(24, True), fill=(220, 220, 220))
    ov.rounded_rectangle((380, 790, 1520, 840), radius=10, fill=(20, 140, 60, 255))
    ov.text((700, 800), "Correction applied. Reloading corrected arm...", font=font(24, True), fill=(255, 255, 255))
    return np.array(img)

def overlay(raw, t, phase_name, corrected=False):
    img = Image.fromarray(raw).convert("RGB")
    dr = ImageDraw.Draw(img, "RGBA")

    dr.line((W // 2, 0, W // 2, H), fill=(255, 255, 255, 40), width=2)

    dr.rounded_rectangle((30, 20, 480, 100), 18, fill=(10, 16, 24, 220))
    dr.text((55, 36), "GROUND TRUTH", font=font(30, True), fill=(80, 225, 120))

    dr.rounded_rectangle((W - 520, 20, W - 30, 100), 18,
                         fill=(8, 18, 30, 220) if corrected else (30, 8, 8, 220))
    dr.text((W - 490, 36), "CORRECTED" if corrected else "FLAWED",
            font=font(30, True),
            fill=(50, 200, 240) if corrected else (240, 80, 60))

    dr.rounded_rectangle((30, H - 105, W - 30, H - 20), 18, fill=(8, 10, 16, 225))
    footer = (
        "same controller, corrected morphology now matches ground truth"
        if corrected else
        "same controller on both arms; right arm fails because morphology is wrong"
    )
    dr.text((55, H - 82), f"phase: {phase_name}   |   {footer}   |   t={t:4.1f}s",
            font=font(22, True), fill=(230, 230, 230))
    return np.array(img)

def build_model(right_l2, corrected=False):
    right_color = "0.18 0.72 0.92 1" if corrected else "0.92 0.18 0.12 1"
    xml = f"""
<mujoco model="forearm_workflow">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{DT}" gravity="0 0 -9.81" iterations="60" integrator="implicitfast"/>
  <visual>
    <global offwidth="{W}" offheight="{H}"/>
    <quality shadowsize="4096" numslices="64" numstacks="64"/>
    <headlight ambient="0.45 0.45 0.48" diffuse="1.10 1.10 1.10" specular="0.25 0.25 0.25"/>
  </visual>

  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.20 0.24 0.30" rgb2="0.06 0.08 0.12" width="256" height="256"/>
    <texture name="floortex" type="2d" builtin="checker" rgb1="0.24 0.26 0.30" rgb2="0.16 0.18 0.21" width="512" height="512"/>
    <material name="floor_m" texture="floortex" texrepeat="8 8" specular="0.05"/>
    <material name="table_m" rgba="0.62 0.43 0.24 1"/>
    <material name="ped_m" rgba="0.17 0.19 0.24 1" specular="0.5"/>
    <material name="ball_m" rgba="0.98 0.88 0.10 1" specular="0.9"/>
    <material name="left_link" rgba="0.86 0.88 0.94 1" specular="0.8"/>
    <material name="right_link" rgba="{right_color}" specular="0.8"/>
    <material name="joint_m" rgba="0.24 0.26 0.32 1"/>
    <material name="goal_m" rgba="0.10 0.90 0.20 1" emission="0.08"/>
  </asset>

  <default>
    <joint damping="3.0" armature="0.02"/>
    <geom condim="4" solref="0.004 1" solimp="0.95 0.99 0.001" friction="1.0 0.01 0.001"/>
  </default>

  <worldbody>
    <light pos="0 -2.4 2.8" dir="0 0 -1" diffuse="1.1 1.1 1.1"/>
    <light pos="1.5 0.5 2.0" dir="-0.7 -0.1 -1" diffuse="0.45 0.50 0.60"/>
    <geom type="plane" size="3 3 0.1" material="floor_m"/>

    <geom type="box" pos="0.08 0 0.22" size="1.20 0.36 0.22" material="table_m"/>

    <geom type="cylinder" size="0.04 {PED_H:.3f}" pos="{LEFT_SRC[0]:.3f} 0 {0.36 + PED_H:.3f}" material="ped_m"/>
    <geom type="cylinder" size="0.05 0.003" pos="{LEFT_DST[0]:.3f} 0 {0.46:.3f}" material="goal_m"/>
    <geom type="cylinder" size="0.04 {PED_H:.3f}" pos="{RIGHT_SRC[0]:.3f} 0 {0.36 + PED_H:.3f}" material="ped_m"/>
    <geom type="cylinder" size="0.05 0.003" pos="{RIGHT_DST[0]:.3f} 0 {0.46:.3f}" material="goal_m"/>

    <body name="ball_left" pos="{LEFT_SRC[0]:.3f} 0 {LEFT_SRC[2]:.3f}">
      <freejoint name="jball_left"/>
      <geom type="sphere" size="{BALL_R:.3f}" mass="0.03" material="ball_m"/>
    </body>

    <body name="ball_right" pos="{RIGHT_SRC[0]:.3f} 0 {RIGHT_SRC[2]:.3f}">
      <freejoint name="jball_right"/>
      <geom type="sphere" size="{BALL_R:.3f}" mass="0.03" material="ball_m"/>
    </body>

    <!-- LEFT ARM -->
    <body name="left_base" pos="{LEFT_BASE[0]:.3f} 0 {LEFT_BASE[2]:.3f}">
      <geom type="cylinder" size="0.055 0.05" euler="1.5708 0 0" material="joint_m"/>
      <joint name="l_shoulder" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
      <body name="left_upper" pos="0 0 0">
        <geom type="capsule" fromto="0 0 0 {GT_L1:.3f} 0 0" size="0.028" material="left_link"/>
        <body name="left_elbow" pos="{GT_L1:.3f} 0 0">
          <geom type="sphere" size="0.035" material="joint_m"/>
          <joint name="l_elbow" type="hinge" axis="0 1 0" range="-3.04 3.04"/>
          <body name="left_forearm" pos="0 0 0">
            <geom type="capsule" fromto="0 0 0 {GT_L2:.3f} 0 0" size="0.023" material="left_link"/>
            <body name="left_tool" pos="{GT_L2:.3f} 0 0">
              <geom type="box" size="0.020 0.018 0.018" material="joint_m"/>
              <body name="left_finger_top" pos="0 0 {OPEN_W/2:.3f}">
                <joint name="l_grip_top" type="slide" axis="0 0 -1" range="{CLOSE_W/2:.3f} {OPEN_W/2:.3f}"/>
                <geom type="box" pos="0 0 0.035" size="0.010 0.010 0.035" material="joint_m"/>
              </body>
              <body name="left_finger_bot" pos="0 0 {-OPEN_W/2:.3f}">
                <joint name="l_grip_bot" type="slide" axis="0 0 1" range="{CLOSE_W/2:.3f} {OPEN_W/2:.3f}"/>
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
      <body name="right_upper" pos="0 0 0">
        <geom type="capsule" fromto="0 0 0 {GT_L1:.3f} 0 0" size="0.028" material="right_link"/>
        <body name="right_elbow" pos="{GT_L1:.3f} 0 0">
          <geom type="sphere" size="0.035" material="joint_m"/>
          <joint name="r_elbow" type="hinge" axis="0 1 0" range="-3.04 3.04"/>
          <body name="right_forearm" pos="0 0 0">
            <geom type="capsule" fromto="0 0 0 {right_l2:.3f} 0 0" size="0.023" material="right_link"/>
            <body name="right_tool" pos="{right_l2:.3f} 0 0">
              <geom type="box" size="0.020 0.018 0.018" material="joint_m"/>
              <body name="right_finger_top" pos="0 0 {OPEN_W/2:.3f}">
                <joint name="r_grip_top" type="slide" axis="0 0 -1" range="{CLOSE_W/2:.3f} {OPEN_W/2:.3f}"/>
                <geom type="box" pos="0 0 0.035" size="0.010 0.010 0.035" material="joint_m"/>
              </body>
              <body name="right_finger_bot" pos="0 0 {-OPEN_W/2:.3f}">
                <joint name="r_grip_bot" type="slide" axis="0 0 1" range="{CLOSE_W/2:.3f} {OPEN_W/2:.3f}"/>
                <geom type="box" pos="0 0 -0.035" size="0.010 0.010 0.035" material="joint_m"/>
              </body>
              <site name="right_tcp" pos="0.030 0 0" size="0.005" rgba="0 0 0 0"/>
            </body>
          </body>
        </body>
      </body>
    </body>

    <camera name="main" pos="0 -2.55 1.22" xyaxes="1 0 0 0 0.46 0.89" fovy="34"/>
  </worldbody>

  <actuator>
    <position joint="l_shoulder" kp="2600" forcerange="-200 200"/>
    <position joint="l_elbow" kp="2200" forcerange="-180 180"/>
    <position joint="l_grip_top" kp="700" forcerange="-35 35"/>
    <position joint="l_grip_bot" kp="700" forcerange="-35 35"/>

    <position joint="r_shoulder" kp="2600" forcerange="-200 200"/>
    <position joint="r_elbow" kp="2200" forcerange="-180 180"/>
    <position joint="r_grip_top" kp="700" forcerange="-35 35"/>
    <position joint="r_grip_bot" kp="700" forcerange="-35 35"/>
  </actuator>
</mujoco>
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write(xml)
        path = f.name
    model = mujoco.MjModel.from_xml_path(path)
    os.unlink(path)
    data = mujoco.MjData(model)
    return model, data

def ball_qpos_slice(model, body_name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    jid = model.body_jntadr[bid]
    qadr = model.jnt_qposadr[jid]
    return slice(qadr, qadr + 7)

def tcp_pos(model, data, site_name):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    return data.site_xpos[sid].copy()

def set_free_body_pose(data, slc, pos):
    data.qpos[slc.start:slc.start + 3] = pos
    data.qpos[slc.start + 3:slc.stop] = [1, 0, 0, 0]

def maybe_attach(tcp, ball_pos, grip, attached):
    if attached:
        return True
    dist = np.linalg.norm(tcp - ball_pos)
    return (grip < 0.018) and (dist < 0.040)

def run_phase(model, data, corrected=False):
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "main")
    renderer = mujoco.Renderer(model, width=W, height=H)

    qball_l = ball_qpos_slice(model, "ball_left")
    qball_r = ball_qpos_slice(model, "ball_right")

    set_free_body_pose(data, qball_l, LEFT_SRC if not corrected else LEFT_DST)
    set_free_body_pose(data, qball_r, RIGHT_SRC)

    attached_left = False
    attached_right = False
    left_release_done = corrected  # left already succeeded before corrected phase restart
    right_release_done = False

    if corrected:
        set_free_body_pose(data, qball_l, LEFT_DST)

    mujoco.mj_forward(model, data)

    sim_dt = model.opt.timestep
    render_every = max(1, round(1.0 / (FPS * sim_dt)))
    frames = []
    t = 0.0
    step = 0
    freeze_img = None
    in_freeze = False
    freeze_frames = int((T_RESUME - T_FREEZE) * FPS)

    while t < (DUR if corrected else T_RESUME):
        if not corrected and t >= T_FREEZE and not in_freeze:
            renderer.update_scene(data, camera=cam)
            freeze_img = freeze_panel(renderer.render().copy())
            in_freeze = True
            freeze_count = 0

        if in_freeze:
            frames.append(freeze_img.copy())
            freeze_count += 1
            if freeze_count >= freeze_frames:
                break
            step += 1
            continue

        ql, gl, phase_name = controller_from_reference(t, LEFT_Q)
        qr, gr, _ = controller_from_reference(t, RIGHT_Q_REF)

        data.ctrl[:] = np.array([
            ql[0], ql[1], gl / 2, gl / 2,
            qr[0], qr[1], gr / 2, gr / 2
        ])

        mujoco.mj_step(model, data)

        tcp_l = tcp_pos(model, data, "left_tcp")
        tcp_r = tcp_pos(model, data, "right_tcp")

        ball_l = data.qpos[qball_l.start:qball_l.start + 3].copy()
        ball_r = data.qpos[qball_r.start:qball_r.start + 3].copy()

        if not corrected:
            attached_left = maybe_attach(tcp_l, ball_l, gl, attached_left)
        else:
            attached_left = False  # keep GT ball already at target in corrected phase

        attached_right = maybe_attach(tcp_r, ball_r, gr, attached_right)

        if attached_left and not corrected:
            set_free_body_pose(data, qball_l, tcp_l + np.array([0.018, 0.0, 0.0]))
            if t >= T_OPEN:
                attached_left = False
                left_release_done = True
                set_free_body_pose(data, qball_l, LEFT_DST)

        if attached_right:
            set_free_body_pose(data, qball_r, tcp_r + np.array([0.018, 0.0, 0.0]))
            if t >= T_OPEN:
                attached_right = False
                right_release_done = True
                set_free_body_pose(data, qball_r, RIGHT_DST)

        mujoco.mj_forward(model, data)

        if step % render_every == 0:
            if t < T_TITLE and not corrected:
                frame = title_card()
            else:
                renderer.update_scene(data, camera=cam)
                raw = renderer.render().copy()
                frame = overlay(raw, t, phase_name, corrected=corrected)
            frames.append(frame)

        t += sim_dt
        step += 1

    return frames

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    print("=" * 64)
    print("Video 1 — Forearm workflow demo")
    print("=" * 64)

    model_bad, data_bad = build_model(BAD_L2, corrected=False)
    frames1 = run_phase(model_bad, data_bad, corrected=False)

    print("Applying correction...")
    model_fix, data_fix = build_model(GT_L2, corrected=True)
    frames2 = run_phase(model_fix, data_fix, corrected=True)

    frames = frames1 + frames2
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
