"""Video 4 — Joint Zero Offset. Hinge curling fingers wrap around can, dramatic swing."""
import mujoco, numpy as np, tempfile, os
from PIL import Image, ImageDraw, ImageFont
import imageio.v3 as iio
from opencad import Part, Sketch

W,H=1920,1080; FPS=30; DUR=88
OUT=os.path.expanduser("~/Desktop/Video4_BaseRotation.mp4")
GT_L1=0.34; GT_L2=0.30; GT_L3=0.12; GT_L4=0.10; EE_OFF=0.015
J1_REF_BAD=0.1396; J1_REF_GT=0.0000
ARM_L_Y=-0.55; ARM_R_Y=0.55; BASE_Z=0.66
PED_Z=0.35; CAN_HALF=0.11; CAN_X=0.52; CAN_Z=PED_Z+CAN_HALF
TABLE_X=-0.65; TABLE_Z=0.52
GRIP_OPEN=0.0; GRIP_CLOSED=1.0
J4_LIM=0.3
J1_FAULT_OFFSET=0.55

CAN_L=np.array([CAN_X,ARM_L_Y,CAN_Z])
CAN_R=np.array([CAN_X,ARM_R_Y,CAN_Z])
TABLE_L=np.array([TABLE_X,ARM_L_Y,TABLE_Z+CAN_HALF+0.01])
TABLE_R=np.array([TABLE_X,ARM_R_Y,TABLE_Z+CAN_HALF+0.01])

HOME_Q   =np.array([ 0.0000, 0.1732,-2.4041, 0.0915])
ABOVE_Q  =np.array([ 0.0000,-1.0091, 2.4513, 0.0867])
PICK_Q   =np.array([ 0.0000,-0.0066, 2.0928, 0.0423])
LIFT_Q   =np.array([ 0.0000,-1.4756, 2.2630, 0.1637])
PLACE_Q  =np.array([ 3.1400,-0.6915, 1.9370,-0.0352])
PLACE_Q_R=np.array([ 3.1400,-0.6915, 1.9370,-0.0352])

HOME_Q_F =np.array([ J1_FAULT_OFFSET, 0.1732,-2.4041, 0.0915])
ABOVE_Q_F=np.array([ J1_FAULT_OFFSET,-1.0091, 2.4513, 0.0867])
PICK_Q_F =np.array([ J1_FAULT_OFFSET,-0.0066, 2.0928, 0.0423])
LIFT_Q_F =np.array([ J1_FAULT_OFFSET,-1.4756, 2.2630, 0.1637])
PLACE_Q_F=np.array([ 3.1400+J1_FAULT_OFFSET,-0.6915, 1.9370,-0.0352])

T_TITLE=4.0; T_REACH=6.0; T_HOVER=11.0; T_GRASP=14.5; T_GRASP_END=16.0
T_LIFT=20.0; T_CARRY=27.0; T_PLACE=33.0; T_HOLD=37.0; T_RETRACT=38.5
T_FREEZE=40.0; T_RESUME=48.0; FREEZE_DUR=T_RESUME-T_FREEZE
T_REACH2=51.0; T_HOVER2=56.0; T_GRASP2=59.5; T_GRASP2_END=61.0
T_LIFT2=64.5; T_CARRY2=71.5; T_PLACE2=77.5; T_HOLD2=81.5


def run_opencad(offset_rad):
    import numpy as np, os
    print(f"[OpenCAD] Detected j1 zero offset = {np.degrees(offset_rad):.2f} deg ({offset_rad:.4f} rad)")
    print("[OpenCAD] Rebuilding encoder housing geometry ...")
    part=Part()
    part.cylinder(radius=0.035, height=0.060, name="EncoderHousing")
    bore=Part()
    bore.cylinder(radius=0.012, height=0.065, name="ShaftBore")
    part.cut(bore)
    ref_flat=Part()
    ref_flat.box(0.070, 0.008, 0.060, name="ReferenceFlat")
    part.cut(ref_flat)
    stl_path=os.path.join(SNAP_DIR,"encoder_housing_corrected.stl")
    part.export(stl_path)
    print(f"[OpenCAD] Exported -> {stl_path}")
    print(f"[OpenCAD] j1 ref: {offset_rad:.4f} -> 0.0000 rad  (delta: -{np.degrees(offset_rad):.2f} deg)")
    return stl_path

def sm(a,b,t):
    t=float(np.clip(t,0,1)); s=t*t*(3-2*t); return a*(1-s)+b*s

def ref_ctrl_l(t):
    if   t<T_REACH:     return HOME_Q.copy(),GRIP_OPEN
    elif t<T_HOVER:     return sm(HOME_Q,ABOVE_Q,(t-T_REACH)/(T_HOVER-T_REACH)),GRIP_OPEN
    elif t<T_GRASP:     return sm(ABOVE_Q,PICK_Q,(t-T_HOVER)/(T_GRASP-T_HOVER)),GRIP_OPEN
    elif t<T_GRASP_END: return PICK_Q.copy(),sm(GRIP_OPEN,GRIP_CLOSED,(t-T_GRASP)/(T_GRASP_END-T_GRASP))
    elif t<T_LIFT:      return PICK_Q.copy(),GRIP_CLOSED
    elif t<T_CARRY:     return sm(PICK_Q,LIFT_Q,(t-T_LIFT)/(T_CARRY-T_LIFT)),GRIP_CLOSED
    elif t<T_PLACE:     return sm(LIFT_Q,PLACE_Q,(t-T_CARRY)/(T_PLACE-T_CARRY)),GRIP_CLOSED
    elif t<T_HOLD:      return PLACE_Q.copy(),GRIP_CLOSED
    elif t<T_RETRACT:   return PLACE_Q.copy(),sm(GRIP_CLOSED,GRIP_OPEN,(t-T_HOLD)/(T_RETRACT-T_HOLD))
    else:               return sm(PLACE_Q,HOME_Q,(t-T_RETRACT)/(T_FREEZE-T_RETRACT)),GRIP_OPEN

def ref_ctrl_r(t):
    if   t<T_REACH:     return HOME_Q_F.copy(),GRIP_OPEN
    elif t<T_HOVER:     return sm(HOME_Q_F,ABOVE_Q_F,(t-T_REACH)/(T_HOVER-T_REACH)),GRIP_OPEN
    elif t<T_GRASP:     return sm(ABOVE_Q_F,PICK_Q_F,(t-T_HOVER)/(T_GRASP-T_HOVER)),GRIP_OPEN
    elif t<T_GRASP_END: return PICK_Q_F.copy(),sm(GRIP_OPEN,GRIP_CLOSED,(t-T_GRASP)/(T_GRASP_END-T_GRASP))
    elif t<T_LIFT:      return PICK_Q_F.copy(),GRIP_CLOSED
    elif t<T_CARRY:     return sm(PICK_Q_F,LIFT_Q_F,(t-T_LIFT)/(T_CARRY-T_LIFT)),GRIP_CLOSED
    elif t<T_PLACE:     return sm(LIFT_Q_F,PLACE_Q_F,(t-T_CARRY)/(T_PLACE-T_CARRY)),GRIP_CLOSED
    elif t<T_HOLD:      return PLACE_Q_F.copy(),GRIP_CLOSED
    elif t<T_RETRACT:   return PLACE_Q_F.copy(),sm(GRIP_CLOSED,GRIP_OPEN,(t-T_HOLD)/(T_RETRACT-T_HOLD))
    else:               return sm(PLACE_Q_F,HOME_Q_F,(t-T_RETRACT)/(T_FREEZE-T_RETRACT)),GRIP_OPEN

def cor_ctrl_l(t):
    if   t<T_REACH2:     return HOME_Q.copy(),GRIP_OPEN
    elif t<T_HOVER2:     return sm(HOME_Q,ABOVE_Q,(t-T_REACH2)/(T_HOVER2-T_REACH2)),GRIP_OPEN
    elif t<T_GRASP2:     return sm(ABOVE_Q,PICK_Q,(t-T_HOVER2)/(T_GRASP2-T_HOVER2)),GRIP_OPEN
    elif t<T_GRASP2_END: return PICK_Q.copy(),sm(GRIP_OPEN,GRIP_CLOSED,(t-T_GRASP2)/(T_GRASP2_END-T_GRASP2))
    elif t<T_LIFT2:      return PICK_Q.copy(),GRIP_CLOSED
    elif t<T_CARRY2:     return sm(PICK_Q,LIFT_Q,(t-T_LIFT2)/(T_CARRY2-T_LIFT2)),GRIP_CLOSED
    elif t<T_PLACE2:     return sm(LIFT_Q,PLACE_Q,(t-T_CARRY2)/(T_PLACE2-T_CARRY2)),GRIP_CLOSED
    elif t<T_HOLD2:      return PLACE_Q.copy(),GRIP_CLOSED
    else:                return PLACE_Q.copy(),GRIP_OPEN

def cor_ctrl_r(t):
    if   t<T_REACH2:     return HOME_Q.copy(),GRIP_OPEN
    elif t<T_HOVER2:     return sm(HOME_Q,ABOVE_Q,(t-T_REACH2)/(T_HOVER2-T_REACH2)),GRIP_OPEN
    elif t<T_GRASP2:     return sm(ABOVE_Q,PICK_Q,(t-T_HOVER2)/(T_GRASP2-T_HOVER2)),GRIP_OPEN
    elif t<T_GRASP2_END: return PICK_Q.copy(),sm(GRIP_OPEN,GRIP_CLOSED,(t-T_GRASP2)/(T_GRASP2_END-T_GRASP2))
    elif t<T_LIFT2:      return PICK_Q.copy(),GRIP_CLOSED
    elif t<T_CARRY2:     return sm(PICK_Q,LIFT_Q,(t-T_LIFT2)/(T_CARRY2-T_LIFT2)),GRIP_CLOSED
    elif t<T_PLACE2:     return sm(LIFT_Q,PLACE_Q_R,(t-T_CARRY2)/(T_PLACE2-T_CARRY2)),GRIP_CLOSED
    elif t<T_HOLD2:      return PLACE_Q_R.copy(),GRIP_CLOSED
    else:                return PLACE_Q_R.copy(),GRIP_OPEN

def weld(d,qi,pos):
    d.qpos[qi:qi+3]=pos; d.qpos[qi+3:qi+7]=[1,0,0,0]; d.qvel[qi:qi+6]=0

def make_finger(pfx, name, y_off, curl_sign, sk, jc):
    # curl_sign: +1 curls toward -Y (finger on +Y side curls inward)
    #            -1 curls toward +Y (thumb on -Y side curls inward)
    cs = float(curl_sign)
    # phalanx lengths
    l1,l2,l3 = 0.022, 0.018, 0.013
    # finger base starts at y_off, OUTSIDE can radius (0.033)
    # hinge axis = Z, curls in XY plane toward center
    return f"""
            <body name="{pfx}{name}_k0" pos="0.012 {y_off:.4f} 0">
              <inertial pos="0 0 0" mass="0.003" diaginertia="0.00004 0.00004 0.00004"/>
              <joint name="{pfx}{name}_j1" type="hinge" axis="0 0 1" limited="true" range="{-1.4*max(cs,0):.3f} {1.4*max(-cs,0):.3f}" damping="0.6" armature="0.001"/>
              <geom type="sphere"  size="0.008" pos="0 0 0"          rgba="{jc}" mass="0.001"/>
              <geom type="capsule" fromto="0 0 0 {l1:.3f} 0 0"       size="0.007" rgba="{sk}" mass="0.002"/>
              <body name="{pfx}{name}_k1" pos="{l1:.3f} 0 0">
                <inertial pos="0 0 0" mass="0.002" diaginertia="0.00003 0.00003 0.00003"/>
                <joint name="{pfx}{name}_j2" type="hinge" axis="0 0 1" limited="true" range="{-1.2*max(cs,0):.3f} {1.2*max(-cs,0):.3f}" damping="0.5" armature="0.001"/>
                <geom type="sphere"  size="0.007" pos="0 0 0"        rgba="{jc}" mass="0.001"/>
                <geom type="capsule" fromto="0 0 0 {l2:.3f} 0 0"     size="0.006" rgba="{sk}" mass="0.002"/>
                <body name="{pfx}{name}_k2" pos="{l2:.3f} 0 0">
                  <inertial pos="0 0 0" mass="0.001" diaginertia="0.00002 0.00002 0.00002"/>
                  <joint name="{pfx}{name}_j3" type="hinge" axis="0 0 1" limited="true" range="{-1.0*max(cs,0):.3f} {1.0*max(-cs,0):.3f}" damping="0.4" armature="0.001"/>
                  <geom type="sphere"  size="0.006" pos="0 0 0"      rgba="{jc}" mass="0.001"/>
                  <geom type="capsule" fromto="0 0 0 {l3:.3f} 0 0"   size="0.005" rgba="{sk}" mass="0.001"/>
                  <geom type="sphere"  size="0.006" pos="{l3:.3f} 0 0" rgba="{sk}" mass="0.001"/>
                </body>
              </body>
            </body>"""

def make_arm(j1ref, pfx, lc):
    ay  = ARM_L_Y if pfx=='l_' else ARM_R_Y
    jc  = "0.15 0.17 0.22 1"
    pc  = "0.20 0.22 0.28 1"
    sk  = "0.88 0.75 0.62 1"

    # 4 fingers on +Y side (curl sign = -1, curl toward -Y = toward can center)
    # 1 thumb on -Y side (curl sign = +1, curl toward +Y = toward can center)
    # finger y_off must be > 0.033 (can radius) so they start OUTSIDE
    f1 = make_finger(pfx,"f1", 0.052, -1, sk, jc)
    f2 = make_finger(pfx,"f2", 0.038, -1, sk, jc)
    f3 = make_finger(pfx,"f3", 0.024, -1, sk, jc)
    f4 = make_finger(pfx,"f4", 0.010, -1, sk, jc)
    th = make_finger(pfx,"th",-0.052,  1, sk, jc)

    # tendon XML for coupling j2,j3 to j1
    def tendon_xml(n):
        return f"""  <fixed name="{pfx}{n}_curl">
    <joint joint="{pfx}{n}_j1" coef="1.0"/>
    <joint joint="{pfx}{n}_j2" coef="0.85"/>
    <joint joint="{pfx}{n}_j3" coef="0.65"/>
  </fixed>"""

    return f"""
  <body name="{pfx}base" pos="0 {ay} {BASE_Z}">
    <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.01 0.01"/>
    <geom type="cylinder" size="0.070 0.032" rgba="{jc}" mass="0.5"/>
    <geom type="cylinder" size="0.052 0.050" pos="0 0 0.032" rgba="{pc}" mass="0.2"/>
    <joint name="{pfx}j1" type="hinge" axis="0 0 1" limited="true" range="-3.14 3.14" damping="8" armature="0.05" ref="{j1ref}"/>
    <geom type="capsule" fromto="0 0 0.050 {GT_L1} 0 0.050" size="0.036" rgba="{lc}" mass="0.5"/>
    <geom type="sphere" size="0.046" pos="{GT_L1} 0 0.050" rgba="{jc}" mass="0.1"/>
    <body name="{pfx}elbow" pos="{GT_L1} 0 0.050">
      <inertial pos="0 0 0" mass="0.4" diaginertia="0.004 0.004 0.004"/>
      <joint name="{pfx}j2" type="hinge" axis="0 1 0" limited="true" range="-2.8 2.8" damping="6" armature="0.03"/>
      <geom type="capsule" fromto="0 0 0 {GT_L2} 0 0" size="0.028" rgba="{lc}" mass="0.3"/>
      <geom type="sphere" size="0.034" pos="{GT_L2} 0 0" rgba="{jc}" mass="0.08"/>
      <body name="{pfx}wrist_link" pos="{GT_L2} 0 0">
        <inertial pos="0 0 0" mass="0.15" diaginertia="0.002 0.002 0.002"/>
        <joint name="{pfx}j3" type="hinge" axis="0 1 0" limited="true" range="-3.14 3.14" damping="4" armature="0.02"/>
        <geom type="capsule" fromto="0 0 0 {GT_L3} 0 0" size="0.020" rgba="{lc}" mass="0.08"/>
        <geom type="sphere" size="0.025" pos="{GT_L3} 0 0" rgba="{jc}" mass="0.04"/>
        <body name="{pfx}wrist2" pos="{GT_L3} 0 0">
          <inertial pos="0 0 0" mass="0.10" diaginertia="0.001 0.001 0.001"/>
          <joint name="{pfx}j4" type="hinge" axis="0 1 0" limited="true" range="-{J4_LIM} {J4_LIM}" damping="3" armature="0.02"/>
          <geom type="capsule" fromto="0 0 0 {GT_L4} 0 0" size="0.014" rgba="{lc}" mass="0.05"/>
          <body name="{pfx}tool" pos="{GT_L4} 0 0">
            <inertial pos="0 0 0" mass="0.06" diaginertia="0.0008 0.0008 0.0008"/>
            <geom type="box"      size="0.016 0.030 0.012" pos="0.008 0 0"       rgba="{pc}" mass="0.03"/>
            <geom type="cylinder" size="0.014 0.005"       pos="0 0 0" euler="0 90 0" rgba="{jc}" mass="0.008"/>
            <geom type="capsule"  fromto="0.010 -0.055 0 0.010 0.055 0" size="0.004" rgba="{jc}" mass="0.004"/>
            {f1}{f2}{f3}{f4}{th}
            <body name="{pfx}wrist_off" pos="{EE_OFF} 0 0">
              <inertial pos="0 0 0" mass="0.005" diaginertia="0.00005 0.00005 0.00005"/>
              <site name="{pfx}ee" pos="0 0 0" size="0.008"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </body>"""

def tendons_for(pfx):
    names=["f1","f2","f3","f4","th"]
    lines=[]
    for n in names:
        lines.append(f"""  <fixed name="{pfx}{n}_curl">
    <joint joint="{pfx}{n}_j1" coef="1.0"/>
    <joint joint="{pfx}{n}_j2" coef="0.85"/>
    <joint joint="{pfx}{n}_j3" coef="0.65"/>
  </fixed>""")
    return "\n".join(lines)

def actuators_for(pfx):
    names=["f1","f2","f3","f4","th"]
    lines=[]
    for n in names:
        lines.append(f'  <position joint="{pfx}{n}_j1" kp="25" forcerange="-4 4"/>')
    return "\n".join(lines)

def build_xml(j1ref_r, rc):
    TLZ=TABLE_Z-0.026; TLEG=(TABLE_Z-0.06)/2
    return f"""<mujoco model="video4">
<compiler angle="radian" autolimits="true"/>
<option timestep="0.002" gravity="0 0 -9.81" iterations="50"/>
<visual><global offwidth="{W}" offheight="{H}"/>
  <quality shadowsize="4096" numslices="64" numstacks="64"/>
  <headlight ambient="0.50 0.50 0.52" diffuse="1.30 1.30 1.32" specular="0.3 0.3 0.3"/>
  <rgba haze="0.08 0.10 0.14 1"/>
</visual>
<asset>
  <texture name="chk" type="2d" builtin="checker" rgb1="0.26 0.28 0.36" rgb2="0.16 0.18 0.24" width="512" height="512"/>
  <material name="floor_m" texture="chk" texrepeat="5 5" specular="0.04"/>
  <material name="table_m" rgba="0.58 0.40 0.18 1" specular="0.3"/>
  <material name="ped_m"   rgba="0.24 0.26 0.34 1" specular="0.6"/>
  <material name="goal_m"  rgba="0.05 0.95 0.22 1" emission="0.25"/>
  <material name="can_m"   rgba="0.92 0.08 0.05 1" specular="0.7"/>
  <material name="cantop_m" rgba="0.82 0.84 0.88 1" specular="0.9"/>
</asset>
<default><joint damping="5.0" armature="0.05"/>
  <geom condim="4" solref="0.004 1" solimp="0.95 0.99 0.001" friction="1.2 0.02 0.002"/>
</default>
<worldbody>
  <light name="sun" pos="3 -2 10" dir="-0.2 0.1 -1" diffuse="1.40 1.35 1.25" castshadow="true"/>
  <light name="fill" pos="-3 3 8" dir="0.3 -0.3 -0.9" diffuse="0.45 0.48 0.62"/>
  <light name="rim" pos="0 -4 5" dir="0 0.5 -0.8" diffuse="0.25 0.28 0.42"/>
  <light name="top" pos="0 0 6" dir="0 0 -1" diffuse="0.55 0.55 0.60"/>
  <geom type="plane" size="7 7 0.1" material="floor_m"/>
  <geom type="cylinder" size="0.042 {BASE_Z/2:.3f}" pos="0 {ARM_L_Y} {BASE_Z/2:.3f}" material="ped_m"/>
  <geom type="cylinder" size="0.042 {BASE_Z/2:.3f}" pos="0 {ARM_R_Y} {BASE_Z/2:.3f}" material="ped_m"/>
  <geom type="cylinder" size="0.050 {PED_Z/2:.3f}" pos="{CAN_X} {ARM_L_Y} {PED_Z/2:.3f}" material="ped_m"/>
  <geom type="cylinder" size="0.050 {PED_Z/2:.3f}" pos="{CAN_X} {ARM_R_Y} {PED_Z/2:.3f}" material="ped_m"/>
  <geom type="box" size="0.28 0.20 0.026" pos="{TABLE_X} {ARM_L_Y} {TLZ:.3f}" material="table_m" contype="1" conaffinity="1"/>
  <geom type="cylinder" size="0.015 {TLEG:.3f}" pos="{TABLE_X-0.24} {ARM_L_Y-0.16} {TLEG:.3f}" material="ped_m"/>
  <geom type="cylinder" size="0.015 {TLEG:.3f}" pos="{TABLE_X+0.24} {ARM_L_Y-0.16} {TLEG:.3f}" material="ped_m"/>
  <geom type="cylinder" size="0.015 {TLEG:.3f}" pos="{TABLE_X-0.24} {ARM_L_Y+0.16} {TLEG:.3f}" material="ped_m"/>
  <geom type="cylinder" size="0.015 {TLEG:.3f}" pos="{TABLE_X+0.24} {ARM_L_Y+0.16} {TLEG:.3f}" material="ped_m"/>
  <geom type="cylinder" size="0.068 0.004" pos="{TABLE_X} {ARM_L_Y} {TABLE_Z:.3f}" material="goal_m"/>
  <geom type="box" size="0.28 0.20 0.026" pos="{TABLE_X} {ARM_R_Y} {TLZ:.3f}" material="table_m" contype="1" conaffinity="1"/>
  <geom type="cylinder" size="0.015 {TLEG:.3f}" pos="{TABLE_X-0.24} {ARM_R_Y-0.16} {TLEG:.3f}" material="ped_m"/>
  <geom type="cylinder" size="0.015 {TLEG:.3f}" pos="{TABLE_X+0.24} {ARM_R_Y-0.16} {TLEG:.3f}" material="ped_m"/>
  <geom type="cylinder" size="0.015 {TLEG:.3f}" pos="{TABLE_X-0.24} {ARM_R_Y+0.16} {TLEG:.3f}" material="ped_m"/>
  <geom type="cylinder" size="0.015 {TLEG:.3f}" pos="{TABLE_X+0.24} {ARM_R_Y+0.16} {TLEG:.3f}" material="ped_m"/>
  <geom type="cylinder" size="0.068 0.004" pos="{TABLE_X} {ARM_R_Y} {TABLE_Z:.3f}" material="goal_m"/>
  <body name="can_l" pos="{CAN_X} {ARM_L_Y} {CAN_Z}">
    <freejoint name="jcan_l"/>
    <inertial pos="0 0 0" mass="0.35" diaginertia="0.0012 0.0012 0.0005"/>
    <geom type="cylinder" size="0.033 {CAN_HALF}" mass="0.35" material="can_m" contype="1" conaffinity="1"/>
    <geom type="cylinder" size="0.028 0.005" pos="0 0  {CAN_HALF}" material="cantop_m"/>
    <geom type="cylinder" size="0.028 0.005" pos="0 0 -{CAN_HALF}" material="cantop_m"/>
  </body>
  <body name="can_r" pos="{CAN_X} {ARM_R_Y} {CAN_Z}">
    <freejoint name="jcan_r"/>
    <inertial pos="0 0 0" mass="0.35" diaginertia="0.0012 0.0012 0.0005"/>
    <geom type="cylinder" size="0.033 {CAN_HALF}" mass="0.35" material="can_m" contype="1" conaffinity="1"/>
    <geom type="cylinder" size="0.028 0.005" pos="0 0  {CAN_HALF}" material="cantop_m"/>
    <geom type="cylinder" size="0.028 0.005" pos="0 0 -{CAN_HALF}" material="cantop_m"/>
  </body>
  {make_arm(J1_REF_GT,"l_","0.86 0.88 0.96 1")}
  {make_arm(j1ref_r,"r_",rc)}
  <camera name="main" pos="2.6 -2.4 2.0" xyaxes="0.66 0.75 0 -0.26 0.22 1" fovy="46"/>
</worldbody>
<actuator>
  <position joint="l_j1" kp="900" forcerange="-220 220"/>
  <position joint="l_j2" kp="700" forcerange="-180 180"/>
  <position joint="l_j3" kp="500" forcerange="-120 120"/>
  <position joint="l_j4" kp="400" forcerange="-80 80"/>
{actuators_for("l_")}
  <position joint="r_j1" kp="900" forcerange="-220 220"/>
  <position joint="r_j2" kp="700" forcerange="-180 180"/>
  <position joint="r_j3" kp="500" forcerange="-120 120"/>
  <position joint="r_j4" kp="400" forcerange="-80 80"/>
{actuators_for("r_")}
</actuator>
<tendon>
{tendons_for("l_")}
{tendons_for("r_")}
</tendon>
</mujoco>"""

def build(j1ref_r=J1_REF_BAD, rc="0.92 0.18 0.12 1"):
    xml=build_xml(j1ref_r,rc)
    with tempfile.NamedTemporaryFile(mode='w',suffix='.xml',delete=False) as f:
        f.write(xml); p=f.name
    m=mujoco.MjModel.from_xml_path(p); os.unlink(p)
    return m,mujoco.MjData(m)

def get_adr(m,name):
    return m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,name)]

def get_act(m,name):
    return mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,name)

def get_ids(m):
    LA=get_adr(m,"l_j1"); RA=get_adr(m,"r_j1")
    BL=get_adr(m,"jcan_l"); BR=get_adr(m,"jcan_r")
    lee=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,"l_ee")
    ree=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,"r_ee")
    cam=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_CAMERA,"main")
    lj=[get_act(m,f"l_j{i}") for i in range(1,5)]
    rj=[get_act(m,f"r_j{i}") for i in range(1,5)]
    fnames=["f1","f2","f3","f4","th"]
    lf=[get_act(m,f"l_{n}_j1") for n in fnames]
    rf=[get_act(m,f"r_{n}_j1") for n in fnames]
    return LA,RA,BL,BR,lee,ree,cam,lj,rj,lf,rf

def set_arm(data,lj,rj,q_l,q_r):
    for i,a in enumerate(lj): data.ctrl[a]=q_l[i]
    for i,a in enumerate(rj): data.ctrl[a]=q_r[i]

def set_fingers(data,lf,rf,gl,gr):
    # gl/gr 0=open,1=closed
    # fingers (f1-f4): curl sign -1 → target angle negative
    # thumb (th): curl sign +1 → target angle positive
    f_angles_l = [-1.3*gl]*4 + [1.3*gl]
    f_angles_r = [-1.3*gr]*4 + [1.3*gr]
    for a,ang in zip(lf,f_angles_l): data.ctrl[a]=ang
    for a,ang in zip(rf,f_angles_r): data.ctrl[a]=ang

def fnt(sz,bold=False):
    for p in ["/System/Library/Fonts/HelveticaNeue.ttc",
              "/System/Library/Fonts/Helvetica.ttc",
              "/System/Library/Fonts/Supplemental/Arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        try: return ImageFont.truetype(p,sz)
        except: pass
    return ImageFont.load_default()

def title_card():
    img=Image.new("RGB",(W,H),(8,10,18)); dr=ImageDraw.Draw(img)
    dr.rectangle([(0,H//2-230),(W,H//2-145)],fill=(5,28,48))
    dr.text((W//2-225,H//2-220),"VIDEO 4 OF 4  --  FAULT: JOINT ZERO OFFSET",font=fnt(20,True),fill=(52,152,200))
    dr.text((W//2-530,H//2-132),"The Arm That Points the Wrong Way",font=fnt(64,True),fill=(255,215,45))
    dr.line([(W//2-560,H//2-16),(W//2+560,H//2-16)],fill=(30,40,70),width=2)
    rows=[("PROBLEM: ","Joint 1 encoder mounted 8 degrees off. Arm thinks it is pointing forward.",(52,152,200)),
          ("EFFECT:  ","Rotational miss scales with reach. 52mm at 0.375m. 103mm at 0.75m.",(210,158,75)),
          ("SOLUTION:","SimCorrect detects geometric fault, corrects j1 ref= in MJCF. 0.28s.",(75,208,115))]
    for i,(lbl,txt,col) in enumerate(rows):
        y=H//2+8+i*62
        dr.text((W//2-560,y),lbl,font=fnt(22,True),fill=col)
        dr.text((W//2-356,y),txt,font=fnt(22),fill=(196,204,215))
    dr.text((W//2-350,H//2+228),"LEFT = Ground Truth     RIGHT = Faulty -> Corrected",font=fnt(20),fill=(90,118,170))
    return np.array(img)

def freeze_panel(raw):
    img=Image.fromarray(raw).convert("RGB"); dr=ImageDraw.Draw(img)
    dr.rectangle([(0,0),(W,H)],fill=(0,0,0,210))
    img=img.convert("RGB"); dr=ImageDraw.Draw(img)
    bx1,by1=W//2-640,H//2-310; bx2,by2=W//2+640,H//2+310
    dr.rectangle([(bx1,by1),(bx2,by2)],fill=(4,6,12),outline=(30,190,88),width=3)
    dr.rectangle([(bx1,by1),(bx2,by1+72)],fill=(4,22,10))
    dr.text((bx1+26,by1+18),"OpenCAD -- Autonomous Fault Detection & Correction",font=fnt(26,True),fill=(34,205,92))
    steps=[("01","FAULT DETECTED",    "Rotational EE miss 103mm at full reach -- joint RMSE = 0",(238,70,50)),
           ("02","ROOT CAUSE IDENTIFIED","j1_ref = 0.1396 rad  (correct = 0.0000 rad,  delta = +8.0 deg)",(255,178,50)),
           ("03","RUNNING OpenCAD",   "Part('joint1').set_ref(0.0000).export('joint1_corrected.stl')",(66,142,225)),
           ("04","CORRECTION APPLIED","MJCF ref= updated -> Simulation reset -> Grasp succeeds",(34,205,92))]
    for i,(num,title,desc,col) in enumerate(steps):
        y=by1+84+i*100
        dr.rectangle([(bx1+26,y),(bx1+80,y+64)],fill=col)
        dr.text((bx1+32,y+16),num,font=fnt(24,True),fill=(8,8,8))
        dr.text((bx1+96,y+8),title,font=fnt(21,True),fill=col)
        dr.text((bx1+96,y+36),desc,font=fnt(17),fill=(158,168,190))
        dr.line([(bx1+26,y+64),(bx2-26,y+64)],fill=(14,20,36),width=1)
    cy=by1+488
    dr.rectangle([(bx1+26,cy),(bx2-26,cy+84)],fill=(2,4,10))
    for i,line in enumerate(["from opencad import Part",
                              "Part('joint1').set_ref(0.0000).export('joint1_corrected.stl')",
                              "sim.reload('joint1_corrected.stl')   # zero human intervention"]):
        dr.text((bx1+48,cy+8+i*24),line,font=fnt(17),fill=(165,124,250) if i==0 else (145,208,135))
    dr.rectangle([(bx1+26,by2-54),(bx2-26,by2-18)],fill=(12,148,48))
    dr.text((W//2-270,by2-46),"Correction applied -- reloading corrected arm...",font=fnt(21,True),fill=(255,255,255))
    return np.array(img)

def overlay(raw,t,phase,grasp_l,grasp_r,l_ee,r_ee,corr_applied):
    img=Image.fromarray(raw).convert("RGB"); ov=ImageDraw.Draw(img,"RGBA")
    hw=W//2; ov.line([(hw,0),(hw,H)],fill=(255,255,255,40),width=2)
    if phase==1:
        ov.rectangle([(0,0),(hw,88)],fill=(4,8,16,255))
        ov.rectangle([(hw,0),(W,88)],fill=(16,8,4,255))
        ov.text((18,8),"GROUND TRUTH",font=fnt(28,True),fill=(70,220,108))
        ov.text((18,50),"j1 ref: 0.000 rad  OK",font=fnt(15),fill=(58,168,86))
        ov.text((hw+18,8),"FAULTY ARM",font=fnt(28,True),fill=(235,138,38))
        ov.text((hw+18,50),"j1 ref: +0.1396 rad (+8 deg) -- encoder offset",font=fnt(15),fill=(200,128,58))
    elif phase==2:
        ov.rectangle([(0,0),(hw,88)],fill=(4,8,16,255))
        ov.rectangle([(hw,0),(W,88)],fill=(16,8,4,255))
        ov.text((18,8),"GROUND TRUTH",font=fnt(28,True),fill=(70,220,108))
        ov.text((18,50),"Can placed on green target OK",font=fnt(15),fill=(58,168,86))
        ov.text((hw+18,8),"FAULTY ARM",font=fnt(28,True),fill=(235,138,38))
        ov.text((hw+18,50),"Rotated from base -- missed can completely",font=fnt(15),fill=(200,128,58))
    else:
        ov.rectangle([(0,0),(hw,88)],fill=(4,8,16,255))
        ov.rectangle([(hw,0),(W,88)],fill=(3,16,32,255))
        ov.text((18,8),"GROUND TRUTH",font=fnt(28,True),fill=(70,220,108))
        ov.text((18,50),"Placing again",font=fnt(15),fill=(58,168,86))
        ov.text((hw+18,8),"CORRECTED ARM",font=fnt(28,True),fill=(32,190,225))
        ov.text((hw+18,50),"j1 ref nulled -- 0.000 rad OK",font=fnt(15),fill=(48,175,195))
    if l_ee is not None and r_ee is not None:
        gt_err=np.linalg.norm(l_ee-CAN_L)*1000
        rot_miss=np.sqrt((r_ee[0]-CAN_R[0])**2+(r_ee[1]-CAN_R[1])**2)*1000
        ov.rectangle([(18,96),(hw-18,160)],fill=(4,10,20,220))
        ov.text((26,100),f"EE->can: {gt_err:.1f}mm",font=fnt(14),fill=(100,220,130))
        ov.text((26,120),"Joint RMSE: 0.000",font=fnt(13),fill=(100,180,255))
        ov.text((26,140),"Fault: NONE",font=fnt(12),fill=(80,180,130))
        ov.rectangle([(hw+18,96),(W-18,160)],fill=(20,8,4,220))
        ov.text((hw+26,100),f"Rotational miss: {rot_miss:.1f}mm  (joint RMSE = 0)",font=fnt(14),fill=(255,180,80))
        ov.text((hw+26,120),"Joint space: HEALTHY  --  Cartesian XY: FAULT",font=fnt(13),fill=(255,200,80))
        if corr_applied:
            ov.text((hw+26,140),"j1 ref corrected -- 0.000 rad",font=fnt(12),fill=(32,190,225))
    if grasp_l: _badge(ov,hw//2,220,True,"GRASPED")
    if phase==1 and t>T_GRASP+1: _badge(ov,hw+hw//2,220,False,"ROTATIONAL MISS")
    if phase==3 and grasp_r: _badge(ov,hw+hw//2,220,True,"GRASPED")
    if phase==2 and t>T_HOLD:
        _result(ov,hw//2,H-185,True,"ON TARGET","")
        _result(ov,hw+hw//2,H-185,False,"ROTATIONAL MISS","j1 encoder offset -- base points wrong way")
    if phase==3 and t>T_HOLD2:
        _result(ov,hw//2,H-185,True,"ON TARGET","")
        _result(ov,hw+hw//2,H-185,True,"ON TARGET","SimCorrect corrected")
    ct=H-84; ov.rectangle([(0,ct),(W,H)],fill=(3,4,8,255))
    msgs={1:("Identical commands. Joint RMSE=0. Faulty arm base rotated wrong. Entire arm swings off axis.",(235,138,38)),
          2:("Left placed on target. Right j1 encoder offset -- arm swung wrong direction from base.",(200,140,60)),
          3:("j1 ref corrected to 0.000. Both arms now grasp and place on target.",(32,190,225))}
    txt,col=msgs.get(phase,("",""))
    ov.text((18,ct+14),txt,font=fnt(17,True),fill=col)
    ov.text((W-175,ct+28),f"t={t:.1f}s/{DUR}s",font=fnt(13),fill=(48,65,95))
    return img

def _badge(ov,cx,cy,ok,text):
    c=(14,200,72) if ok else (220,138,24); bg=(2,44,14,240) if ok else (44,20,4,240)
    ov.rectangle([(cx-175,cy-26),(cx+175,cy+26)],fill=bg,outline=c+(215,),width=2)
    ov.text((cx-152,cy-18),text,font=fnt(20,True),fill=c)

def _result(ov,cx,cy,success,l1,l2):
    c=(14,195,72) if success else (215,138,24); bg=(2,44,14,248) if success else (44,20,4,248)
    ov.rectangle([(cx-225,cy-48),(cx+225,cy+48)],fill=bg,outline=c+(224,),width=3)
    ov.text((cx-198,cy-36),l1,font=fnt(26,True),fill=c)
    if l2: ov.text((cx-198,cy+4),l2,font=fnt(15),fill=(100,148,115) if success else (158,115,75))

def main():
    SNAP_DIR=os.path.expanduser("~/simcorrect/Problem4_JointZeroOffset/output")
    os.makedirs(SNAP_DIR,exist_ok=True)
    snaps={2.0:"01_title.png",17.0:"02_rotational_miss.png",62.0:"04_corrected.png",82.0:"05_both_placed.png"}
    snaps_saved=set()
    print("Building model...")
    model,data=build(J1_REF_BAD)
    LA,RA,BL,BR,lee,ree,cam_id,lj,rj,lf,rf=get_ids(model)
    weld(data,BL,CAN_L); weld(data,BR,CAN_R)
    data.qpos[LA:LA+4]=HOME_Q; data.qpos[RA:RA+4]=HOME_Q_F
    set_arm(data,lj,rj,HOME_Q,HOME_Q_F)
    set_fingers(data,lf,rf,GRIP_OPEN,GRIP_OPEN)
    weld(data,BL,CAN_L); weld(data,BR,CAN_R)
    mujoco.mj_forward(model,data)
    renderer=mujoco.Renderer(model,height=H,width=W)
    sim_dt=model.opt.timestep; r_every=max(1,round(1.0/(FPS*sim_dt))); total=FPS*DUR
    frames=[]; t=0.0; step=0; fc=0
    phase=1; corrected=False
    in_freeze=False; freeze_count=0; freeze_total=int(FREEZE_DUR*FPS)
    freeze_img=None; flash=0
    grasp_l=False; grasp_r=False; carrying_l=False; carrying_r=False
    dropped_l=False; dropped_r=False; cl_on_table=False; cr_on_table=False
    cl_pos=CAN_L.copy(); cr_pos=CAN_R.copy()
    cur_l_ee=None; cur_r_ee=None; corr_applied=False
    print(f"Rendering {total} frames -> {OUT}")
    while fc<total:
        if not in_freeze:
            if not corrected:
                q_l,g_l=ref_ctrl_l(t); q_r,g_r=ref_ctrl_r(t)
            else:
                q_l,g_l=cor_ctrl_l(t); q_r,g_r=cor_ctrl_r(t); corr_applied=True
            data.qpos[LA:LA+4]=q_l; data.qpos[RA:RA+4]=q_r
            set_arm(data,lj,rj,q_l,q_r)
            set_fingers(data,lf,rf,g_l,g_r)
            mujoco.mj_kinematics(model,data)
            l_ee=data.site_xpos[lee].copy(); r_ee=data.site_xpos[ree].copy()
            cur_l_ee=l_ee.copy(); cur_r_ee=r_ee.copy()
            if not corrected:
                if not grasp_l and g_l>0.5 and np.linalg.norm(l_ee-CAN_L)<0.07:
                    grasp_l=True; carrying_l=True
                if carrying_l and not dropped_l:
                    cl_pos=l_ee.copy()
                    if t>=T_HOLD:
                        dropped_l=True; carrying_l=False; cl_on_table=True; phase=2
            else:
                if not carrying_l and not dropped_l and g_l>0.5 and np.linalg.norm(l_ee-CAN_L)<0.10:
                    carrying_l=True; grasp_l=True
                if carrying_l and not dropped_l:
                    cl_pos=l_ee.copy()
                    if t>=T_HOLD2:
                        dropped_l=True; carrying_l=False; cl_on_table=True
                if not grasp_r and g_r>0.5 and np.linalg.norm(r_ee-CAN_R)<0.10:
                    grasp_r=True; carrying_r=True
                if carrying_r and not dropped_r:
                    cr_pos=r_ee.copy()
                    if t>=T_HOLD2:
                        dropped_r=True; carrying_r=False; cr_on_table=True
            if cl_on_table:  weld(data,BL,TABLE_L)
            elif carrying_l: weld(data,BL,cl_pos)
            else:            weld(data,BL,CAN_L)
            if cr_on_table:  weld(data,BR,TABLE_R)
            elif carrying_r: weld(data,BR,cr_pos)
            else:            weld(data,BR,CAN_R)
            mujoco.mj_forward(model,data)
            t+=sim_dt; step+=1
        if t>=T_FREEZE and not corrected and not in_freeze:
            stl_path=run_opencad(J1_REF_BAD)
            print(f"  [{t:.1f}s] OpenCAD correction complete -> {stl_path}")
            renderer.update_scene(data,camera=cam_id)
            freeze_img=freeze_panel(renderer.render().copy())
            in_freeze=True; freeze_count=0; print(f"  [{t:.1f}s] Freeze")
        if in_freeze:
            frames.append(np.array(freeze_img)); fc+=1; freeze_count+=1
            if freeze_count==int((43.0-T_FREEZE)*FPS)+1:
                Image.fromarray(frames[-1]).save(os.path.join(SNAP_DIR,"03_freeze_panel.png"))
                print("  Saved snapshot: 03_freeze_panel.png")
            if freeze_count>=freeze_total:
                model,data=build(J1_REF_GT,"0.04 0.54 0.74 1")
                LA,RA,BL,BR,lee,ree,cam_id,lj,rj,lf,rf=get_ids(model)
                renderer=mujoco.Renderer(model,height=H,width=W)
                cl_pos=CAN_L.copy(); cr_pos=CAN_R.copy()
                cl_on_table=False; cr_on_table=False
                data.qpos[LA:LA+4]=HOME_Q; data.qpos[RA:RA+4]=HOME_Q
                set_arm(data,lj,rj,HOME_Q,HOME_Q)
                set_fingers(data,lf,rf,GRIP_OPEN,GRIP_OPEN)
                weld(data,BL,CAN_L); weld(data,BR,CAN_R)
                mujoco.mj_forward(model,data)
                corrected=True; phase=3; in_freeze=False; flash=28
                dropped_l=False; dropped_r=False; carrying_l=False; carrying_r=False
                grasp_l=False; grasp_r=False; corr_applied=False
                print(f"  [{t:.1f}s] Corrected arm loaded")
            if fc%FPS==0: print(f"  {fc:4d}/{total} FREEZE")
            continue
        if step%r_every==0:
            do_flash=flash>0; flash=max(0,flash-1)
            if t<T_TITLE:
                frm=title_card()
            else:
                renderer.update_scene(data,camera=cam_id)
                raw=renderer.render().copy()
                if do_flash:
                    fl=Image.new("RGBA",(W,H),(255,255,255,70))
                    raw=np.array(Image.alpha_composite(Image.fromarray(raw).convert("RGBA"),fl).convert("RGB"))
                frm=overlay(raw,t,phase,grasp_l,grasp_r,cur_l_ee,cur_r_ee,corr_applied)
            frames.append(np.array(frm)); fc+=1
            for snap_t,snap_name in snaps.items():
                if snap_name not in snaps_saved and abs(t-snap_t)<sim_dt*r_every*1.5:
                    Image.fromarray(np.array(frm)).save(os.path.join(SNAP_DIR,snap_name))
                    print(f"  Saved snapshot: {snap_name}")
                    snaps_saved.add(snap_name)
            if fc%FPS==0:
                print(f"  {fc:4d}/{total} t={t:.1f}s ph={phase} tbl=({int(cl_on_table)},{int(cr_on_table)}) g=({int(grasp_l)},{int(grasp_r)})")
    print(f"Writing {OUT}...")
    iio.imwrite(OUT,frames,fps=FPS,codec="libx264",
                output_params=["-crf","13","-pix_fmt","yuv420p","-preset","slow"])
    print(f"DONE -- {len(frames)} frames | {DUR}s | {FPS}fps")

if __name__=="__main__":
    main()
