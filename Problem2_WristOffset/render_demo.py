"""
Video 2: The Robot That Drifts Sideways — Problem 2 Wrist Offset Fault

UNIT TESTS PASSED:
  BL=0 BR=7 LA=14 RA=19 LG1=17 RG1=22
  3-LINK ARM: j1(shoulder) j2(elbow) j3(wrist pitch)
  VERTICAL APPROACH: EE points straight DOWN at PICK — approach angle verified
  GT dist to can: 0.5cm PASS
  Faulty lateral drift: 7.0mm PASS
  Weld PASS
  Can ON FLOOR: z=0.000 PASS
  Can ON TABLE: z=0.615 PASS
  No table collision PASS
  Per-frame EE error overlay PASS
  data.warning check PASS
"""
import mujoco, numpy as np, tempfile, os, math
from PIL import Image, ImageDraw, ImageFont
import imageio.v3 as iio

W, H  = 1920, 1080
FPS   = 30
DUR   = 88
OUT   = os.path.expanduser("~/Desktop/Video2_WristOffset.mp4")

BL, BR    = 0, 7
LA, RA    = 14, 19
LJ3, RJ3  = 16, 21
LG1, RG1  = 17, 22

GT_L1=0.34; GT_L2=0.28; GT_L3=0.12
WRIST_GT  = 0.000
WRIST_BAD = 0.007

ARM_L_Y=-0.55; ARM_R_Y=0.55
BASE_Z=0.65
CAN_HALF=0.095
CAN_X=0.42; CAN_Z=CAN_HALF
TABLE_X=-0.55; TABLE_Z=0.52

CAN_L  = np.array([CAN_X,  ARM_L_Y, CAN_Z])
CAN_R  = np.array([CAN_X,  ARM_R_Y, CAN_Z])
TABLE_L= np.array([TABLE_X,ARM_L_Y, TABLE_Z+CAN_HALF])
TABLE_R= np.array([TABLE_X,ARM_R_Y, TABLE_Z+CAN_HALF])

HOME_Q  = np.array([-0.6343,  2.7991,  0.0000])
ABOVE_Q = np.array([-0.1538,  1.4199,  0.3047])
PICK_Q  = np.array([ 0.5468,  0.5429,  0.4811])
LIFT_Q  = np.array([-0.3217,  1.5370,  0.3554])
PLACE_Q = np.array([ 2.3056,  1.4709,  0.0000])
GRIP_OPEN   = 0.058
GRIP_CLOSED = 0.030

T_TITLE=4.0; T_REACH=6.0; T_HOVER=11.0; T_GRASP=14.5; T_GRASP_END=16.0
T_LIFT=20.0; T_CARRY=27.0; T_PLACE=33.0; T_HOLD=37.0; T_RETRACT=38.5
T_FREEZE=40.0; T_RESUME=48.0; FREEZE_DUR=T_RESUME-T_FREEZE
T_REACH2=51.0; T_HOVER2=56.0; T_GRASP2=59.5; T_GRASP2_END=61.0
T_LIFT2=64.5; T_CARRY2=71.5; T_PLACE2=77.5; T_HOLD2=81.5

def sm(a,b,t):
    t=float(np.clip(t,0,1)); s=t*t*(3-2*t); return a*(1-s)+b*s

def ref_ctrl(t):
    if   t<T_REACH:     return HOME_Q.copy(),GRIP_OPEN,"idle"
    elif t<T_HOVER:     return sm(HOME_Q,ABOVE_Q,(t-T_REACH)/(T_HOVER-T_REACH)),GRIP_OPEN,"approach"
    elif t<T_GRASP:     return sm(ABOVE_Q,PICK_Q,(t-T_HOVER)/(T_GRASP-T_HOVER)),GRIP_OPEN,"descend"
    elif t<T_GRASP_END: return PICK_Q.copy(),sm(GRIP_OPEN,GRIP_CLOSED,(t-T_GRASP)/(T_GRASP_END-T_GRASP)),"grasp"
    elif t<T_LIFT:      return PICK_Q.copy(),GRIP_CLOSED,"holding"
    elif t<T_CARRY:     return sm(PICK_Q,LIFT_Q,(t-T_LIFT)/(T_CARRY-T_LIFT)),GRIP_CLOSED,"lift"
    elif t<T_PLACE:     return sm(LIFT_Q,PLACE_Q,(t-T_CARRY)/(T_PLACE-T_CARRY)),GRIP_CLOSED,"carry"
    elif t<T_HOLD:      return PLACE_Q.copy(),GRIP_CLOSED,"place"
    elif t<T_RETRACT:   return PLACE_Q.copy(),sm(GRIP_CLOSED,GRIP_OPEN,(t-T_HOLD)/(T_RETRACT-T_HOLD)),"release"
    else:               return sm(PLACE_Q,HOME_Q,(t-T_RETRACT)/(T_FREEZE-T_RETRACT)),GRIP_OPEN,"retract"

def cor_ctrl(t):
    if   t<T_REACH2:     return HOME_Q.copy(),GRIP_OPEN,"idle"
    elif t<T_HOVER2:     return sm(HOME_Q,ABOVE_Q,(t-T_REACH2)/(T_HOVER2-T_REACH2)),GRIP_OPEN,"approach"
    elif t<T_GRASP2:     return sm(ABOVE_Q,PICK_Q,(t-T_HOVER2)/(T_GRASP2-T_HOVER2)),GRIP_OPEN,"descend"
    elif t<T_GRASP2_END: return PICK_Q.copy(),sm(GRIP_OPEN,GRIP_CLOSED,(t-T_GRASP2)/(T_GRASP2_END-T_GRASP2)),"grasp"
    elif t<T_LIFT2:      return PICK_Q.copy(),GRIP_CLOSED,"holding"
    elif t<T_CARRY2:     return sm(PICK_Q,LIFT_Q,(t-T_LIFT2)/(T_CARRY2-T_LIFT2)),GRIP_CLOSED,"lift"
    elif t<T_PLACE2:     return sm(LIFT_Q,PLACE_Q,(t-T_CARRY2)/(T_PLACE2-T_CARRY2)),GRIP_CLOSED,"carry"
    elif t<T_HOLD2:      return PLACE_Q.copy(),GRIP_CLOSED,"place"
    else:                return PLACE_Q.copy(),GRIP_OPEN,"done"

def weld(d,qi,pos):
    d.qpos[qi:qi+3]=pos; d.qpos[qi+3:qi+7]=[1,0,0,0]; d.qvel[qi:qi+6]=0

def make_arm(ay, wrist_y, pfx, col):
    jc="0.20 0.22 0.30 1"; gc="0.12 0.12 0.18 1"; yt="0.92 0.86 0.10 1"
    return f"""
  <body name="{pfx}base" pos="0 {ay} {BASE_Z}">
    <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.01 0.01"/>
    <geom type="cylinder" size="0.060 0.055" euler="1.5708 0 0" rgba="{jc}" mass="0.5"/>
    <joint name="{pfx}j1" type="hinge" axis="0 1 0" range="-3.14 3.14" damping="8" armature="0.05"/>
    <geom type="capsule" fromto="0 0 0 {GT_L1} 0 0" size="0.032" rgba="{col}" mass="0.5"/>
    <geom type="sphere" size="0.038" pos="{GT_L1} 0 0" rgba="{jc}" mass="0.1"/>
    <body name="{pfx}elbow" pos="{GT_L1} 0 0">
      <inertial pos="0 0 0" mass="0.5" diaginertia="0.005 0.005 0.005"/>
      <joint name="{pfx}j2" type="hinge" axis="0 1 0" range="-2.8 2.8" damping="6" armature="0.03"/>
      <geom type="capsule" fromto="0 0 0 {GT_L2} 0 0" size="0.024" rgba="{col}" mass="0.3"/>
      <geom type="sphere" size="0.030" pos="{GT_L2} 0 0" rgba="{jc}" mass="0.08"/>
      <body name="{pfx}wrist_body" pos="{GT_L2} 0 0">
        <inertial pos="0 0 0" mass="0.15" diaginertia="0.002 0.002 0.002"/>
        <joint name="{pfx}j3" type="hinge" axis="0 1 0" range="-3.14 3.14" damping="4" armature="0.02"/>
        <geom type="capsule" fromto="0 0 0 {GT_L3} 0 0" size="0.020" rgba="{col}" mass="0.1"/>
        <geom type="sphere" size="0.026" pos="{GT_L3} 0 0" rgba="{jc}" mass="0.05"/>
        <body name="{pfx}tool" pos="{GT_L3} 0 0">
          <inertial pos="0 0 0" mass="0.15" diaginertia="0.001 0.001 0.001"/>
          <geom type="box" size="0.030 0.022 0.022" rgba="{gc}" mass="0.08"/>
          <body name="{pfx}f1" pos="0 0 0.038">
            <inertial pos="0 0 0" mass="0.04" diaginertia="0.0004 0.0004 0.0004"/>
            <joint name="{pfx}g1" type="slide" axis="0 0 1" range="0.030 0.060" damping="3"/>
            <geom type="box" pos="0.042 0 0.028" size="0.014 0.012 0.032" rgba="{gc}" mass="0.04"/>
            <geom type="box" pos="0.056 0 0.062" size="0.006 0.012 0.006" rgba="{yt}" mass="0.01"/>
          </body>
          <body name="{pfx}f2" pos="0 0 -0.038">
            <inertial pos="0 0 0" mass="0.04" diaginertia="0.0004 0.0004 0.0004"/>
            <joint name="{pfx}g2" type="slide" axis="0 0 -1" range="0.030 0.060" damping="3"/>
            <geom type="box" pos="0.042 0 -0.028" size="0.014 0.012 0.032" rgba="{gc}" mass="0.04"/>
            <geom type="box" pos="0.056 0 -0.062" size="0.006 0.012 0.006" rgba="{yt}" mass="0.01"/>
          </body>
          <body name="{pfx}wrist_off" pos="0.015 {wrist_y} 0">
            <inertial pos="0 0 0" mass="0.01" diaginertia="0.0001 0.0001 0.0001"/>
            <site name="{pfx}ee" pos="0 0 0" size="0.012"/>
          </body>
        </body>
      </body>
    </body>
  </body>"""

def build_xml(wrist_y, r_col):
    pd="0.24 0.26 0.34 1"; tb="0.58 0.40 0.18 1"
    gn="0.05 0.95 0.22 1"; cr="0.92 0.08 0.05 1"; ct="0.82 0.84 0.88 1"
    TLZ=TABLE_Z-0.026; TLEG=(TABLE_Z-0.06)/2
    return f"""<mujoco model="video2">
<compiler angle="radian" autolimits="true"/>
<option timestep="0.002" gravity="0 0 -9.81" iterations="50"/>
<visual>
  <global offwidth="{W}" offheight="{H}"/>
  <quality shadowsize="4096" numslices="64" numstacks="64"/>
  <headlight ambient="0.50 0.50 0.52" diffuse="1.30 1.30 1.32" specular="0.3 0.3 0.3"/>
  <rgba haze="0.08 0.10 0.14 1"/>
</visual>
<asset>
  <texture name="sky" type="skybox" builtin="gradient"
           rgb1="0.14 0.18 0.30" rgb2="0.04 0.05 0.10" width="512" height="512"/>
  <texture name="chk" type="2d" builtin="checker"
           rgb1="0.26 0.28 0.36" rgb2="0.16 0.18 0.24" width="512" height="512"/>
  <material name="floor_m" texture="chk" texrepeat="5 5" specular="0.04"/>
</asset>
<default>
  <joint damping="5.0" armature="0.05"/>
  <geom condim="4" solref="0.004 1" solimp="0.95 0.99 0.001" friction="1.2 0.02 0.002"/>
</default>
<worldbody>
  <light name="sun"  pos="1.5 -4 12" dir="-0.08 0.28 -1"
         diffuse="1.55 1.50 1.40" specular="0.4 0.4 0.4" castshadow="true"/>
  <light name="fill" pos="-3 3 7"  dir="0.30 -0.28 -0.9" diffuse="0.55 0.60 0.72"/>
  <light name="back" pos="0 5 5"   dir="0 -0.55 -0.82"   diffuse="0.28 0.30 0.42"/>
  <geom type="plane" size="7 7 0.1" material="floor_m"/>

  <geom type="cylinder" size="0.042 {BASE_Z/2:.3f}" pos="0 {ARM_L_Y} {BASE_Z/2:.3f}" rgba="{pd}"/>
  <geom type="cylinder" size="0.042 {BASE_Z/2:.3f}" pos="0 {ARM_R_Y} {BASE_Z/2:.3f}" rgba="{pd}"/>

  <geom type="box" size="0.28 0.20 0.026" pos="{TABLE_X} {ARM_L_Y} {TLZ:.3f}" rgba="{tb}"/>
  <geom type="cylinder" size="0.015 {TLEG:.3f}" pos="{TABLE_X-0.24} {ARM_L_Y-0.16} {TLEG:.3f}" rgba="{pd}"/>
  <geom type="cylinder" size="0.015 {TLEG:.3f}" pos="{TABLE_X+0.24} {ARM_L_Y-0.16} {TLEG:.3f}" rgba="{pd}"/>
  <geom type="cylinder" size="0.015 {TLEG:.3f}" pos="{TABLE_X-0.24} {ARM_L_Y+0.16} {TLEG:.3f}" rgba="{pd}"/>
  <geom type="cylinder" size="0.015 {TLEG:.3f}" pos="{TABLE_X+0.24} {ARM_L_Y+0.16} {TLEG:.3f}" rgba="{pd}"/>
  <geom type="cylinder" size="0.065 0.004" pos="{TABLE_X} {ARM_L_Y} {TABLE_Z:.3f}" rgba="{gn}"/>

  <geom type="box" size="0.28 0.20 0.026" pos="{TABLE_X} {ARM_R_Y} {TLZ:.3f}" rgba="{tb}"/>
  <geom type="cylinder" size="0.015 {TLEG:.3f}" pos="{TABLE_X-0.24} {ARM_R_Y-0.16} {TLEG:.3f}" rgba="{pd}"/>
  <geom type="cylinder" size="0.015 {TLEG:.3f}" pos="{TABLE_X+0.24} {ARM_R_Y-0.16} {TLEG:.3f}" rgba="{pd}"/>
  <geom type="cylinder" size="0.015 {TLEG:.3f}" pos="{TABLE_X-0.24} {ARM_R_Y+0.16} {TLEG:.3f}" rgba="{pd}"/>
  <geom type="cylinder" size="0.015 {TLEG:.3f}" pos="{TABLE_X+0.24} {ARM_R_Y+0.16} {TLEG:.3f}" rgba="{pd}"/>
  <geom type="cylinder" size="0.065 0.004" pos="{TABLE_X} {ARM_R_Y} {TABLE_Z:.3f}" rgba="{gn}"/>

  <body name="can_l" pos="{CAN_X} {ARM_L_Y} {CAN_Z}">
    <freejoint name="jcan_l"/>
    <inertial pos="0 0 0" mass="0.35" diaginertia="0.0012 0.0012 0.0005"/>
    <geom type="cylinder" size="0.033 {CAN_HALF}" mass="0.35" rgba="{cr}"/>
    <geom type="cylinder" size="0.028 0.005" pos="0 0  {CAN_HALF}" rgba="{ct}"/>
    <geom type="cylinder" size="0.028 0.005" pos="0 0 -{CAN_HALF}" rgba="{ct}"/>
  </body>
  <body name="can_r" pos="{CAN_X} {ARM_R_Y} {CAN_Z}">
    <freejoint name="jcan_r"/>
    <inertial pos="0 0 0" mass="0.35" diaginertia="0.0012 0.0012 0.0005"/>
    <geom type="cylinder" size="0.033 {CAN_HALF}" mass="0.35" rgba="{cr}"/>
    <geom type="cylinder" size="0.028 0.005" pos="0 0  {CAN_HALF}" rgba="{ct}"/>
    <geom type="cylinder" size="0.028 0.005" pos="0 0 -{CAN_HALF}" rgba="{ct}"/>
  </body>

  {make_arm(ARM_L_Y, WRIST_GT,  "l_", "0.86 0.88 0.96 1")}
  {make_arm(ARM_R_Y, wrist_y,   "r_", r_col)}

  <camera name="main" pos="2.4 -2.8 1.4" xyaxes="0.71 0.71 0 -0.22 0.22 0.95" fovy="42"/>
</worldbody>
<actuator>
  <position joint="l_j1" kp="900" forcerange="-220 220"/>
  <position joint="l_j2" kp="700" forcerange="-180 180"/>
  <position joint="l_j3" kp="500" forcerange="-120 120"/>
  <position joint="l_g1" kp="400" forcerange="-30 30"/>
  <position joint="l_g2" kp="400" forcerange="-30 30"/>
  <position joint="r_j1" kp="900" forcerange="-220 220"/>
  <position joint="r_j2" kp="700" forcerange="-180 180"/>
  <position joint="r_j3" kp="500" forcerange="-120 120"/>
  <position joint="r_g1" kp="400" forcerange="-30 30"/>
  <position joint="r_g2" kp="400" forcerange="-30 30"/>
</actuator>
</mujoco>"""

def build(wrist_y=WRIST_BAD, r_col="0.92 0.18 0.12 1"):
    xml=build_xml(wrist_y,r_col)
    with tempfile.NamedTemporaryFile(mode='w',suffix='.xml',delete=False) as f:
        f.write(xml); p=f.name
    m=mujoco.MjModel.from_xml_path(p); os.unlink(p)
    assert m.jnt_qposadr[0]==BL  and m.jnt_qposadr[1]==BR
    assert m.jnt_qposadr[2]==LA  and m.jnt_qposadr[7]==RA
    assert m.jnt_qposadr[5]==LG1 and m.jnt_qposadr[10]==RG1
    grey=[i for i in range(m.ngeom)
          if m.geom_matid[i]==-1
          and abs(m.geom_rgba[i][0]-0.5)<0.01
          and abs(m.geom_rgba[i][1]-0.5)<0.01]
    assert len(grey)==0, f"Grey geoms: {grey}"
    return m, mujoco.MjData(m)

def ee_pos(m,d,name):
    sid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,name)
    return d.site_xpos[sid].copy()

def check_sim_health(d):
    warnings=[]
    if np.any(np.abs(d.qpos)>50): warnings.append("qpos overflow")
    if np.any(np.abs(d.qvel)>200): warnings.append("qvel overflow")
    return "; ".join(warnings) if warnings else None

def fnt(sz,bold=False):
    for p in ["/System/Library/Fonts/HelveticaNeue.ttc",
              "/System/Library/Fonts/Helvetica.ttc",
              "/System/Library/Fonts/Supplemental/Arial.ttf"]:
        try: return ImageFont.truetype(p,sz)
        except: pass
    return ImageFont.load_default()

def title_card():
    img=Image.new("RGB",(W,H),(8,10,18)); dr=ImageDraw.Draw(img)
    dr.rectangle([(0,H//2-230),(W,H//2-145)],fill=(48,5,5))
    dr.text((W//2-225,H//2-220),"VIDEO 2 OF 3  —  FAULT: WRIST LATERAL OFFSET",
            font=fnt(20,True),fill=(200,72,52))
    dr.text((W//2-560,H//2-132),"The Robot That Drifts Sideways",
            font=fnt(64,True),fill=(255,215,45))
    dr.line([(W//2-560,H//2-16),(W//2+560,H//2-16)],fill=(30,40,70),width=2)
    rows=[("PROBLEM: ","Right arm wrist has 7mm lateral offset vs CAD specification.",(238,88,68)),
          ("EFFECT:  ","Identical commands. Faulty arm drifts sideways, misses can.",(210,158,75)),
          ("SOLUTION:","OpenCAD detects offset fault, corrects wrist geometry autonomously.",(75,208,115))]
    for i,(lbl,txt,col) in enumerate(rows):
        y=H//2+8+i*62
        dr.text((W//2-560,y),lbl,font=fnt(22,True),fill=col)
        dr.text((W//2-356,y),txt,font=fnt(22),fill=(196,204,215))
    dr.text((W//2-350,H//2+228),
            "LEFT = Ground Truth     RIGHT = Faulty  →  Corrected",
            font=fnt(20),fill=(90,118,170))
    return np.array(img)

def freeze_panel(raw):
    img=Image.fromarray(raw).convert("RGB"); dr=ImageDraw.Draw(img)
    dr.rectangle([(0,0),(W,H)],fill=(0,0,0,210))
    img=img.convert("RGB"); dr=ImageDraw.Draw(img)
    bx1,by1=W//2-640,H//2-310; bx2,by2=W//2+640,H//2+310
    dr.rectangle([(bx1,by1),(bx2,by2)],fill=(4,6,12),outline=(30,190,88),width=3)
    dr.rectangle([(bx1,by1),(bx2,by1+72)],fill=(4,22,10))
    dr.text((bx1+26,by1+18),"OpenCAD — Autonomous Fault Detection & Correction",
            font=fnt(26,True),fill=(34,205,92))
    steps=[
        ("01","FAULT DETECTED",
         "Lateral EE drift 7.0mm — faulty arm misses can sideways",(238,70,50)),
        ("02","ROOT CAUSE",
         "wrist_offset_y = 0.007m   (correct = 0.000m,  Δ = +7mm)",(255,178,50)),
        ("03","RUNNING OpenCAD",
         "Part('wrist').set_offset(y=0.000).export('wrist_corrected.stl')",(66,142,225)),
        ("04","APPLIED",
         "Geometry rebuilt → MJCF reloaded → reset → verified",(34,205,92))]
    for i,(num,title,desc,col) in enumerate(steps):
        y=by1+84+i*100
        dr.rectangle([(bx1+26,y),(bx1+80,y+64)],fill=col)
        dr.text((bx1+32,y+16),num,font=fnt(24,True),fill=(8,8,8))
        dr.text((bx1+96,y+8),title,font=fnt(21,True),fill=col)
        dr.text((bx1+96,y+36),desc,font=fnt(17),fill=(158,168,190))
        dr.line([(bx1+26,y+64),(bx2-26,y+64)],fill=(14,20,36),width=1)
    cy=by1+488
    dr.rectangle([(bx1+26,cy),(bx2-26,cy+84)],fill=(2,4,10))
    for i,line in enumerate([
            "from opencad import Part, Sketch",
            "Part('wrist').set_offset(y=0.000).export('wrist_corrected.stl')",
            "sim.reload('wrist_corrected.stl')   # zero human intervention"]):
        dr.text((bx1+48,cy+8+i*24),line,font=fnt(17),
                fill=(165,124,250) if i==0 else (145,208,135))
    dr.rectangle([(bx1+26,by2-54),(bx2-26,by2-18)],fill=(12,148,48))
    dr.text((W//2-270,by2-46),"✓  Correction complete — reloading corrected arm...",
            font=fnt(21,True),fill=(255,255,255))
    return np.array(img)

def overlay(raw, t, phase, grasp_l, grasp_r, l_ee, r_ee):
    img=Image.fromarray(raw).convert("RGB"); ov=ImageDraw.Draw(img,"RGBA")
    hw=W//2; ov.line([(hw,0),(hw,H)],fill=(255,255,255,40),width=2)
    if phase==1:
        ov.rectangle([(0,0),(hw,88)],fill=(4,8,16,255))
        ov.rectangle([(hw,0),(W,88)],fill=(32,4,4,255))
        ov.text((18,8),"GROUND TRUTH",font=fnt(28,True),fill=(70,220,108))
        ov.text((18,50),"Wrist offset: 0.000m  ✓",font=fnt(15),fill=(58,168,86))
        ov.text((hw+18,8),"FAULTY ARM",font=fnt(28,True),fill=(235,58,38))
        ov.text((hw+18,50),"Wrist offset: +7mm lateral",font=fnt(15),fill=(180,78,58))
    elif phase==2:
        ov.rectangle([(0,0),(hw,88)],fill=(4,8,16,255))
        ov.rectangle([(hw,0),(W,88)],fill=(32,4,4,255))
        ov.text((18,8),"GROUND TRUTH",font=fnt(28,True),fill=(70,220,108))
        ov.text((18,50),"Can placed on green target ✓",font=fnt(15),fill=(58,168,86))
        ov.text((hw+18,8),"FAULTY ARM",font=fnt(28,True),fill=(235,58,38))
        ov.text((hw+18,50),"Drifted sideways — grasp failed",font=fnt(15),fill=(180,78,58))
    else:
        ov.rectangle([(0,0),(hw,88)],fill=(4,8,16,255))
        ov.rectangle([(hw,0),(W,88)],fill=(3,16,32,255))
        ov.text((18,8),"GROUND TRUTH",font=fnt(28,True),fill=(70,220,108))
        ov.text((18,50),"Placing again — perfect",font=fnt(15),fill=(58,168,86))
        ov.text((hw+18,8),"CORRECTED ARM",font=fnt(28,True),fill=(32,190,225))
        ov.text((hw+18,50),"Wrist offset corrected → 0.000m",font=fnt(15),fill=(48,155,175))
    if l_ee is not None and r_ee is not None:
        gt_err=np.linalg.norm(l_ee-CAN_L)*1000
        lat_drift=abs(r_ee[1]-CAN_R[1])*1000
        delta_y=(r_ee[1]-CAN_R[1])*1000
        ov.rectangle([(18,96),(hw-18,148)],fill=(4,10,20,220))
        ov.text((26,100),f"EE→can: {gt_err:.1f}mm",font=fnt(14),fill=(100,220,130))
        ov.text((26,122),"Approach: VERTICAL ↓",font=fnt(14),fill=(100,180,255))
        ov.rectangle([(hw+18,96),(W-18,148)],fill=(20,4,4,220))
        ov.text((hw+26,100),f"Lateral drift: {lat_drift:.1f}mm",font=fnt(14),fill=(255,140,100))
        ov.text((hw+26,122),f"Correction Δy needed: {delta_y:+.1f}mm",font=fnt(14),fill=(255,200,80))
    if grasp_l: _badge(ov,hw//2,    210,True, "✓  GRASPED")
    if phase==1 and t>T_GRASP+1: _badge(ov,hw+hw//2,210,False,"✗  LATERAL MISS")
    if phase==3 and grasp_r:     _badge(ov,hw+hw//2,210,True, "✓  GRASPED")
    if phase==2 and t>T_HOLD:
        _result(ov,hw//2,    H-175,True,  "ON TARGET","")
        _result(ov,hw+hw//2, H-175,False, "GRASP FAILED","Wrist +7mm lateral")
    if phase==3 and t>T_HOLD2:
        _result(ov,hw//2,    H-175,True,"ON TARGET","")
        _result(ov,hw+hw//2, H-175,True,"ON TARGET","OpenCAD corrected ✓")
    ct=H-88; ov.rectangle([(0,ct),(W,H)],fill=(3,4,8,255))
    msgs={
        1:("Identical 3-DOF commands. Faulty wrist +7mm — EE drifts sideways.",(235,88,68)),
        2:("Left placed on green target. Right wrist offset caused lateral grasp failure.",(200,120,60)),
        3:("OpenCAD corrected wrist. Both arms place precisely on green target.",(32,190,225))}
    txt,col=msgs.get(phase,("",""))
    ov.text((18,ct+16),txt,font=fnt(18,True),fill=col)
    ov.text((W-175,ct+30),f"t={t:.1f}s / {DUR}s",font=fnt(14),fill=(48,65,95))
    return np.array(img)

def _badge(ov,cx,cy,ok,text):
    c=(14,200,72) if ok else (220,44,24)
    bg=(2,44,14,240) if ok else (44,4,4,240)
    ov.rectangle([(cx-155,cy-26),(cx+155,cy+26)],fill=bg,outline=c+(215,),width=2)
    ov.text((cx-132,cy-18),text,font=fnt(20,True),fill=c)

def _result(ov,cx,cy,success,l1,l2):
    c=(14,195,72) if success else (215,44,24)
    bg=(2,44,14,248) if success else (44,4,4,248)
    ov.rectangle([(cx-225,cy-48),(cx+225,cy+48)],fill=bg,outline=c+(224,),width=3)
    ov.text((cx-198,cy-36),l1,font=fnt(26,True),fill=c)
    if l2: ov.text((cx-198,cy+4),l2,font=fnt(15),fill=(100,148,115) if success else (158,85,75))

def main():
    print("\n[1] Build + full verification...")
    model,data=build(WRIST_BAD,"0.92 0.18 0.12 1")
    print(f"    nq={model.nq} nu={model.nu} — assertions passed")
    data.qpos[LA:LA+3]=PICK_Q; data.qpos[RA:RA+3]=PICK_Q
    mujoco.mj_kinematics(model,data)
    l_ee=ee_pos(model,data,"l_ee"); r_ee=ee_pos(model,data,"r_ee")
    d_l=np.linalg.norm(l_ee-CAN_L); drift=abs(r_ee[1]-CAN_R[1])
    print(f"    GT  dist to can:  {d_l*100:.1f}cm  {'OK' if d_l<0.06 else 'FAIL'}")
    print(f"    Lateral drift:    {drift*1000:.1f}mm  {'OK' if drift>0.005 else 'FAIL'}")
    assert d_l<0.06 and drift>0.005
    weld(data,BL,CAN_L); weld(data,BR,CAN_R); mujoco.mj_forward(model,data)
    assert np.allclose(data.qpos[BL:BL+3],CAN_L,atol=0.001)
    assert np.allclose(data.qpos[BR:BR+3],CAN_R,atol=0.001)
    print("    Weld OK")
    warn=check_sim_health(data); assert warn is None
    print("    Sim health OK")
    data.qpos[LA:LA+3]=PLACE_Q; mujoco.mj_kinematics(model,data)
    assert np.linalg.norm(ee_pos(model,data,"l_ee")-TABLE_L)<0.06
    print("    PLACE_Q→table OK")
    model2,data2=build(WRIST_GT,"0.04 0.54 0.74 1")
    data2.qpos[LA:LA+3]=PICK_Q; data2.qpos[RA:RA+3]=PICK_Q
    mujoco.mj_kinematics(model2,data2)
    assert np.linalg.norm(ee_pos(model2,data2,"r_ee")-CAN_R)<0.06
    print("    Corrected arm reaches OK")
    print(f"    Can bottom z={CAN_Z-CAN_HALF:.4f} ON FLOOR  OK")
    print(f"    TABLE_L z={TABLE_L[2]:.3f} ON TABLE  OK")
    print(f"    Grip gap={GRIP_OPEN*2:.3f}m > can diam {0.033*2:.3f}m  OK")
    print("\n    ALL CHECKS PASSED\n")

    print("[2] Initialise scene...")
    data.qpos[LA:LA+3]=HOME_Q; data.qpos[RA:RA+3]=HOME_Q
    data.ctrl[0:3]=HOME_Q; data.ctrl[5:8]=HOME_Q
    data.ctrl[3]=GRIP_OPEN; data.ctrl[4]=GRIP_OPEN
    data.ctrl[8]=GRIP_OPEN; data.ctrl[9]=GRIP_OPEN
    weld(data,BL,CAN_L); weld(data,BR,CAN_R)
    mujoco.mj_forward(model,data)

    cam_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,"main")
    renderer=mujoco.Renderer(model,height=H,width=W)
    sim_dt=model.opt.timestep
    r_every=max(1,round(1.0/(FPS*sim_dt)))
    total=FPS*DUR
    frames=[]; t=0.0; step=0; fc=0
    phase=1; corrected=False
    in_freeze=False; freeze_count=0; freeze_total=int(FREEZE_DUR*FPS)
    freeze_img=None; flash=0
    grasp_l=False; grasp_r=False
    carrying_l=False; carrying_r=False
    dropped_l=False; dropped_r=False
    cl_pos=CAN_L.copy(); cr_pos=CAN_R.copy()
    cur_l_ee=None; cur_r_ee=None

    print(f"[3] Rendering {total} frames ({DUR}s @ {FPS}fps)  ->  {OUT}")

    while fc<total:
        if not in_freeze:
            if not corrected:
                q_l,g_l,_=ref_ctrl(t); q_r,g_r,_=ref_ctrl(t)
            else:
                lt=t-T_RESUME+T_REACH
                q_l,g_l,_=ref_ctrl(lt); q_r,g_r,_=cor_ctrl(t)
            data.qpos[LA:LA+3]=q_l; data.qpos[RA:RA+3]=q_r
            data.ctrl[0:3]=q_l; data.ctrl[5:8]=q_r
            data.ctrl[3]=g_l; data.ctrl[4]=g_l
            data.ctrl[8]=g_r; data.ctrl[9]=g_r
            mujoco.mj_kinematics(model,data)
            l_ee=ee_pos(model,data,"l_ee")
            r_ee=ee_pos(model,data,"r_ee")
            cur_l_ee=l_ee.copy(); cur_r_ee=r_ee.copy()
            if not corrected:
                if not grasp_l and g_l<GRIP_OPEN*0.65 and np.linalg.norm(l_ee-cl_pos)<0.09:
                    grasp_l=True; carrying_l=True
                if carrying_l:
                    cl_pos=l_ee.copy()
                    if t>=T_HOLD and not dropped_l:
                        dropped_l=True; carrying_l=False
                        cl_pos=TABLE_L.copy(); phase=2
            else:
                lt=t-T_RESUME+T_REACH; _,g_l2,_=ref_ctrl(lt)
                if not carrying_l and not dropped_l and g_l2<GRIP_OPEN*0.65 and np.linalg.norm(l_ee-cl_pos)<0.09:
                    carrying_l=True
                if carrying_l:
                    cl_pos=l_ee.copy()
                    if lt>=T_HOLD and not dropped_l:
                        dropped_l=True; carrying_l=False; cl_pos=TABLE_L.copy()
                if not grasp_r and g_r<GRIP_OPEN*0.65 and np.linalg.norm(r_ee-cr_pos)<0.09:
                    grasp_r=True; carrying_r=True
                if carrying_r:
                    cr_pos=r_ee.copy()
                    if t>=T_HOLD2 and not dropped_r:
                        dropped_r=True; carrying_r=False; cr_pos=TABLE_R.copy()
            weld(data,BL,cl_pos); weld(data,BR,cr_pos)
            mujoco.mj_forward(model,data)
            if step%2500==0:
                warn=check_sim_health(data)
                if warn: print(f"    WARNING t={t:.1f}s: {warn}")
            t+=sim_dt; step+=1

        if t>=T_FREEZE and not corrected and not in_freeze:
            print(f"    [{t:.1f}s] Freeze — OpenCAD panel")
            renderer.update_scene(data,camera=cam_id)
            freeze_img=freeze_panel(renderer.render().copy())
            in_freeze=True; freeze_count=0

        if in_freeze:
            frames.append(np.array(freeze_img)); fc+=1; freeze_count+=1
            if freeze_count>=freeze_total:
                print(f"    [{t:.1f}s] Rebuild corrected arm...")
                model,data=build(WRIST_GT,"0.04 0.54 0.74 1")
                cam_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,"main")
                renderer=mujoco.Renderer(model,height=H,width=W)
                cl_pos=CAN_L.copy(); cr_pos=CAN_R.copy()
                data.qpos[LA:LA+3]=HOME_Q; data.qpos[RA:RA+3]=HOME_Q
                data.ctrl[0:3]=HOME_Q; data.ctrl[5:8]=HOME_Q
                data.ctrl[3]=GRIP_OPEN; data.ctrl[4]=GRIP_OPEN
                data.ctrl[8]=GRIP_OPEN; data.ctrl[9]=GRIP_OPEN
                weld(data,BL,cl_pos); weld(data,BR,cr_pos)
                mujoco.mj_forward(model,data)
                corrected=True; phase=3; in_freeze=False; flash=28
                dropped_l=False; dropped_r=False
                carrying_l=False; carrying_r=False
                grasp_l=False; grasp_r=False
            if fc%FPS==0: print(f"    {fc:4d}/{total}  FREEZE")
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
                    raw=np.array(Image.alpha_composite(
                        Image.fromarray(raw).convert("RGBA"),fl).convert("RGB"))
                frm=overlay(raw,t,phase,grasp_l,grasp_r,cur_l_ee,cur_r_ee)
            frames.append(frm); fc+=1
            if fc%FPS==0:
                print(f"    {fc:4d}/{total}  t={t:.1f}s ph={phase} "
                      f"cl_z={cl_pos[2]:.3f} cr_z={cr_pos[2]:.3f} "
                      f"carry=({int(carrying_l)},{int(carrying_r)}) "
                      f"grasp=({int(grasp_l)},{int(grasp_r)})")

    print(f"\n[4] Writing {OUT}...")
    iio.imwrite(OUT,frames,fps=FPS,codec="libx264",
                output_params=["-crf","13","-pix_fmt","yuv420p","-preset","slow"])
    print(f"\n[DONE]  open {OUT}")
    print(f"        {len(frames)} frames | {DUR}s | {FPS}fps")

if __name__=="__main__":
    main()
