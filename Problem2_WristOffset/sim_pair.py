"""
Problem 2: Wrist Offset Fault — Dual Simulation
Two MuJoCo simulations running in parallel:
  - Sim A: Ground truth (correct wrist lateral offset)
  - Sim B: Faulty model (injected wrist offset error)
Both log joint state trajectories and EE positions for divergence detection.
"""

import mujoco
import numpy as np
import tempfile
import os


ROBOT_XML_TEMPLATE = """
<mujoco model="simple_arm">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.3 0.3 0.3 1"/>
    <body name="can" pos="0.38 0.085 0.1">
      <geom type="cylinder" size="0.033 0.06" rgba="0.85 0.15 0.15 1"/>
      <geom type="cylinder" size="0.033 0.005" pos="0 0 0.065" rgba="0.75 0.75 0.75 1"/>
      <site name="can_site" size="0.01" pos="0 0 0.06"/>
    </body>
    <body name="link1" pos="0 0 0.1">
      <joint name="joint1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
      <geom type="capsule" size="0.04" fromto="0 0 0 0 0 0.30" rgba="0.2 0.6 0.9 1"/>
      <body name="link2" pos="0 0 0.30">
        <joint name="joint2" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
        <geom type="capsule" size="0.03" fromto="0 0 0 0.38 0 0" rgba="0.9 0.4 0.2 1"/>
        <body name="wrist" pos="0.38 {wrist_offset} 0">
          <geom type="sphere" size="0.025" rgba="0.1 0.9 0.3 1"/>
          <site name="ee_site" size="0.01"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="joint1" gear="50" ctrllimited="true" ctrlrange="-1 1"/>
    <motor joint="joint2" gear="50" ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""

GROUND_TRUTH_PARAMS = {"wrist_offset": 0.085}
FAULTY_PARAMS       = {"wrist_offset": 0.092}
INJECTED_ERROR      = {"parameter": "wrist_offset", "true_value": 0.085, "faulty_value": 0.092, "error_magnitude": 0.007}


def build_xml(params):
    return ROBOT_XML_TEMPLATE.format(**params)

def make_model(params):
    xml = build_xml(params)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(xml)
        tmp_path = f.name
    model = mujoco.MjModel.from_xml_path(tmp_path)
    os.unlink(tmp_path)
    return model, mujoco.MjData(model)

def sinusoidal_control(t):
    return np.array([0.4 * np.sin(2.0 * t), 0.3 * np.sin(1.5 * t + 0.5)])

def run_dual_simulation(duration=3.0, log_hz=100.0):
    model_gt, data_gt = make_model(GROUND_TRUTH_PARAMS)
    model_fx, data_fx = make_model(FAULTY_PARAMS)
    dt = model_gt.opt.timestep
    log_every = max(1, int(1.0 / (log_hz * dt)))
    n_steps = int(duration / dt)
    times, log_gt, log_fx, ee_gt, ee_fx = [], [], [], [], []

    print(f"Running dual simulation: {duration}s")
    print(f"  Sim A (ground truth): wrist_offset = {GROUND_TRUTH_PARAMS['wrist_offset']:.3f}m")
    print(f"  Sim B (faulty):       wrist_offset = {FAULTY_PARAMS['wrist_offset']:.3f}m  <- injected error")

    for step in range(n_steps):
        t = step * dt
        ctrl = sinusoidal_control(t)
        data_gt.ctrl[:] = ctrl
        data_fx.ctrl[:] = ctrl
        mujoco.mj_step(model_gt, data_gt)
        mujoco.mj_step(model_fx, data_fx)
        if step % log_every == 0:
            times.append(t)
            log_gt.append(data_gt.qpos[:2].copy())
            log_fx.append(data_fx.qpos[:2].copy())
            ee_id_gt = mujoco.mj_name2id(model_gt, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
            ee_id_fx = mujoco.mj_name2id(model_fx, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
            ee_gt.append(data_gt.site_xpos[ee_id_gt].copy())
            ee_fx.append(data_fx.site_xpos[ee_id_fx].copy())

    return {
        "times": np.array(times),
        "ground_truth": {"joint_states": np.array(log_gt), "ee_positions": np.array(ee_gt), "params": GROUND_TRUTH_PARAMS},
        "faulty_model":  {"joint_states": np.array(log_fx), "ee_positions": np.array(ee_fx), "params": FAULTY_PARAMS},
        "injected_error": INJECTED_ERROR,
    }

if __name__ == "__main__":
    traj = run_dual_simulation(duration=3.0)
    ee_gt = traj["ground_truth"]["ee_positions"]
    ee_fx = traj["faulty_model"]["ee_positions"]
    lateral_error = np.abs(ee_gt[:, 1] - ee_fx[:, 1])
    print(f"\nLateral (Y) EE error — mean: {lateral_error.mean()*1000:.2f} mm  max: {lateral_error.max()*1000:.2f} mm")
    print("Divergence confirmed." if lateral_error.mean() > 0.001 else "WARNING: No divergence detected")
    np.save("/tmp/trajectories_p2.npy", traj, allow_pickle=True)
    print("Trajectories saved to /tmp/trajectories_p2.npy")
    print("Phase 1 complete.")
