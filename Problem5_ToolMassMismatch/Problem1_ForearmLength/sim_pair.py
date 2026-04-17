"""
Phase 1: Dual Simulation
Two MuJoCo simulations running in parallel:
  - Sim A: Ground truth (correct parameters)
  - Sim B: Faulty model (injected link length error)
Both log joint state trajectories for divergence detection.
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
    <body name="link1" pos="0 0 0.1">
      <joint name="joint1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
      <geom type="capsule" size="0.04" fromto="0 0 0 0 0 {link1_length}" rgba="0.2 0.6 0.9 1"/>
      <body name="link2" pos="0 0 {link1_length}">
        <joint name="joint2" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
        <geom type="capsule" size="0.03" fromto="0 0 0 {link2_length} 0 0" rgba="0.9 0.4 0.2 1"/>
        <body name="end_effector" pos="{link2_length} 0 0">
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

GROUND_TRUTH_PARAMS = {"link1_length": 0.30, "link2_length": 0.25}
FAULTY_PARAMS       = {"link1_length": 0.30, "link2_length": 0.22}
INJECTED_ERROR      = {"parameter": "link2_length", "true_value": 0.25, "faulty_value": 0.22, "error_magnitude": 0.03}


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
    print(f"  Sim A (ground truth): link2_length = {GROUND_TRUTH_PARAMS['link2_length']:.3f}m")
    print(f"  Sim B (faulty):       link2_length = {FAULTY_PARAMS['link2_length']:.3f}m  <- injected error")

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
    gt_js = traj["ground_truth"]["joint_states"]
    fx_js = traj["faulty_model"]["joint_states"]
    rmse = np.sqrt(np.mean((gt_js - fx_js) ** 2))
    print(f"\nJoint state RMSE between sims: {rmse:.6f} rad")
    print("Divergence confirmed." if rmse > 0 else "WARNING: No divergence detected")
    np.save("/tmp/trajectories.npy", traj, allow_pickle=True)
    print("Trajectories saved to /tmp/trajectories.npy")
    print("Phase 1 complete.")
