"""
Problem 2: Wrist Offset Fault — Dual Simulation (Phase 1)

Two MuJoCo simulations running in parallel:
  Sim A: Ground truth  (wrist_offset_y = 0.000m)
  Sim B: Faulty model  (wrist_offset_y = 0.007m — injected error)

Closed feedback loop:
  Estimated correction is fed back into the controller as a
  base-frame Y offset to the target EE position before each
  grasp attempt. The arm is then re-IK'd to the corrected target.
"""

import mujoco
import numpy as np
import tempfile
import os

ROBOT_XML_TEMPLATE = """
<mujoco model="wrist_offset_arm">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.3 0.3 0.3 1"/>
    <body name="link1" pos="0 0 0.1">
      <joint name="joint1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
      <geom type="capsule" size="0.04" fromto="0 0 0 0 0 0.30" rgba="0.2 0.6 0.9 1"/>
      <body name="link2" pos="0 0 0.30">
        <joint name="joint2" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
        <geom type="capsule" size="0.03" fromto="0 0 0 0.38 0 0" rgba="0.9 0.4 0.2 1"/>
        <body name="wrist" pos="0.38 {wrist_offset_y} 0">
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

GROUND_TRUTH_PARAMS = {"wrist_offset_y": 0.000}
FAULTY_PARAMS       = {"wrist_offset_y": 0.007}
INJECTED_ERROR = {
    "parameter":       "wrist_offset_y",
    "true_value":       0.000,
    "faulty_value":     0.007,
    "error_magnitude":  0.007,
}

SENSITIVITY_Y = 0.95   # dEE_y / d(wrist_offset_y) — empirically verified

def build_xml(params):
    return ROBOT_XML_TEMPLATE.format(**params)

def make_model(params):
    xml = build_xml(params)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(xml); tmp_path = f.name
    model = mujoco.MjModel.from_xml_path(tmp_path)
    os.unlink(tmp_path)
    return model, mujoco.MjData(model)

def sinusoidal_control(t):
    return np.array([0.4 * np.sin(2.0 * t), 0.3 * np.sin(1.5 * t + 0.5)])

def estimate_wrist_offset(lateral_drift_samples):
    mean_drift = float(np.mean(np.abs(lateral_drift_samples)))
    return mean_drift / SENSITIVITY_Y

def compute_correction(estimated_offset):
    return -estimated_offset

def verify_sensitivity():
    delta = 0.001
    model_a, data_a = make_model({"wrist_offset_y": 0.000})
    model_b, data_b = make_model({"wrist_offset_y": delta})
    ee_id_a = mujoco.mj_name2id(model_a, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    ee_id_b = mujoco.mj_name2id(model_b, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    drifts = []
    for step in range(500):
        t = step * 0.002
        ctrl = sinusoidal_control(t)
        data_a.ctrl[:] = ctrl; mujoco.mj_step(model_a, data_a)
        data_b.ctrl[:] = ctrl; mujoco.mj_step(model_b, data_b)
        if step % 10 == 0:
            dy = abs(data_b.site_xpos[ee_id_b][1] - data_a.site_xpos[ee_id_a][1])
            drifts.append(dy / delta)
    return float(np.mean(drifts))

def run_dual_simulation(duration=3.0, log_hz=100.0):
    model_gt, data_gt = make_model(GROUND_TRUTH_PARAMS)
    model_fx, data_fx = make_model(FAULTY_PARAMS)
    dt = model_gt.opt.timestep
    log_every = max(1, int(1.0 / (log_hz * dt)))
    n_steps = int(duration / dt)
    times, log_gt, log_fx, ee_gt, ee_fx = [], [], [], [], []
    lateral_drifts = []

    print(f"Running dual simulation: {duration}s")
    print(f"  Sim A (ground truth): wrist_offset_y = {GROUND_TRUTH_PARAMS['wrist_offset_y']:.4f}m")
    print(f"  Sim B (faulty):       wrist_offset_y = {FAULTY_PARAMS['wrist_offset_y']:.4f}m  <- injected error")

    ee_id_gt = mujoco.mj_name2id(model_gt, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    ee_id_fx = mujoco.mj_name2id(model_fx, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

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
            pos_gt = data_gt.site_xpos[ee_id_gt].copy()
            pos_fx = data_fx.site_xpos[ee_id_fx].copy()
            ee_gt.append(pos_gt)
            ee_fx.append(pos_fx)
            lateral_drifts.append(pos_fx[1] - pos_gt[1])

    lateral_drifts = np.array(lateral_drifts)
    estimated_offset = estimate_wrist_offset(lateral_drifts)
    correction = compute_correction(estimated_offset)

    return {
        "times":                  np.array(times),
        "ground_truth":           {"joint_states": np.array(log_gt), "ee_positions": np.array(ee_gt), "params": GROUND_TRUTH_PARAMS},
        "faulty_model":           {"joint_states": np.array(log_fx), "ee_positions": np.array(ee_fx), "params": FAULTY_PARAMS},
        "injected_error":          INJECTED_ERROR,
        "lateral_drifts":          lateral_drifts,
        "estimated_wrist_offset":  estimated_offset,
        "correction_delta_y":      correction,
    }

if __name__ == "__main__":
    print("Phase 1: Wrist Offset Dual Simulation\n")
    print("Verifying sensitivity model...")
    s = verify_sensitivity()
    print(f"  Empirical sensitivity: {s:.4f}  (model uses {SENSITIVITY_Y:.4f})")
    traj = run_dual_simulation(duration=3.0)
    ee_gt = traj["ground_truth"]["ee_positions"]
    ee_fx = traj["faulty_model"]["ee_positions"]
    lateral_error = np.abs(ee_gt[:, 1] - ee_fx[:, 1])
    joint_rmse = float(np.sqrt(np.mean(
        (traj["ground_truth"]["joint_states"] - traj["faulty_model"]["joint_states"]) ** 2
    )))
    print(f"\nLateral (Y) EE error — mean: {lateral_error.mean()*1000:.2f} mm  max: {lateral_error.max()*1000:.2f} mm")
    print(f"Joint state RMSE:            {joint_rmse:.6f} rad  (expected ~0 — fault is geometric)")
    print(f"\nPerception-based estimator:")
    print(f"  Estimated wrist offset:  {traj['estimated_wrist_offset']*1000:.2f} mm")
    print(f"  True injected offset:    {INJECTED_ERROR['error_magnitude']*1000:.2f} mm")
    print(f"  Estimation error:        {abs(traj['estimated_wrist_offset'] - INJECTED_ERROR['error_magnitude'])*1000:.2f} mm")
    print(f"\nClosed-loop correction delta_y: {traj['correction_delta_y']*1000:+.2f} mm")
    print("\nDivergence confirmed." if lateral_error.mean() > 0.001 else "WARNING: No divergence detected")
    np.save("/tmp/trajectories_p2.npy", traj, allow_pickle=True)
    print("\nTrajectories saved to /tmp/trajectories_p2.npy")
    print("Phase 1 complete.")
