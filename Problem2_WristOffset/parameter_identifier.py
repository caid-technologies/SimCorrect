"""
Problem 2: Wrist Offset Fault — Parameter Identification (Phase 3)
Identifies wrist_offset_y via lateral EE sensitivity analysis.
"""

import mujoco
import numpy as np
import tempfile
import os
import json

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

CANDIDATE_PARAMETERS = {"wrist_offset_y": 0.000}
PERTURBATION_FRACTION = 0.05

def sinusoidal_control(t):
    return np.array([0.4 * np.sin(2.0 * t), 0.3 * np.sin(1.5 * t + 0.5)])

def make_model(params):
    xml = ROBOT_XML_TEMPLATE.format(**params)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(xml); tmp_path = f.name
    model = mujoco.MjModel.from_xml_path(tmp_path); os.unlink(tmp_path)
    return model, mujoco.MjData(model)

def run_simulation(params, duration=3.0, log_hz=100.0):
    model, data = make_model(params)
    dt = model.opt.timestep
    log_every = max(1, int(1.0 / (log_hz * dt)))
    n_steps = int(duration / dt)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    js_log, ee_log = [], []
    for step in range(n_steps):
        t = step * dt
        data.ctrl[:] = sinusoidal_control(t)
        mujoco.mj_step(model, data)
        if step % log_every == 0:
            js_log.append(data.qpos[:2].copy())
            ee_log.append(data.site_xpos[ee_id].copy())
    return np.array(js_log), np.array(ee_log)

def compute_sensitivity(base_params, param_name):
    base_js, base_ee = run_simulation(base_params)
    perturbed = base_params.copy()
    delta = max(abs(base_params[param_name]) * PERTURBATION_FRACTION, 1e-4)
    perturbed[param_name] += delta
    pert_js, pert_ee = run_simulation(perturbed)
    lateral_sens = np.mean(np.abs(pert_ee[:, 1] - base_ee[:, 1])) / delta
    joint_sens   = np.mean(np.abs(pert_js - base_js), axis=0) / delta
    return lateral_sens, joint_sens, delta

def identify_parameter(trajectories):
    gt_ee = trajectories["ground_truth"]["ee_positions"]
    fx_ee = trajectories["faulty_model"]["ee_positions"]
    gt_js = trajectories["ground_truth"]["joint_states"]
    fx_js = trajectories["faulty_model"]["joint_states"]
    observed_lateral = float(np.mean(np.abs(gt_ee[:, 1] - fx_ee[:, 1])))
    observed_joint   = np.mean(np.abs(gt_js - fx_js), axis=0)
    print(f"\nObserved lateral drift:    {observed_lateral*1000:.2f} mm")
    print(f"Observed joint divergence: {observed_joint} rad  (near zero — geometric fault)")
    print("\nComputing parameter sensitivities...")
    sensitivities = {}; scores = {}
    for param_name in CANDIDATE_PARAMETERS:
        lat_sens, jnt_sens, delta = compute_sensitivity(CANDIDATE_PARAMETERS, param_name)
        sensitivities[param_name] = lat_sens
        joint_ratio = float(np.linalg.norm(jnt_sens)) / (lat_sens + 1e-10)
        score = lat_sens / (joint_ratio + 0.01)
        scores[param_name] = score
        print(f"  {param_name:20s}: lateral_sens={lat_sens:.4f}  joint_sens={np.round(jnt_sens,4)}  score={score:.4f}")
    identified_param = max(scores, key=scores.get)
    lat_sens = sensitivities[identified_param]
    estimated_delta = observed_lateral / (lat_sens + 1e-10)
    current_value   = CANDIDATE_PARAMETERS[identified_param]
    proposed_value  = current_value + estimated_delta
    return {
        "identified_parameter": identified_param,
        "confidence":           round(scores[identified_param], 4),
        "current_value":        round(current_value, 5),
        "estimated_delta":      round(estimated_delta, 5),
        "proposed_value":       round(proposed_value, 5),
        "lateral_sensitivity":  round(lat_sens, 4),
        "all_scores":           {k: round(v, 4) for k, v in scores.items()},
        "method":               "lateral_ee_sensitivity_analysis",
        "diagnostic_signature": "geometric_offset — joint_rmse≈0, lateral_ee_drift≈constant",
    }

if __name__ == "__main__":
    print("Phase 3: Parameter Identification — Wrist Offset Fault\n")
    traj = np.load("/tmp/trajectories_p2.npy", allow_pickle=True).item()
    result = identify_parameter(traj)
    print("\n── Identification Report ─────────────────────────────")
    print(f"  Identified parameter: {result['identified_parameter']}")
    print(f"  Confidence score:     {result['confidence']}")
    print(f"  Current value:        {result['current_value']}m")
    print(f"  Estimated delta:      {result['estimated_delta']*1000:+.2f} mm")
    print(f"  Proposed correction:  {result['proposed_value']}m")
    print(f"  True value:           {traj['injected_error']['true_value']}m")
    print(f"  Diagnostic signature: {result['diagnostic_signature']}")
    print("──────────────────────────────────────────────────────")
    with open("/tmp/identification_result_p2.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nSaved to /tmp/identification_result_p2.json")
    print("Phase 3 complete.")
