"""Phase 3: Parameter Identification via Sensitivity Analysis"""

import mujoco
import numpy as np
import tempfile
import os
import json

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

CANDIDATE_PARAMETERS = {"link1_length": 0.30, "link2_length": 0.22}
PERTURBATION_FRACTION = 0.05

def sinusoidal_control(t):
    return np.array([0.4 * np.sin(2.0 * t), 0.3 * np.sin(1.5 * t + 0.5)])

def make_model(params):
    xml = ROBOT_XML_TEMPLATE.format(**params)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(xml); tmp_path = f.name
    model = mujoco.MjModel.from_xml_path(tmp_path)
    os.unlink(tmp_path)
    return model, mujoco.MjData(model)

def run_simulation(params, duration=3.0, log_hz=100.0):
    model, data = make_model(params)
    dt = model.opt.timestep
    log_every = max(1, int(1.0 / (log_hz * dt)))
    n_steps = int(duration / dt)
    log = []
    for step in range(n_steps):
        t = step * dt
        data.ctrl[:] = sinusoidal_control(t)
        mujoco.mj_step(model, data)
        if step % log_every == 0:
            log.append(data.qpos[:2].copy())
    return np.array(log)

def compute_sensitivity(base_params, param_name):
    base_traj = run_simulation(base_params)
    perturbed_params = base_params.copy()
    delta = base_params[param_name] * PERTURBATION_FRACTION
    perturbed_params[param_name] += delta
    perturbed_traj = run_simulation(perturbed_params)
    traj_diff = np.abs(perturbed_traj - base_traj)
    sensitivity_per_joint = np.mean(traj_diff, axis=0) / delta
    return sensitivity_per_joint, delta

def identify_parameter(trajectories):
    gt_js = trajectories["ground_truth"]["joint_states"]
    fx_js = trajectories["faulty_model"]["joint_states"]
    observed_divergence = np.mean(np.abs(gt_js - fx_js), axis=0)
    print(f"\nObserved divergence per joint: {observed_divergence}")
    obs_norm = observed_divergence / (np.linalg.norm(observed_divergence) + 1e-10)
    sensitivities = {}
    scores = {}
    print("\nComputing parameter sensitivities...")
    for param_name in CANDIDATE_PARAMETERS:
        sens_vec, delta = compute_sensitivity(CANDIDATE_PARAMETERS, param_name)
        sensitivities[param_name] = sens_vec
        sens_norm = sens_vec / (np.linalg.norm(sens_vec) + 1e-10)
        cosine_sim = float(np.dot(obs_norm, sens_norm))
        scores[param_name] = cosine_sim
        print(f"  {param_name:20s}: sensitivity={sens_vec}, cosine_sim={cosine_sim:.4f}")
    identified_param = max(scores, key=scores.get)
    sens = sensitivities[identified_param]
    estimated_delta = float(np.dot(sens, observed_divergence) / (np.dot(sens, sens) + 1e-10))
    current_value = CANDIDATE_PARAMETERS[identified_param]
    proposed_value = current_value + estimated_delta
    return {
        "identified_parameter": identified_param,
        "confidence": round(scores[identified_param], 4),
        "current_value": round(current_value, 5),
        "estimated_delta": round(estimated_delta, 5),
        "proposed_value": round(proposed_value, 5),
        "all_scores": {k: round(v, 4) for k, v in scores.items()},
        "method": "sensitivity_analysis_cosine_similarity",
    }

if __name__ == "__main__":
    print("Phase 3: Parameter Identification")
    traj = np.load("/tmp/trajectories.npy", allow_pickle=True).item()
    result = identify_parameter(traj)
    print("\n── Identification Report ─────────────────────────────")
    print(f"  Identified parameter: {result['identified_parameter']}")
    print(f"  Confidence (cosine):  {result['confidence']}")
    print(f"  Current value:        {result['current_value']}m")
    print(f"  Estimated delta:      {result['estimated_delta']:+.5f}m")
    print(f"  Proposed correction:  {result['proposed_value']}m")
    print(f"  True value:           {traj['injected_error']['true_value']}m")
    print(f"  All scores:           {result['all_scores']}")
    print("──────────────────────────────────────────────────────")
    with open("/tmp/identification_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nSaved to /tmp/identification_result.json")
    print("Phase 3 complete.")
