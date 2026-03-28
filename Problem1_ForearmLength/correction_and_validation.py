"""
Phase 4 + 5: OpenCAD Correction and Validation
Loads identification result from Phase 3.
Uses OpenCAD to correct the parameter.
Reruns simulation and validates convergence.
Plots before vs after.
"""

import mujoco
import numpy as np
import tempfile
import os
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from opencad import Part, Sketch

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
    times, log, ee_log = [], [], []
    for step in range(n_steps):
        t = step * dt
        data.ctrl[:] = sinusoidal_control(t)
        mujoco.mj_step(model, data)
        if step % log_every == 0:
            times.append(t)
            log.append(data.qpos[:2].copy())
            ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
            ee_log.append(data.site_xpos[ee_id].copy())
    return np.array(times), np.array(log), np.array(ee_log)

def opencad_correction(identification_result, current_params):
    """
    Use OpenCAD to apply the parameter correction.
    Records the correction in the feature tree for audit trail.
    """
    param = identification_result["identified_parameter"]
    proposed_value = identification_result["proposed_value"]

    print(f"\nOpenCAD correction:")
    print(f"  Parameter: {param}")
    print(f"  From: {current_params[param]:.5f}m")
    print(f"  To:   {proposed_value:.5f}m")

    # Use OpenCAD feature tree to record the correction
    # This creates an auditable parametric history of the fix
    arm_profile = (
        Sketch(name="Arm Correction Record")
        .rect(proposed_value * 100, 4)   # scaled for CAD units (cm)
    )
    Part(name=f"Corrected_{param}").extrude(
        arm_profile,
        depth=4,
        name=f"Correction: {param} = {proposed_value:.4f}m"
    )

    print(f"  OpenCAD feature tree updated — correction logged.")

    # Apply correction to simulation parameters
    corrected_params = current_params.copy()
    corrected_params[param] = proposed_value
    return corrected_params

def compute_rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

def plot_before_after(traj_gt, traj_before, traj_after, times, identification_result,
                      save_path="/tmp/correction_validation.png"):
    fig = plt.figure(figsize=(14, 10), facecolor="#0a0f14")
    gs  = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)
    style = {"axes.facecolor":"#0d1520","axes.edgecolor":"#1a2a3a","axes.labelcolor":"#7a9ab0",
             "axes.titlecolor":"#e0e8f0","xtick.color":"#3a5060","ytick.color":"#3a5060",
             "grid.color":"#1a2a3a","grid.linestyle":"--","grid.alpha":0.5}

    with plt.rc_context(style):
        for i, (joint_idx, title) in enumerate([( 0, "Joint 1"), (1, "Joint 2")]):
            ax = fig.add_subplot(gs[0, i])
            ax.plot(times, traj_gt[:, joint_idx],     color="#00c896", lw=2.0, label="Ground Truth")
            ax.plot(times, traj_before[:, joint_idx], color="#ff5050", lw=1.5, linestyle="--", label="Before correction")
            ax.plot(times, traj_after[:, joint_idx],  color="#6496ff", lw=1.5, linestyle="-.", label="After correction")
            ax.set_title(f"{title} — Before vs After Correction", fontsize=11, pad=10)
            ax.set_xlabel("Time (s)"); ax.set_ylabel("Position (rad)")
            ax.legend(fontsize=8, facecolor="#0d1520", edgecolor="#1a2a3a", labelcolor="#7a9ab0")
            ax.grid(True)

        # RMSE comparison bar chart
        ax3 = fig.add_subplot(gs[1, 0])
        rmse_before = compute_rmse(traj_gt, traj_before)
        rmse_after  = compute_rmse(traj_gt, traj_after)
        bars = ax3.bar(["Before Correction", "After Correction"],
                       [rmse_before, rmse_after],
                       color=["#ff5050", "#00c896"], width=0.4, edgecolor="#1a2a3a")
        ax3.axhline(0.002, color="#ffb400", lw=1.5, linestyle=":", label="Detection threshold")
        for bar, val in zip(bars, [rmse_before, rmse_after]):
            ax3.text(bar.get_x() + bar.get_width()/2, val + 0.0001,
                     f"{val:.5f}", ha="center", va="bottom", fontsize=9, color="#e0e8f0")
        ax3.set_title("RMSE Comparison — Correction Effectiveness", fontsize=11, pad=10)
        ax3.set_ylabel("RMSE (rad)")
        ax3.legend(fontsize=8, facecolor="#0d1520", edgecolor="#1a2a3a", labelcolor="#7a9ab0")
        ax3.grid(True, axis="y")

        # Summary text panel
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis("off")
        summary = [
            ("Identified parameter",  identification_result["identified_parameter"]),
            ("Confidence",            f"{identification_result['confidence']}"),
            ("Value before",          f"{identification_result['current_value']}m"),
            ("Value after",           f"{identification_result['proposed_value']}m"),
            ("True value",            "0.25000m"),
            ("Error before",          f"{rmse_before:.6f} rad"),
            ("Error after",           f"{rmse_after:.6f} rad"),
            ("Reduction",             f"{(1 - rmse_after/rmse_before)*100:.1f}%"),
            ("Converged",             "YES" if rmse_after < 0.002 else "NO"),
        ]
        y = 0.95
        for label, value in summary:
            color = "#00c896" if label in ("Converged", "Reduction") else "#7a9ab0"
            ax4.text(0.05, y, label + ":", fontsize=10, color="#4a6a7a",
                     transform=ax4.transAxes, va="top")
            ax4.text(0.55, y, value, fontsize=10, color=color,
                     transform=ax4.transAxes, va="top", fontweight="bold")
            y -= 0.1

    fig.suptitle("Autonomous Sim-to-Real Gap Correction — Full Pipeline Result",
                 color="#e0e8f0", fontsize=13, fontweight="bold", y=0.98)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    print("Phase 4 + 5: OpenCAD Correction and Validation")

    # Load data
    traj = np.load("/tmp/trajectories.npy", allow_pickle=True).item()
    with open("/tmp/identification_result.json") as f:
        identification_result = json.load(f)

    faulty_params = {"link1_length": 0.30, "link2_length": 0.22}
    gt_params     = {"link1_length": 0.30, "link2_length": 0.25}

    # Phase 4 — OpenCAD correction
    corrected_params = opencad_correction(identification_result, faulty_params)
    print(f"\nCorrected params: {corrected_params}")

    # Phase 5 — Rerun simulation with corrected params
    print("\nRerunning simulation with corrected parameters...")
    times, traj_gt,     ee_gt     = run_simulation(gt_params)
    times, traj_before, ee_before = run_simulation(faulty_params)
    times, traj_after,  ee_after  = run_simulation(corrected_params)

    rmse_before = compute_rmse(traj_gt, traj_before)
    rmse_after  = compute_rmse(traj_gt, traj_after)
    converged   = rmse_after < 0.002

    print(f"\n── Validation Report ─────────────────────────────────")
    print(f"  RMSE before correction: {rmse_before:.6f} rad")
    print(f"  RMSE after correction:  {rmse_after:.6f} rad")
    print(f"  Reduction:              {(1 - rmse_after/rmse_before)*100:.1f}%")
    print(f"  Converged:              {converged}")
    print(f"──────────────────────────────────────────────────────")

    plot_before_after(traj_gt, traj_before, traj_after, times, identification_result)
    print("\nFull pipeline complete. Zero human intervention.")
