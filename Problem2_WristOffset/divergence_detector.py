"""
Problem 2: Wrist Offset Fault — Divergence Detector (Phase 2)

Joint RMSE is near zero — fault is invisible in joint space.
Detection operates in Cartesian EE lateral position only.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

LATERAL_THRESHOLD = 0.002
WINDOW_SIZE = 20

def compute_sliding_lateral_error(ee_gt, ee_fx, window):
    n = len(ee_gt)
    err = np.zeros(n)
    for i in range(n):
        start = max(0, i - window + 1)
        dy = ee_gt[start:i+1, 1] - ee_fx[start:i+1, 1]
        err[i] = np.sqrt(np.mean(dy ** 2))
    return err

def detect_divergence(trajectories):
    times  = trajectories["times"]
    gt_js  = trajectories["ground_truth"]["joint_states"]
    fx_js  = trajectories["faulty_model"]["joint_states"]
    gt_ee  = trajectories["ground_truth"]["ee_positions"]
    fx_ee  = trajectories["faulty_model"]["ee_positions"]
    joint_rmse      = float(np.sqrt(np.mean((gt_js - fx_js) ** 2)))
    lateral_error   = np.abs(gt_ee[:, 1] - fx_ee[:, 1])
    lateral_sliding = compute_sliding_lateral_error(gt_ee, fx_ee, WINDOW_SIZE)
    ee_3d_error     = np.linalg.norm(gt_ee - fx_ee, axis=1)
    detection_mask  = lateral_sliding > LATERAL_THRESHOLD
    detected        = bool(np.any(detection_mask))
    first_detection_time = float(times[np.argmax(detection_mask)]) if detected else None
    return {
        "detected":              detected,
        "first_detection_time":  first_detection_time,
        "peak_lateral_error_mm": float(np.max(lateral_error)) * 1000,
        "mean_lateral_error_mm": float(np.mean(lateral_error)) * 1000,
        "joint_rmse_rad":        joint_rmse,
        "threshold_m":           LATERAL_THRESHOLD,
        "times":                 times,
        "lateral_sliding":       lateral_sliding,
        "lateral_error":         lateral_error,
        "ee_3d_error":           ee_3d_error,
        "gt_joint_states":       gt_js,
        "fx_joint_states":       fx_js,
        "gt_ee":                 gt_ee,
        "fx_ee":                 fx_ee,
        "injected_error":        trajectories["injected_error"],
        "estimated_offset_mm":   trajectories.get("estimated_wrist_offset", 0) * 1000,
        "correction_delta_y_mm": trajectories.get("correction_delta_y", 0) * 1000,
    }

def plot_divergence(report, save_path="/tmp/divergence_detection_p2.png"):
    times = report["times"]
    gt_js = report["gt_joint_states"]; fx_js = report["fx_joint_states"]
    gt_ee = report["gt_ee"];           fx_ee = report["fx_ee"]
    error = report["injected_error"]
    fig = plt.figure(figsize=(14, 10), facecolor="#0a0f14")
    gs  = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)
    style = {"axes.facecolor":"#0d1520","axes.edgecolor":"#1a2a3a","axes.labelcolor":"#7a9ab0",
             "axes.titlecolor":"#e0e8f0","xtick.color":"#3a5060","ytick.color":"#3a5060",
             "grid.color":"#1a2a3a","grid.linestyle":"--","grid.alpha":0.5}
    with plt.rc_context(style):
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(times, gt_js[:, 0], color="#00c896", lw=1.8, label="Sim A — GT")
        ax1.plot(times, fx_js[:, 0], color="#ff5050", lw=1.8, linestyle="--", label="Sim B — Faulty")
        ax1.set_title("Joint 1 — Near-Identical (Fault Invisible Here)", fontsize=11, pad=10)
        ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Position (rad)")
        ax1.legend(fontsize=8, facecolor="#0d1520", edgecolor="#1a2a3a", labelcolor="#7a9ab0"); ax1.grid(True)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(times, gt_ee[:, 1], color="#00c896", lw=1.8, label="Sim A — GT (Y)")
        ax2.plot(times, fx_ee[:, 1], color="#ff5050", lw=1.8, linestyle="--", label="Sim B — Faulty (Y)")
        ax2.set_title("EE Lateral Position Y — Fault Visible Here", fontsize=11, pad=10)
        ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Y Position (m)")
        ax2.legend(fontsize=8, facecolor="#0d1520", edgecolor="#1a2a3a", labelcolor="#7a9ab0"); ax2.grid(True)
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(times, report["lateral_sliding"]*1000, color="#ffb400", lw=2.0, label="Lateral EE RMSE (mm)")
        ax3.axhline(LATERAL_THRESHOLD*1000, color="#ff5050", lw=1.5, linestyle=":", label=f"Threshold={LATERAL_THRESHOLD*1000:.1f}mm")
        if report["detected"]:
            ax3.axvline(report["first_detection_time"], color="#ff5050", lw=1.2, alpha=0.6, linestyle="--")
            ax3.text(report["first_detection_time"]+0.05, LATERAL_THRESHOLD*1000*1.15,
                     f"Detected @ {report['first_detection_time']:.2f}s", color="#ff5050", fontsize=8)
        ax3.set_title("Lateral EE Divergence Detection", fontsize=11, pad=10)
        ax3.set_xlabel("Time (s)"); ax3.set_ylabel("Lateral RMSE (mm)")
        ax3.legend(fontsize=8, facecolor="#0d1520", edgecolor="#1a2a3a", labelcolor="#7a9ab0"); ax3.grid(True)
        ax4 = fig.add_subplot(gs[1, 1]); ax4.axis("off")
        rows=[("Fault type","Geometric — wrist lateral offset"),
              ("Joint RMSE",f"{report['joint_rmse_rad']:.6f} rad  (≈ 0)"),
              ("Peak lateral drift",f"{report['peak_lateral_error_mm']:.2f} mm"),
              ("Mean lateral drift",f"{report['mean_lateral_error_mm']:.2f} mm"),
              ("Detected",f"{'YES' if report['detected'] else 'NO'} @ {report['first_detection_time']:.2f}s"),
              ("Estimated offset",f"{report['estimated_offset_mm']:.2f} mm"),
              ("Correction Δy",f"{report['correction_delta_y_mm']:+.2f} mm"),
              ("True offset",f"{error['error_magnitude']*1000:.2f} mm")]
        y=0.95
        for label,value in rows:
            col="#00c896" if label in ("Detected","Correction Δy") else "#7a9ab0"
            ax4.text(0.05,y,label+":",fontsize=10,color="#4a6a7a",transform=ax4.transAxes,va="top")
            ax4.text(0.52,y,value,fontsize=10,color=col,transform=ax4.transAxes,va="top",fontweight="bold")
            y-=0.11
    fig.suptitle(f"Problem 2 — Wrist Offset Divergence  ·  Injected: {error['parameter']} = {error['faulty_value']}m",
                 color="#e0e8f0",fontsize=13,fontweight="bold",y=0.98)
    plt.savefig(save_path,dpi=150,bbox_inches="tight",facecolor=fig.get_facecolor()); plt.close()
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    traj = np.load("/tmp/trajectories_p2.npy", allow_pickle=True).item()
    report = detect_divergence(traj)
    print("\n── Problem 2 Divergence Detection Report ──")
    print(f"  Detected:              {report['detected']}")
    print(f"  First detection time:  {report['first_detection_time']:.3f}s")
    print(f"  Joint RMSE:            {report['joint_rmse_rad']:.6f} rad  (near zero — geometric fault)")
    print(f"  Peak lateral drift:    {report['peak_lateral_error_mm']:.2f} mm")
    print(f"  Mean lateral drift:    {report['mean_lateral_error_mm']:.2f} mm")
    print(f"  Estimated offset:      {report['estimated_offset_mm']:.2f} mm")
    print(f"  Correction Δy:         {report['correction_delta_y_mm']:+.2f} mm")
    print("────────────────────────────────────────────")
    plot_divergence(report)
    print("Phase 2 complete.")
