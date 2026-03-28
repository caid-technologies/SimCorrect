"""Phase 2: Divergence Detector"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

RMSE_THRESHOLD = 0.002
WINDOW_SIZE    = 20

def compute_sliding_rmse(gt, fx, window):
    n = len(gt)
    rmse = np.zeros(n)
    for i in range(n):
        start = max(0, i - window + 1)
        diff = gt[start:i+1] - fx[start:i+1]
        rmse[i] = np.sqrt(np.mean(diff ** 2))
    return rmse

def detect_divergence(trajectories):
    times  = trajectories["times"]
    gt_js  = trajectories["ground_truth"]["joint_states"]
    fx_js  = trajectories["faulty_model"]["joint_states"]
    gt_ee  = trajectories["ground_truth"]["ee_positions"]
    fx_ee  = trajectories["faulty_model"]["ee_positions"]
    rmse_j1    = compute_sliding_rmse(gt_js[:, 0:1], fx_js[:, 0:1], WINDOW_SIZE)
    rmse_j2    = compute_sliding_rmse(gt_js[:, 1:2], fx_js[:, 1:2], WINDOW_SIZE)
    rmse_total = compute_sliding_rmse(gt_js, fx_js, WINDOW_SIZE)
    ee_error   = np.linalg.norm(gt_ee - fx_ee, axis=1)
    detection_mask = rmse_total > RMSE_THRESHOLD
    detected = bool(np.any(detection_mask))
    first_detection_time = float(times[np.argmax(detection_mask)]) if detected else None
    return {
        "detected": detected,
        "first_detection_time": first_detection_time,
        "peak_rmse": float(np.max(rmse_total)),
        "threshold": RMSE_THRESHOLD,
        "times": times,
        "rmse_total": rmse_total,
        "rmse_joint1": rmse_j1,
        "rmse_joint2": rmse_j2,
        "ee_error": ee_error,
        "gt_joint_states": gt_js,
        "fx_joint_states": fx_js,
        "gt_ee": gt_ee,
        "fx_ee": fx_ee,
        "injected_error": trajectories["injected_error"],
    }

def plot_divergence(report, save_path="/tmp/divergence_detection.png"):
    times = report["times"]
    gt_js = report["gt_joint_states"]
    fx_js = report["fx_joint_states"]
    error = report["injected_error"]
    fig = plt.figure(figsize=(14, 10), facecolor="#0a0f14")
    gs  = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)
    style = {"axes.facecolor":"#0d1520","axes.edgecolor":"#1a2a3a","axes.labelcolor":"#7a9ab0","axes.titlecolor":"#e0e8f0","xtick.color":"#3a5060","ytick.color":"#3a5060","grid.color":"#1a2a3a","grid.linestyle":"--","grid.alpha":0.5}
    with plt.rc_context(style):
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(times, gt_js[:, 0], color="#00c896", lw=1.8, label="Sim A — Ground Truth")
        ax1.plot(times, fx_js[:, 0], color="#ff5050", lw=1.8, linestyle="--", label="Sim B — Faulty")
        ax1.set_title("Joint 1 — Angular Position", fontsize=11, pad=10)
        ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Position (rad)")
        ax1.legend(fontsize=8, facecolor="#0d1520", edgecolor="#1a2a3a", labelcolor="#7a9ab0")
        ax1.grid(True)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(times, gt_js[:, 1], color="#00c896", lw=1.8, label="Sim A — Ground Truth")
        ax2.plot(times, fx_js[:, 1], color="#ff5050", lw=1.8, linestyle="--", label="Sim B — Faulty")
        ax2.set_title("Joint 2 — Angular Position", fontsize=11, pad=10)
        ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Position (rad)")
        ax2.legend(fontsize=8, facecolor="#0d1520", edgecolor="#1a2a3a", labelcolor="#7a9ab0")
        ax2.grid(True)
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(times, report["rmse_total"],  color="#ffb400", lw=2.0, label="Total RMSE")
        ax3.plot(times, report["rmse_joint1"], color="#6496ff", lw=1.2, alpha=0.7, label="Joint 1 RMSE")
        ax3.plot(times, report["rmse_joint2"], color="#c864ff", lw=1.2, alpha=0.7, label="Joint 2 RMSE")
        ax3.axhline(RMSE_THRESHOLD, color="#ff5050", lw=1.5, linestyle=":", label=f"Threshold={RMSE_THRESHOLD}")
        if report["detected"]:
            ax3.axvline(report["first_detection_time"], color="#ff5050", lw=1.2, alpha=0.6, linestyle="--")
            ax3.text(report["first_detection_time"]+0.05, RMSE_THRESHOLD*1.1, f"Detected @ {report['first_detection_time']:.2f}s", color="#ff5050", fontsize=8)
        ax3.set_title("Sliding Window RMSE — Divergence Detection", fontsize=11, pad=10)
        ax3.set_xlabel("Time (s)"); ax3.set_ylabel("RMSE (rad)")
        ax3.legend(fontsize=8, facecolor="#0d1520", edgecolor="#1a2a3a", labelcolor="#7a9ab0")
        ax3.grid(True)
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.fill_between(times, report["ee_error"], alpha=0.3, color="#00c896")
        ax4.plot(times, report["ee_error"], color="#00c896", lw=1.8)
        ax4.set_title("End-Effector Position Error", fontsize=11, pad=10)
        ax4.set_xlabel("Time (s)"); ax4.set_ylabel("||pos_A - pos_B|| (m)")
        ax4.grid(True)
    fig.suptitle(f"Divergence Detection  ·  Injected Error: {error['parameter']} = {error['faulty_value']}m  (true: {error['true_value']}m)", color="#e0e8f0", fontsize=13, fontweight="bold", y=0.98)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    traj = np.load("/tmp/trajectories.npy", allow_pickle=True).item()
    report = detect_divergence(traj)
    print("\n── Divergence Detection Report ──")
    print(f"  Detected:             {report['detected']}")
    print(f"  First detection time: {report['first_detection_time']:.3f}s")
    print(f"  Peak RMSE:            {report['peak_rmse']:.6f} rad")
    print(f"  Injected error:       {report['injected_error']}")
    print("─────────────────────────────────")
    plot_divergence(report)
    print("Phase 2 complete.")
