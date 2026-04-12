"""
Problem 3 — Joint Friction Fault
divergence_detector.py

Detection signal: sliding-window joint-velocity RMSE.

Friction faults manifest dynamically — the faulty arm lags
behind the reference from the first frame of motion. Position
error is slow to accumulate; velocity error is immediate.

Algorithm:
  1. Per-step velocity proxy: finite difference of joint positions
  2. Compute per-step RMSE between nominal and faulty velocities
  3. Sliding window mean over WINDOW_S seconds
  4. Alarm when window mean crosses THRESHOLD
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from sim_pair import SimPairResult, run_sim_pair, DT

WINDOW_S  = 0.10
THRESHOLD = 0.08
MOTION_T  = 6.05


@dataclass
class DetectionResult:
    detected:        bool
    detection_time:  Optional[float]
    latency:         Optional[float]
    vel_rmse:        np.ndarray
    window_mean:     np.ndarray
    threshold:       float
    alarm_index:     Optional[int]


def detect(result: SimPairResult) -> DetectionResult:
    records = result.records
    n       = len(records)
    W       = max(1, int(WINDOW_S / DT))

    q_nom = np.array([r.q_nom for r in records])
    q_flt = np.array([r.q_flt for r in records])
    vnom  = np.diff(q_nom, axis=0) / DT
    vflt  = np.diff(q_flt, axis=0) / DT
    vel_rmse = np.sqrt(np.mean((vnom - vflt)**2, axis=1))
    vel_rmse = np.append(vel_rmse, vel_rmse[-1])

    window_mean = np.zeros(n)
    for i in range(n):
        lo = max(0, i - W + 1)
        window_mean[i] = vel_rmse[lo:i+1].mean()

    motion_idx = int(MOTION_T / DT)
    alarm_idx  = None
    for i in range(motion_idx, n):
        if window_mean[i] > THRESHOLD:
            alarm_idx = i
            break

    detected = alarm_idx is not None
    t_alarm  = records[alarm_idx].time if detected else None
    latency  = (t_alarm - MOTION_T)   if detected else None

    return DetectionResult(
        detected       = detected,
        detection_time = t_alarm,
        latency        = latency,
        vel_rmse       = vel_rmse,
        window_mean    = window_mean,
        threshold      = THRESHOLD,
        alarm_index    = alarm_idx,
    )


if __name__ == "__main__":
    print("Running sim pair ...")
    sp  = run_sim_pair()
    print("Running detector ...")
    det = detect(sp)
    print(f"  Detected         : {det.detected}")
    if det.detected:
        print(f"  Detection time   : {det.detection_time:.3f} s")
        print(f"  Latency          : {det.latency:.3f} s")
    print(f"  Peak vel RMSE    : {det.vel_rmse.max():.4f} rad/s")
    print(f"  Threshold        : {det.threshold} rad/s")
