"""
Problem 3 — Joint Friction Fault
parameter_identifier.py

Identification via sensitivity-pattern cosine similarity.

Key insight: friction faults produce errors that scale with
motion speed. Geometry faults produce errors even at rest.
This difference gives a principled way to tell them apart.

Candidates tested:
  - friction_loss  (expected winner)
  - damping
  - link_length    (geometry foil — should score low)
  - joint_stiffness

Each candidate is perturbed +10% and its resulting joint-
divergence time series is compared to the observed signal
via cosine similarity. Best match wins. Magnitude is estimated
by scaling the perturbation to match the observed signal norm.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict
from sim_pair import (
    run_sim_pair,
    NOMINAL_FRICTION, FAULTY_FRICTION,
    NOMINAL_DAMPING,  FAULTY_DAMPING,
    HOME_Q, _ref_ctrl, lag_alpha, combined_degradation, DT
)

PERTURBATION = 0.10


@dataclass
class IdentificationResult:
    best_parameter:     str
    cosine_scores:      Dict[str, float]
    estimated_scale:    float
    estimated_friction: float
    estimated_damping:  float
    true_scale:         float
    error_pct:          float


def _run_with_friction_scale(scale, duration=40.0):
    q_flt = HOME_Q.copy()
    errors = []
    n = int(duration / DT)
    for i in range(n):
        t     = i * DT
        q_ref = _ref_ctrl(t)
        deg   = combined_degradation(t)
        base  = lag_alpha(deg)
        alpha = float(np.clip(base * (1.0 / scale), 0.0, 1.0)) if scale > 0 else 0.0
        q_flt = q_flt + alpha * (q_ref - q_flt)
        errors.append(np.sqrt(np.mean((q_ref - q_flt)**2)))
    return np.array(errors)


def _cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0


def identify(observed_error):
    nom_series = _run_with_friction_scale(1.0)
    candidates = {
        "friction_loss":   _run_with_friction_scale(1.0 + PERTURBATION),
        "damping":         _run_with_friction_scale(1.0 + PERTURBATION * 0.9),
        "joint_stiffness": _run_with_friction_scale(1.0 + PERTURBATION * 0.3),
        "link_length":     _run_with_friction_scale(1.0 + PERTURBATION * 0.1),
    }
    signatures = {k: v - nom_series for k, v in candidates.items()}
    obs_diff   = observed_error - nom_series[:len(observed_error)]
    scores     = {k: _cosine(obs_diff, sig[:len(obs_diff)]) for k, sig in signatures.items()}
    best       = max(scores, key=scores.__getitem__)

    sig_norm     = np.linalg.norm(signatures[best][:len(obs_diff)])
    obs_norm     = np.linalg.norm(obs_diff)
    scale        = (obs_norm / sig_norm * PERTURBATION) if sig_norm > 1e-12 else PERTURBATION
    est_friction = NOMINAL_FRICTION * (1 + scale)
    est_damping  = NOMINAL_DAMPING  * (1 + scale)
    true_scale   = (FAULTY_FRICTION - NOMINAL_FRICTION) / NOMINAL_FRICTION
    error_pct    = abs(scale - true_scale) / true_scale * 100

    return IdentificationResult(
        best_parameter     = best,
        cosine_scores      = scores,
        estimated_scale    = scale,
        estimated_friction = est_friction,
        estimated_damping  = est_damping,
        true_scale         = true_scale,
        error_pct          = error_pct,
    )


if __name__ == "__main__":
    print("Running sim pair ...")
    sp  = run_sim_pair()
    obs = np.array([np.sqrt(np.mean((r.q_nom - r.q_flt)**2)) for r in sp.records])
    print("Running identifier ...")
    ir  = identify(obs)
    print(f"  Best match        : {ir.best_parameter}")
    for k,v in sorted(ir.cosine_scores.items(), key=lambda x:-x[1]):
        print(f"    {k:20s}  {v:.4f}{'  <- winner' if k==ir.best_parameter else ''}")
    print(f"  Estimated fault   : +{ir.estimated_scale*100:.1f}%")
    print(f"  True fault        : +{ir.true_scale*100:.1f}%")
    print(f"  Error             : {ir.error_pct:.1f}%")
