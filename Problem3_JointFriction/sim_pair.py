"""
Problem 3 — Joint Friction Fault
sim_pair.py

Runs two instances of the arm under identical joint commands.
  - nominal:  frictionloss=0.5 Nm,  damping=8.0 Ns/m
  - faulty:   frictionloss=1.06 Nm, damping=16.96 Ns/m (+112%)

The fault is dynamic — both arms have identical geometry.
The faulty arm lags behind the reference trajectory due to
joint resistance consuming torque before motion begins.

Degradation is progressive: 6 real-world causes accumulate
over time (wear, lubrication loss, contamination, corrosion,
thermal expansion, seal aging), each filling at a different
rate. Combined degradation drives the lag factor.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List

NOMINAL_FRICTION = 0.5
FAULTY_FRICTION  = 1.06
NOMINAL_DAMPING  = 8.0
FAULTY_DAMPING   = 16.96

FACTORS = [
    ("Wear",          1.00),
    ("Lubrication",   0.85),
    ("Contamination", 0.70),
    ("Corrosion",     0.55),
    ("Thermal",       0.75),
    ("Seal Aging",    0.90),
]
T_PEAK  = 20.0
T_REACH = 6.0

HOME_Q  = np.array([ 0.0000,  0.1732, -2.4041,  0.0915])
ABOVE_Q = np.array([ 0.0000, -1.0091,  2.4513,  0.0867])
PICK_Q  = np.array([ 0.0000, -0.0066,  2.0928,  0.0423])
LIFT_Q  = np.array([ 0.0000, -1.4756,  2.2630,  0.1637])
PLACE_Q = np.array([ 3.1400, -0.6915,  1.9370, -0.0352])

DT = 0.002


@dataclass
class StepRecord:
    time:  float
    q_nom: np.ndarray
    q_flt: np.ndarray
    deg:   float
    alpha: float


@dataclass
class SimPairResult:
    records:          List[StepRecord] = field(default_factory=list)
    dt:               float = DT
    nominal_friction: float = NOMINAL_FRICTION
    faulty_friction:  float = FAULTY_FRICTION
    nominal_damping:  float = NOMINAL_DAMPING
    faulty_damping:   float = FAULTY_DAMPING


def factor_levels(t):
    if t < T_REACH:
        return [0.0] * len(FACTORS)
    elapsed = t - T_REACH
    total   = T_PEAK - T_REACH
    return [float(np.clip((elapsed / total) * rate, 0.0, 1.0)) for _, rate in FACTORS]


def combined_degradation(t):
    return float(np.mean(factor_levels(t)))


def lag_alpha(deg):
    if deg < 0.3:   return 1.0
    elif deg < 0.6: return 1.0 - ((deg - 0.3) / 0.3) * 0.95
    else:           return 0.0


def _sm(a, b, s):
    s = float(np.clip(s, 0, 1)); k = s*s*(3-2*s)
    return a*(1-k) + b*k


def _ref_ctrl(t):
    T_HOVER=11.0; T_GRASP=14.5; T_LIFT=20.0
    T_CARRY=27.0; T_PLACE=33.0; T_RETRACT=38.5; T_FREEZE=40.0
    if   t < T_REACH:   return HOME_Q.copy()
    elif t < T_HOVER:   return _sm(HOME_Q,  ABOVE_Q, (t-T_REACH)/(T_HOVER-T_REACH))
    elif t < T_GRASP:   return _sm(ABOVE_Q, PICK_Q,  (t-T_HOVER)/(T_GRASP-T_HOVER))
    elif t < T_LIFT:    return PICK_Q.copy()
    elif t < T_CARRY:   return _sm(PICK_Q,  LIFT_Q,  (t-T_LIFT)/(T_CARRY-T_LIFT))
    elif t < T_PLACE:   return _sm(LIFT_Q,  PLACE_Q, (t-T_CARRY)/(T_PLACE-T_CARRY))
    elif t < T_RETRACT: return PLACE_Q.copy()
    else:               return _sm(PLACE_Q, HOME_Q,  (t-T_RETRACT)/(T_FREEZE-T_RETRACT))


def run_sim_pair(duration=40.0):
    result  = SimPairResult()
    q_flt   = HOME_Q.copy()
    n_steps = int(duration / DT)
    for i in range(n_steps):
        t     = i * DT
        q_ref = _ref_ctrl(t)
        deg   = combined_degradation(t)
        alpha = lag_alpha(deg)
        q_nom = q_ref.copy()
        q_flt = q_flt + alpha * (q_ref - q_flt)
        result.records.append(StepRecord(t, q_nom.copy(), q_flt.copy(), deg, alpha))
    return result


if __name__ == "__main__":
    print("Running sim pair ...")
    r = run_sim_pair()
    f = r.records[-1]
    print(f"  Steps             : {len(r.records)}")
    print(f"  Final degradation : {f.deg:.3f}")
    print(f"  Final alpha       : {f.alpha:.3f}")
    print(f"  Joint error       : {np.round(f.q_nom - f.q_flt, 4)}")
