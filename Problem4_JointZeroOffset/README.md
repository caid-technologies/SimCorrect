# Problem 4 — Joint Zero Offset

## Fault
Joint 1 encoder mounted **8 degrees (0.1396 rad)** off during assembly.
The arm executes every command perfectly — from the wrong starting angle.

## Signature
| Metric | Value |
|---|---|
| Rotational miss at 0.75m | 103mm |
| Rotational miss at 0.375m | 52mm |
| Scaling ratio | 2.0 (pure rotation) |
| Joint RMSE | **0.000** |
| Fault class | **Geometric** |

## Key distinction
- Miss **scales with reach** (rotation, not translation)
- Joint RMSE = 0 (arm moves correctly, starts from wrong place)
- Differs from Problem 1 (length), Problem 2 (wrist offset), Problem 3 (dynamics)

## Correction
One number. 0.28 seconds. Zero human intervention.

## Run
```bash
cd ~/simcorrect/Problem4_JointZeroOffset
python step0.py
python demo.py
python correction_and_validation.py
python render_demo.py
```
