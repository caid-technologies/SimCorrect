"""Setup and initialization check for Problem 4."""
import os

FOLDER=os.path.expanduser("~/simcorrect/Problem4_JointZeroOffset")
OUTPUT=os.path.join(FOLDER,"output")

def main():
    os.makedirs(OUTPUT,exist_ok=True)
    print(f"Folder:  {FOLDER}")
    print(f"Output:  {OUTPUT}")
    files=["render_demo.py","sim_pair.py","divergence_detector.py",
           "parameter_identifier.py","correction_and_validation.py","demo.py"]
    for f in files:
        p=os.path.join(FOLDER,f)
        status="OK" if os.path.exists(p) else "MISSING"
        print(f"  {f}: {status}")
    try:
        import mujoco; print(f"mujoco:  {mujoco.__version__}  OK")
    except ImportError: print("mujoco: NOT FOUND")
    try:
        import imageio; print("imageio: OK")
    except ImportError: print("imageio: NOT FOUND")
    try:
        from PIL import Image; print("Pillow:  OK")
    except ImportError: print("Pillow: NOT FOUND")
    print("Step 0 complete. Run: python render_demo.py")

if __name__=="__main__":
    main()
