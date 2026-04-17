"""Setup and environment check for Problem 5."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/simcorrect"))

FOLDER = os.path.expanduser("~/simcorrect/Problem5_ToolMassMismatch")
OUTPUT = os.path.join(FOLDER, "output")

def main():
    os.makedirs(OUTPUT, exist_ok=True)
    print(f"Folder:  {FOLDER}")
    print(f"Output:  {OUTPUT}")
    files = ["render_demo.py","sim_pair.py","divergence_detector.py",
             "parameter_identifier.py","correction_and_validation.py",
             "demo.py","step0.py"]
    for f in files:
        p = os.path.join(FOLDER, f)
        print(f"  {f}: {'OK' if os.path.exists(p) else 'MISSING'}")

    opencad_path = os.path.expanduser("~/simcorrect/opencad.py")
    print(f"  opencad.py (root): {'OK' if os.path.exists(opencad_path) else 'MISSING'}")

    try:
        from opencad import Part
        p = Part("grip").set_mass(0.160)
        p.export("/tmp/opencad_test.xml")
        print(f"  OpenCAD import:    OK")
    except Exception as e:
        print(f"  OpenCAD import:    FAILED — {e}")

    try:
        import mujoco; print(f"  mujoco:            {mujoco.__version__}  OK")
    except ImportError: print("  mujoco:            NOT FOUND")
    try:
        import imageio; print("  imageio:           OK")
    except ImportError: print("  imageio:           NOT FOUND")
    try:
        from PIL import Image; print("  Pillow:            OK")
    except ImportError: print("  Pillow:            NOT FOUND")

    print("\nStep 0 complete. Run order:")
    print("  python demo.py")
    print("  python sim_pair.py")
    print("  python correction_and_validation.py")
    print("  python render_demo.py")

if __name__ == "__main__":
    main()
