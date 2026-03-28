import os
import tempfile
import numpy as np
import mujoco
import imageio.v3 as iio

W, H = 1920, 1080
FPS = 30
DUR = 6
OUT = os.path.expanduser("~/Desktop/video1_real_grasp.mp4")

XML = f"""
<mujoco model="render_sanity">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <visual>
    <global offwidth="{W}" offheight="{H}"/>
  </visual>

  <worldbody>
    <light pos="0 -1 2"/>
    <geom type="plane" size="2 2 0.1" rgba="0.2 0.2 0.2 1"/>

    <body name="ball" pos="0 0 0.4">
      <freejoint/>
      <geom type="sphere" size="0.03" mass="0.05" rgba="1 0.8 0 1"/>
    </body>

    <body name="gripper" pos="0 -0.3 0.6">
      <joint name="gy" type="slide" axis="0 1 0" range="-0.3 0.3"/>
      <geom type="box" size="0.05 0.05 0.02" rgba="0.7 0.7 0.8 1"/>
    </body>

    <camera name="main" pos="0 -1.2 0.7" xyaxes="1 0 0 0 0.7 0.7"/>
  </worldbody>

  <actuator>
    <position joint="gy" kp="50"/>
  </actuator>
</mujoco>
"""

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write(XML)
        path = f.name

    model = mujoco.MjModel.from_xml_path(path)
    os.unlink(path)
    data = mujoco.MjData(model)

    renderer = mujoco.Renderer(model, width=W, height=H)
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "main")

    frames = []
    total_steps = int(DUR / model.opt.timestep)

    for step in range(total_steps):
        t = step * model.opt.timestep
        data.ctrl[0] = 0.2 * np.sin(0.8 * t)
        mujoco.mj_step(model, data)

        if step % max(1, round(1.0 / (FPS * model.opt.timestep))) == 0:
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())

    iio.imwrite(
        OUT,
        frames,
        fps=FPS,
        codec="libx264",
        macro_block_size=1,
        output_params=["-crf", "16", "-preset", "slow"]
    )
    print(f"Done: {OUT}")

if __name__ == "__main__":
    main()
