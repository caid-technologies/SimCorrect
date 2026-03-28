import os
import tempfile
import mujoco
import numpy as np
import imageio.v3 as iio

W,H=1600,900
FPS=30
OUT=os.path.expanduser("~/Desktop/video1_corl_physical.mp4")

XML = """
<mujoco model="corl_demo">

<option timestep="0.002" gravity="0 0 -9.81"/>

<visual>
<global offwidth="1600" offheight="900"/>
</visual>

<worldbody>

<geom type="plane" size="3 3 0.1"/>

<!-- tables -->
<geom type="box" pos="-0.4 0 0.2" size="0.2 0.15 0.2"/>
<geom type="box" pos="0.4 0 0.2" size="0.2 0.15 0.2"/>

<!-- targets -->
<geom type="cylinder" pos="-0.4 0 0.42" size="0.05 0.003" rgba="0 1 0 1"/>
<geom type="cylinder" pos="0.4 0 0.42" size="0.05 0.003" rgba="0 1 0 1"/>

<!-- cans -->
<body name="can_left" pos="-0.6 0 0.03">
<freejoint/>
<geom type="cylinder" size="0.03 0.06" mass="0.05" friction="1 0.1 0.1" rgba="0.8 0.2 0.2 1"/>
</body>

<body name="can_right" pos="0.6 0 0.03">
<freejoint/>
<geom type="cylinder" size="0.03 0.06" mass="0.05" friction="1 0.1 0.1" rgba="0.8 0.2 0.2 1"/>
</body>

<!-- LEFT ARM -->
<body name="left_base" pos="-0.2 0 0.6">
<joint type="hinge" axis="0 1 0"/>
<body>
<geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="1"/>
<joint type="hinge" axis="0 1 0"/>
<body>
<geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.025" mass="0.8"/>
<site name="left_tcp" pos="0.3 0 0"/>
</body>
</body>
</body>

<!-- RIGHT ARM -->
<body name="right_base" pos="0.2 0 0.6">
<joint type="hinge" axis="0 1 0"/>
<body>
<geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="1" rgba="1 0 0 1"/>
<joint type="hinge" axis="0 1 0"/>
<body>
<geom type="capsule" fromto="0 0 0 0.18 0 0" size="0.025" mass="0.8" rgba="1 0 0 1"/>
<site name="right_tcp" pos="0.18 0 0"/>
</body>
</body>
</body>

<camera name="main" pos="0 -2.8 1.2"/>

</worldbody>
</mujoco>
"""

def main():

    with tempfile.NamedTemporaryFile(delete=False,suffix=".xml") as f:
        f.write(XML.encode())
        xml=f.name

    model=mujoco.MjModel.from_xml_path(xml)
    data=mujoco.MjData(model)

    renderer=mujoco.Renderer(model,width=W,height=H)
    cam=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,"main")

    frames=[]

    for i in range(6000):
        mujoco.mj_step(model,data)

        if i%2==0:
            renderer.update_scene(data,camera=cam)
            frames.append(renderer.render().copy())

    iio.imwrite(
        OUT,
        frames,
        fps=FPS,
        macro_block_size=1
    )

    print("Done:",OUT)

if __name__=="__main__":
    main()
