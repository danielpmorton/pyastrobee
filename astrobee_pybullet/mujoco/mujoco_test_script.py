"""Quick script to see if we can import a cargo-bag-like deformable using mujoco

Note:
If you want to just load the mjcf in the interactive viewer, run:
python -m mujoco.viewer --mjcf=/home/dan/astrobee_pybullet/astrobee_pybullet/mujoco/my_demo.mjcf

TODO
update the mjcf file (or make a new one) for a cargo bag
- potentially, model the handle with three cylinders or capsules

Note: XML for a soft cylinder:
<body pos="0 0 1">
    <freejoint/>
    <composite type="ellipsoid" count="5 7 9" spacing="0.05">
    <skin texcoord="true" material="matsponge" rgba=".7 .7 .7 1"/>
    <geom type="capsule" size=".015 0.05" rgba=".8 .2 .1 1"/>
    </composite>
</body>

"""

import mujoco
import numpy as np
import cv2

# import time
# import itertools
# from typing import Callable, NamedTuple, Optional, Union, List
# import mediapy as media
# import matplotlib.pyplot as plt


def main():
    # Make model and data
    # model = mujoco.MjModel.from_xml_path("astrobee_pybullet/mujoco/my_demo.mjcf")
    model = mujoco.MjModel.from_xml_path("astrobee_pybullet/mujoco/astrobee_demo.mjcf")
    # mujoco.mj_saveLastXML("astrobee_pybullet/mujoco/saved.xml", model)
    data = mujoco.MjData(model)

    # Make renderer, render and show the pixels
    renderer = mujoco.Renderer(model)
    # media.show_image(renderer.render()) # This would just give a black scene if uncommented

    mujoco.mj_forward(model, data)
    renderer.update_scene(data)

    # TODO comment out? This will just show a single frame I believe
    # media.show_image(renderer.render())

    # Simulation parameters
    duration = 3.8  # (seconds)
    framerate = 60  # (Hz)

    # Simulate and display video.
    frames = []
    mujoco.mj_resetData(model, data)  # Reset state and time.
    while data.time < duration:
        mujoco.mj_step(model, data)
        if len(frames) < data.time * framerate:
            renderer.update_scene(data)
            pixels = renderer.render().copy()
            frames.append(pixels)
    for frame in frames:
        cv2.imshow("Frame", frame)
        if cv2.waitKey(25) & 0xFF == ord("q"):  # TODO improve this
            break


if __name__ == "__main__":
    main()
