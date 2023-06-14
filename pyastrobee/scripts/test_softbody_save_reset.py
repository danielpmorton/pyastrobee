"""Test script to see if we can get the pybullet saveState / restoreState functions working with deformables

This should be run from a terminal window where we're working with the local version of pybullet
For reference, thus can be done by:
1) (optional) Build Bullet after making any changes: In the bullet3 folder (/home/dan/software/bullet3), run
   ./build_cmake_pybullet_double.sh
2) Update the PYTHONPATH: export PYTHONPATH=/home/dan/software/bullet3/build_cmake/examples/pybullet
3) Run this script in that terminal
"""

import time

import numpy as np
import pybullet
import pybullet_data


from pyastrobee.elements.cargo_bag import CargoBag
from pyastrobee.utils.bullet_utils import initialize_pybullet, load_deformable_object


def loop_sim(t=5):
    """Run the sim for a specified amount of time"""
    start_time = time.time()
    while time.time() - start_time < t:
        pybullet.stepSimulation()
        time.sleep(1 / 120)


def run_test(object: str):
    """Pause, save, and reset the sim after user interaction. Object is either "bag" or "cloth"."""
    initialize_pybullet()
    if object == "bag":
        object_id = CargoBag("top_handle_bag", None, [0, 0, 0], [0, 0, 0, 1])
    elif object == "cloth":
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        load_deformable_object("cloth_z_up.obj")
    print("Apply a disturbance force")
    loop_sim()
    input("Press Enter to save the state")
    state_id = pybullet.saveState()
    input("Press Enter to let the sim run for a bit")
    loop_sim()
    input("Press Enter to reset the state")
    pybullet.restoreState(stateId=state_id)
    input("Press Enter to let the sim keep running")
    loop_sim()


if __name__ == "__main__":
    # run_test("bag")
    run_test("cloth")
