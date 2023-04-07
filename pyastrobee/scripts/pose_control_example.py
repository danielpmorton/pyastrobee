"""Quick example of position/constraint control with the Astrobee"""

import time

import numpy as np
import pybullet

from pyastrobee.control.astrobee import Astrobee
from pyastrobee.control.controller import PoseController
from pyastrobee.utils.debug_visualizer import visualize_frame, remove_debug_objects
from pyastrobee.utils.poses import pos_quat_to_tmat
from pyastrobee.utils.quaternion import random_quaternion


def random_pose(max_dist=2):
    p = max_dist * np.random.rand(3)
    q = random_quaternion()
    return np.concatenate((p, q))


np.random.seed(0)
pybullet.connect(pybullet.GUI)
robot = Astrobee()
controller = PoseController(robot)

# Use the position control to navigate to various random poses
for _ in range(10):
    target = random_pose()
    ids = visualize_frame(pos_quat_to_tmat(target))
    controller.go_to_pose(target)
    print("Done, waiting a few seconds before the next one")
    time.sleep(2)
    remove_debug_objects(ids)

# Leave the sim running
print("Looping sim...")
while True:
    pybullet.stepSimulation()
    time.sleep(1 / 120)
