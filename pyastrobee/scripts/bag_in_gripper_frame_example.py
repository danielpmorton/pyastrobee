"""Example of initializing a bag in the correct pose for the handle to be in the Astrobee's gripper

This was originally just supposed to be a temporary debugging script for the transformations but it
is actually quite useful as an example of loading objects in relative locations to the Astrobee.

TODO
- There is some weird stuff happening with the collision between the Astrobee and the bag
  on initialization. It seems like the collision info for the bag isn't perfect and thus the
  Astrobee and the bag are colliding (and the bag is launching away). This behavior should be
  figured out (possibly by updating the collision bodies of the bags, or changing the grasp point)
  - An easy fix might be just inserting a bonus delta-x transform in the math in this script so that
    the bag is slightly further from the gripper
"""

import time
import pybullet
import numpy as np
from pyastrobee.elements.astrobee import Astrobee
from pyastrobee.utils.bullet_utils import load_rigid_object
from pyastrobee.utils.poses import pos_quat_to_tmat
from pyastrobee.utils.rotations import rmat_to_quat
from pyastrobee.utils.debug_visualizer import visualize_frame
from pyastrobee.utils.quaternions import random_quaternion
import pyastrobee.config.bag_properties as bag_props

front_file = "pyastrobee/assets/meshes/bags/front_handle_bag.obj"
side_file = "pyastrobee/assets/meshes/bags/side_handle_bag.obj"
top_file = "pyastrobee/assets/meshes/bags/top_handle_bag.obj"

pybullet.connect(pybullet.GUI)
np.random.seed(0)

FRONT_BAG_TO_EE = bag_props.FRONT_BAG_GRASP_TRANSFORM
SIDE_BAG_TO_EE = bag_props.SIDE_BAG_GRASP_TRANSFORM
TOP_BAG_TO_EE = bag_props.TOP_BAG_GRASP_TRANSFORM


robot = Astrobee(pose=[1, 1, 1, *random_quaternion()])
EE2W = pos_quat_to_tmat(robot.ee_pose)
visualize_frame(EE2W)
FRONT_BAG_TO_WORLD = EE2W @ FRONT_BAG_TO_EE
visualize_frame(FRONT_BAG_TO_WORLD)
load_rigid_object(
    front_file,
    pos=FRONT_BAG_TO_WORLD[:3, 3],
    orn=rmat_to_quat(FRONT_BAG_TO_WORLD[:3, :3]),
)


robot_2 = Astrobee(pose=[1, -1, 1, *random_quaternion()])
EE2W_2 = pos_quat_to_tmat(robot_2.ee_pose)
visualize_frame(EE2W_2)
SIDE_BAG_TO_WORLD = EE2W_2 @ SIDE_BAG_TO_EE
visualize_frame(SIDE_BAG_TO_WORLD)
load_rigid_object(
    side_file, pos=SIDE_BAG_TO_WORLD[:3, 3], orn=rmat_to_quat(SIDE_BAG_TO_WORLD[:3, :3])
)


robot3 = Astrobee(pose=[-1, -1, 1, *random_quaternion()])
EE2W_3 = pos_quat_to_tmat(robot3.ee_pose)
visualize_frame(EE2W_3)
TOP_BAG_TO_WORLD = EE2W_3 @ TOP_BAG_TO_EE
visualize_frame(TOP_BAG_TO_WORLD)
load_rigid_object(
    top_file, pos=TOP_BAG_TO_WORLD[:3, 3], orn=rmat_to_quat(TOP_BAG_TO_WORLD[:3, :3])
)

input("The sim is paused. Press Enter to begin looping it")
while True:
    pybullet.stepSimulation()
    time.sleep(1 / 240)
