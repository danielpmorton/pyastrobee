"""WORK IN PROGRESS"""

import time
import pybullet
import numpy as np
from pyastrobee.core.astrobee import Astrobee

# from pyastrobee.core.rigid_bag import RigidCargoBag
from pyastrobee.utils.bullet_utils import create_box, create_sphere
from pyastrobee.utils.debug_visualizer import visualize_points
from pyastrobee.utils.rotations import (
    euler_xyz_to_quat,
    quat_to_rmat,
    euler_xyz_to_rmat,
)
from pyastrobee.config.astrobee_transforms import (
    GRIPPER_TO_ARM_DISTAL,
    ARM_DISTAL_TO_GRIPPER,
)
from pyastrobee.config.bag_properties import BOX_LENGTH, BOX_WIDTH, BOX_HEIGHT
from pyastrobee.utils.transformations import make_transform_mat, transform_point
from pyastrobee.utils.poses import pos_quat_to_tmat
from pyastrobee.utils.quaternions import random_quaternion


np.random.seed(0)
pybullet.connect(pybullet.GUI)
robot = Astrobee()

robot.reset_to_ee_pose((0, 0, 0.21 + 0.1, *euler_xyz_to_quat((0, -np.pi / 2, 0))))
box = create_box(
    (0, 0, 0),
    (0, 0, 0, 1),  # random_quaternion(),
    1,
    (BOX_LENGTH, BOX_WIDTH, BOX_HEIGHT),
    True,
    (1, 1, 1, 1),
)
sphere = create_sphere(robot.ee_pose[:3], 0.01, 0.03, False, [1, 0, 0, 0.5])
box_pos, box_orn = pybullet.getBasePositionAndOrientation(box)
box_tmat = make_transform_mat(quat_to_rmat(box_orn), box_pos)
handle_offset = 0.1
grasp_pt_box_frame = np.array([0, 0, BOX_HEIGHT / 2 + handle_offset])
grasp_pt_world = transform_point(box_tmat, grasp_pt_box_frame)

# offsets in local gripper frame
constraint_offset_dist = 0.05
offsets = (
    np.array(
        [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    )
    * constraint_offset_dist
)

adjusted_box_offsets = [
    euler_xyz_to_rmat((0, -np.pi / 2, 0)) @ offset for offset in offsets
]

for i in range(len(offsets)):
    robot_pt_in_world = transform_point(pos_quat_to_tmat(robot.ee_pose), offsets[i])
    box_pt_in_world = transform_point(
        box_tmat, grasp_pt_box_frame + adjusted_box_offsets[i]
    )
    visualize_points(box_pt_in_world, (1, 0, 0), 10)
    input()
    visualize_points(robot_pt_in_world, (0, 0, 1), 20)
    input()

input("Press Enter to attach constraints")

cids = []
for i in range(len(offsets)):
    if i == 0:
        max_force = 100
    else:
        max_force = 1
    cid = pybullet.createConstraint(
        robot.id,
        robot.Links.ARM_DISTAL.value,
        box,
        -1,
        pybullet.JOINT_POINT2POINT,
        (0, 0, 1),
        GRIPPER_TO_ARM_DISTAL[:3, 3] + offsets[i],
        grasp_pt_box_frame + adjusted_box_offsets[i],
        # (0, 0, 0, 1),
        # euler_xyz_to_quat((0, -np.pi / 2, 0)),
    )
    pybullet.changeConstraint(cid, maxForce=max_force)
    cids.append(cid)

# pybullet.changeConstraint(cid, maxForce=100)
# cid = pybullet.createConstraint(
#     box,
#     -1,
#     sphere,
#     -1,
#     pybullet.JOINT_POINT2POINT,
#     (0, 0, 1),
#     grasp_pt_box_frame,
#     (0, 0, 0),
#     # (0, 0, 0, 1),
#     # (0, 0, 0, 1),
# )

# input()
while True:
    pybullet.stepSimulation()
    time.sleep(1 / 120)
