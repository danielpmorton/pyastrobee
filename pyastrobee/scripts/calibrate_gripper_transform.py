"""Calibrates the transformation between the arm_distal link and the the gripper point

TODO fix the physics for the visualization of the point
"""

import numpy as np
import pybullet

from pyastrobee.elements.astrobee import Astrobee
from pyastrobee.utils.bullet_utils import initialize_pybullet, run_sim, create_sphere
from pyastrobee.utils.rotations import Ry
from pyastrobee.utils.transformations import make_transform_mat

initialize_pybullet(bg_color=[1, 1, 1])

# Use the half-open gripper to determine the position of the grasp point
robot = Astrobee(gripper_pos=50)
ad_id = 2
gld_id = 4
T_AD2W = robot.get_link_transform(ad_id)  # Arm distal link transform
T_GLD2W = robot.get_link_transform(gld_id)  # Gripper left distal link transform
p_AD_in_world = T_AD2W[:3, 3]
p_GLD_in_world = T_GLD2W[:3, 3]
R_W2AD = Ry(np.pi / 2)  # World to arm rotation
# R_W2A_new = T_AD2W.T[:3, :3]  # TODO use or remove?
ad_to_gld_in_world = p_GLD_in_world - p_AD_in_world
ad_to_gld_in_ad = R_W2AD @ ad_to_gld_in_world
# This will give us a vector between the two frames, but we don't exactly want this because the
# gripper joints have a lateral offset
# So, to remove this offset, we just set this offset (the y component) to be 0
ad_to_grip_in_ad = np.array([1, 0, 1]) * ad_to_gld_in_ad

# We don't really care about the rotation of this point, so assume it's the same as the arm distal frame
T_G2AD = make_transform_mat(np.eye(3), ad_to_grip_in_ad)

# Print the calibrated transformation (grasp to arm_distal link)
print(f"Transform:\n{T_G2AD}")

# To confirm that this is the correct transformation, add an object to help view this grasp point
# Right now, this sphere is acting really strangely (physics-wise at least), BUT the position looks good
# (TODO) - fix the physics for this. But, it isn't totally necessary because we really just want the transform

T_G2W = T_AD2W @ T_G2AD
p_grip_in_world = T_G2W[:3, 3]
# pybullet.addUserDebugPoints([p_grip_in_world], [[1.0, 0.0, 0.0]], 40, )
sphere = create_sphere(p_grip_in_world, 0.01, 0.03, False, [1, 0, 0, 0.5])
pybullet.createConstraint(
    robot.id,
    ad_id,
    sphere,
    -1,
    pybullet.JOINT_FIXED,
    [0, 0, 0],
    ad_to_grip_in_ad,
    [0, 0, 0],
)

run_sim()
