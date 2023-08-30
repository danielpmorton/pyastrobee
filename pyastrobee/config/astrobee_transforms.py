"""Transformation values for the Astrobee

These are either hand-calibrated or directly from NASA's code

We store these as transformation matrices

Sourced from:
- astrobee/config/geometry.config
- astrobee/config/robots/honey.config
"""
# TODO
# - Delete / comment any transforms that aren't used?
# - Deal with placeholder for gripper to arm distal !!!

import numpy as np

from pyastrobee.utils.transformations import make_transform_mat, invert_transform_mat
from pyastrobee.utils.rotations import quat_to_rmat, fixed_xyz_to_rmat

# Transformation between the arm's distal link frame and the center of the grasp point
# See the calibration script for more info
GRIPPER_TO_ARM_DISTAL = make_transform_mat(np.eye(3), [-0.08240211, 0, -0.04971851])
ARM_DISTAL_TO_GRIPPER = invert_transform_mat(GRIPPER_TO_ARM_DISTAL)

# Camera to robot for 3rd person view of the robot as it moves
OBSERVATION_CAM = make_transform_mat(
    fixed_xyz_to_rmat([0, -np.pi / 4, 0]), [-0.7, 0, 0.5]
)

# Camera

# TODO need to verify these. They said these are placeholders and are not valid!
PERCH_CAM = make_transform_mat(
    quat_to_rmat([0, -0.70710678118, 0, 0.70710678118]), [-0.1331, 0.0509, -0.0166]
)
HAZ_CAM = make_transform_mat(
    quat_to_rmat([-0.500, 0.500, -0.500, 0.500]), [0.1328, 0.0362, -0.0826]
)
NAV_CAM = make_transform_mat(
    quat_to_rmat([0.500, 0.500, 0.500, 0.500]), [0.1157 + 0.002, -0.0422, -0.0826]
)
DOCK_CAM = make_transform_mat(
    quat_to_rmat([0.500, -0.500, -0.500, 0.500]), [-0.1032 - 0.0029, -0.0540, -0.0064]
)
SCI_CAM = make_transform_mat(
    quat_to_rmat([0.500, 0.500, 0.500, 0.500]), [0.118, 0.0, -0.096]
)
IMU = make_transform_mat(
    quat_to_rmat([0.000, 0.000, 0.70710678118, 0.70710678118]), [0.0247, 0.0183, 0.0094]
)
# These camera transforms below are unclear as to why they are needed
NAV_CAM_TO_HAZ_CAM = make_transform_mat(
    quat_to_rmat([-0.0030431141, 0.0092646368, 0.99993195, 0.0064039206]),
    [0.071421481, -0.00030319673, 0.0018058249],
)
NAV_CAM_TO_SCI_CAM = make_transform_mat(
    quat_to_rmat([-0.0035414316, 0.0048089362, -0.0071515076, 0.99995659]),
    [-0.076526224, 0.011869553, 0.045409155],
)
# This transform I will assume is a valid transformation matrix, but it should probably be checked (TODO)
# I am also unsure what this refers to
HAZ_CAM_DEPTH_TO_IMAGE = np.array(
    [
        [0.91602851, -0.00031586647, -0.0028485861, 0.0029767338],
        [0.00032189197, 0.91603089, 0.0019373744, -0.0020741879],
        [0.0028479115, -0.0019383659, 0.91602652, -0.0042296964],
        [0, 0, 0, 1],
    ]
)
