"""Cargo bag properties (including softbody info, geometry, ...)"""

import numpy as np

from pyastrobee.utils.transformations import make_transform_mat
from pyastrobee.utils.rotations import Rx, Ry, Rz


# Transformation matrices ("handle-to-bag"). These are defined so that the Astrobee gripper frame will be able
# to grasp the handle (z pointing along the handle, and x pointing out of the bag). "Bag frame" refers to the
# center of mass frame of the bag (what pybullet returns when you query the bag's position/orientation)
# Note: These need to be updated if the CAD changes
FRONT_HANDLE_TRANSFORM = make_transform_mat(
    Ry(-np.pi / 2) @ Rz(-np.pi / 2), [0, -0.16, 0]
)
BACK_HANDLE_TRANSFORM = make_transform_mat(Rz(np.pi / 2) @ Rx(-np.pi / 2), [0, 0.16, 0])
LEFT_HANDLE_TRANSFORM = make_transform_mat(Rz(np.pi), [-0.285, 0, 0])
RIGHT_HANDLE_TRANSFORM = make_transform_mat(np.eye(3), [0.285, 0, 0])
TOP_HANDLE_TRANSFORM = make_transform_mat(Ry(-np.pi / 2), [0, 0, 0.245])
BOTTOM_HANDLE_TRANSFORM = make_transform_mat(Ry(np.pi / 2) @ Rx(np.pi), [0, 0, -0.245])

# Softbody parameters (TODO: these may need to be refined)
MASS = 1.0
BENDING_STIFFNESS = 50.0
DAMPING_STIFFNESS = 0.1
ELASTIC_STIFFNESS = 50.0
FRICTION_COEFF = 0.1

# TODO: THESE ARE OUT OF DATE. Recalibrate these!!
# Indices of the mesh vertices closest to the corners of the bags
# These vertices are ordered as follows:
# 0. Right back top
# 1. Right back bottom
# 2. Right front top
# 3. Right front bottom
# 4. Left back top
# 5. Left back bottom
# 6. Left front top
# 7. Left front bottom
FRONT_BAG_CORNER_VERTS = [228, 331, 138, 372, 279, 223, 166, 201]
SIDE_BAG_CORNER_VERTS = [299, 221, 111, 151, 332, 312, 186, 89]
TOP_BAG_CORNER_VERTS = [296, 243, 281, 99, 237, 151, 171, 262]

# Dimensions of the bounding box around the main compartment of the bag (meters)
BOX_LENGTH = 0.50  # X dimension
BOX_WIDTH = 0.25  # Y dimension
BOX_HEIGHT = 0.42  # Z dimension
