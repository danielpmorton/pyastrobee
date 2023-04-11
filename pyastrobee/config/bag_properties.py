"""Cargo bag properties (including softbody info, geometry, ...)"""

import numpy as np

from pyastrobee.utils.transformations import make_transform_mat
from pyastrobee.utils.rotations import Ry, Rz


# These transformation matrices can be thought of as "Bag to End-Effector" when grasped
# So for instance, we can compose these as "Bag to World" = "EE to World" @ "Bag to EE"
# Where "EE to World" is the pose of the end-effector grasp point
# The rotation component aligns the frame of the OBJ file with the graping frame
# The translation component is based on the CAD - it's the distance from the center of the bag to the handle
# If the CAD changes, these values will also need to be updated
FRONT_BAG_GRASP_TRANSFORM = make_transform_mat(
    Rz(np.pi / 2) @ Ry(-np.pi / 2), [-0.175, 0, 0]
)
SIDE_BAG_GRASP_TRANSFORM = make_transform_mat(np.eye(3), [-0.295, 0, 0])
TOP_BAG_GRASP_TRANSFORM = make_transform_mat(Ry(np.pi / 2), [-0.245, 0, 0])


# Softbody parameters (TODO: these may need to be refined)
MASS = 1.0
BENDING_STIFFNESS = 50.0
DAMPING_STIFFNESS = 0.1
ELASTIC_STIFFNESS = 50.0
FRICTION_COEFF = 0.1

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
