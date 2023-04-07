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
BENDING_STIFFNESS = 50
DAMPING_STIFFNESS = 0.1
ELASTIC_STIFFNESS = 50
FRICTION_COEFF = 0.1


# File locations
# (TODO) Should these even be here, or shoud this file only include the bag properties?
# (TODO) Should the OBJs also be specified?
MESH_DIRECTORY = "pyastrobee/assets/meshes/bags/"
FRONT_BAG_FILEPATH = MESH_DIRECTORY + "front_handle_bag.vtk"
SIDE_BAG_FILEPATH = MESH_DIRECTORY + "side_handle_bag.vtk"
TOP_BAG_FILEPATH = MESH_DIRECTORY + "top_handle_bag.vtk"
