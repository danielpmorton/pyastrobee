"""Tools for determining the (approximate) position/orientation of a volumetric softbody

Pybullet's built-in functions don't work well on softbodies (can't get reliable angular information)
so this will help determine the state of the cargo bag

Here, we use the positions of the eight mesh vertices closest to the corners of the bag's main compartment
to approximate the delta-x/y/z values, and then create an orthonormal basis from that

See the find_corner_verts script for more info on the corner-vertex identification process
"""

import time

import numpy as np
import numpy.typing as npt
import pybullet

from pyastrobee.utils.transformations import make_transform_mat
from pyastrobee.utils.debug_visualizer import visualize_frame
from pyastrobee.utils.mesh_utils import get_mesh_data
from pyastrobee.utils.bullet_utils import load_deformable_object
from pyastrobee.config.bag_properties import TOP_HANDLE_BAG_CORNERS

# For indexing into the list of vertices
# Naming these as constants so their usage is clearer in the functions below
RIGHT_BACK_TOP = 0
RIGHT_BACK_BOT = 1
RIGHT_FRONT_TOP = 2
RIGHT_FRONT_BOT = 3
LEFT_BACK_TOP = 4
LEFT_BACK_BOT = 5
LEFT_FRONT_TOP = 6
LEFT_FRONT_BOT = 7


def get_x_axis(mesh: npt.ArrayLike, corners: list[int]) -> np.ndarray:
    """Calculate the x-axis of the bag frame based on averaging the length-wise vectors between corners

    Args:
        mesh (npt.ArrayLike): Bag mesh vertex positions, shape (num_verts, 3)
        corners (list[int]): Indices of the 8 vertices closest to the corners of the bag

    Returns:
        np.ndarray: Best-fit axis based on the corner positions, shape (3,)
    """
    a = mesh[corners[RIGHT_BACK_TOP]] - mesh[corners[LEFT_BACK_TOP]]
    b = mesh[corners[RIGHT_BACK_BOT]] - mesh[corners[LEFT_BACK_BOT]]
    c = mesh[corners[RIGHT_FRONT_TOP]] - mesh[corners[LEFT_FRONT_TOP]]
    d = mesh[corners[RIGHT_FRONT_BOT]] - mesh[corners[LEFT_FRONT_BOT]]
    return np.average([a, b, c, d], axis=0)


def get_y_axis(mesh: npt.ArrayLike, corners: list[int]):
    """Calculate the y-axis of the bag frame based on averaging the width-wise vectors between corners

    Args:
        mesh (npt.ArrayLike): Bag mesh vertex positions, shape (num_verts, 3)
        corners (list[int]): Indices of the 8 vertices closest to the corners of the bag

    Returns:
        np.ndarray: Best-fit axis based on the corner positions, shape (3,)
    """
    a = mesh[corners[RIGHT_BACK_TOP]] - mesh[corners[RIGHT_FRONT_TOP]]
    b = mesh[corners[RIGHT_BACK_BOT]] - mesh[corners[RIGHT_FRONT_BOT]]
    c = mesh[corners[LEFT_BACK_TOP]] - mesh[corners[LEFT_FRONT_TOP]]
    d = mesh[corners[LEFT_BACK_BOT]] - mesh[corners[LEFT_FRONT_BOT]]
    return np.average([a, b, c, d], axis=0)


def get_z_axis(mesh: npt.ArrayLike, corners: list[int]):
    """Calculate the z-axis of the bag frame based on averaging the height-wise vectors between corners

    Args:
        mesh (npt.ArrayLike): Bag mesh vertex positions, shape (num_verts, 3)
        corners (list[int]): Indices of the 8 vertices closest to the corners of the bag

    Returns:
        np.ndarray: Best-fit axis based on the corner positions, shape (3,)
    """
    a = mesh[corners[RIGHT_BACK_TOP]] - mesh[corners[RIGHT_BACK_BOT]]
    b = mesh[corners[RIGHT_FRONT_TOP]] - mesh[corners[RIGHT_FRONT_BOT]]
    c = mesh[corners[LEFT_BACK_TOP]] - mesh[corners[LEFT_BACK_BOT]]
    d = mesh[corners[LEFT_FRONT_TOP]] - mesh[corners[LEFT_FRONT_BOT]]
    return np.average([a, b, c, d], axis=0)


def orthogonalize(
    x: npt.ArrayLike, y: npt.ArrayLike, z: npt.ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn three linearly independent vectors into an orthogonal/orthonormal basis

    We assume that the direction of x is correct, and base the other calculations on this

    Args:
        x (npt.ArrayLike): Initial x axis, shape (3,)
        y (npt.ArrayLike): Initial y axis, shape (3,)
        z (npt.ArrayLike): Initial z axis, shape (3,)

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The new normalized + orthogonal x, y, and z axes
    """
    x_new = x / np.linalg.norm(x)
    y_new = np.cross(z, x)
    y_new = y_new / np.linalg.norm(y_new)
    z_new = np.cross(x, y)
    z_new = z_new / np.linalg.norm(z_new)
    return x_new, y_new, z_new


def get_bag_frame(mesh: npt.ArrayLike, corners: list[int]) -> np.ndarray:
    """Determine the transformation matrix for the cargo bag frame

    Args:
        mesh (npt.ArrayLike): Bag mesh vertex positions, shape (num_verts, 3)
        corners (list[int]): Indices of the 8 vertices closest to the corners of the bag

    Returns:
        np.ndarray: Transformation matrix, shape (4, 4)
    """
    x = get_x_axis(mesh, corners)
    y = get_y_axis(mesh, corners)
    z = get_z_axis(mesh, corners)
    R = np.column_stack(orthogonalize(x, y, z))
    origin = np.average(mesh, axis=0)
    return make_transform_mat(R, origin)


if __name__ == "__main__":
    pybullet.connect(pybullet.GUI)
    pybullet.resetSimulation(pybullet.RESET_USE_DEFORMABLE_WORLD)
    filename = "pyastrobee/assets/meshes/bags/top_handle.vtk"
    bag_id = load_deformable_object(filename, bending_stiffness=10)
    while True:
        n_verts, bag_mesh = get_mesh_data(bag_id)
        T = get_bag_frame(bag_mesh, TOP_HANDLE_BAG_CORNERS)
        visualize_frame(T, lifetime=1)
        pybullet.stepSimulation()
        time.sleep(1 / 240)
