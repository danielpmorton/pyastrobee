"""Mesh utilities for deformable simulation in PyBullet.

Reference: mesh_utils and anchor_utils in contactrika/dedo
"""

from typing import Union

import numpy as np
import numpy.typing as npt
import pybullet


def get_mesh_data(object_id: int) -> tuple[int, np.ndarray]:
    """Determines the number of vertices and their locations of a given mesh object in Pybullet

    Args:
        object_id (int): ID of the mesh loaded into Pybullet

    Returns:
        tuple of:
            int: Number of vertices in the mesh
            tuple[tuple[float, float, float], ...]: Mesh vertex positions, shape (num_verts, 3)
    """
    kwargs = {}
    if hasattr(pybullet, "MESH_DATA_SIMULATION_MESH"):
        kwargs["flags"] = pybullet.MESH_DATA_SIMULATION_MESH
    num_verts, mesh_vert_positions = pybullet.getMeshData(object_id, **kwargs)
    # Mesh vertices are originally stored in a tuple of tuples, so convert to numpy for ease of use
    return num_verts, np.array(mesh_vert_positions)


def get_closest_mesh_vertex(
    pos: npt.ArrayLike,
    mesh: Union[npt.ArrayLike, tuple[tuple[float, float, float], ...]],
) -> tuple[np.ndarray, int]:
    """Finds the vertex in a mesh closest to the given point

    TODO: If we need to average over multiple vertices, revert to the version in dedo/anchor_utils.
    But, if the new pybullet works best with 1 vertex per anchor, this is a simpler implementation

    Args:
        pos (npt.ArrayLike): The given XYZ position to search for nearby mesh vertices, shape (3,)
        mesh (npt.ArrayLike): Mesh vertices, stored in a (num_verts, 3) array, or a tuple of tuples.
            See get_mesh_data() for more details
    """
    pos = np.array(pos).reshape(1, -1)
    mesh = np.array(mesh)
    dists = np.linalg.norm(mesh - pos, axis=1)
    closest_vert = np.argmin(dists)
    return mesh[closest_vert], closest_vert
