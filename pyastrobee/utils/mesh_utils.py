"""Mesh utilities for deformable simulation in PyBullet.

Reference: mesh_utils and anchor_utils in contactrika/dedo
"""

from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import pybullet
from pybullet_utils.bullet_client import BulletClient


def get_mesh_data(
    object_id: int, client: Optional[BulletClient] = None
) -> tuple[int, np.ndarray]:
    """Determines the number of vertices and their locations of a given mesh object in Pybullet

    Args:
        object_id (int): ID of the mesh loaded into Pybullet
        client (BulletClient, optional): If connecting to multiple physics servers, include the client
            (the class instance, not just the ID) here. Defaults to None (use default connected client)

    Returns:
        tuple[int, np.ndarray]:
            int: Number of vertices in the mesh
            np.ndarray: Mesh vertex positions, shape (num_verts, 3)
    """
    client: pybullet = pybullet if client is None else client
    kwargs = {}
    if hasattr(pybullet, "MESH_DATA_SIMULATION_MESH"):
        kwargs["flags"] = pybullet.MESH_DATA_SIMULATION_MESH
    num_verts, mesh_vert_positions = client.getMeshData(object_id, **kwargs)
    # Mesh vertices are originally stored in a tuple of tuples, so convert to numpy for ease of use
    return num_verts, np.array(mesh_vert_positions)


# TODO: If we need to average over multiple vertices, revert to the version in dedo/anchor_utils.
# But, if the new pybullet works best with 1 vertex per anchor, this is a simpler implementation
def get_closest_mesh_vertex(
    pos: npt.ArrayLike,
    mesh: Union[npt.ArrayLike, tuple[tuple[float, float, float], ...]],
) -> tuple[np.ndarray, int]:
    """Finds the vertex in a mesh closest to the given point

    Args:
        pos (npt.ArrayLike): The given XYZ position to search for nearby mesh vertices, shape (3,)
        mesh (npt.ArrayLike): Mesh vertices, stored in a (num_verts, 3) array, or a tuple of tuples.
            See get_mesh_data() for more details

    Returns:
        tuple[np.ndarray, int]:
            np.ndarray: The world-frame position of the closest vertex, shape (3,)
            int: The index of the closest vertex in the mesh
    """
    pos = np.array(pos).reshape(1, -1)
    mesh = np.array(mesh)
    dists = np.linalg.norm(mesh - pos, axis=1)
    closest_vert = np.argmin(dists)
    return mesh[closest_vert], closest_vert


def get_tet_mesh_data(
    object_id: int, client: Optional[BulletClient] = None
) -> tuple[int, np.ndarray]:
    """Determines the state of all of the tetrahedral elements in a tet mesh

    Args:
        object_id (int): ID of the tet mesh loaded into Pybullet
        client (BulletClient, optional): If connecting to multiple physics servers, include the client
            (the class instance, not just the ID) here. Defaults to None (use default connected client)

    Raises:
        AttributeError: If the Pybullet version does not support this functionality

    Returns:
        tuple[int, np.ndarray]:
            int: Number of tetrahedrons in the mesh
            np.ndarray: The xyz positions of all of the vertices of the tetrahedrons, shape (num_tets, 4, 3)
    """
    client: pybullet = pybullet if client is None else client
    try:
        data = client.getTetraMeshData(object_id)
    except AttributeError as e:
        raise AttributeError(
            "Cannot get tet mesh data. Check that you are using the most recent "
            + "locally-built version of Pybullet, as this is a recent feature"
        ) from e
    n, verts = data
    n_tets = n // 4
    verts = np.reshape(verts, (n_tets, 4, 3))
    return n_tets, verts
