"""Script to construct symmetric meshes for the cargo bag

This will mirror a mesh about a plane and then stitch it together with the mirrored component

The mesh should be pre-constructed so that:
1) The planes to use for mirroring are well-defined (like the standard planes at the origin)
2) The mesh should be non-manifold (it should not be an enclosed region -- since the mirroring will
   otherwise duplicate faces that are up against the mirror plane)

This is a bit of a mess but it seems to work, and shouldn't be needed often

It relies on pymesh, which is a bit annoying to install. Instructions are below. Notably, we need to work
with this in a new virtual environment (NOT the pyastrobee environment) because it only works with
python 3 - 3.7

sudo apt-get install \
   libeigen3-dev \
   libgmp-dev \
   libgmpxx4ldbl \
   libmpfr-dev \
   libboost-dev \
   libboost-thread-dev \
   libtbb-dev \
pyenv install 3.7.15
pyenv virtualenv 3.7.15 pymesh
pyenv shell pymesh
pip install wheel
pip install numpy>=1.10.4
pip install scipy>=0.17.0
pip install nose>=1.3.7
pip install pymesh2
"""

from pathlib import Path

from collections import defaultdict
from typing import Union

import numpy as np
import pymesh


class Plane:
    def __init__(self, a, b, c, d):
        # Plane defined by ax + by + cz + d = 0
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.normal = np.array([a, b, c]) / np.linalg.norm([a, b, c])
        self.origin = np.array([a, b, c]) * d

    def __iter__(self):
        return iter([self.a, self.b, self.c, self.d])


# Origin plane definitions
YZ_PLANE = Plane(1, 0, 0, 0)
XZ_PLANE = Plane(0, 1, 0, 0)
XY_PLANE = Plane(0, 0, 1, 0)


# Unused
def get_face_center(verts, face):
    assert len(face) == 3
    return np.average([verts[i] for i in face], axis=0)


# Based on https://www.geeksforgeeks.org/mirror-of-a-point-through-a-3-d-plane/
def mirror_point(point: np.ndarray, plane: Plane):
    a, b, c, d = plane
    x1, y1, z1 = point
    k = (-a * x1 - b * y1 - c * z1 - d) / float((a * a + b * b + c * c))
    x2 = a * k + x1
    y2 = b * k + y1
    z2 = c * k + z1
    x3 = 2 * x2 - x1
    y3 = 2 * y2 - y1
    z3 = 2 * z2 - z1
    return np.array([x3, y3, z3])


def mirror_and_join_mesh(mesh: Union[pymesh.Mesh, tuple], plane: Plane) -> tuple:
    """Mirrors a mesh about a plane and then merges the mirrored mesh with the original

    Args:
        mesh (Union[pymesh.Mesh, tuple]): Mesh to mirror. If a tuple, entries are (vertices, faces)
        plane (Plane): Plane to mirror about

    Returns:
        tuple[np.ndarray, np.ndarray]: Vertices and faces.
            Vertices: shape (n_verts, 3), containing an XYZ position for each vertex
            Faces: shape (n_faces, 3), containing three vertex indices for each face
    """
    if isinstance(mesh, pymesh.Mesh):
        old_verts = mesh.vertices
        old_faces = mesh.faces
    elif isinstance(mesh, tuple):
        assert len(mesh) == 2
        old_verts, old_faces = mesh
    eps = 1e-6

    # Faces are normally constructed via vertex indices
    # However, we're going to be creating a bunch of new vertices, and deleting some duplicates
    # So, with all of these modifications, we don't really want to work with indices
    # Rather, we'll name the vertices and faces as:
    # v_0 for the 0-indexed original vertex; v_0_m for the mirrored version
    # f_0 for the 0-indexed original face; f_0_m for the mirrored version
    # Now, each face will be a list of three strings (the names of each vertex, rather than its index)

    V = {}  # Original vertex names => Position
    for i, vert in enumerate(old_verts):
        V["v_" + str(i)] = vert
    F = defaultdict(list)  # Original face names => three vertex names
    for i, face in enumerate(old_faces):
        for vid in face:
            F["f_" + str(i)].append("v_" + str(vid))
    VM = {}  # Mirrored vertex names => Position
    for i, vert in enumerate(old_verts):
        VM["v_" + str(i) + "_m"] = mirror_point(vert, plane)
    FM = defaultdict(list)  # Mirrored face names => three vertex names
    for i, face in enumerate(old_faces):
        # Flip vertex order to maintain correct normal direction after mirror
        for vid in face[::-1]:
            FM["f_" + str(i) + "_m"].append("v_" + str(vid) + "_m")
    V2F = defaultdict(list)  # Original vertex name => Connected face names
    for name, face in F.items():
        for vert_name in face:
            V2F[vert_name].append(name)
    VM2FM = defaultdict(list)  # Mirrored vertex name => Connected face names
    for name, face in FM.items():
        for vert_name in face:
            VM2FM[vert_name].append(name)

    # Go through each of the new mirrored points, determine if it is a duplicate
    # If so, remove it from storage and update its associated faces to instead use the non-mirrored point
    for v_m_name, v_m_pt in list(VM.items()):
        orig_pt_name = v_m_name.rstrip("_m")
        orig_pt = V[orig_pt_name]
        if np.linalg.norm(orig_pt - v_m_pt) <= eps:
            VM.pop(v_m_name)
            for associated_face in VM2FM[v_m_name]:
                for j, vert_name in enumerate(FM[associated_face]):
                    if vert_name == v_m_name:
                        FM[associated_face][j] = orig_pt_name

    # Merge the dictionaries to combine the original mesh with the mirrored parts
    V.update(VM)
    F.update(FM)

    # Reconstruct the vertices and faces into the original array-like format
    verts = []
    V2ID = {}  # Vertex name => Final position in the array
    for name, vert in V.items():
        verts.append(vert)
        V2ID[name] = len(verts) - 1
    faces = []
    for name, face in F.items():
        f = []
        for vert_name in face:
            f.append(V2ID[vert_name])
        faces.append(f)
    return np.array(verts), np.array(faces)


# Currently unused... maybe helpful?
def get_normals(verts: np.ndarray, faces: np.ndarray):
    normals = np.zeros(faces.shape)
    for i, face in enumerate(faces):
        vert_a = verts[face[0]]
        vert_b = verts[face[1]]
        vert_c = verts[face[2]]
        normals[i] = np.cross(vert_b - vert_a, vert_c - vert_a)
    return normals


# This is a good test to run to validate that the mesh was generated properly
def check_for_duplicates(vertices):
    num_duplicates = 0
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            if np.linalg.norm(vertices[i] - vertices[j]) < 1e-6:
                num_duplicates += 1
    return num_duplicates


def main():
    filename = "/home/dan/Downloads/top_handle_quarter.stl"
    output_name = "artifacts/top_handle_symmetric"
    output_stl_location = output_name + ".stl"
    output_obj_location = output_name + ".obj"

    if not Path(filename).exists():
        raise FileNotFoundError("Could not find the input mesh")

    mesh = pymesh.load_mesh(filename)
    vertices, faces = mirror_and_join_mesh(mesh, YZ_PLANE)
    vertices, faces = mirror_and_join_mesh((vertices, faces), XZ_PLANE)
    # Note: OBJ is important for pybullet, STL is important for GMSH
    if Path(output_stl_location).exists():
        input(f"WARNING: File {output_stl_location} exists:\n Press Enter to overwrite")
    pymesh.save_mesh_raw(output_stl_location, vertices, faces)
    if Path(output_obj_location).exists():
        input(f"WARNING: File {output_obj_location} exists:\n Press Enter to overwrite")
    pymesh.save_mesh_raw(output_obj_location, vertices, faces)


if __name__ == "__main__":
    main()
