# sudo apt-get install \
#    libeigen3-dev \
#    libgmp-dev \
#    libgmpxx4ldbl \
#    libmpfr-dev \
#    libboost-dev \
#    libboost-thread-dev \
#    libtbb-dev \
# pyenv install 3.7.15
# pyenv virtualenv 3.7.15 pymesh
# pyenv shell pymesh
# pip install wheel
# pip install numpy>=1.10.4
# pip install scipy>=0.17.0
# pip install nose>=1.3.7
# pip install pymesh2


# NOTE THAT THE MESH WILL HAVE TO HAVE BOUNDARY ELEMENTS REMOVED

from collections import defaultdict
from typing import Union

import numpy as np
import pymesh


class Plane:
    def __init__(self, a, b, c, d):
        # ax + by + cz + d = 0
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
    # print("x3 =", x3)
    # print("y3 =", y3)
    # print("z3 =", z3)
    return np.array([x3, y3, z3])


def mirror_and_join_mesh(mesh: Union[pymesh.Mesh, tuple], plane: Plane):
    if isinstance(mesh, pymesh.Mesh):
        old_verts = mesh.vertices
        old_faces = mesh.faces
    elif isinstance(mesh, tuple):
        assert len(mesh) == 2
        old_verts, old_faces = mesh
    eps = 1e-6
    num_original_verts = old_verts.shape[0]
    num_original_faces = old_faces.shape[0]
    old_normals = get_normals(old_verts, old_faces)

    # i don't like that the faces are constructed by the vertex index
    # since we'll be messing with those a bunch
    # Change it to a dict of strings
    V = {}
    for i, vert in enumerate(old_verts):
        V["v_" + str(i)] = vert
    F = defaultdict(list)
    for i, face in enumerate(old_faces):
        for vid in face:
            F["f_" + str(i)].append("v_" + str(vid))
    VM = {}
    for i, vert in enumerate(old_verts):
        VM["v_" + str(i) + "_m"] = mirror_point(vert, plane)
    FM = defaultdict(list)
    for i, face in enumerate(old_faces):
        for vid in face[::-1]:
            FM["f_" + str(i) + "_m"].append("v_" + str(vid) + "_m")

    V2F = defaultdict(list)
    for name, face in F.items():
        for vert_name in face:
            V2F[vert_name].append(name)

    VM2FM = defaultdict(list)
    for name, face in FM.items():
        for vert_name in face:
            VM2FM[vert_name].append(name)

    for v_m_name, v_m_pt in list(VM.items()):
        orig_pt_name = v_m_name.rstrip("_m")
        orig_pt = V[orig_pt_name]
        if np.linalg.norm(orig_pt - v_m_pt) <= eps:
            VM.pop(v_m_name)
            for associated_face in VM2FM[v_m_name]:
                for j, vert_name in enumerate(FM[associated_face]):
                    if vert_name == v_m_name:
                        FM[associated_face][j] = orig_pt_name

    V.update(VM)
    F.update(FM)
    verts = []
    V2ID = {}
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

    mirrored_verts = np.empty_like(old_verts)
    # new_faces =

    associated_faces = defaultdict(list)
    for i, face in enumerate(old_faces):
        for vid in face:
            associated_faces[vid].append(i)

    # Mirror the points
    for i, vert in enumerate(old_verts):
        mirrored_verts[i] = mirror_point(vert, plane)
    all_verts = np.vstack([old_verts, mirrored_verts])
    # Mirror the faces
    mirrored_faces = old_faces[:, ::-1] + num_original_verts
    all_faces = np.vstack([old_faces, mirrored_faces])
    # Detect duplicated points on the boundary
    bad_vert_ids = []
    for i in range(num_original_verts):
        mirrored_index = i + num_original_verts
        # Don't double-add verts that were on the mirror boundary
        if np.linalg.norm(old_verts[i] - mirrored_verts[i]) <= eps:
            bad_vert_ids.append(mirrored_index)
            for face_id in associated_faces[i]:
                face = all_faces[face_id]
                for j, vert_id in face:
                    if vert_id == i:
                        face[j] = mirrored_index

    # Need to update the mirrored faces to use the original point rather than the mirror
    for vid in bad_vert_ids:
        # Delete this vertex from the vertices
        all_verts[vid] = np.nan * all_verts[vid]
        # Change the new faces to use its equivalent from the original meshre

    # Properly delete thigs from the numpy array

    verts = []
    faces = []

    for i in range(num_original_faces):
        normal = old_normals[i]
        center = old_faces[i]
        # Delete faces that were on the mirror boundary
        face_is_parallel_to_boundary = (
            abs(np.dot(old_normals[i], plane.normal)) >= 1 - eps
        )
        face_is_on_boundary = (
            np.dot(
                plane.normal, plane.origin - get_face_center(old_verts, old_faces[i])
            )
            <= eps
        )
        if face_is_parallel_to_boundary and face_is_on_boundary:
            continue  # Don't use this face
        faces.append(old_faces[i])
        faces.append()


def check_mesh(mesh: pymesh.Mesh):
    # # Delete faces that were on the mirror boundary
    # face_is_parallel_to_boundary = (
    #     abs(np.dot(old_normals[i], plane.normal)) >= 1 - eps
    # )
    # face_is_on_boundary = (
    #     np.dot(
    #         plane.normal, plane.origin - get_face_center(old_verts, old_faces[i])
    #     )
    #     <= eps
    # )
    # if face_is_parallel_to_boundary and face_is_on_boundary:
    #     continue  # Don't use this face
    pass


def get_normals(verts: np.ndarray, faces: np.ndarray):
    normals = np.zeros(faces.shape)
    for i, face in enumerate(faces):
        vert_a = verts[face[0]]
        vert_b = verts[face[1]]
        vert_c = verts[face[2]]
        normals[i] = np.cross(vert_b - vert_a, vert_c - vert_a)
    return normals


def main():
    filename = "/home/dan/Downloads/top_handle_quarter_nonmanifold.stl"
    mesh = pymesh.load_mesh(filename)
    vertices, faces = mirror_and_join_mesh(mesh, YZ_PLANE)
    vertices, faces = mirror_and_join_mesh((vertices, faces), XZ_PLANE)
    # num_duplicates = 0
    # for i in range(len(vertices)):
    #     for j in range(i + 1, len(vertices)):
    #         if np.linalg.norm(vertices[i] - vertices[j]) < 1e-5:
    #             num_duplicates += 1
    # print("DUPLICATES: ", num_duplicates)

    # Mirror them a second time about the XZ plane
    # verts = mesh.vertices
    # faces = mesh.faces  # shape num_faces, 3, each row is vertex indices
    # normals = get_normals(verts, faces)

    pymesh.save_mesh_raw(
        "/home/dan/Downloads/top_handle_symmetric.obj", vertices, faces
    )


if __name__ == "__main__":
    main()
