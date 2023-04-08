"""Example script to just load the Astrobee with a softbody cargo bag

This is mainly to validate that the dynamics of the bag attached to the Astrobee "look correct"
if we manually click + drag to interact with the objects

More robust validation techniques will be a future TODO
"""

import os
from typing import Optional

import numpy as np
import pybullet

import pyastrobee.config.bag_properties as bag_props
from pyastrobee.control.astrobee import Astrobee
from pyastrobee.utils.bullet_utils import load_deformable_object, create_anchor
from pyastrobee.utils.mesh_utils import get_mesh_data, get_closest_mesh_vertex
from pyastrobee.utils.poses import pos_quat_to_tmat
from pyastrobee.utils.rotations import rmat_to_quat


class BagProperties:
    pass


class Mesh:
    def __init__(self, filename: str):
        # self.id = id
        self.filename = filename
        # Initialize property attributes
        self._num_mesh_vertices = None
        self._mesh_vertices = None

    @property
    def num_mesh_vertices(self):
        if self.id is None:
            raise AttributeError("Mesh has not been loaded")
        if self._num_mesh_vertices is None:
            self._num_mesh_vertices, self._mesh_vertices = get_mesh_data(self.id)
        return self._num_mesh_vertices

    @property
    def mesh_vertices(self):
        if self.id is None:
            raise AttributeError("Mesh has not been loaded")
        if self._mesh_vertices is None:
            self._num_mesh_vertices, self._mesh_vertices = get_mesh_data(self.id)
        return self._mesh_vertices

    @property
    def mesh_type(self):
        file_ext = os.path.splitext(self.filename)[1]
        if file_ext in {".obj", ".stl"}:
            return "triangular"
        elif file_ext == ".vtk":
            return "tetrahedral"
        raise ValueError(f"Unknown filetype for file {self.filename}")


# AAAH if we want to load a cargo bag properly
class SoftBody(Mesh):
    def __init__(
        self,
        id,
        filename,
        texture,
        mass,
        bending_stiffness,
        damping_stiffness,
        elastic_stiffness,
        friction_coeff,
    ):
        super().__init__(filename)
        self.texture = texture
        self.mass = mass
        self.bending_stiffness = bending_stiffness
        self.damping_stiffness = damping_stiffness
        self.elastic_stiffness = elastic_stiffness
        self.friction_coeff = friction_coeff

    def load(self, pos, orn):
        self_collision = False
        scale = 1
        self.id = load_deformable_object(
            self.filename,
            self.texture,
            scale,
            pos,
            orn,
            self.mass,
            self.bending_stiffness,
            self.damping_stiffness,
            self.elastic_stiffness,
            self.friction_coeff,
            self_collision,
        )

    # def load(self):
    # self.id = load_deformable_object()


# class RigidBodyMesh(Mesh):
#     def __init__(self):
#         super().__init__()
#         self.mass
#         # self.......
#     pass

# TODO figure out a better way than using the magic strings?


MESH_DIR = "pyastrobee/assets/meshes/bags/"
BAG_NAMES = ["front_handle_bag", "side_handle_bag", "top_handle_bag"]
_objs = [MESH_DIR + name + ".obj" for name in BAG_NAMES]
_vtks = [MESH_DIR + name + ".vtk" for name in BAG_NAMES]
OBJS = dict(zip(BAG_NAMES, _objs))
VTKS = dict(zip(BAG_NAMES, _vtks))


def load_cargo_bag(bag_name, pos, orn) -> int:
    # TODO decide if we want to use an object-orientated CargoBag (for now, let's not)
    _validate_bag_name(bag_name)
    obj_file = OBJS[bag_name]
    vtk_file = VTKS[bag_name]
    # We will apply texture via the OBJ/MTL/VTK combo rather than applying a PNG file
    # TODO fix the UV mapping of the VTKs so this process is easier
    texture = None
    scale = 1
    self_collision = False
    bag_id = load_deformable_object(
        obj_file,
        texture,
        scale,
        pos,
        orn,
        bag_props.MASS,
        bag_props.BENDING_STIFFNESS,
        bag_props.DAMPING_STIFFNESS,
        bag_props.ELASTIC_STIFFNESS,
        bag_props.FRICTION_COEFF,
        self_collision,
        vtk_file,
    )
    return bag_id


def _validate_bag_name(name) -> None:
    if name not in BAG_NAMES:
        raise ValueError(f"Invalid bag name: {name}. Must be one of {BAG_NAMES}")


def _get_grasp_transform(bag_name) -> np.ndarray:
    _validate_bag_name(bag_name)
    if bag_name == "front_handle_bag":
        return bag_props.FRONT_BAG_GRASP_TRANSFORM
    elif bag_name == "side_handle_bag":
        return bag_props.SIDE_BAG_GRASP_TRANSFORM
    elif bag_name == "top_handle_bag":
        return bag_props.TOP_BAG_GRASP_TRANSFORM
    else:
        raise LookupError(f"Could not find the associated transform for {bag_name}")


def load_and_attach_bag(robot: Astrobee, bag_name: str):
    # Determine the transformation which dictates where we need to load the bag
    # for the handle to line up with the gripper
    T_ee2world = pos_quat_to_tmat(robot3.ee_pose)
    T_bag2ee = _get_grasp_transform(bag_name)
    T_bag2world = T_ee2world @ T_bag2ee
    # Load the bag at the position/orientation specified by the transform
    pos = T_bag2world[:3, 3]
    orn = rmat_to_quat(T_bag2world[:3, :3])
    bag_id = load_cargo_bag(bag_name, pos, orn)
    # Generate the constraints between the bag and the robot
    # Find the points on the mesh on either side of the handle
    # (Using the left/right gripper link frames as reference points)
    n_vert, bag_mesh = get_mesh_data(bag_id)
    pos_1 = robot.get_link_transform(robot.Links.GRIPPER_LEFT_DISTAL)[:3, 3]
    pos_2 = robot.get_link_transform(robot.Links.GRIPPER_RIGHT_DISTAL)[:3, 3]
    v1_pos, v1_id = get_closest_mesh_vertex(pos_1, bag_mesh)
    v2_pos, v2_id = get_closest_mesh_vertex(pos_2, bag_mesh)
    # (TODO) decide if it makes more sense to anchor to the gripper fingers themselves
    # or if anchoring to the "palm" (the arm distal link) is ok
    link_to_anchor = robot.Links.ARM_DISTAL.value
    anchor1_id, geom1_id = create_anchor(
        bag_id, v1_id, robot.id, link_to_anchor, add_geom=True, geom_pos=v1_pos
    )
    anchor2_id, geom2_id = create_anchor(
        bag_id, v2_id, robot.id, link_to_anchor, add_geom=True, geom_pos=v2_pos
    )
    anchor_ids = (anchor1_id, anchor2_id)
    geom_ids = (geom1_id, geom2_id)
    return bag_id, anchor_ids, geom_ids


class CargoBag:
    # Store all filenames in here??
    def __init__(self, bag_name: str, pos, orn):
        _validate_bag_name(bag_name)  # Check if this is a redundant check
        # TODO make load_cargo_bag a method of this class
        self.obj_file = OBJS[bag_name]
        self.vtk_file = VTKS[bag_name]
        # self.mass = bag_props.MASS
        # self.bending_stiffness = bag_props.BENDING_STIFFNESS
        # self.damping_stiffness = bag_props.DAMPING_STIFFNESS
        # self.elastic_stiffness = bag_props.ELASTIC_STIFFNESS
        # self.friction_coeff = bag_props.FRICTION_COEFF

        # super().__init__(id, VTK_FILENAMES[bag_name], bag_props.MASS, bag_props.BENDING_STIFFNESS, bag_props.DAMPING_STIFFNESS, bag_props.ELASTIC_STIFFNESS, bag_props.FRICTION_COEFF)
        if bag_name == "front_handle":
            self.handle_transform = bag_props.FRONT_BAG_GRASP_TRANSFORM
        elif bag_name == "side_handle":
            self.handle_transform = bag_props.SIDE_BAG_GRASP_TRANSFORM
        else:  # Top handle
            self.handle_transform = bag_props.TOP_BAG_GRASP_TRANSFORM
        self.anchors = []
        self.anchor_objects = []
        self.id = self._load(pos, orn)

    def _load(self, pos, orn):
        texture = None
        scale = 1
        self_collision = False
        bag_id = load_deformable_object(
            self.obj_file,
            texture,
            scale,
            pos,
            orn,
            self.mass,
            self.bending_stiffness,
            self.damping_stiffness,
            self.elastic_stiffness,
            self.friction_coeff,
            self_collision,
            self.vtk_file,
        )
        return bag_id

    # Read-only physical properties defined at initialization of the bag
    @property
    def num_mesh_vertices(self):
        if self.id is None:
            raise AttributeError("Mesh has not been loaded")
        if self._num_mesh_vertices is None:
            self._num_mesh_vertices, self._mesh_vertices = get_mesh_data(self.id)
        return self._num_mesh_vertices

    @property
    def mesh_vertices(self):
        if self.id is None:
            raise AttributeError("Mesh has not been loaded")
        if self._mesh_vertices is None:
            self._num_mesh_vertices, self._mesh_vertices = get_mesh_data(self.id)
        return self._mesh_vertices

    @property
    def mass(self):
        return bag_props.MASS

    @property
    def bending_stiffness(self):
        return bag_props.BENDING_STIFFNESS

    @property
    def damping_stiffness(self):
        return bag_props.DAMPING_STIFFNESS

    @property
    def elastic_stiffness(self):
        return bag_props.ELASTIC_STIFFNESS

    @property
    def friction_coeff(self):
        return bag_props.FRICTION_COEFF

    # @property
    # def mesh_type(self):
    #     file_ext = os.path.splitext(self.filename)[1]
    #     if file_ext in {".obj", ".stl"}:
    #         return "triangular"
    #     elif file_ext == ".vtk":
    #         return "tetrahedral"
    #     raise ValueError(f"Unknown filetype for file {self.filename}")


# Move to CargoBag
def detach_bag(anchors: list[int], anchor_objects: Optional[list[int]]):
    """Remove any currently attached anchors from the bag (and any associated visual geometry)

    Args:
        bag (CargoBag): The bag to detach all anchors from
    """
    for cid in anchors:
        pybullet.removeConstraint(cid)
    if anchor_objects is not None:
        for obj in anchor_objects:
            pybullet.removeBody(obj)
    # bag.anchors = None
    # bag.anchor_objects = None


bag
robot3 = Astrobee(pose=[-1, -1, 1, *random_quaternion()])
EE2W_3 = pos_quat_to_tmat(robot3.ee_pose)
TOP_BAG_TO_WORLD = EE2W_3 @ TOP_BAG_TO_EE
load_rigid_object(
    top_file, pos=TOP_BAG_TO_WORLD[:3, 3], orn=rmat_to_quat(TOP_BAG_TO_WORLD[:3, :3])
)


# Should the bag be initialized on creation of the CargoBag

# From the demo: TODO put this in a better place
# Also TODO determine if it's better to anchor the bag to the arm distal link or the gripper
def load_and_attach_bag(robot: Astrobee, side=0):

    # Load deformable bag and attach the middle of each side of the handle to
    # the middle of each of the astrobee fingers.
    pfx = "pyastrobee/assets/meshes/bags/"
    fnames = ["front_handle_bag.vtk", "side_handle_bag.vtk", "top_handle_bag.vtk"]

    poss = np.array([[-0.05, 0.00, -0.53], [-0.05, 0.00, -0.65]])  # z=-0.53  -0.48
    orns = np.array([[-np.pi / 2, 0, 0], [0, -np.pi / 2, 0]])
    # Remember to update the orientations here to use quaternions
    # Actually nvm all of the above lines should be irrelevant now
    print("poss[side]", poss[side], "orns[side]", orns[side])
    bag_id = load_deformable_object(
        os.path.join(pfx, fnames[side]),
        pos=poss[side],
        orn=orns[side],
        bending_stiffness=50,
        elastic_stiffness=50,
        mass=1.0,
    )
    bag_texture_id = pybullet.loadTexture(
        "pyastrobee/assets/imgs/textile_pixabay_red.jpg"
    )
    kwargs = {}
    if hasattr(pybullet, "VISUAL_SHAPE_DOUBLE_SIDED"):
        kwargs["flags"] = pybullet.VISUAL_SHAPE_DOUBLE_SIDED
    pybullet.changeVisualShape(
        bag_id, -1, rgbaColor=[1, 1, 1, 1], textureUniqueId=bag_texture_id, **kwargs
    )
    n_vert, bag_mesh = get_mesh_data(bag_id)
    finger1_link_id = 4
    finger2_link_id = 6
    finger1_pos = pybullet.getLinkState(robot.id, finger1_link_id)[
        0
    ]  # Use the Astrobee() functions?
    finger2_pos = pybullet.getLinkState(robot.id, finger2_link_id)[0]
    v1_pos, v1_id = get_closest_mesh_vertex(finger1_pos, bag_mesh)
    v2_pos, v2_id = get_closest_mesh_vertex(finger2_pos, bag_mesh)
    anchor1_id, _ = create_anchor(
        bag_id, v1_id, robot.id, finger1_link_id, add_geom=True, geom_pos=v1_pos
    )
    anchor2_id, _ = create_anchor(
        bag_id, v2_id, robot.id, finger2_link_id, add_geom=True, geom_pos=v2_pos
    )


if __name__ == "__main__":
    robot = Astrobee()
    ee_pose = robot.ee_pose
    # Attach the bag based on this ee pose
