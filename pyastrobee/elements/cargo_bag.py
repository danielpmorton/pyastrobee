"""Class for handling loading the cargo bag and managing attributes such as physical properties

TODO decide if the bag_props import can be handled better
TODO decide if the constants should be moved to class attributes
TODO decide if there is a better way to handle the "attached vs freefloating" initialization
TODO decide if we should anchor to the gripper fingers or the arm distal link (currently distal)
TODO decide if we should set up the class attributes to handle multiple bags loaded in sim
    (Like the Astrobee class)
"""
import time
from typing import Optional

import pybullet
import numpy as np
import numpy.typing as npt

import pyastrobee.config.bag_properties as bag_props
from pyastrobee.elements.astrobee import Astrobee
from pyastrobee.utils.bullet_utils import (
    load_deformable_object,
    create_anchor,
    initialize_pybullet,
)
from pyastrobee.utils.mesh_utils import get_mesh_data, get_closest_mesh_vertex
from pyastrobee.utils.poses import pos_quat_to_tmat
from pyastrobee.utils.rotations import rmat_to_quat
from pyastrobee.utils.python_utils import print_green

# Constants. TODO Move these to class attributes?
MESH_DIR = "pyastrobee/assets/meshes/bags/"
BAG_NAMES = ["front_handle_bag", "side_handle_bag", "top_handle_bag"]
_objs = [MESH_DIR + name + ".obj" for name in BAG_NAMES]
_vtks = [MESH_DIR + name + ".vtk" for name in BAG_NAMES]
OBJS = dict(zip(BAG_NAMES, _objs))
VTKS = dict(zip(BAG_NAMES, _vtks))
GRASP_TRANSFORMS = dict(
    zip(
        BAG_NAMES,
        [
            bag_props.FRONT_BAG_GRASP_TRANSFORM,
            bag_props.SIDE_BAG_GRASP_TRANSFORM,
            bag_props.TOP_BAG_GRASP_TRANSFORM,
        ],
    )
)


class CargoBag:
    """Class for loading and managing properties associated with the cargo bags

    Args:
        bag_name (str): Type of cargo bag to load. One of "front_handle_bag", "side_handle_bag", "top_handle_bag"
        attached_robot (Optional[Astrobee], optional): If initially attaching the cargo bag to a robot, include
            the robot here. Defaults to None, in which case pos/orn must be specified
        pos (Optional[npt.ArrayLike], optional): If not attaching to a robot, this is the XYZ position to load
            the bag. Defaults to None, in which case the robot must be speficied
        orn (Optional[npt.ArrayLike], optional): If not attaching to a robot, this is the XYZW quaternion to load
            the bag. Defaults to None, in which case the robot must be speficied
    """

    def __init__(
        self,
        bag_name: str,
        attached_robot: Optional[Astrobee] = None,
        pos: Optional[npt.ArrayLike] = None,
        orn: Optional[npt.ArrayLike] = None,
    ):
        # Validate inputs
        if bag_name not in BAG_NAMES:
            raise ValueError(
                f"Invalid bag name: {bag_name}. Must be one of {BAG_NAMES}"
            )
        if attached_robot is not None:
            if not isinstance(attached_robot, Astrobee):
                raise ValueError("Invalid robot input. Must be an Astrobee() instance")
        else:
            if pos is None or orn is None:
                raise ValueError(
                    "Must provide a position and orientation to load the bag if not attaching to a robot"
                )
        self._name = bag_name
        # Initializations
        self._anchors = []
        self._anchor_objects = []
        self._mesh_vertices = None
        self._num_mesh_vertices = None
        # Load the bag depending on the specified method
        if attached_robot is not None:
            self._load_attached(attached_robot)
        else:
            self._load(pos, orn)
        print_green("Bag is ready")

    # Read-only physical properties defined at initialization of the bag
    @property
    def num_mesh_vertices(self) -> int:
        if self.id is None:
            raise AttributeError("Mesh has not been loaded")
        if self._num_mesh_vertices is None:
            self._num_mesh_vertices, self._mesh_vertices = get_mesh_data(self.id)
        return self._num_mesh_vertices

    @property
    def mesh_vertices(self) -> np.ndarray:
        if self.id is None:
            raise AttributeError("Mesh has not been loaded")
        if self._mesh_vertices is None:
            self._num_mesh_vertices, self._mesh_vertices = get_mesh_data(self.id)
        return self._mesh_vertices

    @property
    def mass(self) -> float:
        return bag_props.MASS

    @property
    def bending_stiffness(self) -> float:
        return bag_props.BENDING_STIFFNESS

    @property
    def damping_stiffness(self) -> float:
        return bag_props.DAMPING_STIFFNESS

    @property
    def elastic_stiffness(self) -> float:
        return bag_props.ELASTIC_STIFFNESS

    @property
    def friction_coeff(self) -> float:
        return bag_props.FRICTION_COEFF

    @property
    def anchors(self) -> tuple[list[int], list[int]]:
        return (self._anchors, self._anchor_objects)

    @property
    def attached_robot(self) -> int:
        return self._attached_robot

    @property
    def name(self) -> str:
        return self._name

    @property
    def obj_file(self) -> str:
        return OBJS[self._name]

    @property
    def vtk_file(self) -> str:
        return VTKS[self._name]

    @property
    def grasp_transform(self) -> np.ndarray:
        return GRASP_TRANSFORMS[self._name]

    def _load(self, pos: npt.ArrayLike, orn: npt.ArrayLike) -> None:
        """Loads a cargo bag at the specified position/orientation

        Args:
            pos (npt.ArrayLike): XYZ position, shape (3,)
            orn (npt.ArrayLike): XYZW quaternion, shape (4,)
        """
        texture = None
        scale = 1
        self_collision = False
        self.id = load_deformable_object(
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
        self._attached_robot = None

    def _load_attached(self, robot: Astrobee) -> None:
        """Load the cargo bag attached to the gripper of an Astrobee

        Args:
            robot (Astrobee): The robot to attach the bag to
        """
        # Determine the transformation which dictates where we need to load the bag
        # for the handle to line up with the gripper
        T_ee2world = pos_quat_to_tmat(robot.ee_pose)
        T_bag2ee = self.grasp_transform
        T_bag2world = T_ee2world @ T_bag2ee
        # Load the bag at the position/orientation specified by the transform
        pos = T_bag2world[:3, 3]
        orn = rmat_to_quat(T_bag2world[:3, :3])
        self._load(pos, orn)
        # Generate the constraints between the bag and the robot
        # Find the points on the mesh on either side of the handle
        # (Using the left/right gripper link frames as reference points)
        pos_1 = robot.get_link_transform(robot.Links.GRIPPER_LEFT_DISTAL)[:3, 3]
        pos_2 = robot.get_link_transform(robot.Links.GRIPPER_RIGHT_DISTAL)[:3, 3]
        v1_pos, v1_id = get_closest_mesh_vertex(pos_1, self.mesh_vertices)
        v2_pos, v2_id = get_closest_mesh_vertex(pos_2, self.mesh_vertices)
        # (TODO) decide if it makes more sense to anchor to the gripper fingers themselves
        # or if anchoring to the "palm" (the arm distal link) is ok
        link_to_anchor = robot.Links.ARM_DISTAL.value
        anchor1_id, geom1_id = create_anchor(
            self.id, v1_id, robot.id, link_to_anchor, add_geom=True, geom_pos=v1_pos
        )
        anchor2_id, geom2_id = create_anchor(
            self.id, v2_id, robot.id, link_to_anchor, add_geom=True, geom_pos=v2_pos
        )
        self._anchors = (anchor1_id, anchor2_id)
        self._anchor_objects = (geom1_id, geom2_id)
        self._attached_robot = robot.id

    def detach(self) -> None:
        """Remove any currently attached anchors from the bag (and any associated visual geometry)"""
        for cid in self._anchors:
            pybullet.removeConstraint(cid)
        for obj in self._anchor_objects:
            pybullet.removeBody(obj)
        self._anchors = []
        self._anchor_objects = []
        self._attached_robot = None

    def unload(self) -> None:
        """Removes the cargo bag from the simulation"""
        pybullet.removeBody(self.id)


if __name__ == "__main__":
    initialize_pybullet()
    robo = Astrobee()
    bag = CargoBag("top_handle_bag", robo)
    init_time = time.time()
    time_lim = 10
    print(f"Provide a disturbance force. Detaching bag in {time_lim} seconds")
    while time.time() - init_time < 10:
        pybullet.stepSimulation()
        time.sleep(1 / 120)
    bag.detach()
    while True:
        pybullet.stepSimulation()
        time.sleep(1 / 120)
