"""Deformable cargo bag, implemented via a tetrahedral (volumetric) mesh

Documentation for inherited methods can be found in the base class
"""

# TODO !! Fix the softbody velocity issue
# TODO ! the unloading mechanic is currently broken since it only removes the softbody and not the visual
# TODO If the reliability of the softbody position/orientation is not good, use the get_bag_frame() function I made
# TODO decide if the bag_props import can be handled better

import time
from typing import Optional

from pybullet_utils.bullet_client import BulletClient
import numpy as np
import numpy.typing as npt

from pyastrobee.core.abstract_bag import CargoBag
import pyastrobee.config.bag_properties as bag_props
from pyastrobee.core.astrobee import Astrobee
from pyastrobee.utils.bullet_utils import (
    load_deformable_object,
    create_anchor,
    initialize_pybullet,
)
from pyastrobee.utils.mesh_utils import get_mesh_data, get_closest_mesh_vertex
from pyastrobee.utils.python_utils import print_green, flatten
from pyastrobee.utils.quaternions import quats_to_angular_velocities


class DeformableCargoBag(CargoBag):
    """Class for loading and managing properties associated with the deformable cargo bags

    Args:
        bag_name (str): Type of cargo bag to load. Single handle: "front_handle", "right_handle", "top_handle".
            Dual handle: "front_back_handle", "right_left_handle", "top_bottom_handle"
        mass (float): Mass of the cargo bag, in kg
        pos (npt.ArrayLike, optional): Initial XYZ position to load the bag. Defaults to (0, 0, 0)
        orn (npt.ArrayLike, optional): Initial XYZW quaternion to load the bag. Defaults to (0, 0, 0, 1)
        client (BulletClient, optional): If connecting to multiple physics servers, include the client
            (the class instance, not just the ID) here. Defaults to None (use default connected client)
    """

    _objs = [CargoBag.MESH_DIR + name + ".obj" for name in CargoBag.BAG_NAMES]
    _vtks = [CargoBag.MESH_DIR + name + ".vtk" for name in CargoBag.BAG_NAMES]
    OBJS = dict(zip(CargoBag.BAG_NAMES, _objs))
    VTKS = dict(zip(CargoBag.BAG_NAMES, _vtks))
    BAG_CORNER_VERTS = {
        "front_handle": bag_props.FRONT_HANDLE_BAG_CORNERS,
        "right_handle": bag_props.RIGHT_HANDLE_BAG_CORNERS,
        "top_handle": bag_props.TOP_HANDLE_BAG_CORNERS,
        "front_back_handle": bag_props.FRONT_BACK_HANDLE_BAG_CORNERS,
        "right_left_handle": bag_props.RIGHT_LEFT_HANDLE_BAG_CORNERS,
        "top_bottom_handle": bag_props.TOP_BOTTOM_HANDLE_BAG_CORNERS,
    }

    def __init__(
        self,
        bag_name: str,
        mass: float,
        pos: npt.ArrayLike = (0, 0, 0),
        orn: npt.ArrayLike = (0, 0, 0, 1),
        client: Optional[BulletClient] = None,
    ):
        super().__init__(bag_name, mass, pos, orn, client)
        # Initializations
        self._anchors = {}
        self._anchor_objects = {}
        self._mesh_vertices = None
        self._num_mesh_vertices = None
        print_green("Bag is ready")

    # Read-only physical properties defined at initialization of the bag
    @property
    def num_mesh_vertices(self) -> int:
        """Number of vertices in the bag's mesh"""
        if self.id is None:
            raise AttributeError("Mesh has not been loaded")
        if self._num_mesh_vertices is None:
            self._num_mesh_vertices, self._mesh_vertices = get_mesh_data(
                self.id, self.client
            )
        return self._num_mesh_vertices

    @property
    def mesh_vertices(self) -> np.ndarray:
        """Positions of the mesh vertices, shape (n, 3)"""
        if self.id is None:
            raise AttributeError("Mesh has not been loaded")
        self._num_mesh_vertices, self._mesh_vertices = get_mesh_data(
            self.id, self.client
        )
        return self._mesh_vertices

    @property
    def bending_stiffness(self) -> float:
        """Softbody bending stiffness parameter"""
        return bag_props.BENDING_STIFFNESS

    @property
    def damping_stiffness(self) -> float:
        """Softbody damping stiffness parameter"""
        return bag_props.DAMPING_STIFFNESS

    @property
    def elastic_stiffness(self) -> float:
        """Softbody elastic stiffness parameter"""
        return bag_props.ELASTIC_STIFFNESS

    @property
    def friction_coeff(self) -> float:
        """Softbody friction coefficient"""
        return bag_props.FRICTION_COEFF

    @property
    def anchors(self) -> tuple[list[int], list[int]]:
        """Anchor IDs and IDs of their associated visual geometries"""
        # Unpack the list of lists in the anchor dictionaries
        return flatten(self._anchors.values()), flatten(self._anchor_objects.values())

    @property
    def obj_file(self) -> str:
        """Path to the .OBJ triangular mesh file"""
        return self.OBJS[self._name]

    @property
    def vtk_file(self) -> str:
        """Path to the .VTK tetrahedral mesh file"""
        return self.VTKS[self._name]

    # NOTE: Any methods associated with velocity or angular velocity have special handling for the deformable

    @property
    def velocity(self) -> np.ndarray:
        """Current [vx, vy, vz] velocity of the cargo bag's COM frame

        - If both velocity and angular velocity are desired, use the dynamics_state property instead
        """
        return self.dynamics_state[2]

    @property
    def angular_velocity(self) -> np.ndarray:
        """Current [wx, wy, wz] angular velocity of the cargo bag's COM frame

        - If both velocity and angular velocity are desired, use the dynamics_state property instead
        """
        return self.dynamics_state[3]

    @property
    def dynamics_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Current state of the bag dynamics: Position, orientation, linear vel, and angular vel

        - NOTE this moves the simulation forward by 1 step
        - Bullet's velocity definition for softbodies is incorrect or not implemented. I've tried
          implementing this myself in the C++, but it is also not reliable. So instead, we'll need to step
          the sim in this call to do our own calculations

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                np.ndarray: Position, shape (3,)
                np.ndarray: XYZW quaternion orientation, shape (4,)
                np.ndarray: Linear velocity, shape (3,)
                np.ndarray: Angular velocity, shape (3,)
        """
        old_pos, old_orn = self.client.getBasePositionAndOrientation(self.id)
        # Step the sim to get a second reference frame we can use to determine velocity
        # This is not ideal, but it's the best way to do this until Pybullet's getBaseVelocity for softbodies works
        self.client.stepSimulation()
        new_pos, new_orn = self.client.getBasePositionAndOrientation(self.id)
        lin_vel = np.subtract(new_pos, old_pos) / self._dt
        ang_vel = quats_to_angular_velocities(
            np.row_stack([old_orn, new_orn]), self._dt
        )
        if ang_vel.ndim > 1:
            ang_vel = ang_vel[0, :]
        # Return the stepped-ahead position since it's the most recent state we know about
        return (
            np.array(new_pos),
            np.array(new_orn),
            np.array(lin_vel),
            np.array(ang_vel),
        )

    # TODO decide if the non-mesh-based implementation from CargoBag works with the deformable??
    @property
    def corner_positions(self) -> list[np.ndarray]:
        return self.mesh_vertices[self.BAG_CORNER_VERTS[self.name]]

    def _load(self, pos: npt.ArrayLike, orn: npt.ArrayLike) -> int:
        texture = None
        scale = 1
        self_collision = False
        return load_deformable_object(
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
            self.client,
        )

    def _attach(self, robot: Astrobee, handle_index: int) -> None:
        del handle_index  # Unused for the deformable bag
        # Generate the constraints between the bag and the robot: First, find the points on the mesh on either side
        # of the handle (Using the left/right gripper link frames as reference points), then create the anchors
        pos_1 = robot.get_link_transform(robot.Links.GRIPPER_LEFT_DISTAL)[:3, 3]
        pos_2 = robot.get_link_transform(robot.Links.GRIPPER_RIGHT_DISTAL)[:3, 3]
        v1_pos, v1_id = get_closest_mesh_vertex(pos_1, self.mesh_vertices)
        v2_pos, v2_id = get_closest_mesh_vertex(pos_2, self.mesh_vertices)
        anchor1_id, geom1_id = create_anchor(
            self.id,
            v1_id,
            robot.id,
            robot.Links.ARM_DISTAL.value,
            add_geom=True,
            geom_pos=v1_pos,
            client=self.client,
        )
        anchor2_id, geom2_id = create_anchor(
            self.id,
            v2_id,
            robot.id,
            robot.Links.ARM_DISTAL.value,
            add_geom=True,
            geom_pos=v2_pos,
            client=self.client,
        )
        self._anchors.update({robot.id: [anchor1_id, anchor2_id]})
        self._anchor_objects.update({robot.id: [geom1_id, geom2_id]})
        self._attached.append(robot.id)

    def detach(self) -> None:
        anchors, anchor_objects = self.anchors
        for cid in anchors:
            self.client.removeConstraint(cid)
        for obj in anchor_objects:
            self.client.removeBody(obj)
        self._anchors = {}
        self._anchor_objects = {}
        self._attached = []

    def detach_robot(self, robot_id: int) -> None:
        if robot_id not in self.attached:
            raise ValueError("Cannot detach robot: ID unknown")
        for cid in self._anchors[robot_id]:
            self.client.removeConstraint(cid)
        for obj in self._anchor_objects[robot_id]:
            self.client.removeBody(obj)
        self._anchors.pop(robot_id)
        self._anchor_objects.pop(robot_id)
        self._attached.remove(robot_id)


def _main():
    # Very simple example of loading the bag and attaching a robot
    client = initialize_pybullet()
    robot = Astrobee()
    bag = DeformableCargoBag("top_handle", 10)
    bag.attach_to(robot)
    while True:
        client.stepSimulation()
        time.sleep(1 / 120)


if __name__ == "__main__":
    _main()
