"""Class for handling loading the cargo bag and managing attributes such as physical properties

TODO !! Fix the softbody velocity issue
TODO ! the unloading mechanic is currently broken since it only removes the softbody and not the visual
TODO If the reliability of the softbody position/orientation is not good, use the get_bag_frame() function I made
TODO decide if the bag_props import can be handled better
TODO decide if the constants should be moved to class attributes
TODO decide if we should anchor to the gripper fingers or the arm distal link (currently distal)
TODO decide if we should set up the class attributes to handle multiple bags loaded in sim
    (Like the Astrobee class)
"""
import time
from typing import Union

import pybullet
import numpy as np
import numpy.typing as npt

import pyastrobee.config.bag_properties as bag_props
from pyastrobee.core.astrobee import Astrobee
from pyastrobee.utils.bullet_utils import (
    load_deformable_object,
    create_anchor,
    initialize_pybullet,
)
from pyastrobee.utils.mesh_utils import get_mesh_data, get_closest_mesh_vertex
from pyastrobee.utils.poses import pos_quat_to_tmat, tmat_to_pos_quat
from pyastrobee.utils.python_utils import print_green, flatten
from pyastrobee.utils.quaternions import quats_to_angular_velocities

# Constants. TODO Move these to class attributes?
MESH_DIR = "pyastrobee/assets/meshes/bags/"
# TODO: update the bag naming
SINGLE_HANDLE_BAGS = ["front_handle_bag", "side_handle_bag", "top_handle_bag"]
DUAL_HANDLE_BAGS = ["front_back_handle", "side_side_handle", "top_bottom_handle"]
BAG_NAMES = SINGLE_HANDLE_BAGS + DUAL_HANDLE_BAGS
_objs = [MESH_DIR + name + ".obj" for name in BAG_NAMES]
_vtks = [MESH_DIR + name + ".vtk" for name in BAG_NAMES]
OBJS = dict(zip(BAG_NAMES, _objs))
VTKS = dict(zip(BAG_NAMES, _vtks))
HANDLE_TRANSFORMS = {
    "front": bag_props.FRONT_HANDLE_TRANSFORM,
    "back": bag_props.BACK_HANDLE_TRANSFORM,
    "left": bag_props.LEFT_HANDLE_TRANSFORM,
    "right": bag_props.RIGHT_HANDLE_TRANSFORM,
    "top": bag_props.TOP_HANDLE_TRANSFORM,
    "bottom": bag_props.BOTTOM_HANDLE_TRANSFORM,
}
# TODO: These need updating, they are out of date
# Also, they're unused for now, so commenting out
# BAG_CORNERS = dict(
#     zip(
#         BAG_NAMES,
#         [
#             bag_props.FRONT_BAG_CORNER_VERTS,
#             bag_props.SIDE_BAG_CORNER_VERTS,
#             bag_props.TOP_BAG_CORNER_VERTS,
#         ],
#     )
# )


class CargoBag:
    """Class for loading and managing properties associated with the cargo bags

    Args:
        bag_name (str): Type of cargo bag to load. One of "front_handle_bag", "side_handle_bag", "top_handle_bag"
        pos (npt.ArrayLike, optional): Initial XYZ position to load the bag. Defaults to [0, 0, 0]
        orn (npt.ArrayLike, optional): Initial XYZW quaternion to load the bag. Defaults to [0, 0, 0, 1]
    """

    def __init__(
        self,
        bag_name: str,
        pos: npt.ArrayLike = [0, 0, 0],
        orn: npt.ArrayLike = [0, 0, 0, 1],
    ):
        if not pybullet.isConnected():
            raise ConnectionError("Need to connect to pybullet before loading a bag")
        if bag_name not in BAG_NAMES:
            raise ValueError(
                f"Invalid bag name: {bag_name}. Must be one of {BAG_NAMES}"
            )
        self._name = bag_name
        # Initializations
        self._anchors = {}
        self._anchor_objects = {}
        self._mesh_vertices = None
        self._num_mesh_vertices = None
        self._attached = []
        self._id = None
        # Get the simulator timestep (used for calculating the velocity of the bag)
        self._dt = pybullet.getPhysicsEngineParameters()["fixedTimeStep"]
        self._load(pos, orn)
        print_green("Bag is ready")

    # Read-only physical properties defined at initialization of the bag
    @property
    def num_mesh_vertices(self) -> int:
        """Number of vertices in the bag's mesh"""
        if self.id is None:
            raise AttributeError("Mesh has not been loaded")
        if self._num_mesh_vertices is None:
            self._num_mesh_vertices, self._mesh_vertices = get_mesh_data(self.id)
        return self._num_mesh_vertices

    @property
    def mesh_vertices(self) -> np.ndarray:
        """Positions of the mesh vertices, shape (n, 3)"""
        if self.id is None:
            raise AttributeError("Mesh has not been loaded")
        self._num_mesh_vertices, self._mesh_vertices = get_mesh_data(self.id)
        return self._mesh_vertices

    @property
    def mass(self) -> float:
        """Softbody mass"""
        return bag_props.MASS

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
    def attached(self) -> list[int]:
        """ID(s) of the robot (or robots) grasping the bag. Empty if no robots are attached"""
        return self._attached

    @property
    def name(self) -> str:
        """Type of cargo bag"""
        return self._name

    @property
    def obj_file(self) -> str:
        """Path to the .OBJ triangular mesh file"""
        return OBJS[self._name]

    @property
    def vtk_file(self) -> str:
        """Path to the .VTK tetrahedral mesh file"""
        return VTKS[self._name]

    @property
    def grasp_transforms(self) -> list[np.ndarray]:
        """Transformation matrices "handle to bag" representing the grasp locations on the handles to the bag COM

        In the case of a single-handled bag, this list will only have one entry
        """
        if self._name == "front_handle_bag":
            return [HANDLE_TRANSFORMS["front"]]
        elif self._name == "side_handle_bag":
            return [HANDLE_TRANSFORMS["right"]]
        elif self._name == "top_handle_bag":
            return [HANDLE_TRANSFORMS["top"]]
        elif self._name == "front_back_handle":
            return [HANDLE_TRANSFORMS["front"], HANDLE_TRANSFORMS["back"]]
        elif self._name == "side_side_handle":
            return [HANDLE_TRANSFORMS["right"], HANDLE_TRANSFORMS["left"]]
        elif self._name == "top_bottom_handle":
            return [HANDLE_TRANSFORMS["top"], HANDLE_TRANSFORMS["bottom"]]
        else:
            raise NotImplementedError(
                f"Grasp transform(s) not available for bag: {self._name}"
            )

    @property
    def pose(self) -> np.ndarray:
        """Current position + XYZW quaternion pose of the bag's COM frame"""
        return np.concatenate(pybullet.getBasePositionAndOrientation(self.id))

    @property
    def position(self) -> np.ndarray:
        """Current XYZ position of the origin (COM frame) of the cargo bag"""
        return np.array(pybullet.getBasePositionAndOrientation(self.id)[0])

    @property
    def orientation(self) -> np.ndarray:
        """Current XYZW quaternion orientation of the cargo bag's COM frame"""
        return np.array(pybullet.getBasePositionAndOrientation(self.id)[1])

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
            Tuple of:
                np.ndarray: Position, shape (3,)
                np.ndarray: XYZW quaternion orientation, shape (4,)
                np.ndarray: Linear velocity, shape (3,)
                np.ndarray: Angular velocity, shape (3,)
        """
        old_pos, old_orn = pybullet.getBasePositionAndOrientation(self.id)
        # Step the sim to get a second reference frame we can use to determine velocity
        # This is not ideal, but it's the best way to do this until Pybullet's getBaseVelocity for softbodies works
        pybullet.stepSimulation()
        new_pos, new_orn = pybullet.getBasePositionAndOrientation(self.id)
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

    @property
    def num_handles(self) -> int:
        """Number of handles on the cargo bag"""
        if self._name in SINGLE_HANDLE_BAGS:
            return 1
        elif self._name in DUAL_HANDLE_BAGS:
            return 2
        else:
            return 0  # This may have an application in the future

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

    def attach_to(
        self, robot_or_robots: Union[Astrobee, list[Astrobee], tuple[Astrobee]]
    ) -> None:
        """Attaches a robot (or multiple robots) to the handle(s) of the bag

        Args:
            robot_or_robots (Union[Astrobee, list[Astrobee], tuple[Astrobee]]): Robot(s) to attach to the bag

        Raises:
            ValueError: For invalid inputs, or if the bag does not have enough handles for each robot
            NotImplementedError: Multi-robot case with >2 robots
        """
        # Handle inputs
        if isinstance(robot_or_robots, Astrobee):  # Single robot
            num_robots = 1
        elif isinstance(robot_or_robots, (list, tuple)):  # Multi-robot
            if not all(isinstance(r, Astrobee) for r in robot_or_robots):
                raise ValueError("Non-Astrobee input detected")
            num_robots = len(robot_or_robots)
            if self.num_handles < num_robots:
                raise ValueError(
                    f"Bag does not have enough handles to support {num_robots} robots"
                )
            if num_robots == 1:  # Edge case: Unpack the list if only one robot
                robot_or_robots = robot_or_robots[0]
        else:
            raise ValueError(
                "Invalid input: Must provide either an Astrobee or a list of multiple Astrobees"
            )

        # We will attach the bag by updating the positions of the Astrobee(s) to interface with the bag
        bag_to_world = pos_quat_to_tmat(self.pose)
        if num_robots == 1:
            robot = robot_or_robots
            handle_to_bag = self.grasp_transforms[0]
            handle_to_world = bag_to_world @ handle_to_bag
            self._attach(robot, tmat_to_pos_quat(handle_to_world))
        elif num_robots == 2:
            robot_1, robot_2 = robot_or_robots
            handle_1_to_bag = self.grasp_transforms[0]
            handle_2_to_bag = self.grasp_transforms[1]
            handle_1_to_world = bag_to_world @ handle_1_to_bag
            handle_2_to_world = bag_to_world @ handle_2_to_bag
            self._attach(robot_1, tmat_to_pos_quat(handle_1_to_world))
            self._attach(robot_2, tmat_to_pos_quat(handle_2_to_world))
        else:
            raise NotImplementedError(
                "The multi-robot case is only implemented for 2 Astrobees"
            )

    def _attach(self, robot: Astrobee, handle_pose: npt.ArrayLike) -> None:
        """Helper function: Connects a single robot to a handle at a specified pose

        TODO decide between moving the bag to the robot, or the robot to the bag

        Args:
            robot (Astrobee): Robot to attach
            handle_pose (npt.ArrayLike): Position + quaternion grasp pose (handle-to-world), shape (7,)
        """
        robot.reset_to_ee_pose(handle_pose)  # Move robot to bag
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
        )
        anchor2_id, geom2_id = create_anchor(
            self.id,
            v2_id,
            robot.id,
            robot.Links.ARM_DISTAL.value,
            add_geom=True,
            geom_pos=v2_pos,
        )
        self._anchors.update({robot.id: [anchor1_id, anchor2_id]})
        self._anchor_objects.update({robot.id: [geom1_id, geom2_id]})
        self._attached.append(robot.id)

    def detach(self) -> None:
        """Remove all anchors (and visuals) from the bag"""

        anchors, anchor_objects = self.anchors
        for cid in anchors:
            pybullet.removeConstraint(cid)
        for obj in anchor_objects:
            pybullet.removeBody(obj)
        self._anchors = {}
        self._anchor_objects = {}
        self._attached = []

    def detach_robot(self, robot_id: int) -> None:
        """Detaches a specific robot from the bag by removing its associated anchors / visuals

        Args:
            robot_id (int): Pybullet ID of the robot to detach
        """
        if robot_id not in self.attached:
            raise ValueError("Cannot detach robot: ID unknown")
        for cid in self._anchors[robot_id]:
            pybullet.removeConstraint(cid)
        for obj in self._anchor_objects[robot_id]:
            pybullet.removeBody(obj)
        self._anchors.pop(robot_id)
        self._anchor_objects.pop(robot_id)
        self._attached.remove(robot_id)

    def unload(self) -> None:
        """Removes the cargo bag from the simulation

        TODO this is kinda broken since it only removes the simulated body rather than the visual
        (if we update how we load the texture for the softbody this might be fixed)
        """
        pybullet.removeBody(self.id)
        self.id = None


# TODO decide if this is useful (or something similar)
# (move the bag to the robot, instead of robot to the bag)
# def load_bag_attached(robot: Astrobee, bag_name: str) -> CargoBag:
#     pass


def _main():
    # Very simple example of loading the bag and attaching a robot
    initialize_pybullet()
    robot = Astrobee()
    bag = CargoBag("top_handle_bag")
    bag.attach_to(robot)
    while True:
        pybullet.stepSimulation()
        time.sleep(1 / 120)


if __name__ == "__main__":
    _main()
