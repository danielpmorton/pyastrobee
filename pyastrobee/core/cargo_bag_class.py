"""WORK IN PROGRESS"""

# TODO use custom methods for the deformable bag velocity / angular velocity
# ^^ Same for dynamics state
# NOTE the deformable methods might not use a handle index

from abc import ABC
from typing import Union, Optional

import pybullet
from pybullet_utils.bullet_client import BulletClient
import numpy as np
import numpy.typing as npt

from pyastrobee.core.astrobee import Astrobee
from pyastrobee.utils.poses import pos_quat_to_tmat, tmat_to_pos_quat
from pyastrobee.utils.transformations import invert_transform_mat
from pyastrobee.utils.rotations import quat_to_rmat
import pyastrobee.config.bag_properties as bag_props


# TODO rename this
class CargoBagABC(ABC):
    LENGTH = 0.50  # meters
    WIDTH = 0.25  # meters
    HEIGHT = 0.42  # meters
    URDF_DIR = "pyastrobee/assets/urdf/bags/"
    MESH_DIR = "pyastrobee/assets/meshes/bags/"
    SINGLE_HANDLE_BAGS = ["front_handle", "right_handle", "top_handle"]
    DUAL_HANDLE_BAGS = ["front_back_handle", "right_left_handle", "top_bottom_handle"]
    BAG_NAMES = SINGLE_HANDLE_BAGS + DUAL_HANDLE_BAGS
    HANDLE_TRANSFORMS = {
        "front": bag_props.FRONT_HANDLE_TRANSFORM,
        "back": bag_props.BACK_HANDLE_TRANSFORM,
        "left": bag_props.LEFT_HANDLE_TRANSFORM,
        "right": bag_props.RIGHT_HANDLE_TRANSFORM,
        "top": bag_props.TOP_HANDLE_TRANSFORM,
        "bottom": bag_props.BOTTOM_HANDLE_TRANSFORM,
    }

    def __init__(
        self,
        bag_name: str,
        mass: float,
        pos: npt.ArrayLike,
        orn: npt.ArrayLike,
        client: Optional[BulletClient] = None,
    ):
        self.client: pybullet = pybullet if client is None else client
        if not self.client.isConnected():
            raise ConnectionError("Need to connect to pybullet before loading a bag")
        if bag_name not in self.BAG_NAMES:
            raise ValueError(
                f"Invalid bag name: {bag_name}. Must be one of {self.BAG_NAMES}"
            )
        self._mass = mass
        self._name = bag_name
        self._attached = []
        self._dt = self.client.getPhysicsEngineParameters()["fixedTimeStep"]
        self.id = self._load(pos, orn)
        if self.id < 0:
            raise ValueError("Bag was not properly loaded!")

    @property
    def mass(self) -> float:
        """Mass of the cargo bag"""
        return self._mass

    @property
    def attached(self) -> list[int]:
        """ID(s) of the robot (or robots) grasping the bag. Empty if no robots are attached"""
        return self._attached

    @property
    def name(self) -> str:
        """Type of cargo bag"""
        return self._name

    @property
    def grasp_transforms(self) -> list[np.ndarray]:
        """Transformation matrices "handle to bag" representing the grasp locations on the handles to the bag COM

        In the case of a single-handled bag, this list will only have one entry
        """
        if self._name == "front_handle":
            return [self.HANDLE_TRANSFORMS["front"]]
        elif self._name == "right_handle":
            return [self.HANDLE_TRANSFORMS["right"]]
        elif self._name == "top_handle":
            return [self.HANDLE_TRANSFORMS["top"]]
        elif self._name == "front_back_handle":
            return [self.HANDLE_TRANSFORMS["front"], self.HANDLE_TRANSFORMS["back"]]
        elif self._name == "right_left_handle":
            return [self.HANDLE_TRANSFORMS["right"], self.HANDLE_TRANSFORMS["left"]]
        elif self._name == "top_bottom_handle":
            return [self.HANDLE_TRANSFORMS["top"], self.HANDLE_TRANSFORMS["bottom"]]
        else:
            raise NotImplementedError(
                f"Grasp transform(s) not available for bag: {self._name}"
            )

    @property
    def pose(self) -> np.ndarray:
        """Current position + XYZW quaternion pose of the bag"""
        return np.concatenate(self.client.getBasePositionAndOrientation(self.id))

    @property
    def tmat(self):
        """Current transformation matrix for the cargo bag"""
        return pos_quat_to_tmat(self.pose)

    @property
    def position(self) -> np.ndarray:
        """Current XYZ position of the origin (COM frame) of the cargo bag"""
        return np.array(self.client.getBasePositionAndOrientation(self.id)[0])

    @property
    def orientation(self) -> np.ndarray:
        """Current XYZW quaternion orientation of the cargo bag's COM frame"""
        return np.array(self.client.getBasePositionAndOrientation(self.id)[1])

    # TODO decide if using dynamics state is actually the best here
    @property
    def velocity(self) -> np.ndarray:
        """Current [vx, vy, vz] velocity of the cargo bag's COM frame

        - If both velocity and angular velocity are desired, use the dynamics_state property instead
        """
        return np.array(self.client.getBaseVelocity(self.id)[0])

    @property
    def angular_velocity(self) -> np.ndarray:
        """Current [wx, wy, wz] angular velocity of the cargo bag's COM frame

        - If both velocity and angular velocity are desired, use the dynamics_state property instead
        """
        return np.array(self.client.getBaseVelocity(self.id)[1])

    @property
    def dynamics_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Current state of the bag dynamics: Position, orientation, linear vel, and angular vel

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                np.ndarray: Position, shape (3,)
                np.ndarray: XYZW quaternion orientation, shape (4,)
                np.ndarray: Linear velocity, shape (3,)
                np.ndarray: Angular velocity, shape (3,)
        """
        pos, orn = self.client.getBasePositionAndOrientation(self.id)
        lin_vel, ang_vel = self.client.getBaseVelocity(self.id)
        return (
            np.array(pos),
            np.array(orn),
            np.array(lin_vel),
            np.array(ang_vel),
        )

    @property
    def num_handles(self) -> int:
        """Number of handles on the cargo bag"""
        if self._name in self.SINGLE_HANDLE_BAGS:
            return 1
        elif self._name in self.DUAL_HANDLE_BAGS:
            return 2
        else:
            return 0  # This may have an application in the future

    @property
    def corner_positions(self) -> list[np.ndarray]:
        """Positions of the 8 corners of the main compartment of the bag, shape (8, 3)"""
        # The main compartment is the base link in all URDFs
        pos, quat = self.client.getBasePositionAndOrientation(self.id)
        rmat = quat_to_rmat(quat)
        l, w, h = self.LENGTH, self.WIDTH, self.HEIGHT
        return (
            pos
            + np.array(
                [
                    [l / 2, w / 2, h / 2],
                    [l / 2, w / 2, -h / 2],
                    [l / 2, -w / 2, h / 2],
                    [l / 2, -w / 2, -h / 2],
                    [-l / 2, w / 2, h / 2],
                    [-l / 2, w / 2, -h / 2],
                    [-l / 2, -w / 2, h / 2],
                    [-l / 2, -w / 2, -h / 2],
                ]
            )
            @ rmat.T
        )

    def _load(self, pos: npt.ArrayLike, orn: npt.ArrayLike) -> int:
        """Loads a cargo bag at the specified position/orientation

        Args:
            pos (npt.ArrayLike): XYZ position, shape (3,)
            orn (npt.ArrayLike): XYZW quaternion, shape (4,)

        Returns:
            int: Pybullet ID for the cargo bag
        """
        return -1  # Dummy value

    def unload(self) -> None:
        """Removes the cargo bag from the simulation"""
        self.detach()
        self.client.removeBody(self.id)
        self.id = None

    def attach_to(
        self,
        robot_or_robots: Union[Astrobee, list[Astrobee], tuple[Astrobee]],
        object_to_move: str = "robot",
    ) -> None:
        """Attaches a robot (or multiple robots) to the handle(s) of the bag

        Args:
            robot_or_robots (Union[Astrobee, list[Astrobee], tuple[Astrobee]]): Robot(s) to attach to the bag
            object_to_move (str, optional): Either "robot" or "bag". This dictates what object will get its position
                reset in order to make the grasp connection. In general, it makes more sense to move the robot to the
                bag (default behavior)

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
        if object_to_move not in {"robot", "bag"}:
            raise ValueError("Invalid object to move: Must be either 'robot' or 'bag'.")

        bag_to_world = pos_quat_to_tmat(self.pose)
        if num_robots == 1:
            robot = robot_or_robots  # Unpack list
            if object_to_move == "robot":
                # Reset the position of the robot to interface with the handle
                handle_to_bag = self.grasp_transforms[0]
                handle_to_world = bag_to_world @ handle_to_bag
                handle_pose = tmat_to_pos_quat(handle_to_world)
                robot.reset_to_ee_pose(handle_pose)
            else:  # Move the bag to the robot
                self.reset_to_handle_pose(robot.ee_pose)
            self._attach(robot, 0)
        elif num_robots == 2:
            robot_1, robot_2 = robot_or_robots  # Unpack list
            if object_to_move == "robot":
                # Reset the position of each robot to interface with the two handles
                handle_1_to_bag = self.grasp_transforms[0]
                handle_2_to_bag = self.grasp_transforms[1]
                handle_1_to_world = bag_to_world @ handle_1_to_bag
                handle_2_to_world = bag_to_world @ handle_2_to_bag
                robot_1.reset_to_ee_pose(tmat_to_pos_quat(handle_1_to_world))
                robot_2.reset_to_ee_pose(tmat_to_pos_quat(handle_2_to_world))
                self._attach(robot_1, 0)
                self._attach(robot_2, 1)
            else:  # Move the bag while leaving the robots static
                raise NotImplementedError(
                    "Attaching the bag to multiple robots requires moving at least 1 robot"
                )
        else:
            raise NotImplementedError(
                "The multi-robot case is only implemented for 2 Astrobees"
            )

    def _attach(self, robot: Astrobee, handle_index: int) -> None:
        """Helper function: Connects a single robot to a handle at a specified pose

        This function assumes that the robot and the bag are already correctly positioned for a grasp, which is why
        it should not be called directly

        Args:
            robot (Astrobee): Robot to attach
            handle_index (int): Index of the handle on the bag
        """
        pass

    def detach(self) -> None:
        """Detach all connections to the bag"""
        pass

    def detach_robot(self, robot_id: int) -> None:
        """Detaches a specific robot from the bag

        Args:
            robot_id (int): Pybullet ID of the robot to detach
        """
        pass

    def reset_to_handle_pose(
        self, handle_pose: npt.ArrayLike, handle_index: int = 0
    ) -> None:
        """Resets the position of the bag so that the handle is positioned at a desired pose

        Args:
            handle_pose (npt.ArrayLike): Desired pose of the handle ("handle-to-world"), shape (7,)
            handle_index (int, optional): Index of the handle to align to the desired pose. Defaults to 0.
        """
        handle_to_world = pos_quat_to_tmat(handle_pose)
        bag_to_handle = invert_transform_mat(self.grasp_transforms[handle_index])
        bag_to_world = handle_to_world @ bag_to_handle
        bag_pose = tmat_to_pos_quat(bag_to_world)
        self.client.resetBasePositionAndOrientation(self.id, bag_pose[:3], bag_pose[3:])
