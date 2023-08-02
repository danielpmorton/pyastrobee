"""Rigid version of the cargo bag, using joints in the URDF to mimic what we would see from a deformable

TODO add support for more than just the top handle bag!!!
TODO tune the position control force on the handle

Additional relevant TODOs from the original CargoBag file:
TODO decide if the bag_props import can be handled better
TODO decide if the constants should be moved to class attributes
TODO decide if we should anchor to the gripper fingers or the arm distal link (currently distal)
"""

import time
from typing import Union, Optional

import pybullet
from pybullet_utils.bullet_client import BulletClient
import numpy as np
import numpy.typing as npt

import pyastrobee.config.bag_properties as bag_props
from pyastrobee.core.astrobee import Astrobee
from pyastrobee.utils.bullet_utils import initialize_pybullet
from pyastrobee.utils.poses import pos_quat_to_tmat, tmat_to_pos_quat
from pyastrobee.utils.python_utils import print_green
from pyastrobee.utils.transformations import invert_transform_mat, make_transform_mat
from pyastrobee.utils.rotations import quat_to_rmat, Ry, rmat_to_quat
from pyastrobee.utils.dynamics import box_inertia

# Constants. TODO Move these to class attributes?
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


class RigidCargoBag:
    """Class for loading and managing properties associated with the rigid URDF-based cargo bags

    Args:
        bag_name (str): Type of cargo bag to load. Single handle: "front_handle", "right_handle", "top_handle".
            Dual handle: "front_back_handle", "right_left_handle", "top_bottom_handle"
        pos (npt.ArrayLike, optional): Initial XYZ position to load the bag. Defaults to (0, 0, 0)
        orn (npt.ArrayLike, optional): Initial XYZW quaternion to load the bag. Defaults to (0, 0, 0, 1)
        client (BulletClient, optional): If connecting to multiple physics servers, include the client
            (the class instance, not just the ID) here. Defaults to None (use default connected client)
    """

    URDF = "pyastrobee/assets/urdf/rigid_bag.urdf"
    LINKS_PER_HANDLE = 4  # 3 dummy links for roll/pitch/yaw, plus the handle itself
    LENGTH = 0.50  # meters
    WIDTH = 0.25  # meters
    HEIGHT = 0.42  # meters

    def __init__(
        self,
        bag_name: str,
        pos: npt.ArrayLike = (0, 0, 0),
        orn: npt.ArrayLike = (0, 0, 0, 1),
        client: Optional[BulletClient] = None,
    ):
        self.client: pybullet = pybullet if client is None else client
        if not self.client.isConnected():
            raise ConnectionError("Need to connect to pybullet before loading a bag")
        if bag_name not in BAG_NAMES:
            raise ValueError(
                f"Invalid bag name: {bag_name}. Must be one of {BAG_NAMES}"
            )
        self.id = self.client.loadURDF(RigidCargoBag.URDF, pos, orn)
        self._dt = self.client.getPhysicsEngineParameters()["fixedTimeStep"]
        self.mass = 5  # kg
        # This inertia is slightly approximate because some mass is in the handle and dummy links
        self.inertia = box_inertia(self.mass, self.LENGTH, self.WIDTH, self.HEIGHT)
        self._name = bag_name
        # Initializations
        self._constraints = {}
        self._attached = []
        # Add position control to the handle(s) so its springs back into its natural position
        # to provide some resistance to motion like we would see in a deformable
        # The dummy joints are the joints associated with the motion of the handle
        if self.num_handles == 1:
            dummy_joint_ids = [0, 1, 2]
        elif self.num_handles == 2:
            dummy_joint_ids = [0, 1, 2, 4, 5, 6]
        pybullet.setJointMotorControlArray(
            self.id,
            dummy_joint_ids,
            pybullet.POSITION_CONTROL,
            [0] * len(dummy_joint_ids),
            forces=[0.1] * len(dummy_joint_ids),  # TODO tune force
        )
        print_green("Bag is ready")

    @property
    def constraints(self) -> list[int]:
        """Active IDs of constraints on the bag"""
        return list(self._constraints.values())

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
            return [HANDLE_TRANSFORMS["front"]]
        elif self._name == "right_handle":
            return [HANDLE_TRANSFORMS["right"]]
        elif self._name == "top_handle":
            return [HANDLE_TRANSFORMS["top"]]
        elif self._name == "front_back_handle":
            return [HANDLE_TRANSFORMS["front"], HANDLE_TRANSFORMS["back"]]
        elif self._name == "right_left_handle":
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
        return np.concatenate(self.client.getBasePositionAndOrientation(self.id))

    @property
    def position(self) -> np.ndarray:
        """Current XYZ position of the origin (COM frame) of the cargo bag"""
        return np.array(self.client.getBasePositionAndOrientation(self.id)[0])

    @property
    def orientation(self) -> np.ndarray:
        """Current XYZW quaternion orientation of the cargo bag's COM frame"""
        return np.array(self.client.getBasePositionAndOrientation(self.id)[1])

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
            Tuple of:
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
        if self._name in SINGLE_HANDLE_BAGS:
            return 1
        elif self._name in DUAL_HANDLE_BAGS:
            return 2
        else:
            return 0  # This may have an application in the future

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
            self._attach(robot)
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
                self._attach(robot_1)
                self._attach(robot_2)
            else:  # Move the bag while leaving the robots static
                raise NotImplementedError(
                    "Attaching the bag to multiple robots requires moving at least 1 robot"
                )
                # This generally leads to undesired behavior unless the 2 robots are carefully positioned before
                # attachment, so we'll throw an error instead. Uncomment the lines below if this is desired later on
                # self.reset_to_handle_pose(robot_1.ee_pose, 0)
                # self._attach(robot_1)
                # self.reset_to_handle_pose(robot_2.ee_pose, 1)
                # self._attach(robot_2)
        else:
            raise NotImplementedError(
                "The multi-robot case is only implemented for 2 Astrobees"
            )

    def _attach(self, robot: Astrobee) -> None:
        """Helper function: Connects a single robot to a handle at a specified pose

        This function assumes that the robot and the bag are already correctly positioned for a grasp, which is why
        it should not be called directly

        Args:
            robot (Astrobee): Robot to attach
        """
        handle_link_index = self.num_handles * self.LINKS_PER_HANDLE - 1
        cid = self.client.createConstraint(
            robot.id,
            robot.Links.ARM_DISTAL.value,
            self.id,
            handle_link_index,
            pybullet.JOINT_FIXED,
            [0, 0, 1],
            robot.TRANSFORMS.GRIPPER_TO_ARM_DISTAL[:3, 3],
            [0, 0, 0],
            # These frame orientations (below) align the grasp frame to the handle link frame
            [0, 0, 0, 1],  # Robot link frame orientation
            rmat_to_quat(Ry(-np.pi / 2)),  # Bag handle frame orientation
        )
        self._constraints.update({robot.id: cid})
        self._attached.append(robot.id)

    def get_handle_transform(self, handle_index: int = 0) -> np.ndarray:
        """Calculates the transformation matrix (w.r.t the world) for a specified handle

        Args:
            handle_index (int): Index of the handle on the bag

        Returns:
            np.ndarray: Transformation matrix (handle to world). Shape = (4,4)
        """
        if (handle_index + 1) > self.num_handles:
            raise ValueError(
                f"Invalid handle index: {handle_index}. Bag only has {self.num_handles} handles"
            )
        handle_link_index = (handle_index + 1) * self.LINKS_PER_HANDLE - 1
        link_state = self.client.getLinkState(
            self.id, handle_link_index, computeForwardKinematics=True
        )
        pos, quat = link_state[:2]
        return make_transform_mat(quat_to_rmat(quat), pos)

    def detach(self) -> None:
        """Remove all constraints from the bag"""
        for cid in self.constraints:
            self.client.removeConstraint(cid)
        self._constraints = {}
        self._attached = []

    def detach_robot(self, robot_id: int) -> None:
        """Detaches a specific robot from the bag by removing its associated constraint

        Args:
            robot_id (int): Pybullet ID of the robot to detach
        """
        if robot_id not in self.attached:
            raise ValueError("Cannot detach robot: ID unknown")
        self.client.removeConstraint(self._constraints[robot_id])
        self._constraints.pop(robot_id)
        self._attached.remove(robot_id)

    def unload(self) -> None:
        """Removes the cargo bag from the simulation"""
        self.detach()
        self.client.removeBody(self.id)
        self.id = None

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


def _main():
    # Very simple example of loading the bag and attaching a robot
    client = initialize_pybullet(bg_color=(0.8, 0.8, 1))
    robot = Astrobee()
    bag = RigidCargoBag("top_handle")
    bag.attach_to(robot)
    input("Press Enter to run the simulation")
    while True:
        client.stepSimulation()
        time.sleep(1 / 120)


if __name__ == "__main__":
    _main()
