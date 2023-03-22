"""Manages the properties of the astrobee and all control-associated functions

In general, we assume that we're working with Honey. Multiple astrobees can be loaded, but
we assume that they all have the exact same configuration

TODO
- Can we rename the joints and the links in the urdf? e.g. remove "honey", "top_aft"
- What's the deal with the "top_aft" fixed joint? Is that part of the KDL workaround?
- Figure out the sleep time in the while loops. What value should we use?
- Get step sizes worked out!!
"""

import time
from enum import Enum
from typing import Optional

import pybullet
import numpy as np
import numpy.typing as npt


from pyastrobee.utils.bullet_utils import initialize_pybullet, run_sim
from pyastrobee.utils.transformations import make_transform_mat
from pyastrobee.utils.rotations import quat_to_rmat
from pyastrobee.utils.poses import tmat_to_pos_quat, pos_quat_to_tmat
from pyastrobee.config import astrobee_transforms
from pyastrobee.utils.python_utils import print_green


class Astrobee:
    """Astrobee class for managing control, states, and properties

    Args:
        pose (npt.ArrayLike, optional): Initial pose of the astrobee when loaded. Defaults to
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] (At origin, pointed forward along x axis).
        arm_joints (npt.ArrayLike, optional): Initial position of the arm's joints. Defaults to
            [0.0, 0.0] (Hanging straight down)
        gripper_pos (float, optional): Initial gripper position, in [0, 100]. Defaults to 100 (fully open)

    Raises:
        ConnectionError: If a pybullet server is not connected before initialization
    """

    # TODO add these to a constants or config file? Probably fine here for now
    URDF = "pyastrobee/assets/urdf/astrobee.urdf"
    LOADED_IDS = []  # Initialization
    NUM_ROBOTS = 0  # Initialization. TODO make this into a property?
    NUM_JOINTS = 7
    NUM_LINKS = 7
    TRANSFORMS = astrobee_transforms  # TODO figure out if this is the best way to store this info
    GRIPPER_JOINT_IDXS = [3, 4, 5, 6]
    ARM_JOINT_IDXS = [1, 2]
    # Joint limit information is extracted from the URDF
    # Joint pos limits are [lower, upper] for each joint
    JOINT_POS_LIMITS = np.array(
        [
            [0.0, 0.0],  # top aft (fixed)
            [-2.0944, 1.57079],  # arm proximal joint
            [-1.57079, 1.57079],  # arm distal joint
            [0.349066, 0.698132],  # gripper left proximal joint
            [-1.22173, -0.69813],  # gripper left distal joint
            [-0.698132, -0.349066],  # gripper right proximal joint
            [0.69813, 1.22173],  # gripper right distal joint
        ]
    )
    JOINT_EFFORT_LIMITS = [
        0.0,  # top aft (fixed)
        1.0,  # arm proximal joint
        1.0,  # arm distal joint
        0.1,  # gripper left proximal joint
        0.1,  # gripper left distal joint
        0.1,  # gripper right proximal joint
        0.1,  # gripper right distal joint
    ]
    JOINT_VEL_LIMITS = [
        0.00,  # top aft (fixed)
        0.12,  # arm proximal joint
        0.12,  # arm distal joint
        0.12,  # gripper left proximal joint
        0.12,  # gripper left distal joint
        0.12,  # gripper right proximal joint
        0.12,  # gripper right distal joint
    ]

    # Should these actually be enums, or would a dict be better? The names can be tricky,
    # so typing those out as strings every time could be not ideal
    class Joints(Enum):
        """Enumerates the different joints on the astrobee via their Pybullet index"""

        # Comments indicate the name of the joint in the URDF
        TOP_AFT = 0  # top_aft
        ARM_PROXIMAL = 1  # top_aft_arm_proximal_joint
        ARM_DISTAL = 2  # top_aft_arm_distal_joint
        GRIPPER_LEFT_PROXIMAL = 3  # top_aft_gripper_left_proximal_joint
        GRIPPER_LEFT_DISTAL = 4  # top_aft_gripper_left_distal_joint
        GRIPPER_RIGHT_PROXIMAL = 5  # top_aft_gripper_right_proximal_joint
        GRIPPER_RIGHT_DISTAL = 6  # top_aft_gripper_right_distal_joint

    class Links(Enum):
        """Enumerates the different links on the astrobee via their Pybullet index

        Note: the URDF technically has 8 links, but it appears that pybullet considers
        the very first link to be the base link
        """

        # Comments indicate the name of the link in the URDF
        TOP_AFT = 0  # honey_top_aft
        ARM_PROXIMAL = 1  # honey_top_aft_arm_proximal_link
        ARM_DISTAL = 2  # honey_top_aft_arm_distal_link
        GRIPPER_LEFT_PROXIMAL = 3  # honey_top_aft_gripper_left_proximal_link
        GRIPPER_LEFT_DISTAL = 4  # honey_top_aft_gripper_left_distal_link
        GRIPPER_RIGHT_PROXIMAL = 5  # honey_top_aft_gripper_right_proximal_link
        GRIPPER_RIGHT_DISTAL = 6  # honey_top_aft_gripper_right_distal_link

    def __init__(
        self,
        pose: npt.ArrayLike = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        arm_joints: npt.ArrayLike = [0.0, 0.0],
        gripper_pos: float = 100,
    ):
        if not pybullet.isConnected():
            raise ConnectionError(
                "Need to connect to pybullet before initializing an astrobee"
            )
        self.id = pybullet.loadURDF(Astrobee.URDF, pose[:3], pose[3:])
        Astrobee.LOADED_IDS.append(self.id)
        Astrobee.NUM_ROBOTS += 1
        self.set_gripper_position(gripper_pos)
        self.set_arm_joints(arm_joints)  # Do this only if nonzero?
        print_green("Astrobee is ready")

    @classmethod
    def unload_robot(cls, robot_id: int) -> None:
        """Removes an Astrobee from the simulation

        Args:
            robot_id (int): ID of the robot to remove (the self.id parameter)
        """
        if robot_id not in Astrobee.LOADED_IDS:
            raise ValueError(f"Invalid ID: {robot_id}, cannot unload the astrobee")
        pybullet.removeBody(robot_id)
        Astrobee.LOADED_IDS.remove(robot_id)
        Astrobee.NUM_ROBOTS -= 1

    @property
    def pose(self) -> np.ndarray:
        """The current robot pose (position + XYZW quaternion) expressed in world frame

        Returns:
            np.ndarray: Position and quaternion, size (7,)
        """
        pos, orn = pybullet.getBasePositionAndOrientation(self.id)
        return np.concatenate([pos, orn])

    @property
    def tmat(self) -> np.ndarray:
        """The current robot pose in world frame, expressed as a transformation matrix

        Returns:
            np.ndarray: Transformation matrix (Robot to World), shape (4,4)
        """
        return pos_quat_to_tmat(self.pose)

    @property
    def position(self) -> np.ndarray:
        """Just the position component of the full pose

        Returns:
            np.ndarray: (3,) position vector
        """
        return self.pose[:3]

    @property
    def orientation(self) -> np.ndarray:
        """Just the quaternion component of the full pose

        Returns:
            np.ndarray: (4,) XYZW quaternion
        """
        return self.pose[3:]

    @property
    def rmat(self) -> np.ndarray:
        """The orientation of the robot expressed as a rotation matrix

        Returns:
            np.ndarray: Rotation matrix (Robot to World), shape (3,3)
        """
        return quat_to_rmat(self.orientation)

    @property
    def heading(self) -> np.ndarray:
        """A unit vector in the forward (x) component of the astrobee

        Some notes:
        - This is NOT a full description of orientation since rotation about this axis is undefined
        - The arm is on the REAR side of the astrobee, not the front
        - The y vector points to the port (left) side of the astrobee, and the z vector is up

        Returns:
            np.ndarray: (3,) unit vector
        """
        R_R2W = quat_to_rmat(self.orientation)  # Robot to world
        return R_R2W[:, 0]  # Robot frame x vector expressed in world

    @property
    def velocity(self) -> np.ndarray:
        """Linear velocity of the Astrobee, with respect to the world frame xyz axes

        TODO check if this has a (1/5) scaling issue with the commanded velocity

        Returns:
            np.ndarray: [vx, vy, vz] linear velocities, shape (3,)
        """
        lin_vel, _ = pybullet.getBaseVelocity(self.id)
        return np.array(lin_vel)

    @property
    def angular_velocity(self) -> np.ndarray:
        """Angular velocity of the Astrobee, about the world frame xyz axes

        TODO check if this has a (1/5) scaling issue with the commanded velocity

        Returns:
            np.ndarray: [wx, wy, wz] angular velocities, shape (3,)
        """
        _, ang_vel = pybullet.getBaseVelocity(self.id)
        return np.array(ang_vel)

    @property
    def ee_pose(self) -> np.ndarray:
        """The current end-effector pose (position + XYZW quaternion) expressed in world frame

        Returns:
            np.ndarray: Position and quaternion, size (7,)
        """
        T_G2D = Astrobee.TRANSFORMS.GRIPPER_TO_ARM_DISTAL  # Gripper to distal
        T_D2W = self.get_link_transform(
            Astrobee.Links.ARM_DISTAL.value
        )  # Distal to world
        T_G2W = T_D2W @ T_G2D  # Gripper to world
        return tmat_to_pos_quat(T_G2W)

    @property
    def joint_angles(self) -> np.ndarray:
        """Angular positions (radians) of each joint on the Astrobee

        Returns:
            np.ndarray: Joint angles, shape (NUM_JOINTS,)
        """
        # States: tuple[tuple], size (7, 4)
        # 7 corresponds to NUM_JOINTS
        # 4 corresponds to position, velocity, reaction forces, and applied torque
        states = pybullet.getJointStates(self.id, list(range(Astrobee.NUM_JOINTS)))
        # Index 0: position
        return np.array([states[i][0] for i in range(Astrobee.NUM_JOINTS)])

    @property
    def joint_vels(self) -> np.ndarray:
        """Angular velocities (radians/sec) of each joint on the Astrobee

        Returns:
            np.ndarray: Joint velocities, shape (NUM_JOINTS,)
        """
        states = pybullet.getJointStates(self.id, list(range(Astrobee.NUM_JOINTS)))
        # Index 1: velocity
        return np.array([states[i][1] for i in range(Astrobee.NUM_JOINTS)])

    @property
    def joint_torques(self) -> np.ndarray:
        """Torques (N-m) applied by each joint on the Astrobee

        Returns:
            np.ndarray: Joint torques, shape (NUM_JOINTS,)
        """
        states = pybullet.getJointStates(self.id, list(range(Astrobee.NUM_JOINTS)))
        # Index 3: torque
        return np.array([states[i][3] for i in range(Astrobee.NUM_JOINTS)])

    @property
    def arm_joint_angles(self) -> np.ndarray:
        """Gives the two joint angles associated with the proximal + distal joints of the arm

        Returns:
            np.ndarray: Arm joint angles, shape (2,)
        """
        return self.joint_angles[Astrobee.ARM_JOINT_IDXS]

    @property
    def gripper_joint_angles(self) -> np.ndarray:
        """Gives the four joint angles associated with the proximal + distal joints of the two gripper fingers

        Returns:
            np.ndarray: Gripper joint angles, shape (4,)
        """
        return self.joint_angles[Astrobee.GRIPPER_JOINT_IDXS]

    def get_link_transform(self, link_index: int) -> np.ndarray:
        """Calculates the transformation matrix (w.r.t the world) for a specified link

        Args:
            link_index (int): Index of the link on the robot

        Returns:
            np.ndarray: Transformation matrix (link to world). Shape = (4,4)
        """
        if link_index >= Astrobee.NUM_LINKS or link_index < 0:
            raise ValueError(f"Invalid link index: {link_index}")
        link_state = pybullet.getLinkState(
            self.id, link_index, computeForwardKinematics=True
        )
        # First two link state values are linkWorldPosition, linkWorldOrientation
        # There are other state positions and orientations, but they're confusing. (TODO check on these)
        pos, quat = link_state[:2]
        return make_transform_mat(quat_to_rmat(quat), pos)

    @property  # Is this better as a property or as a "getter"?
    def gripper_position(self) -> int:
        """The current position of the gripper, in range [0, 100]

        Returns:
            int: Position of the gripper, an integer between 0 (closed) and 100 (open)
        """
        joint_states = pybullet.getJointStates(self.id, Astrobee.GRIPPER_JOINT_IDXS)
        joint_angles = [state[0] for state in joint_states]
        l_angles = joint_angles[:2]
        r_angles = joint_angles[2:]
        l_closed, l_open, r_closed, r_open = self._get_gripper_joint_ranges()
        l_pct = 100 * (l_angles - l_closed) / (l_open - l_closed)
        r_pct = 100 * (r_angles - r_closed) / (r_open - r_closed)
        return np.round(np.average(np.concatenate([l_pct, r_pct]))).astype(int)

    def set_gripper_position(self, position: float, do_step=True) -> None:
        """Sets the gripper to a position between 0 (fully closed) to 100 (fully open)

        TODO decide if we need finer-grain control of the individual joints, or if this integer-position is fine
        TODO update this to use self.set_joint_angles()!

        Args:
            position (float): Gripper position, in range [0, 100]
        """
        if position < 0 or position > 100:
            raise ValueError("Position should be in range [0, 100]")
        l_closed, l_open, r_closed, r_open = self._get_gripper_joint_ranges()
        left_pos = l_closed + (position / 100) * (l_open - l_closed)
        right_pos = r_closed + (position / 100) * (r_open - r_closed)
        angle_cmd = [*left_pos, *right_pos]
        self.set_gripper_joints(angle_cmd, do_step)

    def open_gripper(self) -> None:
        """Fully opens the gripper"""
        self.set_gripper_position(100)

    def close_gripper(self) -> None:
        """Fully closes the gripper

        TODO add force/torque control?
        """
        self.set_gripper_position(0)

    # TODO rework this?
    def _get_gripper_joint_ranges(self) -> tuple[np.ndarray, ...]:
        """Helper function to determine the range of motion (closed -> open) of the gripper joints

        - This is a bit confusing because of how the URDF specifies joint min/max and how this translates
        to an open/closed position on the gripper
        - For a fully-closed gripper, the right side joints are at their max, and the left side joints are at their min
        - Likewise, for a fully-open gripper, the right side is at the joint min, and the left at joint max

        Returns:
            tuple of:
                np.ndarray: Left-side gripper finger angles when closed. Shape (2,)
                np.ndarray: Left-side gripper finger angles when open. Shape (2,)
                np.ndarray: Right-side gripper finger angles when closed. Shape (2,)
                np.ndarray: Right-side gripper finger angles when open. Shape (2,)
        """
        left_joints = Astrobee.GRIPPER_JOINT_IDXS[:2]
        right_joints = Astrobee.GRIPPER_JOINT_IDXS[2:]
        # As a numpy array, each row will correspond to a joint, and the two columns are [min, max]
        left_closed, left_open = Astrobee.JOINT_POS_LIMITS[left_joints].T
        right_open, right_closed = Astrobee.JOINT_POS_LIMITS[right_joints].T
        return left_closed, left_open, right_closed, right_open

    def set_joint_angles(
        self, angles: npt.ArrayLike, indices: Optional[npt.ArrayLike] = None,
        do_step = True
    ):
        """Sets the joint angles for the Astrobee (either all joints, or a specified subset)

        Args:
            angles (npt.ArrayLike): Desired joint angles, in radians
            indices (npt.ArrayLike, optional): Indices of the joints to control. Defaults to None,
                in which case we assume all 7 joints will be set.

        Raises:
            ValueError: If the number of angles provided do not match the number of indices
            ValueError: If the angles are out of the joint limits for the specified indices
        """
        if indices is None:
            indices = list(range(Astrobee.NUM_JOINTS))
        angles = np.atleast_1d(angles)  # If scalar, ensure we don't have a 0-D array
        indices = np.atleast_1d(indices)
        if indices.shape != angles.shape:
            raise ValueError(
                "Number of angles must match with the number of provided indices"
            )
        if np.any(angles < Astrobee.JOINT_POS_LIMITS[indices, 0]) or np.any(
            angles > Astrobee.JOINT_POS_LIMITS[indices, 1]
        ):
            raise ValueError(
                f"Joint angle command is outside of joint limits.\nGot: {angles} for joints {indices}"
            )
        pybullet.setJointMotorControlArray(
            self.id, indices, pybullet.POSITION_CONTROL, angles
        )
        if do_step:
            tol = 0.01  # TODO TOTALLY ARBITRARY FOR NOW
            while np.any(np.abs(self.get_joint_angles(indices) - angles) > tol):
                pybullet.stepSimulation()
                time.sleep(1 / 120)  # TODO determine timestep

    def get_joint_angles(self, indices: Optional[npt.ArrayLike] = None) -> np.ndarray:
        """Gives the current joint angles for the Astrobee

        Args:
            indices (npt.ArrayLike, optional): Indices of the joints of interest. Defaults to None,
                in which case all joint angles will be returned

        Returns:
            np.ndarray: Joint angles (in radians), length = len(indices) or Astrobee.NUM_JOINTS
        """
        states = pybullet.getJointStates(self.id, indices)
        return np.array([state[0] for state in states])

    def set_arm_joints(self, angles: npt.ArrayLike) -> None:
        """Sets the joint angles for the arm (proximal + distal)

        Args:
            angles (npt.ArrayLike): Arm joint angles, length = 2
        """
        self.set_joint_angles(angles, Astrobee.ARM_JOINT_IDXS)

    def set_gripper_joints(self, angles: npt.ArrayLike, do_step=True) -> None:
        """Sets the joint angles for the gripper (left + right, proximal + distal)

        Args:
            angles (npt.ArrayLike): Gripper joint angles, length = 4
        """
        self.set_joint_angles(angles, Astrobee.GRIPPER_JOINT_IDXS, do_step)

    # **** TO IMPLEMENT: (maybe... some of these are just random ideas) ****
    #
    # def step(self, constraint=None, joint_pos=None, joint_vel=None, joint_torques=None):
    #     # TODO. Remove while loops from individual functions and use this instead?
    #     raise NotImplementedError

    # def set_joint_torques(self, torques, indices):
    #     raise NotImplementedError
    # def set_joint_vels(self, vels, indices):
    #     raise NotImplementedError

    # def is_near(
    #     self, pose: npt.ArrayLike, pos_tol: float = 1e-3, orn_tol: float = 1e-5
    # ) -> bool:
    #     """Confirms if the Astrobee is near a desired pose or not
    #     TODO finish this, and decide how we're setting poses
    #     Args:
    #         pose (npt.ArrayLike): _description_
    #         pos_tol (float, optional): _description_. Defaults to 1e-3.
    #         orn_tol (float, optional): _description_. Defaults to 1e-5.

    #     Returns:
    #         bool: _description_
    #     """
    #     raise NotImplementedError

    # TODO figure this out (use the gripper/distal transform?)
    # @property
    # def tcp_offset(self):
    #     return self._tcp_offset

    # Does this even need to be a property??
    # @tcp_offset.setter
    # def tcp_offset(self, offset):
    #     self._tcp_offset = offset

    # @property
    # def joint_reaction_forces(self) -> np.ndarray:
    #     # TODO: decide if we should use this
    #     # Need to figure out if we are enabling torque sensors on these joints
    #     raise NotImplementedError
    #     states = pybullet.getJointStates(self.id, list(range(Astrobee.NUM_JOINTS)))
    #     return [
    #         states[i][1] for i in range(Astrobee.NUM_JOINTS)
    #     ]  # Index 2: reaction forces

    # def set_ee_pose(self, pose):
    #     pass

    # TODO: should refine these states so it is clear what is going on
    # Should it be possible for the robot to be in multiple states? e.g. moving and manipulating??
    # Should we keep multiple states in separate enumerations?
    # These states are all just ideas for now
    # Add an error state?
    # class States(Enum):
    #     IDLE = 1
    #     PLANNING = 2
    #     MOVING = 3
    #     MANIPULATING = 4

    # def set_arm_pose_world(self, pose):
    #     pass


if __name__ == "__main__":
    initialize_pybullet()
    robot = Astrobee()
    run_sim()
