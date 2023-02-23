"""Manages the properties of the astrobee and all control-associated functions

In general, we assume that we're working with Honey. Multiple astrobees can be loaded, but
we assume that they all have the exact same configuration


TO ADD:
follow_trajectory -- given a sequence of poses for the constraints, continually update the state of the constraint to follow this
plan_trajectory -- given a pose target, plan a sequence of constraint values to get from the current pose to desired

TODO
- Can we rename the joints and the links in the urdf? e.g. remove "honey", "top_aft"
- What's the deal with the "top_aft" fixed joint? Is that part of the KDL workaround?
- Decide how to define the gripper position. Map it to a range between 0-100 (fully closed vs fully open?)
- Figure out the sleep time in the while loops. What value should we use?
"""

import time
from enum import Enum
from typing import Optional

import pybullet
import numpy as np
import numpy.typing as npt

# TODO REMOVE THIS (integrate it into the rotations file)
from pytransform3d.rotations import (
    axis_angle_from_two_directions,
    quaternion_from_axis_angle,
)


from pyastrobee.utils.bullet_utils import initialize_pybullet, run_sim
from pyastrobee.utils.transformations import make_transform_mat
from pyastrobee.utils.rotations import quat_to_rmat, quaternion_dist, quaternion_interp
from pyastrobee.utils.poses import tmat_to_pos_quat
from pyastrobee.config import astrobee_transforms
from pyastrobee.utils.quaternion import Quaternion
from pyastrobee.utils.math_utils import normalize


# TODO: should refine these states so it is clear what is going on
# Should it be possible for the robot to be in multiple states? e.g. moving and manipulating??
# Should we keep multiple states in separate enumerations?
# These states are all just ideas for now
# Add an error state?
class States(Enum):
    IDLE = 1
    PLANNING = 2
    MOVING = 3
    MANIPULATING = 4


class Astrobee:
    # TODO docstring

    # TODO add these to a constants or config file? Probably fine here for now
    URDF = "pyastrobee/urdf/astrobee.urdf"
    LOADED_IDS = []  # Initialization
    NUM_ROBOTS = 0  # Initialization. TODO make this into a property?
    NUM_JOINTS = 7
    NUM_LINKS = 7
    TRANSFORMS = astrobee_transforms  # TODO figure out if this is the best way to store this info

    GRIPPER_JOINT_IDXS = [3, 4, 5, 6]
    ARM_JOINT_IDXS = [1, 2]

    # Constants from geometry.config in the NASA code
    # LENGTH = 0.2
    # REACH = 0.155

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

    # Joint limit information is extracted from the URDF
    # Joint pos limits are [lower, upper] for each joint
    JOINT_POS_LIMITS = [
        [0.0, 0.0],  # top aft (fixed)
        [-2.0944, 1.57079],  # arm proximal joint
        [-1.57079, 1.57079],  # arm distal joint
        [0.349066, 0.698132],  # gripper left proximal joint
        [-1.22173, -0.69813],  # gripper left distal joint
        [-0.698132, -0.349066],  # gripper right proximal joint
        [0.69813, 1.22173],  # gripper right distal joint
    ]
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

    # TODO change default initialization values
    def __init__(
        self,
        pose: npt.ArrayLike = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        arm_joints: npt.ArrayLike = None,
        gripper_state=None,  # TODO figure out what type this should be
    ):
        if not pybullet.isConnected():
            raise ConnectionError(
                "Need to connect to pybullet before initializing an astrobee"
            )
        self.id = pybullet.loadURDF(Astrobee.URDF, pose[:3], pose[3:])
        Astrobee.LOADED_IDS.append(self.id)
        Astrobee.NUM_ROBOTS += 1
        # TODO set the arm joints
        # TODO set the gripper

        # Property internal variables
        self._tcp_offset = None  # TODO
        # Constraint is for position control
        # TODO check on the input parameters here
        self.constraint_id = pybullet.createConstraint(
            self.id, -1, -1, -1, pybullet.JOINT_FIXED, None, (0, 0, 0), (0, 0, 0)
        )

        # TODO we should probably initialize the astrobee with the gripper open
        # It initializes in a weird state if you just load the URDF

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

    # TODO figure this out (use the gripper/distal transform?)
    @property
    def tcp_offset(self):
        return self._tcp_offset

    # Does this even need to be a property??
    @tcp_offset.setter
    def tcp_offset(self, offset):
        self._tcp_offset = offset

    @property
    def pose(self) -> np.ndarray:
        """The current robot pose (position + XYZW quaternion) expressed in world frame

        Returns:
            np.ndarray: Position and quaternion, size (7,)
        """
        pos, orn = pybullet.getBasePositionAndOrientation(self.id)
        return np.concatenate([pos, orn])

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
        return [states[i][0] for i in range(Astrobee.NUM_JOINTS)]  # Index 0: position

    @property
    def joint_vels(self) -> np.ndarray:
        """Angular velocities (radians/sec) of each joint on the Astrobee

        Returns:
            np.ndarray: Joint velocities, shape (NUM_JOINTS,)
        """
        states = pybullet.getJointStates(self.id, list(range(Astrobee.NUM_JOINTS)))
        return [states[i][1] for i in range(Astrobee.NUM_JOINTS)]  # Index 1: velocity

    @property
    def joint_reaction_forces(self) -> np.ndarray:
        # TODO: decide if we should use this
        # Need to figure out if we are enabling torque sensors on these joints
        # states = pybullet.getJointStates(self.id, list(range(Astrobee.NUM_JOINTS)))
        # return [
        #     states[i][1] for i in range(Astrobee.NUM_JOINTS)
        # ]  # Index 2: reaction forces
        raise NotImplementedError

    @property
    def joint_torques(self) -> np.ndarray:
        """Torques (N-m) applied by each joint on the Astrobee

        Returns:
            np.ndarray: Joint torques, shape (NUM_JOINTS,)
        """
        states = pybullet.getJointStates(self.id, list(range(Astrobee.NUM_JOINTS)))
        return [states[i][3] for i in range(Astrobee.NUM_JOINTS)]  # Index 3: torque

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

    def set_ee_pose(self, pose):
        pass

    def set_gripper_position(self, position: float) -> None:
        """Sets the gripper to a position between 0 (fully closed) to 100 (fully open)

        TODO decide if we need finer-grain control of the individual joints, or if this integer-position is fine

        Args:
            position (float): Gripper position, in range [0, 100]
        """
        if position < 0 or position > 100:
            raise ValueError("Position should be in range [0, 100]")
        l_closed, l_open, r_closed, r_open = self._get_gripper_joint_ranges()
        left_pos = l_closed + (position / 100) * (l_open - l_closed)
        right_pos = r_closed + (position / 100) * (r_open - r_closed)
        angle_cmd = [*left_pos, *right_pos]
        pybullet.setJointMotorControlArray(
            self.id, Astrobee.GRIPPER_JOINT_IDXS, pybullet.POSITION_CONTROL, angle_cmd
        )
        # TODO this is a lot of computations for just setting the gripper
        # (because we recalculate the gripper position at each step). Is there a better way to do this?
        while self.gripper_position != position:  # Do we need to add any tolerance?
            pybullet.stepSimulation()
            time.sleep(1 / 120)

    def _get_gripper_joint_ranges(self) -> tuple[np.ndarray, ...]:
        """Helper function to determine the range of motion (closed -> open) of the gripper joints

        - This is a bit confusing because of how the URDF specifies joint min/max and how this translates
        to an open/closed position on the gripper
        - For a fully-closed gripper, the right side joints are at their max, and the left side joints are at their min
        - Likewise, for a fully-open gripper, the right side is at the joint min, and the left at joint max

        Returns:
            _type_: _description_ TODO
        """
        left_gripper_joints = Astrobee.GRIPPER_JOINT_IDXS[:2]
        right_gripper_joints = Astrobee.GRIPPER_JOINT_IDXS[2:]
        # As a numpy array, each row will correspond to a joint, and the two columns are [min, max]
        joint_lims_array = np.array(Astrobee.JOINT_POS_LIMITS)
        left_closed, left_open = joint_lims_array[left_gripper_joints].T
        right_open, right_closed = joint_lims_array[right_gripper_joints].T
        return left_closed, left_open, right_closed, right_open

    def open_gripper(self) -> None:
        """Fully opens the gripper"""
        self.set_gripper_position(100)

    def close_gripper(self) -> None:
        """Fully closes the gripper

        TODO add force/torque control?
        """
        self.set_gripper_position(0)

    @property  # Is this better as a property or as a "getter"?
    def gripper_position(self) -> int:
        """The current position of the gripper, in range [0, 100]

        Returns:
            int: Position of the gripper, an integer between 0 (closed) and 100 (open)
        """
        joint_states = pybullet.getJointStates(self.id, Astrobee.GRIPPER_JOINT_IDXS)
        joint_angles = [state[0] for state in joint_states]
        l_angles, r_angles = np.split(joint_angles, [2])
        l_closed, l_open, r_closed, r_open = self._get_gripper_joint_ranges()
        l_pct = 100 * (l_angles - l_closed) / (l_open - l_closed)
        r_pct = 100 * (r_angles - r_closed) / (r_open - r_closed)
        return np.round(np.average(np.concatenate([l_pct, r_pct]))).astype(int)

    def set_arm_pose_world(self, pose):
        pass

    def set_robot_pose(self, pose):
        pass

    def set_joint_angles(
        self, angles: npt.ArrayLike, indices: Optional[npt.ArrayLike] = None
    ):
        if indices is None:
            indices = list(range(Astrobee.NUM_JOINTS))
        if len(indices) != len(angles):
            raise ValueError(
                "Number of angles must match with the length of provided indices"
            )
        pybullet.setJointMotorControlArray(
            self.id, indices, pybullet.POSITION_CONTROL, angles
        )

    def set_joint_torques(self, torques):
        raise NotImplementedError

    def set_joint_torques_by_index(self, torques, indices):
        raise NotImplementedError

    def set_joint_vels(self, vels):
        raise NotImplementedError

    def set_joint_vels_by_index(self, vels, indices):
        raise NotImplementedError

    def is_near(
        self, pose: npt.ArrayLike, pos_tol: float = 1e-3, orn_tol: float = 1e-5
    ) -> bool:
        """Confirms if the Astrobee is near a desired pose or not
        TODO finish this, and decide how we're setting poses
        Args:
            pose (npt.ArrayLike): _description_
            pos_tol (float, optional): _description_. Defaults to 1e-3.
            orn_tol (float, optional): _description_. Defaults to 1e-5.

        Returns:
            bool: _description_
        """
        raise NotImplementedError

    def step(self, constraint=None, joint_pos=None, joint_vel=None, joint_torques=None):
        # TODO. Remove while loops from individual functions and use this instead?
        raise NotImplementedError

    def align_to(self, orn: npt.ArrayLike) -> None:
        """Rotates the Astrobee about its current position to align with a specified orientation

        TODO this is super hacky, needs to be cleaned up significantly!!
        And, need to improve the stepping mechanic

        Args:
            goal_orn (npt.ArrayLike): Desired XYZW quaternion orientation
        """
        # TODO: use a check_quaternion function instead of this
        if len(orn) != 4:
            raise ValueError(f"Invalid quaternion.\nGot: {orn}")
        # CLEAN THIS UP!!!
        initial_pose = self.pose
        pos = initial_pose[:3]
        tol = 0.03  # placeholder
        # These don't seem to need to be Quaternion objects?
        q1 = Quaternion(xyzw=self.orientation)
        q2 = Quaternion(xyzw=orn)
        # This method of stepping through the interpolated values is not ideal
        # It should use some sort of stepsize or enforce velocity constraints (TODO)
        pct = 0.01
        dpct = 0.01
        while quaternion_dist(self.orientation, orn) > tol:
            q = quaternion_interp(q1, q2, pct)
            # TODO decide if any other inputs to the change constraint function are needed
            pybullet.changeConstraint(self.constraint_id, pos, q)
            pybullet.stepSimulation()
            time.sleep(1 / 120)
            pct = min(pct + dpct, 1)

    def follow_line_to(self, position: npt.ArrayLike) -> None:
        if len(position) != 3:
            raise ValueError(f"Invalid position.\nGot: {position}")
        pos, orn = np.split(self.pose, [3])
        step = 0.1  # Totally arbitrary for now
        # Need to improve the stepping mechanic.. increase and decrease speed/stepsize as needed
        while np.linalg.norm(pos - position) > step:
            direction = normalize(position - pos)  # Unit vector
            new_pos = pos + step * direction
            pybullet.changeConstraint(self.constraint_id, new_pos, orn)
            pybullet.stepSimulation()
            time.sleep(1 / 120)
            pos = self.position  # Update after moving

    def go_to_pose(self, pose: npt.ArrayLike) -> None:
        # TODO improve this pose checking. Check if Pose() instance too?
        if len(pose) != 7:
            raise ValueError(f"Invalid pose.\nGot: {pose}")
        goal_pos, goal_orn = np.split(pose, [3])
        direction = goal_pos - self.position
        # Move to the next position specified if we are far from it
        # If we are within a stepsize, we should just orient ourselves to avoid unnecessary rotation
        tol = 1e-3  # UPDATE THIS ONCE YOU DECIDE ON A STEPSIZE
        if np.linalg.norm(direction) > tol:
            axis_and_angle = axis_angle_from_two_directions(self.heading, direction)
            axis, angle = np.split(axis_and_angle, [3])
            q = Quaternion(wxyz=quaternion_from_axis_angle(axis_and_angle))
            self.align_to(q.xyzw)
            self.follow_line_to(goal_pos)
        # Orient the astrobee to the desired ending orientation
        self.align_to(goal_orn)


if __name__ == "__main__":
    initialize_pybullet()
    robot = Astrobee()
    run_sim()
