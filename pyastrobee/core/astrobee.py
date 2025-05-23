"""Manages the properties of the astrobee and all control-associated functions

In general, we assume that we're working with Honey. Multiple astrobees can be loaded, but
we assume that they all have the exact same configuration
"""

from typing import Optional, Union
from enum import Enum

import pybullet
from pybullet_utils.bullet_client import BulletClient
import numpy as np
import numpy.typing as npt


from pyastrobee.utils.bullet_utils import initialize_pybullet, run_sim
from pyastrobee.utils.transformations import (
    make_transform_mat,
    invert_transform_mat,
    transform_point,
)
from pyastrobee.utils.rotations import quat_to_rmat
from pyastrobee.utils.poses import tmat_to_pos_quat, pos_quat_to_tmat
from pyastrobee.config import astrobee_transforms
from pyastrobee.config.astrobee_geom import COLLISION_RADIUS
from pyastrobee.utils.python_utils import print_green
from pyastrobee.utils.dynamics import (
    inertial_transformation,
    state_matrix,
    control_matrix,
)


class Astrobee:
    """Astrobee class for managing control, states, and properties

    Args:
        pose (npt.ArrayLike, optional): Initial pose of the astrobee when loaded. Defaults to
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0) (At origin, pointed forward along x axis).
        arm_joints (npt.ArrayLike, optional): Initial position of the arm's joints. Defaults to
            (0.0, 0.0) (Hanging straight down)
        gripper_pos (float, optional): Initial gripper position, in [0, 100]. Defaults to 100 (fully open)
        client (BulletClient, optional): If connecting to multiple physics servers, include the client
            (the class instance, not just the ID) here. Defaults to None (use default connected client)

    Raises:
        ConnectionError: If a pybullet server is not connected before initialization
    """

    URDF = "pyastrobee/assets/urdf/astrobee/astrobee.urdf"
    NUM_JOINTS = 7
    NUM_LINKS = 8
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
    JOINT_EFFORT_LIMITS = np.array(
        [
            0.0,  # top aft (fixed)
            1.0,  # arm proximal joint
            1.0,  # arm distal joint
            0.1,  # gripper left proximal joint
            0.1,  # gripper left distal joint
            0.1,  # gripper right proximal joint
            0.1,  # gripper right distal joint
        ]
    )
    JOINT_VEL_LIMITS = np.array(
        [
            0.00,  # top aft (fixed)
            0.12,  # arm proximal joint
            0.12,  # arm distal joint
            0.12,  # gripper left proximal joint
            0.12,  # gripper left distal joint
            0.12,  # gripper right proximal joint
            0.12,  # gripper right distal joint
        ]
    )

    # Bounding sphere for collision modeling
    COLLISION_RADIUS = COLLISION_RADIUS

    class Joints(Enum):
        """Enumerates the different joints on the astrobee via their Pybullet index"""

        # Comments indicate the name of the joint in the URDF
        ARM_BASE = 0  # top_aft (fixed joint)
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
        BODY = -1  # honey_body
        ARM_BASE = 0  # honey_top_aft
        ARM_PROXIMAL = 1  # honey_top_aft_arm_proximal_link
        ARM_DISTAL = 2  # honey_top_aft_arm_distal_link
        GRIPPER_LEFT_PROXIMAL = 3  # honey_top_aft_gripper_left_proximal_link
        GRIPPER_LEFT_DISTAL = 4  # honey_top_aft_gripper_left_distal_link
        GRIPPER_RIGHT_PROXIMAL = 5  # honey_top_aft_gripper_right_proximal_link
        GRIPPER_RIGHT_DISTAL = 6  # honey_top_aft_gripper_right_distal_link

    def __init__(
        self,
        pose: npt.ArrayLike = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        arm_joints: npt.ArrayLike = (0.0, 0.0),
        gripper_pos: float = 100,
        client: Optional[BulletClient] = None,
    ):
        self.client: pybullet = pybullet if client is None else client
        if not self.client.isConnected():
            raise ConnectionError(
                "Need to connect to pybullet before initializing an astrobee"
            )
        self.id = self.client.loadURDF(
            Astrobee.URDF, pose[:3], pose[3:], flags=pybullet.URDF_USE_INERTIA_FROM_FILE
        )
        self._dt = self.client.getPhysicsEngineParameters()["fixedTimeStep"]
        self.set_gripper_position(gripper_pos, force=True)
        self.set_arm_joints(arm_joints, force=True)
        # Initialize dynamics info with estimated numbers from NASA, we can recompute based on the sim state later
        # if desired. Values are from A Brief Guide to Astrobee
        self._mass = 9.58  # kg
        self._inertia = np.diag([0.153, 0.143, 0.162])  # kg-m^2
        self._inv_inertia = np.linalg.inv(self._inertia)
        self._local_com_position = np.zeros(3)  # Init, not accurate
        # TODO decide if we should recompute automatically???
        # self.recompute_inertial_properties()
        print_green("Astrobee is ready")

    def unload(self) -> None:
        """Remove the Astrobee from the simulation"""
        self.client.removeBody(self.id)

    @property
    def pose(self) -> np.ndarray:
        """The current robot pose (position + XYZW quaternion) expressed in world frame

        Returns:
            np.ndarray: Position and quaternion, size (7,)
        """
        pos, orn = self.client.getBasePositionAndOrientation(self.id)
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

        Returns:
            np.ndarray: [vx, vy, vz] linear velocities, shape (3,)
        """
        lin_vel, _ = self.client.getBaseVelocity(self.id)
        return np.array(lin_vel)

    @property
    def angular_velocity(self) -> np.ndarray:
        """Angular velocity of the Astrobee, about the world frame xyz axes

        Returns:
            np.ndarray: [wx, wy, wz] angular velocities, shape (3,)
        """
        _, ang_vel = self.client.getBaseVelocity(self.id)
        return np.array(ang_vel)

    @property
    def ee_pose(self) -> np.ndarray:
        """The current end-effector pose (position + XYZW quaternion) expressed in world frame

        Returns:
            np.ndarray: Position and quaternion, size (7,)
        """
        return tmat_to_pos_quat(self.ee_tmat)

    @property
    def ee_tmat(self) -> np.ndarray:
        """The current end-effector transformation matrix (gripper-to-world)

        Returns:
            np.ndarray: Transformation matrix, shape (4, 4)
        """
        T_G2D = Astrobee.TRANSFORMS.GRIPPER_TO_ARM_DISTAL  # Gripper to distal
        T_D2W = self.get_link_transform(
            Astrobee.Links.ARM_DISTAL.value
        )  # Distal to world
        return T_D2W @ T_G2D

    @property
    def joint_angles(self) -> np.ndarray:
        """Angular positions (radians) of each joint on the Astrobee

        Returns:
            np.ndarray: Joint angles, shape (NUM_JOINTS,)
        """
        # States: tuple[tuple], size (7, 4)
        # 7 corresponds to NUM_JOINTS
        # 4 corresponds to position, velocity, reaction forces, and applied torque
        states = self.client.getJointStates(self.id, list(range(Astrobee.NUM_JOINTS)))
        # Index 0: position
        return np.array([states[i][0] for i in range(Astrobee.NUM_JOINTS)])

    @property
    def joint_vels(self) -> np.ndarray:
        """Angular velocities (radians/sec) of each joint on the Astrobee

        Returns:
            np.ndarray: Joint velocities, shape (NUM_JOINTS,)
        """
        states = self.client.getJointStates(self.id, list(range(Astrobee.NUM_JOINTS)))
        # Index 1: velocity
        return np.array([states[i][1] for i in range(Astrobee.NUM_JOINTS)])

    @property
    def joint_torques(self) -> np.ndarray:
        """Torques (N-m) applied by each joint on the Astrobee

        Returns:
            np.ndarray: Joint torques, shape (NUM_JOINTS,)
        """
        states = self.client.getJointStates(self.id, list(range(Astrobee.NUM_JOINTS)))
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

    @property
    def inertia(self) -> np.ndarray:
        """Body inertia tensor for the Astrobee, shape (3, 3)"""
        return self._inertia

    @property
    def inv_inertia(self) -> np.ndarray:
        """Inverse of the Astrobee's body inertia tensor, shape (3, 3)"""
        return self._inv_inertia

    @property
    def world_inertia(self) -> np.ndarray:
        """World-frame inertia tensor of the Astrobee, shape (3, 3)

        This takes into account the current rotation of Astrobee
        """
        R = self.rmat
        return R @ self.inertia @ R.T

    @property
    def world_inv_inertia(self) -> np.ndarray:
        """Inverse of the world-frame inertia tensor of the Astrobee, shape (3, 3)

        This takes into account the current rotation of Astrobee
        """
        R = self.rmat
        return R @ self.inv_inertia @ R.T

    @property
    def mass(self) -> float:
        """Mass of the Astrobee"""
        return self._mass

    @property
    def local_com_position(self) -> np.ndarray:
        """Position of the center of mass of the robot w.r.t. the base, in local frame. Shape (3,)"""
        return self._local_com_position

    @property
    def world_com_position(self) -> np.ndarray:
        """Position of the center of mass of the robot, in world frame. Shape (3,)"""
        return transform_point(self.tmat, self.local_com_position)

    @property
    def state_space_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """The state and control matrices A and B, such that x_dot = Ax + Bu

        We assume that the state x = [position, velocity, quaternion, angular velocity] ∈ R13
        and that the control u = [force, torque] ∈ R6

        We linearize the system about the current state

        Returns:
            tuple[np.ndarray, np.ndarray]:
                np.ndarray: A: State matrix, shape (13, 13)
                np.ndarray: B: Control matrix, shape (13, 6)
        """
        # TODO: decide if using the true joint-angle-based inertia tensor
        _, q, _, w = self.dynamics_state
        R = self.rmat
        # Use inertias defined in the world frame
        inertia = R @ self.inertia @ R.T
        inv_inertia = R @ self.inv_inertia @ R.T
        return (
            state_matrix(q, w, inertia, inv_inertia),
            control_matrix(self.mass, inv_inertia),
        )

    @property
    def state_vector(self) -> np.ndarray:
        """The state vector x, such that x_dot = Ax + Bu

        We compose the state as [position, velocity, quaternion, angular velocity] ∈ R13
        """
        pos, orn = self.client.getBasePositionAndOrientation(self.id)
        lin_vel, ang_vel = self.client.getBaseVelocity(self.id)
        return np.concatenate([pos, lin_vel, orn, ang_vel])

    @property
    def dynamics_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Current state of the Astrobee's dynamics: Position, orientation, linear vel, and angular vel

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                np.ndarray: Position, shape (3,)
                np.ndarray: XYZW quaternion orientation, shape (4,)
                np.ndarray: Linear velocity, shape (3,)
                np.ndarray: Angular velocity, shape (3,)
        """
        pos, orn = self.client.getBasePositionAndOrientation(self.id)
        lin_vel, ang_vel = self.client.getBaseVelocity(self.id)
        return np.array(pos), np.array(orn), np.array(lin_vel), np.array(ang_vel)

    @property
    def mass_matrix(self) -> np.ndarray:
        """Mass/Inertia matrix for the Astrobee, given its current configuration

        - This is used to determine the kinetic energy (K = (1/2) * qdot.T @ M @ qdot) or the relationship between
          joint accelerations and torque (M * joint_accels + centrifugal_coriolis_vec + gravity_vec = torque)

        Returns:
            np.ndarray: The mass matrix, shape (12, 12). (12 is the number of degrees of freedom
                of the Astrobee - 6 DOF for a floating base, plus 6 for the six non-fixed joints)
        """
        # Inputs must be a lists (no numpy) or else pybullet will seg fault
        M = self.client.calculateMassMatrix(self.id, list(self.joint_angles))
        return np.array(M)

    def get_jacobians(
        self, link: Union[Links, int], local_pos: npt.ArrayLike = (0.0, 0.0, 0.0)
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate the linear and angular jacobians (Jv and Jw) for a point on a link

        - These relate joint motion and task-space motion: [v; w] = [Jv; Jw] * dq
        - These jacobians will have a 12 columns corresponding to 6 DOF from the Astrobee's floating base, plus another
          6 DOF from the non-fixed joints

        Args:
            link (Union[Links, int]): Link or link index of interest.
                Common links: For the base, set this to -1. For the arm distal link, set this to 2
            local_pos (npt.ArrayLike, optional): Position in the link's reference frame. Defaults to [0.0, 0.0, 0.0].
                For the grasp point, use the position from the calibrated distal/grasp transformation

        Returns:
            tuple[np.ndarray, np.ndarray]:
                np.ndarray: Jv: Linear jacobian, shape (3, 12)
                np.ndarray: Jw: Angular jacobian, shape (3, 12)
        """
        if isinstance(link, Astrobee.Links):
            link = link.value
        ndof = 6  # 7 joints, but 1 fixed
        # The quickstart guide says that the joint velocities and desired accelerations are just there
        # for an internal call to calculateInverseDynamics (and maybe aren't really meaningful?)
        desired_accels = ndof * [0.0]
        # All inputs must be lists (no numpy) or else pybullet will seg fault
        Jv, Jw = self.client.calculateJacobian(
            self.id,
            link,
            list(local_pos),
            list(self.joint_angles)[1:],  # Don't include the first fixed joint
            list(self.joint_vels)[1:],  # Don't include the first fixed joint
            desired_accels,
        )
        return np.array(Jv), np.array(Jw)

    def get_link_transform(self, link_index: Union[Links, int]) -> np.ndarray:
        """Calculates the transformation matrix (w.r.t the world) for a specified link

        Args:
            link_index (int): Index of the link on the robot

        Returns:
            np.ndarray: Transformation matrix (link to world). Shape = (4,4)
        """
        if isinstance(link_index, Astrobee.Links):
            link_index = link_index.value
        # We have 8 links, indexed from -1 to 6
        # Pybullet does not allow access to the base link (-1) through getLinkState
        # So, use this only for non-base links
        if link_index > 6 or link_index < 0:
            raise ValueError(f"Invalid link index: {link_index}")
        link_state = self.client.getLinkState(
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
        joint_states = self.client.getJointStates(self.id, Astrobee.GRIPPER_JOINT_IDXS)
        joint_angles = [state[0] for state in joint_states]
        l_angles = joint_angles[:2]
        r_angles = joint_angles[2:]
        l_closed, l_open, r_closed, r_open = self._get_gripper_joint_ranges()
        l_pct = 100 * (l_angles - l_closed) / (l_open - l_closed)
        r_pct = 100 * (r_angles - r_closed) / (r_open - r_closed)
        return np.round(np.average(np.concatenate([l_pct, r_pct]))).astype(int)

    # TODO decide if we need finer-grain control of the individual joints, or if this integer-position is fine
    def set_gripper_position(
        self, position: float, force: bool = False, wait: bool = False
    ) -> None:
        """Sets the gripper to a position between 0 (fully closed) to 100 (fully open)

        Args:
            position (float): Gripper position, in range [0, 100]
            force (bool, optional): Whether to (non-physically) instantly reset the gripper position, instead
                of stepping the sim. Should only be used at initialization. Defaults to False
            wait (bool, optional): Whether to wait until the arm reaches the desired state by stepping the sim
                forwards in a blocking manner. Defaults to False
        """
        if position < 0 or position > 100:
            raise ValueError("Position should be in range [0, 100]")
        l_closed, l_open, r_closed, r_open = self._get_gripper_joint_ranges()
        left_pos = l_closed + (position / 100) * (l_open - l_closed)
        right_pos = r_closed + (position / 100) * (r_open - r_closed)
        angle_cmd = [*left_pos, *right_pos]
        self.set_gripper_joints(angle_cmd, force, wait)

    def open_gripper(self) -> None:
        """Fully opens the gripper"""
        self.set_gripper_position(100)

    # TODO add force/torque control?
    def close_gripper(self) -> None:
        """Fully closes the gripper"""
        self.set_gripper_position(0)

    # TODO rework this?
    def _get_gripper_joint_ranges(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Helper function to determine the range of motion (closed -> open) of the gripper joints

        - This is a bit confusing because of how the URDF specifies joint min/max and how this translates
        to an open/closed position on the gripper
        - For a fully-closed gripper, the right side joints are at their max, and the left side joints are at their min
        - Likewise, for a fully-open gripper, the right side is at the joint min, and the left at joint max

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        self,
        angles: npt.ArrayLike,
        indices: Optional[npt.ArrayLike] = None,
        force: bool = False,
        wait: bool = False,
    ):
        """Sets the joint angles for the Astrobee (either all joints, or a specified subset)

        Args:
            angles (npt.ArrayLike): Desired joint angles, in radians
            indices (npt.ArrayLike, optional): Indices of the joints to control. Defaults to None,
                in which case we assume all 7 joints will be set.
            force (bool, optional): Whether to (non-physically) instantly reset the joint state.
                Should only be used at initialization. Defaults to False
            wait (bool, optional): Whether to wait until the arm reaches the desired state by stepping the sim
                forwards in a blocking manner. Defaults to False

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
        if force:
            for ind, angle in zip(indices, angles):
                self.client.resetJointState(self.id, ind, angle)
        # Set the position control for the arm so Pybullet will correct for disturbances
        self.client.setJointMotorControlArray(
            self.id,
            indices,
            self.client.POSITION_CONTROL,
            angles,
            forces=self.JOINT_EFFORT_LIMITS[indices],
        )
        # Step the sim until the arm is at the desired angle, if waiting for it to reach the position
        if wait:
            tol = 0.01  # TODO TOTALLY ARBITRARY FOR NOW
            while np.any(np.abs(self.get_joint_angles(indices) - angles) > tol):
                self.client.stepSimulation()

    def get_joint_angles(self, indices: Optional[npt.ArrayLike] = None) -> np.ndarray:
        """Gives the current joint angles for the Astrobee

        Args:
            indices (npt.ArrayLike, optional): Indices of the joints of interest. Defaults to None,
                in which case all joint angles will be returned

        Returns:
            np.ndarray: Joint angles (in radians), length = len(indices) or Astrobee.NUM_JOINTS
        """
        states = self.client.getJointStates(self.id, indices)
        return np.array([state[0] for state in states])

    def set_arm_joints(
        self, angles: npt.ArrayLike, force: bool = False, wait: bool = False
    ) -> None:
        """Sets the joint angles for the arm (proximal + distal)

        Args:
            angles (npt.ArrayLike): Arm joint angles, length = 2
            force (bool, optional): Whether to (non-physically) instantly reset the joint states.
                Should only be used at initialization. Defaults to False
            wait (bool, optional): Whether to wait until the arm reaches the desired state by stepping the sim
                forwards in a blocking manner. Defaults to False
        """
        self.set_joint_angles(angles, Astrobee.ARM_JOINT_IDXS, force, wait)

    def set_gripper_joints(
        self, angles: npt.ArrayLike, force: bool = False, wait: bool = False
    ) -> None:
        """Sets the joint angles for the gripper (left + right, proximal + distal)

        Args:
            angles (npt.ArrayLike): Gripper joint angles, length = 4
            force (bool, optional): Whether to (non-physically) instantly reset the gripper joints.
                Should only be used at initialization. Defaults to False
            wait (bool, optional): Whether to wait until the arm reaches the desired state by stepping the sim
                forwards in a blocking manner. Defaults to False
        """
        self.set_joint_angles(angles, Astrobee.GRIPPER_JOINT_IDXS, force, wait)

    def reset_to_ee_pose(self, pose: npt.ArrayLike) -> None:
        """Resets the position of the robot to achieve a target end-effector pose

        This will currently NOT adjust any of the joints in a "smart" way, it will just reset the position of the base
        given the current joint configuration

        Args:
            pose (npt.ArrayLike): Desired position + XYZW quaternion end-effector pose, shape (7,)
        """
        # Notation: EE: End effector, B: Base, W: World
        des_EE2W = pos_quat_to_tmat(pose)
        cur_B2W = pos_quat_to_tmat(self.pose)
        cur_EE2W = pos_quat_to_tmat(self.ee_pose)
        cur_W2EE = invert_transform_mat(cur_EE2W)
        cur_B2EE = cur_W2EE @ cur_B2W
        des_B2W = des_EE2W @ cur_B2EE
        self.reset_to_base_pose(tmat_to_pos_quat(des_B2W))

    def reset_to_base_pose(self, pose: npt.ArrayLike) -> None:
        """Resets the base of the robot to a target pose

        Args:
            pose (npt.ArrayLike): Desired position + XYZW quaternion pose of the Astrobee's base, shape (7,)
        """
        self.client.resetBasePositionAndOrientation(self.id, pose[:3], pose[3:])

    def localize(self):
        raise NotImplementedError()  # TODO.. see dynamics state. Should have a noise parameter

    def recompute_inertial_properties(self) -> None:
        """Calculate the inertial properties based on the current state of the robot in sim

        This is more accurate than the fixed, base-only values from NASA's documentation, but it is fairly expensive to
        compute and should NOT be done on every simulation step.

        This will update the mass, inertia, inv_inertia, and center of mass
        """
        # Note: Mass will be fixed, but it is not necessarily the same value as provided by NASA
        mass = 0.0
        inertia = np.zeros((3, 3))
        com = np.zeros(3)
        T_B2W = self.tmat  # Base to world
        for link in Astrobee.Links:
            link_info = pybullet.getDynamicsInfo(self.id, link.value)
            link_mass = link_info[0]
            link_inertia_diagonal = link_info[2]
            if link.value == -1:  # Separate handling for base link
                inertia += np.diag(link_inertia_diagonal)
                com += link_mass * T_B2W[:3, 3]
            else:
                T_L2W = self.get_link_transform(link.value)  # Link to world
                T_L2B = invert_transform_mat(T_B2W) @ T_L2W  # Link to base
                inertia += inertial_transformation(
                    link_mass, np.diag(link_inertia_diagonal), T_L2B
                )
                com += link_mass * T_L2W[:3, 3]
            mass += link_mass
        com /= mass
        self._local_com_position = T_B2W[:3, :3].T @ (com - T_B2W[:3, 3])
        self._mass = mass
        self._inertia = inertia
        self._inv_inertia = np.linalg.inv(inertia)

    def store_arm(self, force: bool = False, wait: bool = False):
        """Folds the Astrobee's arm into its body

        Note: Storing the arm reduces the products of inertia, so this is the preferable
        configuration if not manipulating any objects

        Args:
            force (bool, optional): Whether to (non-physically) instantly reset the joints.
                Should only be used at initialization. Defaults to False
            wait (bool, optional): Whether to wait until the arm reaches the desired state by stepping the sim
                forwards in a blocking manner. Defaults to False
        """
        self.set_arm_joints([Astrobee.JOINT_POS_LIMITS[1, 1], 0], force, wait)
        self.set_gripper_position(0, force, wait)

    def deploy_arm(self, force: bool = False, wait: bool = False):
        """Sets the arm to the default position, with the gripper fully open

        Args:
            force (bool, optional): Whether to (non-physically) instantly reset the joints.
                Should only be used at initialization. Defaults to False
            wait (bool, optional): Whether to wait until the arm reaches the desired state by stepping the sim
                forwards in a blocking manner. Defaults to False
        """
        self.set_arm_joints([0, 0], force, wait)
        self.set_gripper_position(100, force, wait)

    @property
    def full_state(self) -> tuple[np.ndarray, ...]:
        """All information required to fully reset the state of the Astrobee

        Returns:
            tuple[np.ndarray, ...]:
                np.ndarray: Position, shape (3,)
                np.ndarray: Orientation (XYZW quaternion), shape (4,)
                np.ndarray: Linear velocity, shape (3,)
                np.ndarray: Angular velocity, shape (3,)
                np.ndarray: Joint positions, shape (NUM_JOINTS,)
                np.ndarray: Joint velocities, shape (NUM_JOINTS,)
        """
        pos, orn = self.client.getBasePositionAndOrientation(self.id)
        vel, ang_vel = self.client.getBaseVelocity(self.id)
        joint_states = self.client.getJointStates(
            self.id, list(range(Astrobee.NUM_JOINTS))
        )
        joint_positions = np.empty(Astrobee.NUM_JOINTS)
        joint_vels = np.empty(Astrobee.NUM_JOINTS)
        for i in range(Astrobee.NUM_JOINTS):
            joint_positions[i] = joint_states[i][0]
            joint_vels[i] = joint_states[i][1]
        return (
            np.array(pos),
            np.array(orn),
            np.array(vel),
            np.array(ang_vel),
            joint_positions,
            joint_vels,
        )

    def reset_full_state(
        self,
        pos: npt.ArrayLike,
        orn: npt.ArrayLike,
        vel: npt.ArrayLike,
        omega: npt.ArrayLike,
        q: npt.ArrayLike,
        qdot: npt.ArrayLike,
    ):
        """Fully resets the state of the Astrobee

        Args:
            pos (npt.ArrayLike): Position, shape (3,)
            orn (npt.ArrayLike): Orientation (XYZW quaternion), shape (4,)
            vel (npt.ArrayLike): Linear velocity, shape (3,)
            omega (npt.ArrayLike): Angular velocity, shape (3,)
            q (npt.ArrayLike): Joint positions, shape (NUM_JOINTS,)
            qdot (npt.ArrayLike): Joint velocities, shape (NUM_JOINTS,)
        """
        self.client.resetBasePositionAndOrientation(self.id, pos, orn)
        self.client.resetBaseVelocity(self.id, vel, omega)
        for i in range(Astrobee.NUM_JOINTS):
            self.client.resetJointState(self.id, i, q[i], qdot[i])

    @property
    def bounding_box(self) -> np.ndarray:
        """Current axis-aligned bounding box of the Astrobee body (Not including the arm), shape (2, 3)"""
        return np.array(self.client.getAABB(self.id, -1))


def _main():
    client = initialize_pybullet(bg_color=[1, 1, 1])
    robot = Astrobee()
    run_sim()


if __name__ == "__main__":
    _main()
