"""Manages the properties of the astrobee and all control-associated functions

In general, we assume that we're working with Honey. Multiple astrobees can be loaded, but
we assume that they all have the exact same configuration


TO ADD:
follow_trajectory -- given a sequence of poses for the constraints, continually update the state of the constraint to follow this
plan_trajectory -- given a pose target, plan a sequence of constraint values to get from the current pose to desired



"""

from enum import Enum

import numpy as np
import pybullet
import numpy.typing as npt

from pyastrobee.utils.bullet_utils import initialize_pybullet, run_sim
from pyastrobee.utils.transformations import make_transform_mat
from pyastrobee.utils.rotations import euler_angles_to_rmat


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

    # TODO add these to a constants or config file? Probably fine here for now
    URDF = "pyastrobee/urdf/astrobee.urdf"
    LOADED_IDS = []
    NUM_LOADED = 0  # TODO make this into a property?
    NUM_JOINTS = 6
    NUM_LINKS = 7
    ARM_DISTAL_TO_GRIPPER_OFFSET = None  # TODO!!!

    # Joint limit information is extracted from the URDF
    # Joint pos limits are [lower, upper] for each joint
    JOINT_POS_LIMITS = [
        [-2.0944, 1.57079],  # arm proximal joint
        [-1.57079, 1.57079],  # arm distal joint
        [0.349066, 0.698132],  # gripper left proximal joint
        [-1.22173, -0.69813],  # gripper left distal joint
        [-0.698132, -0.349066],  # gripper right proximal joint
        [0.69813, 1.22173],  # gripper right distal joint
    ]
    JOINT_EFFORT_LIMITS = [
        1.0,  # arm proximal joint
        1.0,  # arm distal joint
        0.1,  # gripper left proximal joint
        0.1,  # gripper left distal joint
        0.1,  # gripper right proximal joint
        0.1,  # gripper right distal joint
    ]
    JOINT_VEL_LIMITS = [
        0.12,  # arm proximal joint
        0.12,  # arm distal joint
        0.12,  # gripper left proximal joint
        0.12,  # gripper left distal joint
        0.12,  # gripper right proximal joint
        0.12,  # gripper right distal joint
    ]

    def __init__(
        self,
        pos: npt.ArrayLike = [0, 0, 0],
        orn: npt.ArrayLike = [0, 0, 0],
    ):
        if not pybullet.isConnected():
            raise ConnectionError(
                "Need to connect to pybullet before initializing an astrobee"
            )
        # TODO update default values for pos/orn or improve how we specify initial location
        # TODO need to finish this
        self.id = pybullet.loadURDF(Astrobee.URDF)
        Astrobee.LOADED_IDS.append(self.id)
        Astrobee.NUM_LOADED += 1

        # Property internal variables
        self._tcp_offset = None  # TODO
        # Constraint is for position control
        self.constraint_id = pybullet.createConstraint(self.id, -1, -1, -1, pybullet.JOINT_FIXED, None, (0, 0, 0), (0, 0, 0))

    # Is an unloading method needed?
    def _unload(self, robot_id: int):
        if robot_id not in Astrobee.LOADED_IDS:
            raise ValueError(f"Invalid ID: {robot_id}, cannot unload the astrobee")
        pybullet.removeBody(robot_id)
        Astrobee.LOADED_IDS.remove(robot_id)
        Astrobee.NUM_LOADED -= 1

    # Control might be tricky, but NASA already has path planning and stuff implemented
    # Should we communicate back and forth with ROS for this?
    # If so, when we're in contact with the bag and working with the Bullet physics, how can
    # we tell ROS and Gazebo to properly update the state of the system?

    # determine how getters and setters should work in conjunction with ROS!

    @property
    def tcp_offset(self):
        return self._tcp_offset

    # Does this even need to be a property??
    @tcp_offset.setter
    def tcp_offset(self, offset):
        self._tcp_offset = offset

    # TODO this function needs some work!! Need to get transformations as well as confirm where the frames are
    def get_arm_pose_base(self):
        base_to_world = self.get_robot_pose()
        # Link index 2 should be the distal joint of the arm (AKA the one nearest to the gripper but not the fingers)
        link_state = pybullet.getLinkState(
            self.id,
            linkIndex=2,
            computeLinkVelocity=False,
            computeForwardKinematics=True,
        )
        (
            COM_world_pos,
            COM_world_orn,
            local_pos,
            local_orn,
            world_pos,
            world_orn,
        ) = link_state
        return NotImplementedError
        arm_to_base = NotImplemented

        arm_tcp_to_distal = NotImplemented
        arm_distal_to_world = NotImplemented

        return base_to_world @ arm_to_base
        return arm_distal_to_world @ arm_tcp_to_distal

    def set_arm_pose_base(self, pose):
        pass

    def get_arm_pose_world(self):
        pass

    # need to decide on how to deal with gripper because each side has two joints
    def set_gripper_pos(self, angles):
        # angles will need to be an array of 4 values
        pass

    def get_gripper_pos(self):
        pass

    def open_gripper(self):
        pass

    def close_gripper(self):
        # Should this implement a mix of position and torque control?
        pass

    def set_arm_pose_world(self, pose):
        pass

    def get_robot_pose(self):
        # TODO need to clear up any confusion on base frame vs link 0 frame!
        # Note pybullet orientation is quaternions: [x, y, z, w]
        pos, orn = pybullet.getBasePositionAndOrientation(self.id)
        rmat = pybullet.getMatrixFromQuaternion(orn)
        return make_transform_mat(rmat, pos)

    def set_robot_pose(self, pose):
        pass

    def get_joint_angles(self):
        # States: tuple[tuple], size (6, 4)
        # 6 corresponds to NUM_JOINTS
        # 4 corresponds to position, velocity, reaction forces, and applied torque
        states = pybullet.getJointStates(self.id, list(range(Astrobee.NUM_JOINTS)))
        return [states[i][0] for i in range(Astrobee.NUM_JOINTS)]  # Index 0: position

    def get_joint_vels(self):
        states = pybullet.getJointStates(self.id, list(range(Astrobee.NUM_JOINTS)))
        return [states[i][1] for i in range(Astrobee.NUM_JOINTS)]  # Index 1: velocity

    def get_joint_torques(self):
        states = pybullet.getJointStates(self.id, list(range(Astrobee.NUM_JOINTS)))
        return [states[i][3] for i in range(Astrobee.NUM_JOINTS)]  # Index 3: torque

    def set_joint_angles(self, angles: npt.ArrayLike):
        # TODO Add a target velocity?
        if len(angles) != Astrobee.NUM_JOINTS:
            raise ValueError(
                f"Incorrect number of angles ({len(angles)}). Must = {Astrobee.NUM_JOINTS}"
            )
        self.set_joint_angles_by_index(angles, list(range(Astrobee.NUM_JOINTS)))

    def set_joint_angles_by_index(self, angles: npt.ArrayLike, indices: npt.ArrayLike):
        if not len(angles) == len(indices):
            raise Exception(
                "Number of angles must match with the number of provided indices"
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
        pass


    def plan_trajectory(self, desired_pose):
        cur_pose = self.get_robot_pose()
        cur_xyz = cur_pose[:3]
        vec_from_to = 






if __name__ == "__main__":
    initialize_pybullet()
    robot = Astrobee()
    run_sim()
