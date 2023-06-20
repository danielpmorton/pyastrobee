"""Motion planning

** This general concept is likely out of date with how the Controller class(es) are set up
Figure out if all of the motion planning implementation should just go into the controller?
Or generate a trajectory plan here and then pass it in to the controller

TODO
- Add support for multiple astrobees?

Check out these options:
https://github.com/stanford-iprl-lab/mob-manip/blob/main/mob_manip/utils/common/plan_control_traj.py
https://github.com/krishauser/Klampt
https://github.com/caelan/pybullet-planning
https://github.com/caelan/motion-planners
https://arxiv.org/pdf/2205.04422.pdf
https://ompl.kavrakilab.org/
https://github.com/lyfkyle/pybullet_ompl
https://github.com/schmrlng/MotionPlanning.jl/tree/master/src (convert to python?)

Easy alternatives:
- Turn + move in straight line to waypoint
- Try pybullet.rayTest for checking collisions?
"""

import numpy as np
import numpy.typing as npt

from pyastrobee.core.astrobee import Astrobee
from pyastrobee.utils.quaternions import (
    quaternion_slerp,
    get_closest_heading_quat,
    quaternion_dist,
)
from pyastrobee.utils.math_utils import normalize


def point_and_move_pose_traj(
    start_pose: npt.ArrayLike,
    end_pose: npt.ArrayLike,
    pos_step: float,
    orn_step: float,
) -> np.ndarray:
    """Simple pose-only trajectory where the robot points at the goal, moves along a straight line,
    then aligns with the goal

    Args:
        start_pose (npt.ArrayLike): Starting position + xyzw quaternion pose, shape (7,)
        end_pose (npt.ArrayLike): Ending position + xyzw quaternion pose, shape (7,)
        pos_step (float): Position stepsize (meters)
        orn_step (float): Orientation stepsize (quaternion distance)

    Returns:
        np.ndarray: Trajectory, shape (n1 + n2 + n3, 7)
    """
    # Interpolate the orientation, interpolate the position, interpolate the orientation
    start_position = start_pose[:3]
    start_quat = start_pose[3:]
    end_position = end_pose[:3]
    end_quat = end_pose[3:]
    # Trajectory part 1: Maintain same pose, turn to point towards the final position
    # First, need to find the quaternion to point in the right direction
    heading = normalize(end_position - start_position)
    heading_quat = get_closest_heading_quat(start_quat, heading)
    # Also determine the discretization based on this intermediate orientation
    n1 = int(np.ceil(quaternion_dist(start_quat, heading_quat) / orn_step))
    traj_1 = fixed_pos_pose_traj(start_position, start_quat, heading_quat, n1)
    # Trajectory part 2: Maintain the same orientation, move to final position
    n2 = int(np.ceil(np.linalg.norm(end_position - start_position) / pos_step))
    traj_2 = fixed_orn_pose_traj(start_position, end_position, heading_quat, n2)
    # Trajectory part 3: Maintain the same final position, turn to goal orientation
    n3 = int(np.ceil(quaternion_dist(heading_quat, end_quat) / orn_step))
    traj_3 = fixed_pos_pose_traj(end_position, heading_quat, end_quat, n3)
    # Merge the trajectory components together
    return np.vstack((traj_1, traj_2, traj_3))


def fixed_orn_pose_traj(
    pos_1: npt.ArrayLike, pos_2: npt.ArrayLike, quat: npt.ArrayLike, n: int
) -> np.ndarray:
    """Simple pose-only trajectory interpolated between two positions with a fixed orientation

    Args:
        pos_1 (npt.ArrayLike): Starting XYZ position, shape (3,)
        pos_2 (npt.ArrayLike): Ending XYZ position, shape (3,)
        quat (npt.ArrayLike): Fixed orientation (XYZW quaternion), shape (4,)
        n (int): Number of timesteps

    Returns:
        np.ndarray: Trajectory, shape (n, 7)
    """
    positions = np.linspace(pos_1, pos_2, n)
    quats = quat * np.ones((n, 4))
    return np.hstack((positions, quats))


def fixed_pos_pose_traj(
    pos: npt.ArrayLike, q1: npt.ArrayLike, q2: npt.ArrayLike, n: int
) -> np.ndarray:
    """Simple pose-only trajectory interpolated between two orientations with a fixed position

    Args:
        pos (npt.ArrayLike): Fixed XYZ position, shape (3,)
        q1 (npt.ArrayLike): Starting orientation (XYZW quaternion), shape (4,)
        q2 (npt.ArrayLike): Ending orientation (XYZW quaternion), shape (4,)
        n (int): Number of timesteps

    Returns:
        np.ndarray: Trajectory, shape (n, 7)
    """
    positions = pos * np.ones((n, 3))
    quats = quaternion_slerp(q1, q2, np.linspace(0, 1, n))
    return np.hstack((positions, quats))


def interpolation_pose_traj(
    start_pose: npt.ArrayLike, end_pose: npt.ArrayLike, n: int
) -> np.ndarray:
    """Simple pose-only trajectory, interpolates between two poses across n timesteps

    Args:
        start_pose (npt.ArrayLike): Starting position + xyzw quaternion pose, shape (7,)
        end_pose (npt.ArrayLike): Ending position + xyzw quaternion pose, shape (7,)
        n (int): Number of timesteps

    Returns:
        np.ndarray: Trajectory, shape (n, 7)
    """
    start_position = start_pose[:3]
    start_quat = start_pose[3:]
    end_position = end_pose[:3]
    end_quat = end_pose[3:]
    positions = np.linspace(start_position, end_position, n)
    quats = quaternion_slerp(start_quat, end_quat, np.linspace(0, 1, n))
    return np.hstack((positions, quats))


class Planner:
    def __init__(self, robot: Astrobee):
        self.robot = robot

    def plan_trajectory(self, pose_goal):
        pass
