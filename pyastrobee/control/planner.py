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

from pyastrobee.control.astrobee import Astrobee
from pyastrobee.utils.rotations import quaternion_slerp, get_closest_heading_quat
from pyastrobee.utils.math_utils import normalize


def simple_turn_and_move_traj(start_pose, end_pose):
    # Interpolate the orientation, interpolate the position, interpolate the orientation
    start_position = start_pose[:3]
    start_quat = start_pose[3:]
    end_position = end_pose[:3]
    end_quat = end_pose[3:]
    # Trajectory part 1: Maintain same pose, turn to point towards the final position
    # Need to determine the quaternion pointing towards the end
    heading = normalize(end_position - start_position)
    heading_quat = get_closest_heading_quat(start_quat, heading)
    n1 = 10
    traj_1 = fixed_position_traj(start_position, start_quat, heading_quat, n1)
    # Trajectory part 2: Maintain the same orientation, move to final position
    n2 = 10
    traj_2 = fixed_orientation_traj(start_position, end_position, heading_quat, n2)
    # Trajectory part 3: Maintain the same final position, turn to goal orientation
    n3 = 10
    traj_3 = fixed_position_traj(end_position, heading_quat, end_quat, n3)
    # Merge the trajectory components together
    traj = np.vstack((traj_1, traj_2, traj_3))
    return traj


def fixed_orientation_traj(pos_1, pos_2, quat, n):
    positions = np.linspace(pos_1, pos_2, n)
    quats = quat * np.ones((n, 4))
    return np.hstack((positions, quats))


def fixed_position_traj(pos, q1, q2, n):
    positions = pos * np.ones((n, 3))
    quats = quaternion_slerp(q1, q2, np.linspace(0, 1, n))
    return np.hstack((positions, quats))


def simple_interp_traj(start_pose, end_pose, n):
    # Simplest possible trajectory, just interpolate the position and orientation components
    # together without accounting for velocity or velocity/accel/force limits
    pcts = np.linspace(0, 1, n)
    start_position = start_pose[:3]
    start_quat = start_pose[3:]
    end_position = end_pose[:3]
    end_quat = end_pose[3:]
    positions = np.linspace(start_position, end_position, n)
    quats = quaternion_slerp(start_quat, end_quat, pcts)
    return np.hstack((positions, quats))


class Planner:
    def __init__(self, robot: Astrobee):
        self.robot = robot

    def plan_trajectory(self, pose_goal):
        pass
