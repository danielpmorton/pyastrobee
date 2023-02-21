"""Motion planning

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

from pyastrobee.control.astrobee import Astrobee


class Planner:
    def __init__(self, robot: Astrobee):
        self.robot = robot

    def plan_trajectory(self, pose_goal):
        pass
