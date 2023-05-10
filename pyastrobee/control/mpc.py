"""Model predictive control"""

from pyastrobee.elements.astrobee import Astrobee
from pyastrobee.elements.cargo_bag import CargoBag
from pyastrobee.utils.bullet_utils import initialize_pybullet


class SimState:
    def __init__(self, robot_state, bag_state):
        pass


class BagState:
    def __init__(self, pos, orn, lin_vel, ang_vel):
        self.pos = pos
        self.orn = orn
        self.lin_vel = lin_vel
        self.ang_vel = ang_vel


class RobotState:
    def __init__(self, pos, orn, lin_vel, ang_vel, joint_pos, joint_vels):
        self.pos = pos
        self.orn = orn
        self.lin_vel = lin_vel
        self.ang_vel = ang_vel
        self.joint_pos = joint_pos
        self.joint_vels = joint_vels


def reset_state(state_id, bag_pos, bag_orn, bag_lin_vel, bag_ang_vel):
    pass


def init(robot_pose, use_gui: bool = True):
    client = initialize_pybullet(use_gui)
    robot = Astrobee(robot_pose)
    bag = CargoBag("top_handle_bag", robot)
