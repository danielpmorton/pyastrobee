"""Model predictive control"""

import time
from typing import Optional

import pybullet
import numpy as np

from pyastrobee.elements.astrobee import Astrobee
from pyastrobee.elements.cargo_bag import CargoBag
from pyastrobee.utils.bullet_utils import initialize_pybullet


# class SimState:
#     def __init__(self, robot_state, bag_state):
#         pass


# class BagState:
#     def __init__(self, pos, orn, lin_vel, ang_vel):
#         self.pos = pos
#         self.orn = orn
#         self.lin_vel = lin_vel
#         self.ang_vel = ang_vel


# class RobotState:
#     def __init__(self, pos, orn, lin_vel, ang_vel, joint_pos, joint_vels):
#         self.pos = pos
#         self.orn = orn
#         self.lin_vel = lin_vel
#         self.ang_vel = ang_vel
#         self.joint_pos = joint_pos
#         self.joint_vels = joint_vels


def reset_state(state_id, bag_id, bag_pos, bag_orn, bag_lin_vel, bag_ang_vel):
    pybullet.restoreState(stateId=state_id)
    pybullet.resetBasePositionAndOrientation(bag_id, bag_pos, bag_orn)
    pybullet.resetBaseVelocity(bag_id, bag_lin_vel, bag_ang_vel)


def save_state(robot: Astrobee, bag: Optional[CargoBag] = None):
    state_id = pybullet.saveState()
    robot_pos, robot_orn, robot_vel, robot_ang_vel = robot.dynamics_state
    if bag is not None:
        bag_pos, bag_orn, bag_vel, bag_ang_vel = bag.dynamics_state
        anchors, anchor_objects = bag.anchors
    else:
        bag_pos, bag_orn, bag_vel, bag_ang_vel = [None, None, None, None]
    return state_id, bag_pos, bag_orn, bag_vel, bag_ang_vel


def init(robot_pose, use_gui: bool = True):
    client = initialize_pybullet(use_gui)
    robot = Astrobee(robot_pose)
    bag = CargoBag("top_handle_bag", robot)


if __name__ == "__main__":
    # Quick test script to see if the save/reset state works
    initialize_pybullet()
    robot = Astrobee()
    bag = CargoBag("top_handle_bag", robot)
    start_time = time.time()
    print("5 seconds to adjust the robot and bag")
    while time.time() - start_time < 5:
        pybullet.stepSimulation()
        time.sleep(1 / 240)
    input("Press Enter to SAVE")
    state_id, bag_pos, bag_orn, bag_vel, bag_ang_vel = save_state(robot, bag)
    input("Press Enter to RANDOMIZE")
    print("5 seconds to adjust the robot and bag")
    start_time = time.time()
    while time.time() - start_time < 5:
        pybullet.stepSimulation()
        time.sleep(1 / 240)
    input("Press Enter to RESET")
    reset_state(state_id, bag.id, bag_pos, bag_orn, bag_vel, bag_ang_vel)
    while True:
        pybullet.stepSimulation()
        time.sleep(1 / 240)
