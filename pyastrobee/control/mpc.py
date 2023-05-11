"""Model predictive control"""

import time
from typing import Optional

import pybullet
import numpy as np
import numpy.typing as npt

from pyastrobee.elements.astrobee import Astrobee
from pyastrobee.elements.cargo_bag import CargoBag
from pyastrobee.elements.iss import load_iss
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


def reset_state(
    state_id: int,
    bag_id: int,
    bag_pos: npt.ArrayLike,
    bag_orn: npt.ArrayLike,
    bag_lin_vel: npt.ArrayLike,
    bag_ang_vel: npt.ArrayLike,
) -> None:
    """Resets the state of the simulation (including the state of the cargo bag)

    Args:
        state_id (int): Pybullet ID for the saveState object
        bag_id (int): Pybullet ID for the cargo bag
        bag_pos (npt.ArrayLike): Position of the cargo bag, shape (3,)
        bag_orn (npt.ArrayLike): Orientation of the cargo bag (XYZW quaternion), shape (4,)
        bag_lin_vel (npt.ArrayLike): Linear velocity of the cargo bag, shape (3,)
        bag_ang_vel (npt.ArrayLike): Angular velocity of the cargo bag, shape (3,)
    """
    pybullet.restoreState(stateId=state_id)
    pybullet.resetBasePositionAndOrientation(bag_id, bag_pos, bag_orn)
    pybullet.resetBaseVelocity(bag_id, bag_lin_vel, bag_ang_vel)


def save_state(
    bag: CargoBag,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Save the state of the simulation, including the state of the bag

    - Normally, you could just call saveState, but this does not save the state of deformable objects
    - saveState will save info about the Astrobee and the environment, but we need to manually save information
      about the current dyanmics of the bag

    Args:
        bag (CargoBag): The cargo bag in the current simulation

    Returns:
        tuple of:
            int: The Pybullet ID of the saveState object
            np.ndarray: Position of the cargo bag, shape (3,)
            np.ndarray: Orientation of the cargo bag (XYZW quaternion), shape (4,)
            np.ndarray: Linear velocity of the cargo bag, shape (3,)
            np.ndarray: Angular velocity of the cargo bag, shape (3,)
    """
    state_id = pybullet.saveState()
    bag_pos, bag_orn, bag_vel, bag_ang_vel = bag.dynamics_state
    return state_id, bag_pos, bag_orn, bag_vel, bag_ang_vel


def init(robot_pose, use_gui: bool = True):
    client = initialize_pybullet(use_gui)
    load_iss()
    robot = Astrobee(robot_pose)
    bag = CargoBag("top_handle_bag", robot)


if __name__ == "__main__":
    pass
