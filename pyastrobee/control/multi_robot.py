"""Multi-robot control"""

import time
from typing import Optional

import numpy as np
import pybullet
from pybullet_utils.bullet_client import BulletClient

from pyastrobee.trajectories.trajectory import Trajectory
from pyastrobee.control.force_torque_control import ForceTorqueController


def multi_robot_control(
    controllers: list[ForceTorqueController],
    trajs: list[Trajectory],
    stop_at_end: bool,
    client: Optional[BulletClient] = None,
):
    """Control multiple Astrobees to follow their own respective trajectories simultaneously

    Args:
        controllers (list[ForceTorqueController]): Robot controllers (1 per Astrobee)
        trajs (list[Trajectory]): Trajectories to follow (1 per Astrobee)
        stop_at_end (bool): Whether to command the robots to stop at the end of the trajectory
        client (BulletClient, optional): If connecting to multiple physics servers, include the client
            (the class instance, not just the ID) here. Defaults to None (use default connected client)

    Raises:
        ValueError: If there is a mismatch in the number of trajectories/controllers
    """
    if not len(trajs) == len(controllers):
        raise ValueError(
            "Mismatched inputs: Each controller must correspond to a trajectory"
        )
    client: pybullet = pybullet if client is None else client
    num_timesteps = trajs[0].num_timesteps
    num_controllers = len(controllers)
    for t in range(num_timesteps):
        des_states = [
            [
                trajs[i].positions[t, :],
                trajs[i].linear_velocities[t, :],
                trajs[i].linear_accels[t, :],
                trajs[i].quaternions[t, :],
                trajs[i].angular_velocities[t, :],
                trajs[i].angular_accels[t, :],
            ]
            for i in range(num_controllers)
        ]
        step_controllers(controllers, des_states, client)
        # time.sleep(1 / 120)

    if stop_at_end:
        des_states = [
            [
                trajs[i].positions[t, :],
                np.zeros(3),
                np.zeros(3),
                trajs[i].quaternions[t, :],
                np.zeros(3),
                np.zeros(3),
            ]
            for i in range(num_controllers)
        ]
        while True:
            step_controllers(controllers, des_states, client)
            # time.sleep(1 / 120)
    else:
        while True:
            client.stepSimulation()
            # time.sleep(1 / 120)


def step_controllers(
    controllers: list[ForceTorqueController],
    des_states: list[np.ndarray],
    client: BulletClient,
):
    """Step a series of controllers within the same simulation simultaneously

    Args:
        controllers (list[ForceTorqueController]): Active controllers in the simulation
        des_states (list[np.ndarray]): Desired states for each robot. Length = len(controllers).
            Each state should include (in order) the desired position, velocity, acceleration,
            quaternion, angular velocity, and angular acceleration
        client (BulletClient): Pybullet simulation client containing the controllers
    """
    if len(des_states) != len(controllers):
        raise ValueError(
            "Mismatched input lengths. Each controller should have a corresponding goal state"
        )
    # Check that the quaternion input is in the correct location
    if len(des_states[0][3]) != 4:
        raise ValueError(
            "Desired states appear to be mis-ordered.\n"
            + "Each state should include position, velocity, acceleration, quaternion, omega, alpha"
        )
    for controller, des_state in zip(controllers, des_states):
        pos, orn, vel, omega = controller.get_current_state()
        controller.step(pos, vel, orn, omega, *des_state, step_sim=False)
    client.stepSimulation()
