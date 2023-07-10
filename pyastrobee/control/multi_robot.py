"""Multi-robot control"""

import time
from typing import Optional

import numpy as np
import pybullet
from pybullet_utils.bullet_client import BulletClient

from pyastrobee.core.astrobee import Astrobee
from pyastrobee.trajectories.trajectory import Trajectory
from pyastrobee.control.force_torque_control import ForceTorqueController


def multi_robot_control(
    robots: list[Astrobee],
    trajs: list[Trajectory],
    controllers: list[ForceTorqueController],
    stop_at_end: bool,
    client: Optional[BulletClient] = None,
):
    """Control multiple Astrobees to follow their own respective trajectories simultaneously

    Args:
        robots (list[Astrobee]): Astrobees to control
        trajs (list[Trajectory]): Trajectories to follow (1 per Astrobee)
        controllers (list[ForceTorqueController]): Robot controllers (1 per Astrobee)
        stop_at_end (bool): Whether to command the robots to stop at the end of the trajectory
        client (BulletClient, optional): If connecting to multiple physics servers, include the client
            (the class instance, not just the ID) here. Defaults to None (use default connected client)

    Raises:
        ValueError: If there is a mismatch in the number of robots/trajectories/controllers
    """
    if not len(robots) == len(trajs) == len(controllers):
        raise ValueError(
            "Mismatched inputs: Each robot must have a trajectory and a controller"
        )
    client: pybullet = pybullet if client is None else client
    num_timesteps = trajs[0].num_timesteps
    num_robots = len(robots)
    for t in range(num_timesteps):
        for i in range(num_robots):
            pos, orn, lin_vel, ang_vel = robots[i].dynamics_state
            controllers[i].step(
                pos,
                lin_vel,
                orn,
                ang_vel,
                trajs[i].positions[t, :],
                trajs[i].linear_velocities[t, :],
                trajs[i].linear_accels[t, :],
                trajs[i].quaternions[t, :],
                trajs[i].angular_velocities[t, :],
                trajs[i].angular_accels[t, :],
            )
        time.sleep(1 / 120)

    if stop_at_end:
        while True:
            for i in range(num_robots):
                pos, orn, lin_vel, ang_vel = robots[i].dynamics_state
                controllers[i].step(
                    pos,
                    lin_vel,
                    orn,
                    ang_vel,
                    trajs[i].positions[-1, :],
                    np.zeros(3),
                    np.zeros(3),
                    trajs[i].quaternions[-1, :],
                    np.zeros(3),
                    np.zeros(3),
                )
            time.sleep(1 / 120)
    else:
        while True:
            client.stepSimulation()
            time.sleep(1 / 120)
