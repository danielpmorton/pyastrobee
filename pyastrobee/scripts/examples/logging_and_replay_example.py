"""Example of logging and replaying a trajectory tracking example"""

import time

import pybullet
import numpy as np

from pyastrobee.core.astrobee import Astrobee
from pyastrobee.core.rigid_bag import RigidCargoBag
from pyastrobee.control.force_torque_control import ForceTorqueController
from pyastrobee.utils.bullet_utils import initialize_pybullet
from pyastrobee.utils.quaternions import random_quaternion
from pyastrobee.trajectories.polynomials import polynomial_trajectory
from pyastrobee.trajectories.trajectory import visualize_traj
from pyastrobee.utils.bullet_utils import playback_from_log


def run_example():
    client = initialize_pybullet(bg_color=[1, 1, 1])
    np.random.seed(0)
    robot = Astrobee()
    bag = RigidCargoBag("top_handle", 10)
    bag.attach_to(robot)
    log_id = client.startStateLogging(
        client.STATE_LOGGING_GENERIC_ROBOT,
        "artifacts/logging_test_2.bullet",
        [robot.id, bag.id],
    )
    pose_1 = robot.pose
    pose_2 = [1, 2, 3, *random_quaternion()]
    max_time = 10
    dt = pybullet.getPhysicsEngineParameters()["fixedTimeStep"]
    traj = polynomial_trajectory(pose_1, pose_2, max_time, dt)
    visualize_traj(traj, 20)
    kp = 20
    kv = 5
    kq = 1
    kw = 0.1
    controller = ForceTorqueController(
        robot.id, robot.mass, robot.inertia, kp, kv, kq, kw, dt, 1e-1, 1e-1, 1e-1, 1e-1
    )
    controller.follow_traj(traj)
    client.stopStateLogging(log_id)
    client.disconnect()


def playback():
    # Set up pybullet in the same way as in the original execution
    client = initialize_pybullet()
    robot = Astrobee()
    bag = RigidCargoBag("top_handle", 10)
    bag.attach_to(robot)

    playback_from_log("artifacts/logging_test_2.bullet", real_time=True)
    input("Playback complete")
    while True:
        pybullet.stepSimulation()
        time.sleep(1 / 120)


if __name__ == "__main__":
    # run_example()
    playback()
