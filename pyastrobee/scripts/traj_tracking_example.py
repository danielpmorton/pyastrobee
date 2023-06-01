"""Script to give examples of trajectory tracking

Current examples include:
- Following a trajectory with a unit-mass cube
- Following a trajectory with Astrobee
- Following a trajectory (defined for the Astrobee base) with a cargo bag attached

This can also be used as a sandbox for experimenting with the PID tuning
"""

import pybullet
import numpy as np

from pyastrobee.elements.astrobee import Astrobee
from pyastrobee.elements.cargo_bag import CargoBag
from pyastrobee.control.force_controller_new import ForcePIDController
from pyastrobee.utils.bullet_utils import create_box, initialize_pybullet
from pyastrobee.utils.quaternions import random_quaternion
from pyastrobee.trajectories.polynomials import polynomial_trajectory
from pyastrobee.trajectories.trajectory import visualize_traj, compare_trajs


def box_inertia(m, l, w, h):
    return (1 / 12) * m * np.diag([w**2 + h**2, l**2 + h**2, l**2 + w**2])


def unit_mass_cube_example():
    pybullet.connect(pybullet.GUI)
    np.random.seed(0)
    pose_1 = [0, 0, 0, 0, 0, 0, 1]
    pose_2 = [1, 2, 3, *random_quaternion()]
    mass = 10
    sidelengths = [0.25, 0.25, 0.25]
    box = create_box(pose_1[:3], pose_1[3:], mass, sidelengths, True)
    max_time = 10
    dt = pybullet.getPhysicsEngineParameters()["fixedTimeStep"]
    traj = polynomial_trajectory(pose_1, pose_2, max_time, dt)
    visualize_traj(traj, 20)
    kp = 20
    kv = 5
    kq = 1
    kw = 0.1
    inertia = box_inertia(mass, *sidelengths)
    controller = ForcePIDController(box, mass, inertia, kp, kv, kq, kw, dt)
    controller.follow_traj(traj)
    pybullet.disconnect()
    compare_trajs(traj, controller.traj_log)
    controller.control_log.plot()


def astrobee_example():
    pybullet.connect(pybullet.GUI)
    np.random.seed(0)
    pose_1 = [0, 0, 0, 0, 0, 0, 1]
    pose_2 = [1, 2, 3, *random_quaternion()]
    robot = Astrobee()
    max_time = 10
    dt = pybullet.getPhysicsEngineParameters()["fixedTimeStep"]
    traj = polynomial_trajectory(pose_1, pose_2, max_time, dt)
    visualize_traj(traj, 20)
    kp = 20
    kv = 5
    kq = 1
    kw = 0.1
    controller = ForcePIDController(
        robot.id, robot.mass, robot.inertia, kp, kv, kq, kw, dt
    )
    controller.follow_traj(traj)
    pybullet.disconnect()
    compare_trajs(traj, controller.traj_log)
    controller.control_log.plot()


def astrobee_with_bag_example():
    initialize_pybullet(bg_color=[1, 1, 1])
    np.random.seed(0)
    pose_1 = [0, 0, 0, 0, 0, 0, 1]
    pose_2 = [1, 2, 3, *random_quaternion()]
    robot = Astrobee()
    bag = CargoBag("top_handle_bag", robot)
    max_time = 10
    dt = pybullet.getPhysicsEngineParameters()["fixedTimeStep"]
    traj = polynomial_trajectory(pose_1, pose_2, max_time, dt)
    visualize_traj(traj, 20)
    kp = 20
    kv = 5
    kq = 1
    kw = 0.1
    controller = ForcePIDController(
        robot.id, robot.mass, robot.inertia, kp, kv, kq, kw, dt, 1e-1, 1e-1, 1e-1, 1e-1
    )
    controller.follow_traj(traj)
    pybullet.disconnect()
    compare_trajs(traj, controller.traj_log)
    controller.control_log.plot()


if __name__ == "__main__":
    # unit_mass_cube_example()
    # astrobee_example()
    astrobee_with_bag_example()
