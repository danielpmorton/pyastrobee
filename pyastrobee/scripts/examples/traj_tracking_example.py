"""Script to give examples of trajectory tracking

Current examples include:
- Following a trajectory with a unit-mass cube
- Following a trajectory with Astrobee
- Following a trajectory (defined for the Astrobee base) with a cargo bag attached

This can also be used as a sandbox for experimenting with the PID tuning
"""

import pybullet
import numpy as np

from pyastrobee.core.astrobee import Astrobee
from pyastrobee.core.deformable_bag import DeformableCargoBag
from pyastrobee.core.constraint_bag import ConstraintCargoBag
from pyastrobee.control.force_torque_control import ForceTorqueController
from pyastrobee.utils.bullet_utils import create_box, initialize_pybullet
from pyastrobee.utils.quaternions import random_quaternion
from pyastrobee.trajectories.polynomials import polynomial_trajectory
from pyastrobee.trajectories.trajectory import visualize_traj, compare_trajs
from pyastrobee.utils.dynamics import box_inertia


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
    controller = ForceTorqueController(box, mass, inertia, kp, kv, kq, kw, dt)
    controller.follow_traj(traj)
    input("Press Enter to disconnect and show plots")
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
    controller = ForceTorqueController(
        robot.id, robot.mass, robot.inertia, kp, kv, kq, kw, dt
    )
    controller.follow_traj(traj)
    input("Press Enter to disconnect and show plots")
    pybullet.disconnect()
    compare_trajs(traj, controller.traj_log)
    controller.control_log.plot()


def astrobee_with_bag_example():
    client = initialize_pybullet(bg_color=[1, 1, 1])
    np.random.seed(0)
    robot = Astrobee()
    bag = DeformableCargoBag("top_handle", 10)
    bag.attach_to(robot)
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
    input("Press Enter to disconnect and show plots")
    pybullet.disconnect()
    compare_trajs(traj, controller.traj_log)
    controller.control_log.plot()


def astrobee_with_rigid_bag_example():
    client = initialize_pybullet(bg_color=[1, 1, 1])
    np.random.seed(0)
    robot = Astrobee()
    bag = ConstraintCargoBag("top_handle", mass=5)
    bag.attach_to(robot)
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
    controller.follow_traj(traj, stop_at_end=True)
    input("Press Enter to disconnect and show plots")
    pybullet.disconnect()
    compare_trajs(traj, controller.traj_log)
    controller.control_log.plot()


if __name__ == "__main__":
    # unit_mass_cube_example()
    # astrobee_example()
    # astrobee_with_bag_example()
    astrobee_with_rigid_bag_example()
