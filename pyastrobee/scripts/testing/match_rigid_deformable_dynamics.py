"""Test to see if we can match the dynamics between a deformable and rigid cargo bag

NOTE
- Using the URDF-based bag is not ideal because when the reset occurs, it seems to often exceed the joint limits on the
  "spherical joint", leading to really fast corrections which throw off the dynamics
- The constraint-based bag seems to handle this better because of the inherent compliance in the system
"""

import time
from typing import Optional, Union

import pybullet
from pybullet_utils.bullet_client import BulletClient
import numpy as np
import numpy.typing as npt

from pyastrobee.core.astrobee import Astrobee
from pyastrobee.core.deformable_bag import DeformableCargoBag
from pyastrobee.core.rigid_bag import RigidCargoBag
from pyastrobee.core.constraint_bag import ConstraintCargoBag
from pyastrobee.utils.bullet_utils import initialize_pybullet
from pyastrobee.utils.quaternions import random_quaternion
from pyastrobee.trajectories.polynomials import polynomial_trajectory
from pyastrobee.trajectories.trajectory import visualize_traj
from pyastrobee.control.force_torque_control import ForceTorqueController


def get_state(deformable_bag: DeformableCargoBag, robot: Optional[Astrobee] = None):
    robot_state = robot.full_state if robot is not None else None
    return deformable_bag.dynamics_state, robot_state


def reset_state(
    rigid_bag: Union[RigidCargoBag, ConstraintCargoBag],
    bag_state: tuple[np.ndarray, ...],
    robot: Optional[Astrobee] = None,
    robot_state: Optional[tuple[np.ndarray, ...]] = None,
    pos_offset: Optional[npt.ArrayLike] = None,
    client: Optional[BulletClient] = None,
) -> None:
    """Resets the state of the rigid cargo bag to the dynamics recorded from the deformable bag

    Args:
        rigid_bag (RigidCargoBag): The rigid cargo bag
        bag_state (tuple[np.ndarray, ...]): Position, orientation, velocity, and angular velocity of the
            deformable bag
        pos_offset (Optional[npt.ArrayLike], optional): Positional offset for the reset (to keep the two bags apart from
            eachother). Defaults to None.
        client (BulletClient, optional): If connecting to multiple physics servers, include the client
            (the class instance, not just the ID) here. Defaults to None (use default connected client)
    """
    client: pybullet = pybullet if client is None else client
    # Handle the bag
    assert len(bag_state) == 4
    bag_pos, bag_orn, bag_vel, bag_omega = bag_state
    if pos_offset is not None:
        bag_pos = np.add(bag_pos, pos_offset)
    client.resetBasePositionAndOrientation(rigid_bag.id, bag_pos, bag_orn)
    client.resetBaseVelocity(rigid_bag.id, bag_vel, bag_omega)
    # Handle the robot (if needed)
    if robot is not None:
        assert robot_state is not None
        assert len(robot_state) == 6
        robot_pos, robot_orn, robot_vel, robot_omega, robot_q, robot_qdot = robot_state
        if pos_offset is not None:
            robot_pos = np.add(robot_pos, pos_offset)
        robot.reset_full_state(
            robot_pos, robot_orn, robot_vel, robot_omega, robot_q, robot_qdot
        )


def bag_only_test():
    bag_name = "top_handle"
    mass = 10
    deformable_p0 = np.zeros(3)
    pos_offset = np.array([0, 0, 1])
    rigid_p0 = deformable_p0 + pos_offset
    client = initialize_pybullet()
    dt = client.getPhysicsEngineParameters()["fixedTimeStep"]
    deformable_bag = DeformableCargoBag(bag_name, mass, deformable_p0, client=client)
    rigid_bag = ConstraintCargoBag(bag_name, mass, rigid_p0, client=client)

    time_per_reset = 1  # seconds
    steps_per_reset = round(time_per_reset / dt)

    print("Apply a disturbance force to the deformable cargo bag")
    while True:
        for _ in range(steps_per_reset):
            pybullet.stepSimulation()
            time.sleep(1 / 120)
        bag_state, _ = get_state(deformable_bag)
        reset_state(rigid_bag, bag_state, pos_offset=pos_offset)


def astrobee_and_bag_test():
    bag_name = "top_handle_symmetric"
    mass = 10
    deformable_p0 = np.zeros(3)
    pos_offset = np.array([1, 0, 0])
    rigid_p0 = deformable_p0 + pos_offset
    client = initialize_pybullet()
    dt = client.getPhysicsEngineParameters()["fixedTimeStep"]
    deformable_bag = DeformableCargoBag(bag_name, mass, deformable_p0, client=client)
    robot_1 = Astrobee()
    deformable_bag.attach_to(robot_1, "robot")
    rigid_bag = ConstraintCargoBag(bag_name, mass, rigid_p0, client=client)
    robot_2 = Astrobee()
    rigid_bag.attach_to(robot_2, "robot")

    time_per_reset = 1  # seconds
    steps_per_reset = round(time_per_reset / dt)

    print("Apply a disturbance force to the deformable cargo bag")
    while True:
        for _ in range(steps_per_reset):
            pybullet.stepSimulation()
            time.sleep(1 / 120)
        bag_state, robot_state = get_state(deformable_bag, robot_1)
        reset_state(rigid_bag, bag_state, robot_2, robot_state, pos_offset)


def dual_tracking_example():
    np.random.seed(0)
    bag_name = "top_handle"
    mass = 10
    deformable_p0 = np.zeros(3)
    pos_offset = np.array([2, 0, 0])
    rigid_p0 = deformable_p0 + pos_offset
    client = initialize_pybullet()
    dt = client.getPhysicsEngineParameters()["fixedTimeStep"]
    deformable_bag = DeformableCargoBag(bag_name, mass, deformable_p0, client=client)
    robot_1 = Astrobee()
    deformable_bag.attach_to(robot_1, "robot")
    rigid_bag = ConstraintCargoBag(bag_name, mass, rigid_p0, client=client)
    robot_2 = Astrobee()
    rigid_bag.attach_to(robot_2, "robot")

    robot_1_start_pose = robot_1.pose
    q_goal = random_quaternion()
    robot_1_end_pose = [1, 2, 3, *q_goal]
    robot_2_start_pose = robot_2.pose
    robot_2_end_pose = [*(robot_1_end_pose[:3] + pos_offset), *q_goal]
    max_time = 10
    dt = pybullet.getPhysicsEngineParameters()["fixedTimeStep"]
    traj_1 = polynomial_trajectory(robot_1_start_pose, robot_1_end_pose, max_time, dt)
    traj_2 = polynomial_trajectory(robot_2_start_pose, robot_2_end_pose, max_time, dt)
    visualize_traj(traj_1, 20)
    visualize_traj(traj_2, 20)
    kp = 20
    kv = 5
    kq = 1
    kw = 0.1
    controller_1 = ForceTorqueController(
        robot_1.id,
        robot_1.mass,
        robot_1.inertia,
        kp,
        kv,
        kq,
        kw,
        dt,
        1e-1,
        1e-1,
        1e-1,
        1e-1,
    )
    controller_2 = ForceTorqueController(
        robot_2.id,
        robot_2.mass,
        robot_2.inertia,
        kp,
        kv,
        kq,
        kw,
        dt,
        1e-1,
        1e-1,
        1e-1,
        1e-1,
    )
    assert traj_1.num_timesteps == traj_2.num_timesteps
    for i in range(traj_1.num_timesteps):
        pos_1, orn_1, lin_vel_1, ang_vel_1 = controller_1.get_current_state()
        controller_1.step(
            pos_1,
            lin_vel_1,
            orn_1,
            ang_vel_1,
            traj_1.positions[i, :],
            traj_1.linear_velocities[i, :],
            traj_1.linear_accels[i, :],
            traj_1.quaternions[i, :],
            traj_1.angular_velocities[i, :],
            traj_1.angular_accels[i, :],
        )
        pos_2, orn_2, lin_vel_2, ang_vel_2 = controller_2.get_current_state()
        controller_2.step(
            pos_2,
            lin_vel_2,
            orn_2,
            ang_vel_2,
            traj_2.positions[i, :],
            traj_2.linear_velocities[i, :],
            traj_2.linear_accels[i, :],
            traj_2.quaternions[i, :],
            traj_2.angular_velocities[i, :],
            traj_2.angular_accels[i, :],
        )
    # Stopping
    des_vel = np.zeros(3)
    des_accel = np.zeros(3)
    des_omega = np.zeros(3)
    des_alpha = np.zeros(3)
    try:
        while True:
            pos_1, orn_1, lin_vel_1, ang_vel_1 = controller_1.get_current_state()
            controller_1.step(
                pos_1,
                lin_vel_1,
                orn_1,
                ang_vel_1,
                traj_1.positions[-1],
                des_vel,
                des_accel,
                traj_1.quaternions[-1],
                des_omega,
                des_alpha,
            )
            pos_2, orn_2, lin_vel_2, ang_vel_2 = controller_2.get_current_state()
            controller_2.step(
                pos_2,
                lin_vel_2,
                orn_2,
                ang_vel_2,
                traj_2.positions[-1],
                des_vel,
                des_accel,
                traj_2.quaternions[-1],
                des_omega,
                des_alpha,
            )
    except KeyboardInterrupt:
        pybullet.disconnect()


if __name__ == "__main__":
    # bag_only_test()
    astrobee_and_bag_test()
    # dual_tracking_example()
