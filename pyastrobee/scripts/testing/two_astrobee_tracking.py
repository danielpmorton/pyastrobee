"""Testing out various ways of commanding two astrobees to manipulate the bag while tracking a trajectory"""


import time

import pybullet
import numpy as np

from pyastrobee.core.astrobee import Astrobee
from pyastrobee.core.cargo_bag import CargoBag
from pyastrobee.trajectories.planner import plan_trajectory
from pyastrobee.control.force_torque_control import ForceTorqueController
from pyastrobee.utils.rotations import Rx, rmat_to_quat
from pyastrobee.utils.bullet_utils import initialize_pybullet
from pyastrobee.utils.transformations import make_transform_mat
from pyastrobee.trajectories.leader_follower import offset_trajectory


def independent_example():
    """Simple example of two robots each tracking their own independent trajectories, without any consideration of the
    interaction between the two robots or the connection to the bag"""
    pybullet.connect(pybullet.GUI)
    duration = 5
    dt = pybullet.getPhysicsEngineParameters()["fixedTimeStep"]
    start_pose_1 = [0, -0.5, 0, *rmat_to_quat(Rx(np.pi / 2))]
    start_pose_2 = [0, 0.5, 0, *rmat_to_quat(Rx(-np.pi / 2))]
    start_vel_1 = [0.3, 0.3, 0.3]
    start_vel_2 = [0.3, 0.3, 0.3]
    start_omega_1 = np.zeros(3)
    start_omega_2 = np.zeros(3)
    start_accel_1 = np.zeros(3)
    start_accel_2 = np.zeros(3)
    start_alpha_1 = np.zeros(3)
    start_alpha_2 = np.zeros(3)
    end_pose_1 = [2, -0.5, 1, *rmat_to_quat(Rx(np.pi / 2))]
    end_pose_2 = [2, 0.5, 1, *rmat_to_quat(Rx(-np.pi / 2))]
    end_vel_1 = start_vel_1
    end_vel_2 = start_vel_2
    end_omega_1 = np.zeros(3)
    end_omega_2 = np.zeros(3)
    end_accel_1 = np.zeros(3)
    end_accel_2 = np.zeros(3)
    end_alpha_1 = np.zeros(3)
    end_alpha_2 = np.zeros(3)
    robot_1 = Astrobee(start_pose_1)
    robot_2 = Astrobee(start_pose_2)
    kp, kv, kq, kw = 20, 5, 1, 0.1
    controller_1 = ForceTorqueController(
        robot_1.id, robot_1.mass, robot_1.inertia, kp, kv, kq, kw, dt
    )
    controller_2 = ForceTorqueController(
        robot_2.id, robot_2.mass, robot_2.inertia, kp, kv, kq, kw, dt
    )
    traj_1 = plan_trajectory(
        start_pose_1[:3],
        start_pose_1[3:],
        start_vel_1,
        start_omega_1,
        start_accel_1,
        start_alpha_1,
        end_pose_1[:3],
        end_pose_1[3:],
        end_vel_1,
        end_omega_1,
        end_accel_1,
        end_alpha_1,
        duration,
        dt,
    )
    traj_2 = plan_trajectory(
        start_pose_2[:3],
        start_pose_2[3:],
        start_vel_1,
        start_omega_2,
        start_accel_2,
        start_alpha_2,
        end_pose_2[:3],
        end_pose_2[3:],
        end_vel_2,
        end_omega_2,
        end_accel_2,
        end_alpha_2,
        duration,
        dt,
    )
    traj_1.visualize(10)
    traj_2.visualize(10)

    assert traj_1.num_timesteps == traj_2.num_timesteps
    for i in range(traj_1.num_timesteps):
        pos_1, orn_1, lin_vel_1, ang_vel_1 = robot_1.dynamics_state
        pos_2, orn_2, lin_vel_2, ang_vel_2 = robot_2.dynamics_state
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

    while True:
        pybullet.stepSimulation()
        time.sleep(1 / 120)


def bag_example():
    """Example with the two robots connected to the bag

    Note: This currently seems to be unstable
    """
    client = initialize_pybullet()
    bag = CargoBag("right_left_handle")
    robot_1 = Astrobee()
    robot_2 = Astrobee()
    bag.attach_to([robot_1, robot_2])
    duration = 5
    dt = pybullet.getPhysicsEngineParameters()["fixedTimeStep"]
    start_pose_1 = robot_1.pose
    start_pose_2 = robot_2.pose
    start_vel_1 = [0.3, 0.3, 0.3]
    start_vel_2 = [0.3, 0.3, 0.3]
    start_omega_1 = np.zeros(3)
    start_omega_2 = np.zeros(3)
    start_accel_1 = np.zeros(3)
    start_accel_2 = np.zeros(3)
    start_alpha_1 = np.zeros(3)
    start_alpha_2 = np.zeros(3)
    end_pose_1 = start_pose_1 + np.array([3, 2, 1, 0, 0, 0, 0])
    end_pose_2 = start_pose_2 + np.array([3, 2, 1, 0, 0, 0, 0])
    end_vel_1 = start_vel_1
    end_vel_2 = start_vel_2
    end_omega_1 = np.zeros(3)
    end_omega_2 = np.zeros(3)
    end_accel_1 = np.zeros(3)
    end_accel_2 = np.zeros(3)
    end_alpha_1 = np.zeros(3)
    end_alpha_2 = np.zeros(3)
    kp, kv, kq, kw = 20, 5, 1, 0.1
    controller_1 = ForceTorqueController(
        robot_1.id, robot_1.mass, robot_1.inertia, kp, kv, kq, kw, dt
    )
    controller_2 = ForceTorqueController(
        robot_2.id, robot_2.mass, robot_2.inertia, kp, kv, kq, kw, dt
    )
    traj_1 = plan_trajectory(
        start_pose_1[:3],
        start_pose_1[3:],
        start_vel_1,
        start_omega_1,
        start_accel_1,
        start_alpha_1,
        end_pose_1[:3],
        end_pose_1[3:],
        end_vel_1,
        end_omega_1,
        end_accel_1,
        end_alpha_1,
        duration,
        dt,
    )
    traj_2 = plan_trajectory(
        start_pose_2[:3],
        start_pose_2[3:],
        start_vel_1,
        start_omega_2,
        start_accel_2,
        start_alpha_2,
        end_pose_2[:3],
        end_pose_2[3:],
        end_vel_2,
        end_omega_2,
        end_accel_2,
        end_alpha_2,
        duration,
        dt,
    )
    traj_1.visualize(10)
    traj_2.visualize(10)

    assert traj_1.num_timesteps == traj_2.num_timesteps
    for i in range(traj_1.num_timesteps):
        pos_1, orn_1, lin_vel_1, ang_vel_1 = robot_1.dynamics_state
        pos_2, orn_2, lin_vel_2, ang_vel_2 = robot_2.dynamics_state
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

    while True:
        pybullet.stepSimulation()
        # time.sleep(1 / 120)


def leader_follower_example():

    # TODO add info
    # TODO tracking seems like it could be improved

    pybullet.connect(pybullet.GUI)
    duration = 5
    dt = pybullet.getPhysicsEngineParameters()["fixedTimeStep"]
    start_pose_1 = [0, -0.5, 0, *rmat_to_quat(Rx(np.pi / 2))]
    start_vel_1 = [0.3, 0.3, 0.3]
    start_omega_1 = np.zeros(3)
    start_accel_1 = np.zeros(3)
    start_alpha_1 = np.zeros(3)
    end_pose_1 = [2, -0.5, 1, *rmat_to_quat(Rx(-np.pi / 2))]
    end_vel_1 = np.zeros(3)
    end_omega_1 = np.zeros(3)
    end_accel_1 = np.zeros(3)
    end_alpha_1 = np.zeros(3)

    leader_traj = plan_trajectory(
        start_pose_1[:3],
        start_pose_1[3:],
        start_vel_1,
        start_omega_1,
        start_accel_1,
        start_alpha_1,
        end_pose_1[:3],
        end_pose_1[3:],
        end_vel_1,
        end_omega_1,
        end_accel_1,
        end_alpha_1,
        duration,
        dt,
    )
    # TODO make the offset distance a parameter to tune
    # (get the correct values depending on the bag)
    # The rotation will probably always be the same unless we mess with the arm joints
    offset_distance = 1
    T_F2L = make_transform_mat(Rx(np.pi), [0, 0, -1 * offset_distance])

    follower_traj = offset_trajectory(leader_traj, T_F2L)
    leader = Astrobee(start_pose_1)
    follower = Astrobee(follower_traj.poses[0])

    kp, kv, kq, kw = 20, 5, 5, 5  # 20, 5, 1, 0.1
    leader_controller = ForceTorqueController(
        leader.id, leader.mass, leader.inertia, kp, kv, kq, kw, dt
    )
    follower_controller = ForceTorqueController(
        follower.id, follower.mass, follower.inertia, kp, kv, kq, kw, dt
    )

    leader_traj.visualize(10)
    follower_traj.visualize(10)

    for i in range(leader_traj.num_timesteps):
        pos_1, orn_1, lin_vel_1, ang_vel_1 = leader.dynamics_state
        pos_2, orn_2, lin_vel_2, ang_vel_2 = follower.dynamics_state
        leader_controller.step(
            pos_1,
            lin_vel_1,
            orn_1,
            ang_vel_1,
            leader_traj.positions[i, :],
            leader_traj.linear_velocities[i, :],
            leader_traj.linear_accels[i, :],
            leader_traj.quaternions[i, :],
            leader_traj.angular_velocities[i, :],
            leader_traj.angular_accels[i, :],
        )
        follower_controller.step(
            pos_2,
            lin_vel_2,
            orn_2,
            ang_vel_2,
            follower_traj.positions[i, :],
            follower_traj.linear_velocities[i, :],
            follower_traj.linear_accels[i, :],
            follower_traj.quaternions[i, :],
            follower_traj.angular_velocities[i, :],
            follower_traj.angular_accels[i, :],
        )
        time.sleep(1 / 120)

    # Stopping mode
    while True:
        pos_1, orn_1, lin_vel_1, ang_vel_1 = leader.dynamics_state
        pos_2, orn_2, lin_vel_2, ang_vel_2 = follower.dynamics_state
        leader_controller.step(
            pos_1,
            lin_vel_1,
            orn_1,
            ang_vel_1,
            leader_traj.positions[-1, :],
            leader_traj.linear_velocities[-1, :],
            leader_traj.linear_accels[-1, :],
            leader_traj.quaternions[-1, :],
            leader_traj.angular_velocities[-1, :],
            leader_traj.angular_accels[-1, :],
        )
        follower_controller.step(
            pos_2,
            lin_vel_2,
            orn_2,
            ang_vel_2,
            follower_traj.positions[-1, :],
            follower_traj.linear_velocities[-1, :],
            follower_traj.linear_accels[-1, :],
            follower_traj.quaternions[-1, :],
            follower_traj.angular_velocities[-1, :],
            follower_traj.angular_accels[-1, :],
        )
        time.sleep(1 / 120)


if __name__ == "__main__":
    # independent_example()
    # bag_example()
    leader_follower_example()
