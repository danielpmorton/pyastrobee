"""Testing out various ways of commanding two astrobees to manipulate the bag while tracking a trajectory"""

from pathlib import Path
from datetime import datetime

import pybullet
import numpy as np

from pyastrobee.core.astrobee import Astrobee
from pyastrobee.core.deformable_bag import DeformableCargoBag
from pyastrobee.core.rigid_bag import RigidCargoBag
from pyastrobee.core.constraint_bag import ConstraintCargoBag
from pyastrobee.trajectories.planner import local_planner
from pyastrobee.control.force_torque_control import ForceTorqueController
from pyastrobee.utils.rotations import Rx, rmat_to_quat
from pyastrobee.utils.bullet_utils import initialize_pybullet
from pyastrobee.utils.transformations import make_transform_mat, invert_transform_mat
from pyastrobee.trajectories.multi_robot_trajs import (
    offset_trajectory,
    multi_trajectory,
)
from pyastrobee.utils.debug_visualizer import visualize_path
from pyastrobee.control.multi_robot import multi_robot_control


RECORD_VIDEO = True
VIDEO_LOCATION = (
    f"artifacts/{Path(__file__).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.mp4"
)
# Debug visualizer camera parameters: Dist, yaw, pitch, target
CAMERA_VIEW = (1.40, 0, -7.80, (1.15, 0, 0.50))


def independent_example():
    """Simple example of two robots each tracking their own independent trajectories, without any consideration of the
    interaction between the two robots or the connection to the bag"""
    pybullet.connect(pybullet.GUI)
    duration = 5
    dt = pybullet.getPhysicsEngineParameters()["fixedTimeStep"]
    start_pose_1 = [0, -0.5, 0, *rmat_to_quat(Rx(np.pi / 2))]
    start_pose_2 = [0, 0.5, 0, *rmat_to_quat(Rx(-np.pi / 2))]
    start_vel_1 = np.zeros(3)
    start_vel_2 = np.zeros(3)
    start_omega_1 = np.zeros(3)
    start_omega_2 = np.zeros(3)
    start_accel_1 = np.zeros(3)
    start_accel_2 = np.zeros(3)
    start_alpha_1 = np.zeros(3)
    start_alpha_2 = np.zeros(3)
    end_pose_1 = [2, -0.5, 1, *rmat_to_quat(Rx(np.pi / 2))]
    end_pose_2 = [2, 0.5, 1, *rmat_to_quat(Rx(-np.pi / 2))]
    end_vel_1 = np.zeros(3)
    end_vel_2 = np.zeros(3)
    end_omega_1 = np.zeros(3)
    end_omega_2 = np.zeros(3)
    end_accel_1 = np.zeros(3)
    end_accel_2 = np.zeros(3)
    end_alpha_1 = np.zeros(3)
    end_alpha_2 = np.zeros(3)
    robot_1 = Astrobee(start_pose_1)
    robot_2 = Astrobee(start_pose_2)
    kp, kv, kq, kw = 20, 10, 5, 5
    controller_1 = ForceTorqueController(
        robot_1.id, robot_1.mass, robot_1.inertia, kp, kv, kq, kw, dt
    )
    controller_2 = ForceTorqueController(
        robot_2.id, robot_2.mass, robot_2.inertia, kp, kv, kq, kw, dt
    )
    traj_1 = local_planner(
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
    traj_2 = local_planner(
        start_pose_2[:3],
        start_pose_2[3:],
        start_vel_2,
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

    multi_robot_control([controller_1, controller_2], [traj_1, traj_2], True)


# TODO tracking seems like it could be improved
def leader_follower_example():
    """Example with one astrobee following a set 'leader' trajectory, and a second astrobee following at a fixed
    offset from the leader
    """
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

    leader_traj = local_planner(
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

    kp, kv, kq, kw = 20, 10, 5, 5  # 20, 5, 1, 0.1
    leader_controller = ForceTorqueController(
        leader.id, leader.mass, leader.inertia, kp, kv, kq, kw, dt
    )
    follower_controller = ForceTorqueController(
        follower.id, follower.mass, follower.inertia, kp, kv, kq, kw, dt
    )

    leader_traj.visualize(10)
    follower_traj.visualize(10)

    multi_robot_control(
        [leader_controller, follower_controller],
        [leader_traj, follower_traj],
        True,
    )


def dual_trajectory_example():
    """Example with two astrobees each tracking with a symetric offset about a reference trajectory"""

    pybullet.connect(pybullet.GUI)
    duration = 5
    dt = pybullet.getPhysicsEngineParameters()["fixedTimeStep"]
    start_pose_1 = [0, -0.5, 0, 0, 0, 0, 1]
    start_vel_1 = np.zeros(3)
    start_omega_1 = np.zeros(3)
    start_accel_1 = np.zeros(3)
    start_alpha_1 = np.zeros(3)
    end_pose_1 = [2, -0.5, 1, *rmat_to_quat(Rx(np.pi))]
    end_vel_1 = np.zeros(3)
    end_omega_1 = np.zeros(3)
    end_accel_1 = np.zeros(3)
    end_alpha_1 = np.zeros(3)

    reference_traj = local_planner(
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
    T_A = make_transform_mat(Rx(np.pi), [0, 0, -1 * offset_distance])
    T_B = make_transform_mat(Rx(0), [0, 0, 1 * offset_distance])

    traj_A, traj_B = multi_trajectory(reference_traj, [T_A, T_B])
    robot_A = Astrobee(traj_A.poses[0])
    robot_B = Astrobee(traj_B.poses[0])

    # These need to be better tuned for this example
    # I think some of the tracking error is due to high centrifugal forces in this example
    # which causes some error due to the COM offset from the arm
    kp, kv, kq, kw = 20, 10, 5, 5
    leader_controller = ForceTorqueController(
        robot_A.id, robot_A.mass, robot_A.inertia, kp, kv, kq, kw, dt
    )
    follower_controller = ForceTorqueController(
        robot_B.id, robot_B.mass, robot_B.inertia, kp, kv, kq, kw, dt
    )

    n_viz = 20
    traj_A.visualize(n_viz)
    visualize_path(traj_A.positions, n_viz, (1, 1, 1))
    traj_B.visualize(n_viz)
    visualize_path(traj_B.positions, n_viz, (1, 1, 1))
    visualize_path(reference_traj.positions, n_viz, (1, 1, 1))

    multi_robot_control(
        [leader_controller, follower_controller],
        [traj_A, traj_B],
        True,
    )


def bag_example():
    client = initialize_pybullet()
    client.resetDebugVisualizerCamera(*CAMERA_VIEW)
    duration = 5
    dt = client.getPhysicsEngineParameters()["fixedTimeStep"]
    start_pose_1 = [0, 0, 0, 0, 0, 0, 1]
    start_vel_1 = np.zeros(3)
    start_omega_1 = np.zeros(3)
    start_accel_1 = np.zeros(3)
    start_alpha_1 = np.zeros(3)
    end_pose_1 = [2, 0, 1, *rmat_to_quat(Rx(np.pi))]
    end_vel_1 = np.zeros(3)
    end_omega_1 = np.zeros(3)
    end_accel_1 = np.zeros(3)
    end_alpha_1 = np.zeros(3)

    reference_traj = local_planner(
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
    # bag = DeformableCargoBag("right_left_handle", 10)
    bag = ConstraintCargoBag("right_left_handle", 10)
    robot_A = Astrobee()
    robot_B = Astrobee()
    bag.attach_to([robot_A, robot_B])
    # R: Robot, G: Gripper, C: Center of the bag
    T_G2C_A, T_G2C_B = bag.grasp_transforms
    T_R2W_A = robot_A.tmat
    T_R2W_B = robot_B.tmat
    T_G2W_A = robot_A.ee_tmat
    T_G2W_B = robot_B.ee_tmat
    T_R2C_A = T_G2C_A @ invert_transform_mat(T_G2W_A) @ T_R2W_A
    T_R2C_B = T_G2C_B @ invert_transform_mat(T_G2W_B) @ T_R2W_B
    traj_A, traj_B = multi_trajectory(reference_traj, [T_R2C_A, T_R2C_B])

    kp, kv, kq, kw = 20, 10, 5, 5
    leader_controller = ForceTorqueController(
        robot_A.id, robot_A.mass + bag.mass / 2, robot_A.inertia, kp, kv, kq, kw, dt
    )
    follower_controller = ForceTorqueController(
        robot_B.id, robot_B.mass + bag.mass / 2, robot_B.inertia, kp, kv, kq, kw, dt
    )

    n_viz = 20
    reference_traj.visualize(n_viz, 0.25)

    if RECORD_VIDEO:
        input("Press Enter to begin")
        log_id = client.startStateLogging(
            client.STATE_LOGGING_VIDEO_MP4, VIDEO_LOCATION
        )
    try:
        multi_robot_control(
            [leader_controller, follower_controller],
            [traj_A, traj_B],
            True,
        )
    finally:
        client.stopStateLogging(log_id)
        client.disconnect()


if __name__ == "__main__":
    # independent_example()
    bag_example()
    # leader_follower_example()
    # dual_trajectory_example()
