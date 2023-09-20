"""Test script to evaluate integrating arm motion into the trajectory

The robot will grab the bag in one orientation but will need to drag the
bag along, ideally face forward. So, the arm should be in the "reached back"
state for better dragging

When positioning the bag at the end of the trajectory, the arm should also go back
to the original state
"""
import time
import pybullet
import numpy as np
import matplotlib.pyplot as plt

from pyastrobee.core.astrobee import Astrobee
from pyastrobee.config.astrobee_motion import MAX_FORCE_MAGNITUDE, MAX_TORQUE_MAGNITUDE
from pyastrobee.core.constraint_bag import ConstraintCargoBag
from pyastrobee.utils.bullet_utils import initialize_pybullet
from pyastrobee.trajectories.planner import global_planner
from pyastrobee.trajectories.polynomials import fifth_order_poly
from pyastrobee.control.force_torque_control import ForceTorqueController
from pyastrobee.utils.transformations import invert_transform_mat


# This outputs: X position delta:  -0.23162640743995172
def get_arm_pos_delta():
    client = initialize_pybullet(bg_color=(1, 1, 1))
    robot = Astrobee()
    client.changeDynamics(robot.id, -1, 0)  # Fix base
    pose_1 = robot.ee_pose
    input("Press Enter to set the arm")
    robot.set_joint_angles([-1.57079], [Astrobee.ARM_JOINT_IDXS[0]])
    pose_2 = robot.ee_pose
    print("Pose 1: ", pose_1)
    print("Pose 2: ", pose_2)
    print("X position delta: ", (pose_2[:3] - pose_1[:3])[0])
    input("Press Enter to exit")
    client.disconnect()


def get_bag_local_offset(robot: Astrobee, bag: ConstraintCargoBag, client: pybullet):
    T_R2W = robot.tmat
    T_B2W = bag.tmat
    T_B2R = invert_transform_mat(T_R2W) @ T_B2W
    return T_B2R[:3, 3]


# Output: GRAB: Bag offset [-0.07968587  0.         -0.55740857]
# DRAG: Bag offset [-0.55684116  0.         -0.17977188]
def get_bag_positions():
    client = initialize_pybullet(bg_color=(1, 1, 1))
    robot = Astrobee()
    bag = ConstraintCargoBag("top_handle", 10)
    client.changeDynamics(robot.id, -1, 0)  # Fix base
    pose_1 = robot.ee_pose
    bag.attach_to(robot, "bag")
    print("GRAB Bag offset", get_bag_local_offset(robot, bag, client))
    input("Press Enter to set the arm")
    bag.detach()
    robot.set_joint_angles([-1.57079], [Astrobee.ARM_JOINT_IDXS[0]], force=True)
    bag.attach_to(robot, "bag")
    print("DRAGBag offset", get_bag_local_offset(robot, bag, client))
    input("Press Enter to exit")
    client.disconnect()


def _run_test():
    np.random.seed(0)
    start_pose = [0, 0, 0, 0, 0, 0, 1]
    end_pose = [6, 0, 0.2, 0, 0, 0, 1]  # Easy-to-reach location in JPM
    bag_name = "top_handle"
    client = initialize_pybullet()
    robot = Astrobee()
    bag_mass = 10
    bag = ConstraintCargoBag(bag_name, bag_mass)
    bag.attach_to(robot, "bag")
    dt = client.getPhysicsEngineParameters()["fixedTimeStep"]
    # p = get_bag_local_offset(robot, bag, client)
    p_grab = np.array([-0.07968587, 0, -0.55740857])
    p_drag = np.array([-0.55684116, 0, -0.17977188])
    grab_inertia = bag_mass * (
        np.dot(p_grab, p_grab) * np.eye(3) - np.outer(p_grab, p_grab)
    )
    drag_inertia = bag_mass * (
        np.dot(p_drag, p_drag) * np.eye(3) - np.outer(p_drag, p_drag)
    )
    kp, kv, kq, kw = 20, 5, 5, 0.1
    controller = ForceTorqueController(
        robot.id,
        robot.mass + bag.mass,
        robot.inertia + grab_inertia,
        kp,
        kv,
        kq,
        kw,
        dt,
        max_force=MAX_FORCE_MAGNITUDE,
        max_torque=MAX_TORQUE_MAGNITUDE,
        client=client,
    )
    traj = global_planner(
        start_pose[:3],
        start_pose[3:],
        end_pose[:3],
        end_pose[3:],
        dt,
    )
    d_pos_start = np.linalg.norm(traj.positions - start_pose[:3], axis=1)
    d_pos_end = np.linalg.norm(traj.positions - end_pose[:3], axis=1)
    dx_arm = 0.23162640743995172
    dx_arm *= 2  # TESTING
    grab_shoulder_angle = robot.joint_angles[robot.ARM_JOINT_IDXS[0]]
    drag_shoulder_angle = -1.57079
    grab_to_drag_idx = np.searchsorted(d_pos_start, dx_arm)
    drag_to_grab_idx = np.searchsorted(-d_pos_end, -dx_arm)
    T_drag = traj.times[grab_to_drag_idx]
    T_grab = traj.times[drag_to_grab_idx]
    tf = traj.times[-1]
    grab_to_drag_shoulder_poly = fifth_order_poly(
        0, T_drag, grab_shoulder_angle, drag_shoulder_angle, 0, 0, 0, 0
    )
    drag_to_grab_shoulder_poly = fifth_order_poly(
        T_grab, tf, drag_shoulder_angle, grab_shoulder_angle, 0, 0, 0, 0
    )
    grab_to_drag_traj = grab_to_drag_shoulder_poly(traj.times[:grab_to_drag_idx])
    drag_to_grab_traj = drag_to_grab_shoulder_poly(traj.times[drag_to_grab_idx:])
    shoulder_traj = np.ones_like(traj.times) * drag_shoulder_angle
    shoulder_traj[:grab_to_drag_idx] = grab_to_drag_traj
    shoulder_traj[drag_to_grab_idx:] = drag_to_grab_traj

    # if bag_name.startswith("top_handle"):
    #     p_grab =  + np.array([0, 0, ConstraintCargoBag.HEIGHT / 2]) + ConstraintCargoBag.grasp_transforms
    #     p_drag =

    # plt.figure()
    # plt.plot(traj.times, shoulder_traj)
    # plt.show()
    # This traj has a constant orientation component so we don't need to face-forward it
    # TODO get this working

    # TODO  make sure that the traj is long enough that it makes sense to move the arm
    for i in range(traj.num_timesteps):
        pos, orn, lin_vel, ang_vel = controller.get_current_state()
        client.setJointMotorControlArray(
            robot.id,
            [robot.ARM_JOINT_IDXS[0]],
            client.POSITION_CONTROL,
            [shoulder_traj[i]],
        )
        controller.step(
            pos,
            lin_vel,
            orn,
            ang_vel,
            traj.positions[i, :],
            traj.linear_velocities[i, :],
            traj.linear_accels[i, :],
            traj.quaternions[i, :],
            traj.angular_velocities[i, :],
            traj.angular_accels[i, :],
        )
        if i == grab_to_drag_idx:
            # Update the inertia
            controller.inertia = robot.inertia + drag_inertia
        if i == drag_to_grab_idx:  # Does it make sense to update it here?
            controller.inertia = robot.inertia + grab_inertia
        time.sleep(1 / 240)


if __name__ == "__main__":
    # get_arm_pos_delta()
    _run_test()
    # get_bag_positions()
