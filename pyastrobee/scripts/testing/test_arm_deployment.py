"""Test script to evaluate integrating arm motion into the trajectory

The robot will grab the bag in one orientation but will need to drag the
bag along, ideally face forward. So, the arm should be in the "reached back"
state for better dragging

When positioning the bag at the end of the trajectory, the arm should also go back
to the original state
"""

# TODO
# make sure that the trajectory orientation component is face-forwards!

import numpy as np

from pyastrobee.core.astrobee import Astrobee
from pyastrobee.config.astrobee_motion import MAX_FORCE_MAGNITUDE, MAX_TORQUE_MAGNITUDE
from pyastrobee.core.constraint_bag import ConstraintCargoBag
from pyastrobee.utils.bullet_utils import initialize_pybullet
from pyastrobee.trajectories.planner import global_planner
from pyastrobee.control.force_torque_control import ForceTorqueController
from pyastrobee.utils.transformations import invert_transform_mat
from pyastrobee.trajectories.arm_planner import plan_arm_traj


# The idea here was to determine how much the arm moves back when it is moved from "grasp" to "drag"
# This outputs: X position delta:  -0.23162640743995172
def get_arm_pos_delta():
    client = initialize_pybullet(bg_color=(1, 1, 1))
    robot = Astrobee()
    client.changeDynamics(robot.id, -1, 0)  # Fix base
    pose_1 = robot.ee_pose
    input("Press Enter to set the arm")
    robot.set_joint_angles([-1.57079], [Astrobee.ARM_JOINT_IDXS[0]], wait=True)
    pose_2 = robot.ee_pose
    print("Pose 1: ", pose_1)
    print("Pose 2: ", pose_2)
    print("X position delta: ", (pose_2[:3] - pose_1[:3])[0])
    input("Press Enter to exit")
    client.disconnect()


def get_bag_local_offset(robot: Astrobee, bag: ConstraintCargoBag):
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
    print("GRAB Bag offset", get_bag_local_offset(robot, bag))
    input("Press Enter to set the arm")
    bag.detach()
    robot.set_joint_angles([-1.57079], [Astrobee.ARM_JOINT_IDXS[0]], force=True)
    bag.attach_to(robot, "bag")
    print("DRAG Bag offset", get_bag_local_offset(robot, bag))
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
    # Determine what the moment of inertia contribution from the bag is...
    # TODO this is kinda weird. Right now this is like we assume that this is a point mass that is rigidly connected
    # to the arm at the center of mass of the bag. This might be an OK model for the rigid analogs... but the deformable
    # probably not. Does it make sense to have a notion of inertia there? Perhaps the deformable should be modeled
    # as a point mass applied at the grasp point.
    # Also, these positions are for the TOP HANDLE bag only... TODO get the values for all bags
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
    traj.visualize(10, client=client)
    arm_traj = plan_arm_traj(traj)
    arm_traj.plot()

    for i, t in enumerate(traj.times):
        pos, orn, lin_vel, ang_vel = controller.get_current_state()
        client.setJointMotorControlArray(
            robot.id,
            arm_traj.indices,
            client.POSITION_CONTROL,
            arm_traj.angles[i, :],
            forces=Astrobee.JOINT_EFFORT_LIMITS[arm_traj.indices],
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
        # Update inertia values after the arm is fully deployed to a new state
        if abs(t - arm_traj.key_times["end_drag_motion"]) <= dt / 2:
            print("DONE WITH DRAG MOTION")
            controller.inertia = robot.inertia + drag_inertia
        if abs(t - arm_traj.key_times["end_grasp_motion"]) <= dt / 2:
            print("DONE WITH GRASP MOTION")
            controller.inertia = robot.inertia + grab_inertia
        # time.sleep(1 / 240)
    while True:
        pos, orn, lin_vel, ang_vel = controller.get_current_state()
        controller.step(
            pos,
            lin_vel,
            orn,
            ang_vel,
            traj.positions[-1, :],
            traj.linear_velocities[-1, :],
            traj.linear_accels[-1, :],
            traj.quaternions[-1, :],
            traj.angular_velocities[-1, :],
            traj.angular_accels[-1, :],
        )
        # time.sleep(1 / 240)


if __name__ == "__main__":
    # get_arm_pos_delta()
    _run_test()
    # get_bag_positions()
