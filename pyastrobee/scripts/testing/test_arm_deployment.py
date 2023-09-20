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
from pyastrobee.trajectories.polynomials import fifth_order_poly
from pyastrobee.control.force_torque_control import ForceTorqueController
from pyastrobee.utils.transformations import invert_transform_mat
from pyastrobee.trajectories.trajectory import Trajectory, ArmTrajectory


# The idea here was to determine how much the arm moves back when it is moved from "grasp" to "drag"
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


def plan_arm_traj(base_traj: Trajectory):
    """Plans a motion for the arm so that we are dragging the bag behind the robot for most of the trajectory, but we
    start and end from the grasping position

    This will only use the shoulder joint -- other joints are less relevant for the motion of the bag/robot system

    This is essentially comprised of three parts:
    1) Moving the arm from an initial grasp position to the "dragging behind" position
    2) Maintaining the "dragging behind" position for most of the trajectory
    3) Moving the arm to the grasp position again at the end of the trajectory

    Args:
        base_traj (Trajectory): Trajectory of the Astrobee base

    Returns:
        _type_: _description_ # TODO!!!
    """
    # Shoulder joint parameters
    grasp_angle = 0
    drag_angle = -1.57079
    shoulder_index = Astrobee.ARM_JOINT_IDXS[0]
    # Find the transition times/indices for the arm motion planning
    # We could define a fixed amount of time to allocate to the arm motion, but it makes a bit more sense to define this
    # with respect to the motion of the base. The arm should move jointly with the base, so we can say that it should
    # be fully deployed after a certain displacement of the robot
    dists_from_start = np.linalg.norm(
        base_traj.positions - base_traj.positions[0], axis=1
    )
    dists_from_end = np.linalg.norm(
        base_traj.positions - base_traj.positions[-1], axis=1
    )
    # A little over 1 meter seems to work well for the deployment distance (TODO probably needs tuning)
    # For reference, the 0.23 number is how much the x position of the arm changes when it moves back
    dx_arm = 0.23162640743995172 * 5
    # Note: originally I used searchsorted to find these indices, but these distances are not guaranteed to be sorted
    drag_idx = np.flatnonzero(dists_from_start >= dx_arm)[0]
    grasp_idx = np.flatnonzero(dists_from_end <= dx_arm)[0]
    drag_time = base_traj.times[drag_idx]
    grasp_time = base_traj.times[grasp_idx]
    end_time = base_traj.times[-1]
    drag_poly = fifth_order_poly(0, drag_time, grasp_angle, drag_angle, 0, 0, 0, 0)
    grasp_poly = fifth_order_poly(
        grasp_time, end_time, drag_angle, grasp_angle, 0, 0, 0, 0
    )
    arm_motion = np.ones_like(base_traj.times) * drag_angle
    arm_motion[:drag_idx] = drag_poly(base_traj.times[:drag_idx])
    arm_motion[grasp_idx:] = grasp_poly(base_traj.times[grasp_idx:])
    # Edge case for short trajectories: not enough time to fully move arm there and back again
    overlap = drag_idx > grasp_idx
    if overlap:
        # Blend the two motions in the overlapping region
        overlap_poly = (
            (drag_poly - drag_angle) + (grasp_poly - drag_angle)
        ) + drag_angle
        arm_motion[grasp_idx:drag_idx] = np.clip(
            overlap_poly(base_traj.times[grasp_idx:drag_idx]), drag_angle, grasp_angle
        )
    # TODO figure out if this info is even useful
    info = {
        "drag_index": drag_idx,
        "drag_time": base_traj.times[drag_idx],
        "grasp_index": grasp_idx,
        "grasp_time": base_traj.times[grasp_idx],
    }
    return (
        ArmTrajectory(arm_motion, [shoulder_index], base_traj.times),
        info,
    )


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
    arm_traj, info = plan_arm_traj(traj)
    arm_traj.plot()

    for i in range(traj.num_timesteps):
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
        if i == info["drag_index"]:
            controller.inertia = robot.inertia + drag_inertia
        # if i == info["grasp_index"]:  # Does it make sense to update it here? Or at the end of the nominal traj?
        if i == traj.num_timesteps - 1:
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
