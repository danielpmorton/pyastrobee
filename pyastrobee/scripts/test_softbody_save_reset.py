"""Test script to see if we can get the pybullet saveState / restoreState functions working with deformables

This should be run from a terminal window where we're working with the local version of pybullet
For reference, thus can be done by:
1) (optional) Build Bullet after making any changes: In the bullet3 folder (/home/dan/software/bullet3), run
   ./build_cmake_pybullet_double.sh
2) Update the PYTHONPATH: export PYTHONPATH=/home/dan/software/bullet3/build_cmake/examples/pybullet
3) Run this script in that terminal
"""

import time

import numpy as np
import pybullet
import pybullet_data

from pyastrobee.elements.astrobee import Astrobee
from pyastrobee.trajectories.polynomials import polynomial_trajectory
from pyastrobee.utils.quaternions import random_quaternion
from pyastrobee.control.force_controller_new import ForcePIDController
from pyastrobee.elements.cargo_bag import CargoBag
from pyastrobee.utils.rotations import euler_xyz_to_quat
from pyastrobee.utils.bullet_utils import (
    initialize_pybullet,
    load_deformable_object,
    load_floor,
)


def loop_sim(t=5):
    """Run the sim for a specified amount of time"""
    start_time = time.time()
    while time.time() - start_time < t:
        pybullet.stepSimulation()
        time.sleep(1 / 120)


def run_test(object: str):
    """Pause, save, and reset the sim after user interaction. Object is either "bag", "cloth", or "astrobee"."""
    initialize_pybullet()
    if object == "bag":
        bag = CargoBag("top_handle_bag", None, [0, 0, 0], [0, 0, 0, 1])
    elif object == "cloth":
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        cloth_id = load_deformable_object("cloth_z_up.obj")
    elif object == "astrobee":
        robot = Astrobee()
        bag = CargoBag("top_handle_bag", robot)
    print("Apply a disturbance force")
    loop_sim()
    input("Press Enter to save the state")
    state_id = pybullet.saveState()
    input("Press Enter to let the sim run for a bit")
    loop_sim()
    input("Press Enter to reset the state")
    pybullet.restoreState(stateId=state_id)
    input("Press Enter to let the sim keep running")
    loop_sim()


def multi_reset_test():
    """Test to see how multiple successive resets affects the deformable behavior.

    With previous implementations of save/restoreState, this lead to accumulated error in the handle of the bag.
    However, with the new internal Bullet implementation of this, it should not lead to this issue
    """
    initialize_pybullet()
    robot = Astrobee()
    start_pose = [0, 0, 0, 0, 0, 0, 1]
    end_pose = [1, 2, 3, *random_quaternion()]
    duration = 10
    dt = 1 / 350
    traj = polynomial_trajectory(start_pose, end_pose, duration, dt)
    kp = 20
    kv = 5
    kq = 1
    kw = 0.1
    controller = ForcePIDController(
        robot.id, robot.mass, robot.inertia, kp, kv, kq, kw, dt
    )
    bag = CargoBag("top_handle_bag", robot)
    steps_between_resets = 10  # Adjust to change reset frequency
    for i in range(traj.num_timesteps):
        pos, orn, lin_vel, ang_vel = controller.get_current_state()
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
        if (i % steps_between_resets) == 0:
            print("Saving")
            state_id = pybullet.saveState()
            print("Resetting")
            pybullet.restoreState(stateId=state_id)


def multi_object_test():
    """Test with a floor and two deformables to check that Bullet finds the correct UniqueIDs in the restore process"""
    initialize_pybullet()
    floor_id = load_floor(z_pos=-2)
    pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
    cloth_1 = load_deformable_object(
        "cloth_z_up.obj",
        "pyastrobee/assets/imgs/textile_pixabay_gray.jpg",
        pos=[0, 0, 0],
        orn=euler_xyz_to_quat([1.57, 3.14, 0]),
    )
    cloth_2 = load_deformable_object(
        "cloth_z_up.obj",
        "pyastrobee/assets/imgs/textile_pixabay_red.jpg",
        pos=[1, 1, 1],
        orn=euler_xyz_to_quat([1.57, 3.14, 0]),
    )
    print(f"Cloth IDs: {cloth_1}, {cloth_2}")
    print(f"Floor ID: {floor_id}")
    print("Apply a disturbance force")
    loop_sim()
    input("Press Enter to save the state")
    state_id = pybullet.saveState()
    input("Press Enter to let the sim run for a bit")
    loop_sim()
    input("Press Enter to reset the state")
    pybullet.restoreState(stateId=state_id)
    input("Press Enter to let the sim keep running")
    loop_sim()


if __name__ == "__main__":
    # run_test("bag")
    # run_test("cloth")
    # run_test("astrobee")
    multi_object_test()
    # multi_reset_test()
