"""Script to test to see if the state save/reset mechanic works with deformables"""


import time

import pybullet

from pyastrobee.elements.astrobee import Astrobee
from pyastrobee.elements.cargo_bag import CargoBag
from pyastrobee.utils.bullet_utils import initialize_pybullet
from pyastrobee.control.mpc import save_state, reset_state
from pyastrobee.trajectories.polynomials import polynomial_trajectory
from pyastrobee.utils.quaternions import random_quaternion
from pyastrobee.control.force_controller_new import ForcePIDController


def simple_test():
    # Quick test script to see if the save/reset state works
    # This will run the sim for a bit and allow the user to interact, then save an arbitrary state and reset it
    # after some time has passed
    initialize_pybullet()
    robot = Astrobee()
    bag = CargoBag("top_handle_bag", robot)
    start_time = time.time()
    print("5 seconds to adjust the robot and bag")
    while time.time() - start_time < 5:
        pybullet.stepSimulation()
        time.sleep(1 / 240)
    input("Press Enter to SAVE")
    state_id, bag_pos, bag_orn, bag_vel, bag_ang_vel = save_state(bag)
    input("Press Enter to RANDOMIZE")
    print("5 seconds to adjust the robot and bag")
    start_time = time.time()
    while time.time() - start_time < 5:
        pybullet.stepSimulation()
        time.sleep(1 / 240)
    input("Press Enter to RESET")
    reset_state(state_id, bag.id, bag_pos, bag_orn, bag_vel, bag_ang_vel)
    while True:
        pybullet.stepSimulation()
        time.sleep(1 / 240)


def test_multiple_resets():
    # Test to see how multiple successive resets changes the behavior of the deformable
    # It appears that if we don't significantly mess with the deformable then the handle remains "ok"
    # However if we introduce large amounts of stretch to the handle, after the next reset, this stretch will remain
    # in a weird state. This seems to be what causes some of the issue with MPC when the sampled trajectories stretch
    # the bag in odd ways
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
            state_id, bag_pos, bag_orn, bag_vel, bag_ang_vel = save_state(bag)
            print("Resetting")
            reset_state(state_id, bag.id, bag_pos, bag_orn, bag_vel, bag_ang_vel)


if __name__ == "__main__":
    # simple_test()
    test_multiple_resets()
