"""Simple scripts to test some of the attributes and functions in the CargoBag class"""

import time

import pybullet

from pyastrobee.core.astrobee import Astrobee
from pyastrobee.core.cargo_bag import CargoBag
from pyastrobee.utils.bullet_utils import initialize_pybullet


def test_attach_and_detach():
    client = initialize_pybullet()
    robot = Astrobee()
    bag = CargoBag("top_handle")
    bag.attach_to(robot)
    init_time = time.time()
    detach_time_lim = 10
    dt = pybullet.getPhysicsEngineParameters()["fixedTimeStep"]
    print(f"Provide a disturbance force. Detaching bag in {detach_time_lim} seconds")
    while time.time() - init_time < detach_time_lim:
        pybullet.stepSimulation()
        time.sleep(dt)
    bag.detach()
    detach_time = time.time()
    unload_time_lim = 10
    print(f"Detached. Unloading the bag in {unload_time_lim} seconds")
    while time.time() - detach_time < unload_time_lim:
        pybullet.stepSimulation()
        time.sleep(dt)
    bag.unload()  # Currently kinda weird
    print("Unloaded")
    while True:
        pybullet.stepSimulation()
        time.sleep(dt)


def test_dynamics():
    client = initialize_pybullet()
    robot = Astrobee()
    bag = CargoBag("top_handle")
    bag.attach_to(robot)
    dt = pybullet.getPhysicsEngineParameters()["fixedTimeStep"]
    while True:
        pos, orn, lin_vel, ang_vel = bag.dynamics_state
        print("Position: ", pos)
        print("Orientation: ", orn)
        print("Velocity: ", lin_vel)
        print("Angular velocity: ", ang_vel)
        # visualize_frame(pos_quat_to_tmat([*pos, *orn]), lifetime=0.5)
        pybullet.stepSimulation()
        time.sleep(dt)


def test_multi_robot():
    client = initialize_pybullet()
    robot_1 = Astrobee()
    robot_2 = Astrobee()
    bag = CargoBag("top_bottom_handle")
    # bag = CargoBag("front_back_handle")
    # bag = CargoBag("right_left_handle")
    bag.attach_to([robot_1, robot_2])
    while True:
        pybullet.stepSimulation()
        time.sleep(1 / 120)


if __name__ == "__main__":
    # test_attach_and_detach()
    # test_dynamics()
    test_multi_robot()
