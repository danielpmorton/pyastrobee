"""Script to test to see if the state save/reset mechanic works with deformables"""


import time

import pybullet

from pyastrobee.elements.astrobee import Astrobee
from pyastrobee.elements.cargo_bag import CargoBag
from pyastrobee.utils.bullet_utils import initialize_pybullet
from pyastrobee.control.mpc import save_state, reset_state

if __name__ == "__main__":
    # Quick test script to see if the save/reset state works
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
