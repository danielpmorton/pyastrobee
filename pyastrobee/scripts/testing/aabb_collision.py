"""Testing out using axis aligned bounding boxes for conservative 'collision' detection

'collision' here just referring to departure from the safe set
"""

import time

import numpy as np

from pyastrobee.core.astrobee import Astrobee
from pyastrobee.core.constraint_bag import ConstraintCargoBag
from pyastrobee.utils.bullet_utils import initialize_pybullet
from pyastrobee.config.iss_safe_boxes import FULL_SAFE_SET
from pyastrobee.utils.boxes import visualize_3D_box


def main():
    client = initialize_pybullet()
    robot = Astrobee()
    bag = ConstraintCargoBag("top_handle", 10)
    bag.attach_to(robot)
    for box in FULL_SAFE_SET.values():
        visualize_3D_box(box)
    while True:
        client.stepSimulation()
        robot_base_aabb = client.getAABB(robot.id, -1)
        bag_aabb = client.getAABB(bag.id, -1)
        robot_is_safe = any(
            np.all(robot_base_aabb[0] > box.lower)
            and np.all(robot_base_aabb[1] < box.upper)
            for box in FULL_SAFE_SET.values()
        )
        bag_is_safe = any(
            np.all(bag_aabb[0] > box.lower) and np.all(bag_aabb[1] < box.upper)
            for box in FULL_SAFE_SET.values()
        )
        if not robot_is_safe:
            print("Robot collided")
        if not bag_is_safe:
            print("Bag collided")
        time.sleep(1 / 120)


if __name__ == "__main__":
    main()
