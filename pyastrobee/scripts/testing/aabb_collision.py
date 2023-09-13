"""Testing out using axis aligned bounding boxes for conservative 'collision' detection

'collision' here just referring to departure from the safe set
"""

import time

from pyastrobee.core.astrobee import Astrobee
from pyastrobee.core.constraint_bag import ConstraintCargoBag
from pyastrobee.utils.bullet_utils import initialize_pybullet
from pyastrobee.config.iss_safe_boxes import FULL_SAFE_SET
from pyastrobee.utils.boxes import visualize_3D_box, check_box_containment


def main():
    client = initialize_pybullet()
    robot = Astrobee()
    bag = ConstraintCargoBag("top_handle", 10)
    bag.attach_to(robot)
    for box in FULL_SAFE_SET.values():
        visualize_3D_box(box)
    while True:
        client.stepSimulation()
        robot_is_safe = check_box_containment(
            robot.bounding_box, FULL_SAFE_SET.values()
        )
        bag_is_safe = check_box_containment(bag.bounding_box, FULL_SAFE_SET.values())
        if not robot_is_safe:
            print("Robot collided")
        if not bag_is_safe:
            print("Bag collided")
        time.sleep(1 / 120)


if __name__ == "__main__":
    main()
