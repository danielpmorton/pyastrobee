"""Demo: Load an Astrobee holding a cargo bag in the ISS and move around

TODO
- Calibrate the GRIPPER_TO_ARM_DISTAL transformation in pyastrobee/config/astrobee_transforms.py
  (This transform will dictate our offset between the last frame on the arm, and where we're holding the bag handle)
- Find the orientation(s) to load the bag and astrobee together, initially connected
- Attach an anchor to the bag handle
"""

import numpy as np
import pybullet

from pyastrobee.control.astrobee import Astrobee
from pyastrobee.control.controller import PoseController
from pyastrobee.utils.iss_utils import load_iss
from pyastrobee.utils.bullet_utils import (
    initialize_pybullet,
    load_deformable_object,
    run_sim,
)


def load_bag(robot_id):
    # Load deformable bag and attach the middle of each side of the handle to
    # the middle of each of the astrobee fingers.
    bag_id = load_deformable_object(
        "pyastrobee/resources/meshes/bag_thick_handle_sparse.obj",
        pos=[-0.05, -0.01, -0.52],
        orn=[-np.pi / 2, 0, 0],
    )
    bag_texture_id = pybullet.loadTexture(
        "pyastrobee/resources/meshes/textile_pixabay_red.jpg"
    )
    kwargs = {}
    if hasattr(pybullet, "VISUAL_SHAPE_DOUBLE_SIDED"):
        kwargs["flags"] = pybullet.VISUAL_SHAPE_DOUBLE_SIDED
    pybullet.changeVisualShape(
        bag_id, -1, rgbaColor=[1, 1, 1, 1], textureUniqueId=bag_texture_id, **kwargs
    )
    pybullet.createSoftBodyAnchor(bag_id, 787, robot_id, 4)
    pybullet.createSoftBodyAnchor(bag_id, 818, robot_id, 5)


def demo_with_iss():
    """A simple demo of loading the astrobee in the ISS and moving it around in various ways

    TODO: the WPs might need to be refined, there seems to be a lot of weird rotating going on
    Or it could be a quaternion issue? Quaternion ambiguity?

    This could just in general be cleaned up and refined, but it works ok for now
    """
    # Hardcoded waypoints and positions found from keyboard-controlling the Astrobee
    # fmt: off
    wp0 = [0, 0, 0, 0, 0, 0, 1]
    wp1 = [0.44631294, -1.33893871, 0.44631287, 0.08824572, 0.06790329, -0.78759863, 0.60604474]
    wp2 = [0.05603137, -2.81145659, 0.10060672, -0.06176491, -0.0185934, -0.69867597, 0.71252457]
    wp3 = [-0.31709299, 0.31352898, 0.53193288, -0.03191529, 0.0062923, -0.83105266, 0.55524166]
    # fmt: on
    new_arm_joints = [-1.34758381, 0.99330411]
    initialize_pybullet()
    # pybullet.connect(pybullet.GUI)  # a simple version without deformables
    # Bring the camera close to the action (another just random hardcoded position I found)
    pybullet.resetDebugVisualizerCamera(1.6, 206, -26.2, [0, 0, 0])
    load_iss()
    robot = Astrobee()
    load_bag(robot.id)
    controller = PoseController(robot)
    # Go about a small set of actions to show what we can do so far
    while True:
        controller.go_to_pose(wp1)
        controller.go_to_pose(wp2)
        controller.go_to_pose(wp3)
        robot.set_arm_joints(new_arm_joints)
        robot.close_gripper()
        input("Press enter to repeat")
        robot.set_arm_joints([0, 0])
        robot.open_gripper()
        controller.go_to_pose(wp0)
    # Keep the sim spinning
    # run_sim()


def demo_with_bag():
    initialize_pybullet()
    cam_args = {
        "cameraDistance": 1.6,  # use 0.7 to look at anchor attachment closely
        "cameraPitch": 160,
        "cameraYaw": 80,
        "cameraTargetPosition": np.array([0, 0, 0]),
    }
    pybullet.resetDebugVisualizerCamera(**cam_args)
    # It seems at the moment, something weird is happening with the initial motion
    # The URDF loads it at the specified pose, but then it goes back to the origin
    # I think this has something to do with how the constraint is being set
    # (need to debug this a little bit, )
    # robot = Astrobee(pose=[1, 1, 1, 0, 0, 0, 1])
    # Instead, we'll just keep the astrobee started off at the origin, and load the bag in a different spot
    robot = Astrobee()
    load_bag(robot.id)
    controller = PoseController(robot)
    # As just an arbitrary example, have the astrobee move to a new position to show the motion
    # Right now this has no correlation with the bag other than it doesn't collide with the bag
    target_pose = np.array([-0.5, -0.4, 0.8, 0, 0, 0, 1])
    controller.go_to_pose(target_pose)

    # Loop the simulation until closed
    run_sim()


if __name__ == "__main__":
    demo_with_iss()
