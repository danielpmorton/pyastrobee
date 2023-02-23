"""Demo: Load an Astrobee holding a cargo bag in the ISS and move around

TODO
- Calibrate the GRIPPER_TO_ARM_DISTAL transformation in pyastrobee/config/astrobee_transforms.py
  (This transform will dictate our offset between the last frame on the arm, and where we're holding the bag handle)
- Find the orientation(s) to load the bag and astrobee together, initially connected
- Attach an anchor to the bag handle
- Load the ISS (I can do this, right now it's loading in the wrong orientation, so I've just got to rotate/position it
  in a good way, and then figure out the right place to load the astrobee)
"""
import numpy as np
import pybullet

from pyastrobee.control.astrobee import Astrobee
from pyastrobee.utils.astrobee_utils import load_iss
from pyastrobee.utils.bullet_utils import (
    initialize_pybullet,
    load_deformable_object,
    run_sim,
)


def main():
    initialize_pybullet()
    # It seems at the moment, something weird is happening with the initial motion
    # The URDF loads it at the specified pose, but then it goes back to the origin
    # I think this has something to do with how the constraint is being set
    # (need to debug this a little bit, )
    # robot = Astrobee(pose=[1, 1, 1, 0, 0, 0, 1])
    # Instead, we'll just keep the astrobee started off at the origin, and load the bag in a different spot
    robot = Astrobee()
    # I haven't done much work with the meshes in about a month but I'm pretty sure this was the best one I made
    # (I need to clean up my mesh files)
    # Some of the physical parameters will likely need some tuning
    bag_id = load_deformable_object(
        "pyastrobee/resources/meshes/bag_thick_handle_sparse.obj", pos=[1, -1, 1]
    )
    # As just an arbitrary example, have the astrobee move to a new position to show the motion
    # Right now this has no correlation with the bag other than it doesn't collide with the bag
    target_pose = np.array([-1, -1, 1, 0, 0, 0, 1])
    robot.go_to_pose(target_pose)

    # Loop the simulation until closed
    run_sim()


if __name__ == "__main__":
    main()
