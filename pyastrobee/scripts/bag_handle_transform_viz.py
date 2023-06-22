"""Script to visualize the locations of the grasp frames on the bag handles

Note: the Astrobee gripper frame is:
z along handle
x pointed out of the bag
y perpendicular to handle

This visualization should match that description if the transforms are calibrated properly
"""

import pybullet
import numpy as np
from pyastrobee.utils.bullet_utils import load_rigid_object
from pyastrobee.utils.debug_visualizer import visualize_frame, remove_debug_objects
import pyastrobee.config.bag_properties as bag_props

front_file = "pyastrobee/assets/meshes/bags/front_handle.obj"
side_file = "pyastrobee/assets/meshes/bags/right_handle.obj"
top_file = "pyastrobee/assets/meshes/bags/top_handle.obj"

front_back_file = "pyastrobee/assets/meshes/bags/front_back_handle.obj"
side_side_file = "pyastrobee/assets/meshes/bags/right_left_handle.obj"
top_bottom_file = "pyastrobee/assets/meshes/bags/top_bottom_handle.obj"


def main():
    pybullet.connect(pybullet.GUI)
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, False)
    np.random.seed(0)

    obj_id = load_rigid_object(front_back_file)
    frame_1_ids = visualize_frame(bag_props.FRONT_HANDLE_TRANSFORM)
    frame_2_ids = visualize_frame(bag_props.BACK_HANDLE_TRANSFORM)
    input("Press Enter to continue")
    remove_debug_objects([*frame_1_ids, *frame_2_ids])
    pybullet.removeBody(obj_id)

    obj_id = load_rigid_object(side_side_file)
    frame_1_ids = visualize_frame(bag_props.RIGHT_HANDLE_TRANSFORM)
    frame_2_ids = visualize_frame(bag_props.LEFT_HANDLE_TRANSFORM)
    input("Press Enter to continue")
    remove_debug_objects([*frame_1_ids, *frame_2_ids])
    pybullet.removeBody(obj_id)

    obj_id = load_rigid_object(top_bottom_file)
    frame_1_ids = visualize_frame(bag_props.TOP_HANDLE_TRANSFORM)
    frame_2_ids = visualize_frame(bag_props.BOTTOM_HANDLE_TRANSFORM)
    input("Press Enter to continue")
    remove_debug_objects([*frame_1_ids, *frame_2_ids])
    pybullet.removeBody(obj_id)

    pybullet.disconnect()


if __name__ == "__main__":
    main()
