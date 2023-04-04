"""Quick script to view the frames of each link on the Astrobee"""

from pyastrobee.control.astrobee import Astrobee
from pyastrobee.utils.debug_visualizer import (
    visualize_link_frame,
    remove_debug_objects,
)
from pyastrobee.utils.bullet_utils import initialize_pybullet, run_sim

initialize_pybullet(bg_color=[1, 1, 1])
robot = Astrobee()
line_lengths = [0.5, 0.5, 0.25, 0.25, 0.1, 0.1, 0.1, 0.1]
names = [
    "body",
    "top_aft",
    "arm_proximal",
    "arm_distal",
    "gripper_left_proximal",
    "gripper_left_distal",
    "gripper_right_proximal",
    "gripper_right_distal",
]
idxs = range(-1, 7)
num_links = 8

print("Now viewing all of the frames on the Astrobee")
line_ids = []
for i, length in zip(idxs, line_lengths):
    line_ids += visualize_link_frame(robot.id, i, length)

input("Press enter to step through the links individually")
remove_debug_objects(line_ids)

for i in range(num_links):
    print("Now viewing link: ", names[i])
    line_ids = visualize_link_frame(robot.id, idxs[i], line_lengths[i])
    input("Press Enter to continue to the next link")
    remove_debug_objects(line_ids)

print("Done. Looping sim...")
run_sim()
