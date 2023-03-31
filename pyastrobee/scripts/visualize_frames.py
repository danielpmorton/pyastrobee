"""Quick test script to view some transformation matrices and quaternions"""

import numpy as np
import pybullet

from pyastrobee.vision.debug_visualizer import (
    visualize_frame,
    visualize_quaternion,
    remove_debug_objects,
)
from pyastrobee.utils.bullet_utils import run_sim
from pyastrobee.utils.transformations import make_transform_mat
from pyastrobee.utils.rotations import Rx, Ry, Rz, rmat_to_quat
from pyastrobee.utils.python_utils import set_small_vals_to_zero


pybullet.connect(pybullet.GUI)
pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, False)

tmat1 = make_transform_mat(np.eye(3), [0, 0, 0])
tmat2 = make_transform_mat(Rx(np.pi / 2), [0, 0, 0])
tmat3 = make_transform_mat(Ry(np.pi / 2), [0, 0, 0])
tmat4 = make_transform_mat(Rz(np.pi / 2), [0, 0, 0])
tmats = [tmat1, tmat2, tmat3, tmat4]
print("Visualizing transformation matrices (frames)")
for tmat in tmats:
    ids = visualize_frame(tmat)
    print(f"Now viewing:\n{set_small_vals_to_zero(tmat)}")
    input("Press Enter to continue to the next frame")
    remove_debug_objects(ids)

q1 = np.array([0, 0, 0, 1])
q2 = rmat_to_quat(Rx(np.pi / 2))
q3 = rmat_to_quat(Ry(np.pi / 2))
q4 = rmat_to_quat(Rz(np.pi / 2))
quats = [q1, q2, q3, q4]
print("\nVisualizing quaternions")
for quat in quats:
    ids = visualize_quaternion(quat)
    print(f"Now viewing: {quat}")
    input("Press Enter to continue to the next quaternion")
    remove_debug_objects(ids)

print("Done. Looping the sim...")
run_sim()
