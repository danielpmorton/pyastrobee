"""Simple script for loading a tetrahedral mesh version of the cargo bag

This is also useful for confirming if a new VTK file has been constructed properly (change the filename with the VTK
to test, and see if it imports without errors)
"""

import pybullet
import time
from pyastrobee.utils.bullet_utils import load_deformable_object, load_floor

pybullet.connect(pybullet.GUI)
pybullet.resetSimulation(pybullet.RESET_USE_DEFORMABLE_WORLD)
load_floor()
load_deformable_object(
    "pyastrobee/assets/meshes/bags/front_handle.vtk",
    pos=[0, 0, 1],
    bending_stiffness=10,
)
while True:
    pybullet.stepSimulation()
    time.sleep(1 / 240)
