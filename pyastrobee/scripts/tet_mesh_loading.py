"""Simple script for loading a tetrahedral mesh version of the cargo bag"""

import pybullet
import time
from pyastrobee.utils.bullet_utils import load_deformable_object, load_floor

pybullet.connect(pybullet.GUI)
pybullet.resetSimulation(pybullet.RESET_USE_DEFORMABLE_WORLD)
load_floor()
load_deformable_object(
    "pyastrobee/assets/meshes/tet_bag.vtk", pos=[0, 0, 1], bending_stiffness=10
)
while True:
    pybullet.stepSimulation()
    time.sleep(1 / 120)
