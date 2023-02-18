from typing import Optional
import pybullet
import time
import math
import os
import pybullet_data

from pyastrobee.control.astrobee import Astrobee


pybullet.connect(pybullet.GUI)
pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
# pybullet.setAdditionalSearchPath(os.getcwd())


# pybullet.loadURDF("plane.urdf")
robotID = pybullet.loadURDF(
    "pyastrobee/urdf/astrobee2.urdf", 0, 0, 1
)  # Check on these extra numbers
# pybullet.setGravity(0, 0, -10)
pybullet.setRealTimeSimulation(1)
cid = pybullet.createConstraint(
    robotID, -1, -1, -1, pybullet.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 1]
)
print(cid)
print(pybullet.getConstraintUniqueId(0))
a = -math.pi
while 1:
    a = a + 0.01
    if a > math.pi:
        a = -math.pi
    time.sleep(0.01)
    #   pybullet.setGravity(0, 0, -10)
    pivot = [a, 0, 1]
    orn = pybullet.getQuaternionFromEuler([a, 0, 0])
    pybullet.changeConstraint(cid, pivot, jointChildFrameOrientation=orn, maxForce=50)

pybullet.removeConstraint(cid)
