"""Hello World example from the Pybullet quickstart guide

https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA
"""

import time

import pybullet
import pybullet_data

physicsClient = pybullet.connect(pybullet.GUI)  # or p.DIRECT for non-graphical version
pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
pybullet.setGravity(0, 0, -10)
planeId = pybullet.loadURDF("plane.urdf")
startPos = [0, 0, 1]
startOrientation = pybullet.getQuaternionFromEuler([0, 0, 0])
boxId = pybullet.loadURDF("r2d2.urdf", startPos, startOrientation)
# set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
for i in range(10000):
    pybullet.stepSimulation()
    time.sleep(1.0 / 240.0)
cubePos, cubeOrn = pybullet.getBasePositionAndOrientation(boxId)
print(cubePos, cubeOrn)
pybullet.disconnect()
