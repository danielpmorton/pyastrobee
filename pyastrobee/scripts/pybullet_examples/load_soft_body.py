"""https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/load_soft_body.py
"""

import pybullet
from time import sleep
import pybullet_data


physicsClient = pybullet.connect(pybullet.GUI)

pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
pybullet.setGravity(0, 0, -10)
planeId = pybullet.loadURDF("plane.urdf", [0, 0, -2])
boxId = pybullet.loadURDF("cube.urdf", [0, 3, 2], useMaximalCoordinates=True)
bunnyId = pybullet.loadSoftBody("bunny.obj")  # .obj")#.vtk")

# meshData = p.getMeshData(bunnyId)
# print("meshData=",meshData)
# p.loadURDF("cube_small.urdf", [1, 0, 1])
useRealTimeSimulation = 1

if useRealTimeSimulation:
    pybullet.setRealTimeSimulation(1)

print(pybullet.getDynamicsInfo(planeId, -1))
# print(p.getDynamicsInfo(bunnyId, 0))
print(pybullet.getDynamicsInfo(boxId, -1))
pybullet.changeDynamics(boxId, -1, mass=10)
while pybullet.isConnected():
    pybullet.setGravity(0, 0, -10)
    if useRealTimeSimulation:

        sleep(0.01)  # Time in seconds.
        # p.getCameraImage(320,200,renderer=p.ER_BULLET_HARDWARE_OPENGL )
    else:
        pybullet.stepSimulation()
