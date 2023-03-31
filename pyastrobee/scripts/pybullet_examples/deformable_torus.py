"""https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/deformable_torus.py
"""

import pybullet
from time import sleep
import pybullet_data

physicsClient = pybullet.connect(pybullet.GUI)

pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())

pybullet.resetSimulation(pybullet.RESET_USE_DEFORMABLE_WORLD)
pybullet.resetDebugVisualizerCamera(3, -420, -30, [0.3, 0.9, -2])
pybullet.setGravity(0, 0, -10)

tex = pybullet.loadTexture("uvmap.png")
planeId = pybullet.loadURDF("plane.urdf", [0, 0, -2])

boxId = pybullet.loadURDF("cube.urdf", [0, 3, 2], useMaximalCoordinates=True)

bunnyId = pybullet.loadSoftBody(
    "torus/torus_textured.obj",
    # simFileName="torus.vtk",
    mass=3,
    useNeoHookean=1,
    NeoHookeanMu=180,
    NeoHookeanLambda=600,
    NeoHookeanDamping=0.01,
    collisionMargin=0.006,
    useSelfCollision=1,
    frictionCoeff=0.5,
    repulsionStiffness=800,
)
pybullet.changeVisualShape(
    bunnyId, -1, rgbaColor=[1, 1, 1, 1], textureUniqueId=tex, flags=0
)


# bunny2 = p.loadURDF("torus_deform.urdf", [0, 1, 0.5], flags=p.URDF_USE_SELF_COLLISION)

# p.changeVisualShape(bunny2, -1, rgbaColor=[1, 1, 1, 1], textureUniqueId=tex, flags=0)
pybullet.setPhysicsEngineParameter(sparseSdfVoxelSize=0.25)
pybullet.setRealTimeSimulation(0)

while pybullet.isConnected():
    pybullet.stepSimulation()
    pybullet.getCameraImage(320, 200)
    pybullet.setGravity(0, 0, -10)
