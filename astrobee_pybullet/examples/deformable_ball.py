"""https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/deformable_ball.py"""

from time import sleep

import pybullet
import pybullet_data

physicsClient = pybullet.connect(pybullet.GUI)
pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())

pybullet.resetSimulation(pybullet.RESET_USE_DEFORMABLE_WORLD)

pybullet.setGravity(0, 0, -10)

planeOrn = [0, 0, 0, 1]  # p.getQuaternionFromEuler([0.3,0,0])
planeId = pybullet.loadURDF("plane.urdf", [0, 0, -2], planeOrn)

boxId = pybullet.loadURDF("cube.urdf", [0, 3, 2], useMaximalCoordinates=True)

ballId = pybullet.loadSoftBody(
    "ball.obj",
    simFileName="ball.vtk",
    basePosition=[0, 0, -1],
    scale=0.5,
    mass=4,
    useNeoHookean=1,
    NeoHookeanMu=400,
    NeoHookeanLambda=600,
    NeoHookeanDamping=0.001,
    useSelfCollision=1,
    frictionCoeff=0.5,
    collisionMargin=0.001,
)
pybullet.setTimeStep(0.001)
pybullet.setPhysicsEngineParameter(sparseSdfVoxelSize=0.25)


# logId = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "perf.json")

while pybullet.isConnected():

    pybullet.stepSimulation()
    # there can be some artifacts in the visualizer window,
    # due to reading of deformable vertices in the renderer,
    # while the simulators updates the same vertices
    # it can be avoided using
    # p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
    # but then things go slower
    pybullet.setGravity(0, 0, -10)
    # sleep(1./240.)

# p.resetSimulation()
# p.stopStateLogging(logId)
