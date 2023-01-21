"""Simple debugging script to test out softbody parameters"""

import time
import pybullet
import pybullet_data


def main():
    cli = pybullet.connect(pybullet.GUI)
    pybullet.resetSimulation(pybullet.RESET_USE_DEFORMABLE_WORLD)
    pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
    pybullet.setTimeStep(1 / 350)
    pybullet.loadSoftBody(
        mass=1,
        fileName="duck.obj",
        scale=1,
        basePosition=[0, 0, 0],
        baseOrientation=pybullet.getQuaternionFromEuler([3.14159 / 2, 0, 0]),
        springElasticStiffness=100,
        springDampingStiffness=0.2,
        springBendingStiffness=10,
        frictionCoeff=0.1,
        # collisionMargin=0.003,  # how far apart do two objects begin interacting
        useSelfCollision=False,  # True in dedo, but seemed like this caused the mesh to collapse
        springDampingAllDirections=1,
        useFaceContact=True,
        useNeoHookean=0,
        useMassSpring=True,
        useBendingSprings=True,
        # repulsionStiffness=10000000,
    )
    while True:
        pybullet.stepSimulation()
        time.sleep(1 / 240)


if __name__ == "__main__":
    main()
