"""Test script to see if adjusting the linear/angular damping will change 'air resistance'

Results:
- It seems like this is the case if we adjust this parameter for the base link
  (if we really turn this up, it's like the robot is moving in molasses, and we can also
  turn it down to 0)
- The real value is probably very low since there is some air in the ISS (0 would be space)
- Adjusting the damping value for deformables does not seem to work right now, but the damping
  seems to be low, just from interacting with the deformable with the mouse
"""

import time

import pybullet
import pybullet_data


def test_changing_damping(object_id):
    base_link = -1

    print("Interact with the sim: Using default linear/angular damping of 0.04")
    start_time = time.time()
    while time.time() - start_time < 10:
        pybullet.stepSimulation()
        time.sleep(1 / 120)

    input("Press Enter to change the linear/angular damping")
    linear_damping = 0.0
    angular_damping = 0.0
    pybullet.changeDynamics(
        object_id,
        base_link,
        linearDamping=linear_damping,
        angularDamping=angular_damping,
    )

    print(
        f"Now using linear damping of {linear_damping} and angular damping of {angular_damping}"
    )
    start_time = time.time()
    while time.time() - start_time < 10:
        pybullet.stepSimulation()
        time.sleep(1 / 120)
    input("Press Enter to continue/exit")


def main():

    pybullet.connect(pybullet.GUI)
    pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
    robot = pybullet.loadURDF("r2d2.urdf")

    test_changing_damping(robot)

    input("Press Enter to try out deformables")
    pybullet.removeBody(robot)
    pybullet.resetSimulation(pybullet.RESET_USE_DEFORMABLE_WORLD)
    bag = pybullet.loadSoftBody(
        "pyastrobee/assets/meshes/bags/top_handle.vtk",
        (0, 0, 0),
        (0, 0, 0, 1),
        1,
        1.0,
        springElasticStiffness=50.0,
        springDampingStiffness=0.1,
        springBendingStiffness=50.0,
        frictionCoeff=0.1,
        useSelfCollision=False,
        springDampingAllDirections=1,
        useFaceContact=True,
        useNeoHookean=0,
        useMassSpring=True,
        useBendingSprings=True,
    )

    test_changing_damping(bag)


if __name__ == "__main__":
    main()
