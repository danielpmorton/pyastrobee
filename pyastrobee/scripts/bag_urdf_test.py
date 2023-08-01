"""Test script to load the rigid cargo bag into simulation so that we can interact with it, see
how the joints behave, and tune parameters in the URDF as needed."""


import time
import pybullet
import pybullet_data


def main():
    pybullet.connect(pybullet.GUI)
    pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())

    bag_id = pybullet.loadURDF(
        "pyastrobee/assets/urdf/rigid_bag.urdf",
        basePosition=[0, 0, 1],
        useFixedBase=False,
    )
    pybullet.loadURDF("plane.urdf")
    # Set gravity so it's a little easier to see how the joints behave
    pybullet.setGravity(0, 0, -9.8)

    # This will unlock the joints so that they can freely move
    # A small force allows for some damping/friction
    pybullet.setJointMotorControlArray(
        bag_id, [0, 1, 2], pybullet.VELOCITY_CONTROL, forces=[0.1, 0.1, 0.1]
    )
    # This will cause the handle to spring back into its natural position
    pybullet.setJointMotorControlArray(
        bag_id, [0, 1, 2], pybullet.POSITION_CONTROL, [0, 0, 0], forces=[0.1, 0.1, 0.1]
    )
    # ^^ TODO: We really only need one of these commands, so decide which behavior is desired
    while True:
        pybullet.stepSimulation()
        time.sleep(1 / 120)


if __name__ == "__main__":
    main()
