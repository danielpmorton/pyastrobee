"""Test script to load the rigid cargo bag into simulation so that we can interact with it, see
how the joints behave, and tune parameters in the URDF as needed.

Note: if we just want to "unlock" the joints without using the PD control to maintain the handle position,
we can use something like this (below) to reset the original velocity controller:
pybullet.setJointMotorControlArray(
    bag_id,
    joint_ids,
    pybullet.VELOCITY_CONTROL,
    forces=joint_forces, # A small "friction" force, if desired
)
"""

import time
import pybullet
import pybullet_data


def main(name: str):
    if name in {"top_handle", "front_handle", "right_handle"}:
        # Single handle
        joint_ids = [0, 1, 2]
    elif name in {"top_bottom_handle", "front_back_handle", "right_left_handle"}:
        # Dual handle
        joint_ids = [0, 1, 2, 4, 5, 6]
    else:
        raise ValueError("Invalid bag name: ", name)

    pybullet.connect(pybullet.GUI)
    pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())

    bag_id = pybullet.loadURDF(
        f"pyastrobee/assets/urdf/bags/{name}_rigid_bag.urdf",
        basePosition=[0, 0, 1],
        useFixedBase=False,
    )
    pybullet.loadURDF("plane.urdf")
    # Set gravity so it's a little easier to see how the joints behave
    pybullet.setGravity(0, 0, -9.8)

    # This will cause the handle to spring back into its natural position
    pybullet.setJointMotorControlArray(
        bag_id,
        joint_ids,
        pybullet.POSITION_CONTROL,
        [0] * len(joint_ids),
        forces=[0.1] * len(joint_ids),
    )
    while True:
        pybullet.stepSimulation()
        time.sleep(1 / 120)


if __name__ == "__main__":
    main("top_handle")
