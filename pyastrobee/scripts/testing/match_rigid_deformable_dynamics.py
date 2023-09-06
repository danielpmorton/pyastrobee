"""Test to see if we can match the dynamics between a deformable and rigid cargo bag"""

# TODO:
# - We probably need to adjust the joint limits of the cargo bag or maybe the collision info on the handle because
#   it gets weird when we need to reset the position of the bag to something that causes collision or being out of
#   joint limits

import time
from typing import Optional

import pybullet
from pybullet_utils.bullet_client import BulletClient
import numpy as np
import numpy.typing as npt

from pyastrobee.core.astrobee import Astrobee
from pyastrobee.core.deformable_cargo_bag import DeformableCargoBag
from pyastrobee.core.rigid_bag import RigidCargoBag
from pyastrobee.utils.bullet_utils import initialize_pybullet


def get_state(deformable_bag: DeformableCargoBag, robot: Optional[Astrobee] = None):
    robot_state = robot.full_state if robot is not None else None
    return deformable_bag.dynamics_state, robot_state


def reset_state(
    rigid_bag: RigidCargoBag,
    bag_state: tuple[np.ndarray, ...],
    robot: Optional[Astrobee] = None,
    robot_state: Optional[tuple[np.ndarray, ...]] = None,
    pos_offset: Optional[npt.ArrayLike] = None,
    client: Optional[BulletClient] = None,
) -> None:
    """Resets the state of the rigid cargo bag to the dynamics recorded from the deformable bag

    Args:
        rigid_bag (RigidCargoBag): The rigid cargo bag
        bag_state (tuple[np.ndarray, ...]): Position, orientation, velocity, and angular velocity of the
            deformable bag
        pos_offset (Optional[npt.ArrayLike], optional): Positional offset for the reset (to keep the two bags apart from
            eachother). Defaults to None.
        client (BulletClient, optional): If connecting to multiple physics servers, include the client
            (the class instance, not just the ID) here. Defaults to None (use default connected client)
    """
    client: pybullet = pybullet if client is None else client
    # Handle the bag
    assert len(bag_state) == 4
    bag_pos, bag_orn, bag_vel, bag_omega = bag_state
    if pos_offset is not None:
        bag_pos = np.add(bag_pos, pos_offset)
    client.resetBasePositionAndOrientation(rigid_bag.id, bag_pos, bag_orn)
    client.resetBaseVelocity(rigid_bag.id, bag_vel, bag_omega)
    # Handle the robot (if needed)
    if robot is not None:
        assert robot_state is not None
        assert len(robot_state) == 6
        robot_pos, robot_orn, robot_vel, robot_omega, robot_q, robot_qdot = robot_state
        if pos_offset is not None:
            robot_pos = np.add(robot_pos, pos_offset)
        robot.reset_full_state(
            robot_pos, robot_orn, robot_vel, robot_omega, robot_q, robot_qdot
        )


def bag_only_test():
    bag_name = "top_handle"
    deformable_p0 = np.zeros(3)
    pos_offset = np.array([0, 0, 1])
    rigid_p0 = deformable_p0 + pos_offset
    client = initialize_pybullet()
    dt = client.getPhysicsEngineParameters()["fixedTimeStep"]
    deformable_bag = DeformableCargoBag(bag_name, deformable_p0, client=client)
    rigid_bag = RigidCargoBag(bag_name, rigid_p0, client=client)

    time_per_reset = 1  # seconds
    steps_per_reset = round(time_per_reset / dt)

    print("Apply a disturbance force to the deformable cargo bag")
    while True:
        for _ in range(steps_per_reset):
            pybullet.stepSimulation()
            time.sleep(1 / 120)
        bag_state, _ = get_state(deformable_bag)
        reset_state(rigid_bag, bag_state, pos_offset=pos_offset)


def astrobee_and_bag_test():
    bag_name = "top_handle"
    deformable_p0 = np.zeros(3)
    pos_offset = np.array([1, 0, 0])
    rigid_p0 = deformable_p0 + pos_offset
    client = initialize_pybullet()
    dt = client.getPhysicsEngineParameters()["fixedTimeStep"]
    deformable_bag = DeformableCargoBag(bag_name, deformable_p0, client=client)
    robot_1 = Astrobee()
    deformable_bag.attach_to(robot_1, "robot")
    rigid_bag = RigidCargoBag(bag_name, rigid_p0, client=client)
    robot_2 = Astrobee()
    rigid_bag.attach_to(robot_2, "robot")

    time_per_reset = 1  # seconds
    steps_per_reset = round(time_per_reset / dt)

    print("Apply a disturbance force to the deformable cargo bag")
    while True:
        for _ in range(steps_per_reset):
            pybullet.stepSimulation()
            time.sleep(1 / 120)
        bag_state, robot_state = get_state(deformable_bag, robot_1)
        reset_state(rigid_bag, bag_state, robot_2, robot_state, pos_offset)


if __name__ == "__main__":
    # bag_only_test()
    astrobee_and_bag_test()
