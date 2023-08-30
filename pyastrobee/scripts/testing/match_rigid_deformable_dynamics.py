"""Test to see if we can match the dynamics between a deformable and rigid cargo bag"""

# TODO:
# - See how well the reset works when things are connected to Astrobees

import time
from typing import Optional

import pybullet
from pybullet_utils.bullet_client import BulletClient
import numpy as np
import numpy.typing as npt

from pyastrobee.core.cargo_bag import CargoBag
from pyastrobee.core.rigid_bag import RigidCargoBag
from pyastrobee.utils.bullet_utils import initialize_pybullet


def get_state(deformable_bag: CargoBag):
    return deformable_bag.dynamics_state


def reset_state(
    rigid_bag: RigidCargoBag,
    deformable_dynamics: tuple[np.ndarray, ...],
    pos_offset: Optional[npt.ArrayLike] = None,
    client: Optional[BulletClient] = None,
) -> None:
    """Resets the state of the rigid cargo bag to the dynamics recorded from the deformable bag

    Args:
        rigid_bag (RigidCargoBag): The rigid cargo bag
        deformable_dynamics (tuple[np.ndarray, ...]): Position, orientation, velocity, and angular velocity of the
            deformable bag
        pos_offset (Optional[npt.ArrayLike], optional): Positional offset for the reset (to keep the two bags apart from
            eachother). Defaults to None.
        client (BulletClient, optional): If connecting to multiple physics servers, include the client
            (the class instance, not just the ID) here. Defaults to None (use default connected client)
    """
    client: pybullet = pybullet if client is None else client
    assert len(deformable_dynamics) == 4
    pos, orn, vel, omega = deformable_dynamics
    if pos_offset is not None:
        pos = np.add(pos, pos_offset)
    client.resetBasePositionAndOrientation(rigid_bag.id, pos, orn)
    client.resetBaseVelocity(rigid_bag.id, vel, omega)


def main():
    bag_name = "top_handle"
    deformable_p0 = np.zeros(3)
    pos_offset = np.array([0, 0, 1])
    rigid_p0 = deformable_p0 + pos_offset
    client = initialize_pybullet()
    dt = client.getPhysicsEngineParameters()["fixedTimeStep"]
    deformable_bag = CargoBag(bag_name, deformable_p0, client=client)
    rigid_bag = RigidCargoBag(bag_name, rigid_p0, client=client)

    time_per_reset = 1  # seconds
    steps_per_reset = round(time_per_reset / dt)

    print("Apply a disturbance force to the deformable cargo bag")
    while True:
        for _ in range(steps_per_reset):
            pybullet.stepSimulation()
            time.sleep(1 / 120)
        state = get_state(deformable_bag)
        reset_state(rigid_bag, state, pos_offset)


if __name__ == "__main__":
    main()
