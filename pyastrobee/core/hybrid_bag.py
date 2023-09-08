"""Testing to see if we can attach a deformable handle to a rigid bag

WORK IN PROGRESS!!!!!!
"""

import time

import numpy as np
import numpy.typing as npt

from pyastrobee.utils.bullet_utils import (
    initialize_pybullet,
    load_deformable_object,
    create_box,
    create_anchor,
)
from pyastrobee.utils.mesh_utils import get_closest_mesh_vertex, get_mesh_data
from pyastrobee.config.bag_properties import BOX_LENGTH, BOX_WIDTH, BOX_HEIGHT


handle_corner_pos_local = [
    # Outer corners
    (0.075, 0.025, 0),  # ID 32
    (0.075, -0.025, 0),  # ID 1
    (-0.075, 0.025, 0),  # ID 21
    (-0.075, -0.025, 0),  # ID 19
    # Inner corners
    (0.06138287, 0.025, 0),  # ID 31
    (0.06138287, -0.025, 0),  # ID 0
    (-0.06138287, 0.025, 0),  # ID 22
    (-0.06138287, -0.025, 0),  # ID 20
]


client = initialize_pybullet()
box = create_box(
    (0, 0, -BOX_HEIGHT / 2),
    (0, 0, 0, 1),
    5,
    (BOX_LENGTH, BOX_WIDTH, BOX_HEIGHT),
    True,
    client=client,
)
handle = load_deformable_object(
    "pyastrobee/assets/meshes/handle_only.vtk", pos=(0, 0, 0), mass=0.2, client=client
)
n_verts, vert_positions = get_mesh_data(handle, client=client)

ids = []
for i, pos in enumerate(handle_corner_pos_local):
    actual_pos, vert_id = get_closest_mesh_vertex(pos, vert_positions)
    print(f"Corner #{i}: ID = {vert_id}. Distance: {np.linalg.norm(pos - actual_pos)}")
    ids.append(vert_id)

anchor_ids = []
for id in ids:
    aid, _ = create_anchor(handle, id, box, -1)
    anchor_ids.append(aid)

while True:
    client.stepSimulation()
    time.sleep(1 / 120)


class HybridCargoBag(CargoBag):
    def __init__(
        self,
        bag_name: str,
        mass: float,
        pos: npt.ArrayLike,
        orn: npt.ArrayLike,
        client: Optional[BulletClient] = None,
    ):
        super().__init__(bag_name, mass, pos, orn, client)

    @property
    def pose(self) -> np.ndarray:
        # return super().pose
        pass

    @property
    def position(self) -> np.ndarray:
        # return super().position
        pass

    @property
    def orientation(self) -> np.ndarray:
        # return super().orientation
        pass

    @property
    def velocity(self) -> np.ndarray:
        # return super().velocity
        pass

    @property
    def angular_velocity(self) -> np.ndarray:
        # return super().angular_velocity
        pass

    @property
    def dynamics_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # return super().dynamics_state
        pass

    @property
    def corner_positions(self) -> list[np.ndarray]:
        # return super().corner_positions
        pass

    def _load(
        self,
        pos: npt.ArrayLike,
        orn: npt.ArrayLike,
    ) -> int:
        # return super()._load(pos, orn)
        pass

    def unload(self) -> None:
        # return super().unload()
        pass

    def _attach(self, robot: Astrobee, handle_index: int) -> None:
        # return super()._attach(robot, handle_index)
        pass

    def detach(self) -> None:
        # return super().detach()
        pass

    def detach_robot(self, robot_id: int) -> None:
        # return super().detach_robot(robot_id)
        pass

    def reset_dynamics(
        self,
        pos: npt.ArrayLike,
        orn: npt.ArrayLike,
        lin_vel: npt.ArrayLike,
        ang_vel: npt.ArrayLike,
    ) -> None:
        # return super().reset_dynamics(pos, orn, lin_vel, ang_vel)
        pass
