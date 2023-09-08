"""Test to see if we can attach a bunch of rigid bodies via constraints to mimic deformables"""

import time
from typing import Optional
from collections import defaultdict

import numpy as np
import numpy.typing as npt
import pybullet
from pybullet_utils.bullet_client import BulletClient

from pyastrobee.core.abstract_bag import CargoBag
from pyastrobee.core.astrobee import Astrobee
from pyastrobee.utils.bullet_utils import create_box
from pyastrobee.utils.transformations import make_transform_mat, transform_point
from pyastrobee.utils.rotations import quat_to_rmat


class CompositeCargoBag(CargoBag):
    def __init__(
        self,
        bag_name: str,
        mass: float,
        pos: npt.ArrayLike,
        orn: npt.ArrayLike,
        divisions: tuple[int, int, int],
        client: Optional[BulletClient] = None,
    ):
        self.divisions = self._check_divisions(bag_name, divisions)
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
        self._form_structure(pos, orn)
        return self.handle_block_ids[0]

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

    def _check_divisions(
        self, name: str, divisions: npt.ArrayLike
    ) -> tuple[int, int, int]:
        assert len(divisions) == 3
        assert all(d > 0 for d in divisions)
        assert all(isinstance(d, int) for d in divisions)
        dl, dw, dh = divisions
        if name in {"top_handle", "top_bottom_handle"}:
            requires_odd = ["length", "width"]
            is_invalid = dl % 2 != 1 or dw % 2 != 1
        elif name in {"front_handle", "front_back_handle"}:
            requires_odd = ["length", "height"]
            is_invalid = dl % 2 != 1 or dh % 2 != 1
        elif name in {"right_handle", "right_left_handle"}:
            requires_odd = ["width", "height"]
            is_invalid = dw % 2 != 1 or dh % 2 != 1
        else:
            raise ValueError("Bag name not recognized")
        if is_invalid:
            raise ValueError(
                f"Invalid divisions:\nBag: {name} requires an odd number of divisions in "
                + f"{requires_odd[0]} and {requires_odd[1]}.\nGot: {divisions}"
            )
        return dl, dw, dh

    def _form_structure(
        self,
        pos: npt.ArrayLike,
        orn: npt.ArrayLike,
    ):
        # Transformation matrix to the *central* point
        rmat = quat_to_rmat(orn)
        center_tmat = make_transform_mat(rmat, pos)
        # Construct things based on the bottom left front corner (min x y z)
        corner_point = transform_point(
            center_tmat, np.array([-self.LENGTH / 2, -self.WIDTH / 2, -self.HEIGHT / 2])
        )
        corner_tmat = make_transform_mat(rmat, corner_point)

        # Number of blocks in each dimension
        nx, ny, nz = self.divisions
        num_blocks = nx * ny * nz
        # Dimension of the blocks along each axis
        l = self.LENGTH / nx
        w = self.WIDTH / ny
        h = self.HEIGHT / nz
        m = self.mass / num_blocks
        # Create the blocks
        ids = []
        self.handle_block_ids = []
        ijk_to_id = {}
        if self.name == "top_handle":
            handle_ijks = [(nx // 2, ny // 2, nz - 1)]
        elif self.name == "front_handle":
            handle_ijks = [(nx // 2, 0, nz // 2)]
        elif self.name == "right_handle":
            handle_ijks = [(nx - 1, ny // 2, nz // 2)]
        elif self.name == "top_bottom_handle":
            handle_ijks = [(nx // 2, ny // 2, nz - 1), (nx // 2, ny // 2, 0)]
        elif self.name == "front_back_handle":
            handle_ijks = [(nx // 2, 0, nz // 2), (nx // 2, ny - 1, nz // 2)]
        elif self.name == "right_left_handle":
            handle_ijks = [(nx - 1, ny // 2, nz // 2), (0, ny // 2, nz // 2)]

        block_to_center = {}
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    is_handle_block = (i, j, k) in handle_ijks
                    rgba = (1, 0, 0, 1) if is_handle_block else (1, 1, 1, 1)
                    local_pos = np.array(
                        [(2 * i + 1) * l / 2, (2 * j + 1) * w / 2, (2 * k + 1) * h / 2]
                    )
                    block_pos = transform_point(corner_tmat, local_pos)
                    block_id = create_box(
                        block_pos,
                        orn,
                        # 0 if is_handle_block else m,  # DEBUGGING
                        m,
                        (l, w, h),
                        True,
                        rgba,
                    )
                    ijk_to_id[(i, j, k)] = block_id
                    ids.append(block_id)
                    block_to_center[block_id] = (
                        np.array([self.LENGTH / 2, self.WIDTH / 2, self.HEIGHT / 2])
                        - local_pos
                    )
                    if is_handle_block:
                        self.handle_block_ids.append(block_id)
        # Form constraints between adjacent blocks
        # Actually what if we just create constraints between the blocks and the middle?
        # Also consider disabling collisions between all of the blocks with themselves
        # Trying just constraining to the handle block right now
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    is_handle_block = (i, j, k) in handle_ijks
                    # Constrain to center point using handle block as a reference
                    # TODO FIXME improve this for the two-handle case
                    if not is_handle_block:
                        self.client.createConstraint(
                            self.handle_block_ids[0],
                            -1,
                            ijk_to_id[(i, j, k)],
                            -1,
                            self.client.JOINT_FIXED,
                            (0, 0, 1),
                            block_to_center[self.handle_block_ids[0]],
                            block_to_center[ijk_to_id[(i, j, k)]],
                            (0, 0, 0, 1),
                            (0, 0, 0, 1),
                        )
        # Disable internal collisions between adjacent blocks
        # TODO see if we can use setCollisionFilterGroupMask
        neighbors = defaultdict(list)
        # Define neighbors by all linear and diagonal nearest blocks
        # The most neighbors a block can have is 26
        # Think of this like a rubiks cube but with no central block
        # TODO there is definitely a better way to do this
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    possible_neighbors = [
                        (i - 1, j, k),  # Left
                        (i + 1, j, k),  # Right
                        (i, j - 1, k),  # Front
                        (i, j + 1, k),  # Back
                        (i, j, k - 1),  # Bottom
                        (i, j, k + 1),  # Top
                        (i, j - 1, k - 1),
                        (i, j - 1, k + 1),
                        (i, j + 1, k - 1),
                        (i, j + 1, k + 1),
                        (i - 1, j, k - 1),
                        (i - 1, j, k + 1),
                        (i + 1, j, k - 1),
                        (i + 1, j, k + 1),
                        (i - 1, j - 1, k),
                        (i - 1, j + 1, k),
                        (i + 1, j - 1, k),
                        (i + 1, j + 1, k),
                        (i - 1, j - 1, k - 1),  # Left front bottom
                        (i - 1, j - 1, k + 1),  # Left front top
                        (i - 1, j + 1, k - 1),  # Left back bottom
                        (i - 1, j + 1, k + 1),  # Left back top
                        (i + 1, j - 1, k - 1),  # Right front bottom
                        (i + 1, j - 1, k + 1),  # Right front top
                        (i + 1, j + 1, k - 1),  # Right back bottom
                        (i + 1, j + 1, k + 1),  # Right back top
                    ]
                    for neighbor in possible_neighbors:
                        ni, nj, nk = neighbor
                        if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                            neighbors[(i, j, k)].append((ni, nj, nk))
        pairs = set()
        for ijk, neighbor_ijks in neighbors.items():
            for nijk in neighbor_ijks:
                if (nijk, ijk) not in pairs:
                    pairs.add((ijk, nijk))
        for pair in pairs:
            pair_ids = ijk_to_id[pair[0]], ijk_to_id[pair[1]]
            pybullet.setCollisionFilterPair(pair_ids[0], pair_ids[1], -1, -1, 0)


def _main():
    from pyastrobee.utils.bullet_utils import load_floor

    name = "top_handle"
    pos = (0, 0, 1)
    orn = (0, 0, 0, 1)
    mass = 5
    divisions = (3, 3, 3)
    pybullet.connect(pybullet.GUI)
    pybullet.setGravity(0, 0, -9.81)
    load_floor()
    bag = CompositeCargoBag(name, mass, pos, orn, divisions)
    while True:
        pybullet.stepSimulation()
        time.sleep(1 / 120)


if __name__ == "__main__":
    _main()
