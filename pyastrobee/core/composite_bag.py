"""Test to see if we can attach a bunch of rigid bodies via constraints to mimic deformables

WORK IN PROGRESS
"""

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
from pyastrobee.core.constraint_bag import form_constraint_grasp


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
        if (
            len(divisions) != 3
            or any(d <= 0 for d in divisions)
            or any(d % 2 != 1 for d in divisions)
            or any(not isinstance(d, int) for d in divisions)
        ):
            raise ValueError(
                f"Invalid divisions: These must be three positive odd integers.\nGot: {divisions}"
            )
        self.divisions = divisions
        super().__init__(bag_name, mass, pos, orn, client)

        self._handle_constraints = {}

    @property
    def corner_positions(self) -> list[np.ndarray]:
        # TODO: currently, this assumes that there is no deformation going on
        # Should we do something smarter by looking at the corners of each of the corner boxes?
        return super().corner_positions

    def unload(self) -> None:
        self.detach()
        for id in self.block_ids:
            self.client.removeBody(id)
        self.block_ids = None
        self.handle_block_ids = None
        self.center_block_id = None
        self.corner_block_ids = None
        self.ijk_to_id = None
        self.id = None

    def _attach(self, robot: Astrobee, handle_index: int) -> None:
        # We'll use the same handle modeling as with the constraint bag
        # BUT we'll attach to the handle block rather than the center point of the bag,
        # so we need to update the grasp transformation
        handle_id = self.handle_block_ids[handle_index]
        handle_ijk = self.id_to_ijk[handle_id]
        original_grasp_transform = self.grasp_transforms[handle_index]
        orig_rmat = original_grasp_transform[:3, :3]
        pos = self._center_aligned_block_structure()[handle_ijk]
        adjusted_grasp_transform = make_transform_mat(orig_rmat, pos)
        constraints = form_constraint_grasp(
            robot, handle_id, self.mass, adjusted_grasp_transform, client=self.client
        )
        self._handle_constraints.update({robot.id: constraints})
        self._attached.append(robot.id)

    def detach(self) -> None:
        # return super().detach()
        for robot_id, cids in self._handle_constraints.items():
            for cid in cids:
                self.client.removeConstraint(cid)
        self._attached = []
        self._handle_constraints = {}

    def detach_robot(self, robot_id: int) -> None:
        # return super().detach_robot(robot_id)
        if robot_id not in self._handle_constraints:
            raise ValueError("Cannot detach robot: ID unknown")
        for cid in self._handle_constraints[robot_id]:
            self.client.removeConstraint(cid)
        self._attached.remove(robot_id)
        self._handle_constraints.pop(robot_id)

    def reset_dynamics(
        self,
        pos: npt.ArrayLike,
        orn: npt.ArrayLike,
        lin_vel: npt.ArrayLike,
        ang_vel: npt.ArrayLike,
    ) -> None:
        block_positions = self.get_init_block_positions(pos, orn)
        for block_ijk, block_id in self.ijk_to_id.items():
            self.client.resetBasePositionAndOrientation(
                block_id, block_positions[block_ijk], orn
            )
            r = block_positions[block_ijk] - pos
            self.client.resetBaseVelocity(
                block_id,
                lin_vel + np.cross(ang_vel, r),
                ang_vel,
            )
        pass

    def get_init_block_positions(
        self, pos: npt.ArrayLike, orn: npt.ArrayLike
    ) -> dict[tuple[int, int, int], np.ndarray]:
        # Deformation free
        rmat = quat_to_rmat(orn)
        center_tmat = make_transform_mat(rmat, pos)

        positions = self._center_aligned_block_structure()
        for block_ijk in positions:
            positions[block_ijk] = transform_point(center_tmat, positions[block_ijk])
        return positions

    def _corner_aligned_block_structure(self):
        nx, ny, nz = self.divisions
        l = self.LENGTH / nx
        w = self.WIDTH / ny
        h = self.HEIGHT / nz
        positions = {}
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    positions[(i, j, k)] = np.array(
                        [(2 * i + 1) * l / 2, (2 * j + 1) * w / 2, (2 * k + 1) * h / 2]
                    )
        return positions

    def _center_aligned_block_structure(self):
        positions = self._corner_aligned_block_structure()
        corner_to_center = np.array([self.LENGTH / 2, self.WIDTH / 2, self.HEIGHT / 2])
        for block_ijk in positions:
            positions[block_ijk] -= corner_to_center
        return positions

    def _load(
        self,
        pos: npt.ArrayLike,
        orn: npt.ArrayLike,
    ) -> int:
        # Number of blocks in each dimension
        nx, ny, nz = self.divisions
        num_blocks = nx * ny * nz
        # Dimension of the blocks along each axis
        l = self.LENGTH / nx
        w = self.WIDTH / ny
        h = self.HEIGHT / nz
        # Mass of each individual block
        m = self.mass / num_blocks
        # Create the blocks
        self.block_ids = []
        self.handle_block_ids = []
        self.corner_block_ids = []
        self.center_block_id = None
        self.ijk_to_id = {}
        self.id_to_ijk = {}
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

        center_block_ijk = (nx // 2, ny // 2, nz // 2)
        block_to_center = {}
        block_positions = self.get_init_block_positions(pos, orn)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    is_handle_block = (i, j, k) in handle_ijks
                    is_corner_block = (
                        (i in {0, nx - 1}) and (j in {0, ny - 1}) and (k in {0, nz - 1})
                    )
                    is_center_block = (i, j, k) == center_block_ijk
                    if is_handle_block:
                        rgba = (1, 0, 0, 1)
                    elif is_corner_block:
                        rgba = (0, 1, 0, 1)
                    elif is_center_block:
                        rgba = (0, 0, 1, 1)
                    else:
                        rgba = (1, 1, 1, 1)
                    local_pos = np.array(
                        [(2 * i + 1) * l / 2, (2 * j + 1) * w / 2, (2 * k + 1) * h / 2]
                    )
                    block_id = create_box(
                        block_positions[(i, j, k)],
                        orn,
                        m,
                        (l, w, h),
                        True,
                        rgba,
                    )
                    self.ijk_to_id[(i, j, k)] = block_id
                    self.id_to_ijk[block_id] = (i, j, k)
                    self.block_ids.append(block_id)
                    block_to_center[block_id] = (
                        np.array([self.LENGTH / 2, self.WIDTH / 2, self.HEIGHT / 2])
                        - local_pos
                    )
                    if is_handle_block:
                        self.handle_block_ids.append(block_id)
                    if is_corner_block:
                        self.corner_block_ids.append(block_id)
                    if is_center_block:
                        self.center_block_id = block_id
        # Form constraints between the blocks
        # We constrain each block to the central block, with the handle blocks having a larger force
        cids = []
        center_block_id = self.ijk_to_id[center_block_ijk]
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    is_handle_block = (i, j, k) in handle_ijks
                    is_center_block = (i, j, k) == center_block_ijk
                    if not is_center_block:
                        cid = self.client.createConstraint(
                            center_block_id,
                            -1,
                            self.ijk_to_id[(i, j, k)],
                            -1,
                            self.client.JOINT_FIXED,
                            (0, 0, 1),
                            (0, 0, 0),  # Center block origin
                            block_to_center[self.ijk_to_id[(i, j, k)]],
                            (0, 0, 0, 1),
                            (0, 0, 0, 1),
                        )
                        cids.append(cid)
                        # TODO TUNE THESE FORCES
                        constraint_force = self.mass * (20 if is_handle_block else 2)
                        self.client.changeConstraint(cid, maxForce=constraint_force)
        # Disable internal collisions between adjacent blocks
        # Define neighbors via a kind of voxel grid (like a Rubiks cube without the central block)
        neighbors = defaultdict(list)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # Find the 26 neighboring voxel coordinates
                    for delta_i in [-1, 0, 1]:
                        for delta_j in [-1, 0, 1]:
                            for delta_k in [-1, 0, 1]:
                                if delta_i == 0 and delta_j == 0 and delta_k == 0:
                                    pass  # Skip the center point
                                else:
                                    # Add the neighbor if it is within the bounds of the box
                                    ni = i + delta_i
                                    nj = j + delta_j
                                    nk = k + delta_k
                                    if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                                        neighbors[(i, j, k)].append((ni, nj, nk))
        # Get all unique pairs between neighboring blocks
        pairs = set()
        for ijk, neighbor_ijks in neighbors.items():
            for nijk in neighbor_ijks:
                # Ensure no duplicate pairs in the reverse order
                if (nijk, ijk) not in pairs:
                    pairs.add((ijk, nijk))
        for pair in pairs:
            # Disable collision
            pybullet.setCollisionFilterPair(
                self.ijk_to_id[pair[0]], self.ijk_to_id[pair[1]], -1, -1, 0
            )

        # Correct the handle block ID order
        # We build the blocks in order of increasing xyz position, but the way we've ordered the handles in the
        # two-handle bags are top->bottom, front->back, right->left which don't necessarily adhere to this order
        if self.num_handles == 2:
            # Reverse the order for the bags where the first handle is not at the minimum of the relevant dimension
            # Note: the front/back handle bag adheres to the correct order since the front handle is at min y
            if self.name in {"top_bottom_handle", "right_left_handle"}:
                self.handle_block_ids = self.handle_block_ids[::-1]

        # We need an ID to assign to the object
        # In general, the center block makes the most sense because we can query this for dynamics info of the bag
        return self.center_block_id


def _main():
    # pylint: disable=import-outside-toplevel
    from pyastrobee.utils.bullet_utils import load_floor

    name = "front_handle"
    pos = (0, 0, 1)
    orn = (0, 0, 0, 1)
    mass = 5
    divisions = (3, 3, 3)

    pybullet.connect(pybullet.GUI)
    # pybullet.setGravity(0, 0, -9.81)
    load_floor()
    robot = Astrobee()
    # robot2 = Astrobee()
    bag = CompositeCargoBag(name, mass, pos, orn, divisions)
    # bag.attach_to([robot, robot2])
    bag.attach_to(robot)
    while True:
        pybullet.stepSimulation()
        time.sleep(1 / 120)


if __name__ == "__main__":
    _main()
