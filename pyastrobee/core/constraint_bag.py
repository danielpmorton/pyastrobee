"""Modeling a cargo bag as a single rigid box with a handle constructed from multiple point-to-point constraints

Documentation for most methods can be found in the base class
"""

import time
from typing import Optional
import numpy as np
import numpy.typing as npt
import pybullet
from pybullet_utils.bullet_client import BulletClient
from pyastrobee.core.astrobee import Astrobee
from pyastrobee.core.cargo_bag_class import CargoBagABC
from pyastrobee.utils.rotations import (
    euler_xyz_to_quat,
    quat_to_rmat,
    euler_xyz_to_rmat,
)
from pyastrobee.utils.bullet_utils import create_box
from pyastrobee.utils.transformations import make_transform_mat, transform_point
from pyastrobee.utils.poses import pos_quat_to_tmat, tmat_to_pos_quat


class ConstraintCargoBag(CargoBagABC):
    def __init__(
        self,
        bag_name: str,
        mass: float,
        pos: npt.ArrayLike = (0, 0, 0),
        orn: npt.ArrayLike = (0, 0, 0, 1),
        client: BulletClient | None = None,
    ):
        super().__init__(bag_name, mass, pos, orn, client)

        self._constraints = {}
        constraint_offset_dist = 0.05
        self.constraint_structure = (
            np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [-1, 0, 0],
                    [0, 1, 0],
                    [0, -1, 0],
                    [0, 0, 1],
                    [0, 0, -1],
                ]
            )
            * constraint_offset_dist
        )
        self.num_contraints = len(self.constraint_structure)

    # Implement abstract methods

    def _load(self, pos, orn):
        # TODO ALLOW FOR MORE VERSIONS OF THE BAG
        # Tune the naming of the length/width/height things too.. make properties based on bag name?
        return create_box(
            pos,
            orn,
            self.mass,
            (self.LENGTH, self.WIDTH, self.HEIGHT),
            True,
            (1, 1, 1, 1),
        )

    @property
    def dynamics_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pos, orn = self.client.getBasePositionAndOrientation(self.id)
        lin_vel, ang_vel = self.client.getBaseVelocity(self.id)
        return (
            np.array(pos),
            np.array(orn),
            np.array(lin_vel),
            np.array(ang_vel),
        )

    def _attach(self, robot: Astrobee, handle_index: int) -> None:
        # Disable collisions with the end of the gripper
        # Otherwise if we use the standard handle location, it will always be in collision
        # TODO decide if proximal joints should also be disabled
        for link_id in [
            robot.Links.GRIPPER_LEFT_DISTAL.value,
            robot.Links.GRIPPER_RIGHT_DISTAL.value,
        ]:
            pybullet.setCollisionFilterPair(robot.id, self.id, link_id, -1, 0)

        robot_local_constraint_pos = np.array(
            [
                transform_point(Astrobee.TRANSFORMS.GRIPPER_TO_ARM_DISTAL, pt)
                for pt in self.constraint_structure
            ]
        )

        bag_local_constraint_pos = self.get_local_constraint_pos(handle_index)

        constraint_ids = []
        for i in range(len(self.constraint_structure)):
            if i == 0:
                max_force = 20  # TUNE THESE!!!!!!
            else:
                max_force = 1
            cid = pybullet.createConstraint(
                robot.id,
                robot.Links.ARM_DISTAL.value,
                self.id,
                -1,
                pybullet.JOINT_POINT2POINT,
                (0, 0, 1),
                robot_local_constraint_pos[i],
                bag_local_constraint_pos[i],
            )
            pybullet.changeConstraint(cid, maxForce=max_force)
            constraint_ids.append(cid)
        self._constraints.update({robot.id: constraint_ids})
        self._attached.append(robot.id)
        pass  # TODO

    def detach(self) -> None:
        for robot_id, cids in self.constraints.items():
            for cid in cids:
                self.client.removeConstraint(cid)
        self._attached = []
        self._constraints = {}

    def detach_robot(self, robot_id: int) -> None:
        if robot_id not in self.constraints:
            raise ValueError("Cannot detach robot: ID unknown")
        for cid in self.constraints[robot_id]:
            self.client.removeConstraint(cid)
        self._attached.remove(robot_id)
        self._constraints.pop(robot_id)

    # Functions and properties specific to the constraint-based bag

    @property
    def constraints(self) -> dict[int, list[int]]:
        return self._constraints

    def get_local_constraint_pos(self, handle_index: int) -> np.ndarray:
        return np.array(
            [
                transform_point(self.grasp_transforms[handle_index], pt)
                for pt in self.constraint_structure
            ]
        )

    def get_world_constraint_pos(self, handle_index: int) -> np.ndarray:
        tmat = self.tmat
        local_constraint_pos = self.get_local_constraint_pos(handle_index)
        return np.array([transform_point(tmat, pos) for pos in local_constraint_pos])

    @property
    def constraint_forces(self) -> dict[int, float]:
        # TODO add check that these constraints actually exist
        # TODO be clearer about the insertion order and how this corresponds to the constraints...
        forces = {}
        for robot_id, cids in self.constraints.items():
            for cid in cids:
                forces[cid] = pybullet.getConstraintState(cid)
        return forces


def main():
    pybullet.connect(pybullet.GUI)
    robot = Astrobee()
    # robot2 = Astrobee()
    bag = ConstraintCargoBag("top_handle", 1)
    # bag.attach_to([robot, robot2])
    bag.attach_to(robot)
    points_uid = None
    while True:
        forces = np.array(list(bag.constraint_forces.values()))
        force_mags = np.linalg.norm(forces, axis=1)
        rgbs = []
        for i in range(bag.num_contraints):
            # TODO use the numbers from the actual class
            if i == 0:
                fmax = 100
            else:
                fmax = 1
            r = min(1, force_mags[i] / fmax)
            rgbs.append((r, 1 - r, 0))
        world_constraint_pos = bag.get_world_constraint_pos(0)
        if points_uid is None:
            points_uid = pybullet.addUserDebugPoints(world_constraint_pos, rgbs, 10, 0)
        else:
            points_uid = pybullet.addUserDebugPoints(
                world_constraint_pos, rgbs, 10, 0, replaceItemUniqueId=points_uid
            )
        pybullet.stepSimulation()
        time.sleep(1 / 120)


if __name__ == "__main__":
    main()
