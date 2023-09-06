"""Modeling a cargo bag as a single rigid box with a handle constructed from multiple point-to-point constraints

Documentation for inherited methods can be found in the base class
"""

import time

import numpy as np
import numpy.typing as npt
import pybullet
from pybullet_utils.bullet_client import BulletClient

from pyastrobee.core.astrobee import Astrobee
from pyastrobee.core.abstract_bag import CargoBag
from pyastrobee.utils.bullet_utils import create_box
from pyastrobee.utils.transformations import transform_point
from pyastrobee.utils.python_utils import print_green

# Different geometries of the constraint "constellation"
# First point is the central (primary) constraint
# Notes: tetrahedron seems to give a bit better behavior than diamond -- diamond will sometimes "snap" into place,
# which doesn't really make sense for a handle. Plus, tetrahedron has fewer constraints, which is better for sim
UNIT_CONSTRAINT_STRUCTURES = {
    "diamond": np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ]
    ),
    "tetrahedron": np.array(
        [
            [0, 0, 0],
            [-1 / 3, np.sqrt(8 / 9), 0],
            [-1 / 3, -np.sqrt(2 / 9), np.sqrt(2 / 3)],
            [-1 / 3, -np.sqrt(2 / 9), -np.sqrt(2 / 3)],
            [1, 0, 0],
        ]
    ),
}


class ConstraintCargoBag(CargoBag):
    """Class for loading and managing properties associated with the constraint-based cargo bags

    Args:
        bag_name (str): Type of cargo bag to load. Single handle: "front_handle", "right_handle", "top_handle".
            Dual handle: "front_back_handle", "right_left_handle", "top_bottom_handle"
        mass (float): Mass of the cargo bag
        pos (npt.ArrayLike, optional): Initial XYZ position to load the bag. Defaults to (0, 0, 0)
        orn (npt.ArrayLike, optional): Initial XYZW quaternion to load the bag. Defaults to (0, 0, 0, 1)
        client (BulletClient, optional): If connecting to multiple physics servers, include the client
            (the class instance, not just the ID) here. Defaults to None (use default connected client)
    """

    def __init__(
        self,
        bag_name: str,
        mass: float,
        pos: npt.ArrayLike = (0, 0, 0),
        orn: npt.ArrayLike = (0, 0, 0, 1),
        client: BulletClient | None = None,
    ):
        # Heuristic for constraint forces that scales with the mass of the bag
        # These are intended to roughly match the forces from the deformable for the small displacements that we might
        # see when controlling the robot
        self.primary_constraint_force = 5 * mass
        self.secondary_constraint_force = 0.05 * mass
        super().__init__(bag_name, mass, pos, orn, client)
        self._constraints = {}
        # Set up the geometric structure of the constraint-based handle
        constraint_scaling = 0.05
        self.constraint_structure = (
            UNIT_CONSTRAINT_STRUCTURES["tetrahedron"] * constraint_scaling
        )
        self.num_contraints = len(self.constraint_structure)
        print_green("Bag is ready")

    # Implement abstract methods

    def _load(self, pos: npt.ArrayLike, orn: npt.ArrayLike) -> int:
        return create_box(
            pos,
            orn,
            self.mass,
            (self.LENGTH, self.WIDTH, self.HEIGHT),
            True,
            (1, 1, 1, 1),
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

        # Get the constraint attachment positions in each of the local frames
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
                max_force = self.primary_constraint_force
            else:
                max_force = self.secondary_constraint_force
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
        """Constraints between the robot(s) and the handle(s). Key: robot ID; Value: list of constraint IDs"""
        return self._constraints

    def get_local_constraint_pos(self, handle_index: int) -> np.ndarray:
        """Determine the position of the handle's constraints in the bag frame

        Args:
            handle_index (int): Index of the handle of interest

        Returns:
            np.ndarray: Constraint positions, shape (n_constraints, 3)
        """
        return np.array(
            [
                transform_point(self.grasp_transforms[handle_index], pt)
                for pt in self.constraint_structure
            ]
        )

    def get_world_constraint_pos(self, handle_index: int) -> np.ndarray:
        """Determine the position of the handle's constraints in the world frame

        Args:
            handle_index (int): Index of the handle of interest

        Returns:
            np.ndarray: Constraint positions, shape (n_constraints, 3)
        """
        tmat = self.tmat
        local_constraint_pos = self.get_local_constraint_pos(handle_index)
        return np.array([transform_point(tmat, pos) for pos in local_constraint_pos])

    @property
    def constraint_forces(self) -> dict[int, float]:
        """Forces on each constraint. Key: constraint ID; Value: Force, shape (3,)"""
        # NOTE: this dictionary will maintain insertion order so we can also associate
        # these constraint forces in the same order as the original structure
        forces = {}
        for robot_id, cids in self.constraints.items():
            for cid in cids:
                forces[cid] = pybullet.getConstraintState(cid)
        return forces


def _main():
    pybullet.connect(pybullet.GUI)
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_WIREFRAME, 1)
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
            if i == 0:
                fmax = bag.primary_constraint_force
            else:
                fmax = bag.secondary_constraint_force
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
    _main()
