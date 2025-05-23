"""Modeling a cargo bag as a single rigid box with a handle constructed from multiple point-to-point constraints

Documentation for inherited methods can be found in the base class
"""

import time
from typing import Optional

import numpy as np
import numpy.typing as npt
import pybullet
from pybullet_utils.bullet_client import BulletClient

from pyastrobee.core.astrobee import Astrobee
from pyastrobee.core.abstract_bag import CargoBag
from pyastrobee.utils.bullet_utils import create_box
from pyastrobee.utils.transformations import transform_point
from pyastrobee.utils.python_utils import print_green
from pyastrobee.utils.bullet_utils import initialize_pybullet

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
    "xy_cross": np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
        ]
    ),
    "xz_cross": np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [-1, 0, 0],
            [0, 0, 1],
            [0, 0, -1],
        ]
    ),
    "yz_cross": np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ]
    ),
    "x_inline": np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [-1, 0, 0],
        ]
    ),
    "y_inline": np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
        ]
    ),
    "z_inline": np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, -1],
        ]
    ),
}


class ConstraintCargoBag(CargoBag):
    """Class for loading and managing properties associated with the constraint-based cargo bags

    Args:
        bag_name (str): Type of cargo bag to load. Single handle: "front_handle", "right_handle", "top_handle".
            Dual handle: "front_back_handle", "right_left_handle", "top_bottom_handle"
        mass (float): Mass of the cargo bag, in kg
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
        # Set up the geometric structure of the constraint-based handle
        self.constraint_scaling = 0.05
        self.constraint_structure_type = "tetrahedron"
        self.constraint_structure = (
            UNIT_CONSTRAINT_STRUCTURES[self.constraint_structure_type]
            * self.constraint_scaling
        )
        # Define the forces applied by the constraints
        self.primary_constraint_force = 3
        self.secondary_constraint_force = 2
        self.max_constraint_forces = np.concatenate(
            [
                [self.primary_constraint_force],
                self.secondary_constraint_force
                * np.ones(len(self.constraint_structure) - 1),
            ]
        )
        self._constraints = {}
        self.num_contraints = len(self.constraint_structure)
        super().__init__(bag_name, mass, pos, orn, client)
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
        # Disable collisions with the arm for stability when resetting the position w.r.t the deformable
        for link_id in [
            robot.Links.GRIPPER_LEFT_DISTAL.value,
            robot.Links.GRIPPER_RIGHT_DISTAL.value,
            robot.Links.GRIPPER_LEFT_PROXIMAL.value,
            robot.Links.GRIPPER_RIGHT_PROXIMAL.value,
            robot.Links.ARM_DISTAL.value,
            robot.Links.ARM_PROXIMAL.value,
        ]:
            self.client.setCollisionFilterPair(robot.id, self.id, link_id, -1, 0)

        constraints = form_constraint_grasp(
            robot,
            self.id,
            self.grasp_transforms[handle_index],
            self.constraint_structure_type,
            self.constraint_scaling,
            self.max_constraint_forces,
            client=self.client,
        )
        self._constraints.update({robot.id: constraints})
        self._attached.append(robot.id)

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
                forces[cid] = self.client.getConstraintState(cid)
        return forces


def form_constraint_grasp(
    robot: Astrobee,
    body_id: int,
    grasp_transform: np.ndarray,
    structure_type: str,
    structure_scaling: float,
    max_forces: list[float],
    client: Optional[BulletClient] = None,
) -> list[int]:
    """Connects the Astrobee's gripper to an object via point-to-point constraints to mimic a non-rigid grasp

    NOTE: Depending on the grasp transform used, it may be recommended to disable collisions between the Astrobee
    gripper and the object before forming these constraints

    Args:
        robot (Astrobee): Astrobee performing the grasp
        body_id (int): Pybullet ID of the object being grasped
        grasp_transform (np.ndarray): Transformation matrix defining the grasp pose w.r.t the base frame of the object
        structure_type (str, optional): Type of geometry to construct the series of constraints. Defaults to
            "tetrahedron". Other options include "diamond"
        structure_scaling (float, optional): Scale on the size of the constraint structure (if set to 1, the
            constraints will be spaced along a unit (1 meter) sphere). Defaults to 0.05.
        max_forces (list[float]): Maximum applied force for each constraint. Length must match with the number of
            constraints in the desired structure type
        client (BulletClient, optional): If connecting to multiple physics servers, include the client
            (the class instance, not just the ID) here. Defaults to None (use default connected client)

    Returns:
        list[int]: Pybullet IDs of the constraints
    """
    client: pybullet = pybullet if client is None else client
    constraint_structure = (
        UNIT_CONSTRAINT_STRUCTURES[structure_type] * structure_scaling
    )
    if len(max_forces) != len(constraint_structure):
        raise ValueError(
            f"Invalid number of forces: Must be of length {len(constraint_structure)} "
            + f"for structure type {structure_type}.\nGot: {max_forces}"
        )
    body_local_constraint_pos = np.array(
        [transform_point(grasp_transform, pt) for pt in constraint_structure]
    )
    robot_local_constraint_pos = np.array(
        [
            transform_point(Astrobee.TRANSFORMS.GRIPPER_TO_ARM_DISTAL, pt)
            for pt in constraint_structure
        ]
    )
    constraints = []
    for i in range(len(constraint_structure)):
        cid = client.createConstraint(
            robot.id,
            robot.Links.ARM_DISTAL.value,
            body_id,
            -1,
            client.JOINT_POINT2POINT,
            (0, 0, 1),
            robot_local_constraint_pos[i],
            body_local_constraint_pos[i],
        )
        client.changeConstraint(cid, maxForce=max_forces[i])
        constraints.append(cid)
    return constraints


def _main():
    client = initialize_pybullet(bg_color=(0.5, 0.5, 1))
    client.configureDebugVisualizer(client.COV_ENABLE_WIREFRAME, 1)
    robot = Astrobee()
    # robot2 = Astrobee()
    bag = ConstraintCargoBag("top_handle", 10)
    # bag.attach_to([robot, robot2])
    bag.attach_to(robot)
    points_uid = None
    while True:
        forces = np.array(list(bag.constraint_forces.values()))
        force_mags = np.linalg.norm(forces, axis=1)
        rgbs = []
        for i in range(bag.num_contraints):
            r = min(1, force_mags[i] / bag.max_constraint_forces[i])
            rgbs.append((r, 1 - r, 0))
        world_constraint_pos = bag.get_world_constraint_pos(0)
        if points_uid is None:
            points_uid = client.addUserDebugPoints(world_constraint_pos, rgbs, 10, 0)
        else:
            points_uid = client.addUserDebugPoints(
                world_constraint_pos, rgbs, 10, 0, replaceItemUniqueId=points_uid
            )
        client.stepSimulation()
        time.sleep(1 / 120)


if __name__ == "__main__":
    _main()
