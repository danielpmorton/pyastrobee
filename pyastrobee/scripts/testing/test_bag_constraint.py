"""Testing a new model of the cargo bag where we model the handle as multiple point-to-point constraints

WORK IN PROGRESS
"""

import time

import pybullet
import numpy as np

from pyastrobee.core.astrobee import Astrobee
from pyastrobee.utils.bullet_utils import create_box, create_sphere
from pyastrobee.utils.debug_visualizer import visualize_points
from pyastrobee.utils.rotations import (
    euler_xyz_to_quat,
    quat_to_rmat,
    euler_xyz_to_rmat,
)
from pyastrobee.config.bag_properties import BOX_LENGTH, BOX_WIDTH, BOX_HEIGHT
from pyastrobee.utils.transformations import make_transform_mat, transform_point
from pyastrobee.utils.poses import pos_quat_to_tmat


# TODO LESS HARDCODING, MORE INPUTS
# TODO allow for different bag types
# TODO figure out how to deal with constraint info when robot is not connected
# TODO add better visual info for the bag
# TODO tune forces
# TODO structure this in the same way as the CargoBag and RigidCargoBag classes
# TODO include client
class ConstraintCargoBag:
    def __init__(self):
        self.id = create_box(
            (0, 0, 0),
            (0, 0, 0, 1),
            1,
            (BOX_LENGTH, BOX_WIDTH, BOX_HEIGHT),
            True,
            (1, 1, 1, 1),
        )
        handle_offset = 0.1
        self.local_handle_location = np.array([0, 0, BOX_HEIGHT / 2 + handle_offset])
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

    @property
    def pose(self):
        return np.concatenate(pybullet.getBasePositionAndOrientation(self.id))

    @property
    def tmat(self):
        return pos_quat_to_tmat(self.pose)

    @property
    def local_constraint_pos(self):
        # TODO add check that these constraints actually exist
        return self.local_handle_location + self.constraint_structure

    @property
    def world_constraint_pos(self):
        # TODO add check that these constraints actually exist
        # TODO this can probably be more efficient
        tmat = self.tmat
        return np.array(
            [transform_point(tmat, pos) for pos in self.local_constraint_pos]
        )

    @property
    def constraint_forces(self):
        # TODO add check that these constraints actually exist
        return np.array(
            [pybullet.getConstraintState(cid) for cid in self.constraint_ids]
        )

    @property
    def handle_pose(self):
        # SUPER HACKY RIGHT NOW, TODO!!
        pose = self.pose
        tmat = pos_quat_to_tmat(pose)
        pos = transform_point(tmat, self.local_handle_location)
        # orn = pose[3:]  # BAD BAD BAD
        orn = euler_xyz_to_quat((0, -np.pi / 2, 0))  # BAD BAD BAD
        return np.concatenate([pos, orn])

    def attach_robot(self, robot: Astrobee):
        # A rotation is needed for the constraint positions due to the alignment of the Astrobee gripper frame
        # TODO this will vary depending on the type of the bag
        # TODO decide if moving robot or bag

        robot.reset_to_ee_pose(self.handle_pose)

        rmat = euler_xyz_to_rmat((0, np.pi / 2, 0))
        robot_constraint_structure = np.array(
            [rmat @ pos for pos in self.constraint_structure]
        )
        robot_local_constraint_pos = (
            Astrobee.TRANSFORMS.GRIPPER_TO_ARM_DISTAL[:3, 3]
            + robot_constraint_structure
        )
        self.constraint_ids = []
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
                self.local_constraint_pos[i],
            )
            pybullet.changeConstraint(cid, maxForce=max_force)
            self.constraint_ids.append(cid)

        # TODO add log of which robots are connected and to which constraints


def main():
    pybullet.connect(pybullet.GUI)
    robot = Astrobee()
    bag = ConstraintCargoBag()
    bag.attach_robot(robot)
    points_uid = None
    while True:
        forces = bag.constraint_forces
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
        if points_uid is None:
            points_uid = pybullet.addUserDebugPoints(
                bag.world_constraint_pos, rgbs, 10, 0
            )
        else:
            points_uid = pybullet.addUserDebugPoints(
                bag.world_constraint_pos, rgbs, 10, 0, replaceItemUniqueId=points_uid
            )
        pybullet.stepSimulation()
        time.sleep(1 / 120)


if __name__ == "__main__":
    main()
