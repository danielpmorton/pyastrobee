"""Rigid version of the cargo bag, using joints in the URDF to mimic what we would see from a deformable

Documentation for inherited methods can be found in the base class
"""

# TODO make mass variable
# TODO tune the position control force on the handle

import time
from typing import Optional

import pybullet
from pybullet_utils.bullet_client import BulletClient
import numpy as np
import numpy.typing as npt

from pyastrobee.core.astrobee import Astrobee
from pyastrobee.core.abstract_bag import CargoBag
from pyastrobee.utils.bullet_utils import initialize_pybullet
from pyastrobee.utils.python_utils import print_green
from pyastrobee.utils.transformations import make_transform_mat
from pyastrobee.utils.rotations import quat_to_rmat
from pyastrobee.utils.dynamics import box_inertia


class RigidCargoBag(CargoBag):
    """Class for loading and managing properties associated with the rigid URDF-based cargo bags

    Args:
        bag_name (str): Type of cargo bag to load. Single handle: "front_handle", "right_handle", "top_handle".
            Dual handle: "front_back_handle", "right_left_handle", "top_bottom_handle"
        pos (npt.ArrayLike, optional): Initial XYZ position to load the bag. Defaults to (0, 0, 0)
        orn (npt.ArrayLike, optional): Initial XYZW quaternion to load the bag. Defaults to (0, 0, 0, 1)
        client (BulletClient, optional): If connecting to multiple physics servers, include the client
            (the class instance, not just the ID) here. Defaults to None (use default connected client)
    """

    LINKS_PER_HANDLE = 4  # 3 dummy links for roll/pitch/yaw, plus the handle itself
    _urdfs = [
        CargoBag.URDF_DIR + name + "_rigid_bag.urdf" for name in CargoBag.BAG_NAMES
    ]
    URDFS = dict(zip(CargoBag.BAG_NAMES, _urdfs))

    def __init__(
        self,
        bag_name: str,
        pos: npt.ArrayLike = (0, 0, 0),
        orn: npt.ArrayLike = (0, 0, 0, 1),
        client: Optional[BulletClient] = None,
    ):
        mass = 5  # From URDF. TODO make this an input
        super().__init__(bag_name, mass, pos, orn, client)
        # This inertia is slightly approximate because some mass is in the handle and dummy links
        self.inertia = box_inertia(self.mass, self.LENGTH, self.WIDTH, self.HEIGHT)
        # Initializations
        self._constraints = {}
        # Add position control to the handle(s) so its springs back into its natural position
        # to provide some resistance to motion like we would see in a deformable
        # The dummy joints are the joints associated with the motion of the handle
        if self.num_handles == 1:
            self.num_joints = 4  # 3 for rpy, 1 for handle
            self.num_links = 5  # Base, 3 dummies for rpy, handle
            dummy_joint_ids = [0, 1, 2]
        elif self.num_handles == 2:
            self.num_joints = 8
            self.num_links = 9
            dummy_joint_ids = [0, 1, 2, 4, 5, 6]
        self.client.setJointMotorControlArray(
            self.id,
            dummy_joint_ids,
            pybullet.POSITION_CONTROL,
            [0] * len(dummy_joint_ids),
            forces=[0.1] * len(dummy_joint_ids),  # TODO tune force
        )
        print_green("Bag is ready")

    # TODO make this structure consistent with the constraint bag
    @property
    def constraints(self) -> list[int]:
        """Active IDs of constraints on the bag"""
        return list(self._constraints.values())

    def _load(self, pos: npt.ArrayLike, orn: npt.ArrayLike) -> int:
        return self.client.loadURDF(self.URDFS[self.name], pos, orn)

    def _attach(self, robot: Astrobee, handle_index: int) -> None:
        handle_link_index = (handle_index + 1) * self.LINKS_PER_HANDLE - 1
        cid = self.client.createConstraint(
            robot.id,
            robot.Links.ARM_DISTAL.value,
            self.id,
            handle_link_index,
            pybullet.JOINT_FIXED,
            [0, 0, 1],
            robot.TRANSFORMS.GRIPPER_TO_ARM_DISTAL[:3, 3],
            [0, 0, 0],
        )
        self._constraints.update({robot.id: cid})
        self._attached.append(robot.id)

    def detach(self) -> None:
        for cid in self.constraints:
            self.client.removeConstraint(cid)
        self._constraints = {}
        self._attached = []

    def detach_robot(self, robot_id: int) -> None:
        if robot_id not in self.attached:
            raise ValueError("Cannot detach robot: ID unknown")
        self.client.removeConstraint(self._constraints[robot_id])
        self._constraints.pop(robot_id)
        self._attached.remove(robot_id)

    def get_handle_transform(self, handle_index: int = 0) -> np.ndarray:
        """Calculates the transformation matrix (w.r.t the world) for a specified handle

        Args:
            handle_index (int): Index of the handle on the bag

        Returns:
            np.ndarray: Transformation matrix (handle to world). Shape = (4,4)
        """
        if (handle_index + 1) > self.num_handles:
            raise ValueError(
                f"Invalid handle index: {handle_index}. Bag only has {self.num_handles} handles"
            )
        handle_link_index = (handle_index + 1) * self.LINKS_PER_HANDLE - 1
        link_state = self.client.getLinkState(
            self.id, handle_link_index, computeForwardKinematics=True
        )
        pos, quat = link_state[:2]
        return make_transform_mat(quat_to_rmat(quat), pos)


def _single_handle_test(bag_name: str):
    # Very simple example of loading the bag and attaching a robot
    client = initialize_pybullet(bg_color=(0.8, 0.8, 1))
    robot = Astrobee()
    bag = RigidCargoBag(bag_name)
    bag.attach_to(robot)
    while True:
        client.stepSimulation()
        time.sleep(1 / 120)


def _two_handle_test(bag_name: str):
    # Load the bag and attach to two robots
    client = initialize_pybullet(bg_color=(0.8, 0.8, 1))
    robot_1 = Astrobee()
    robot_2 = Astrobee()
    bag = RigidCargoBag(bag_name)
    bag.attach_to([robot_1, robot_2])
    while True:
        client.stepSimulation()
        time.sleep(1 / 120)


if __name__ == "__main__":
    # _single_handle_test("top_handle")
    _two_handle_test("top_bottom_handle")
