# """Class for handling loading the cargo bag and managing attributes such as physical properties

# TODO !! Fix the softbody velocity issue
# TODO ! the unloading mechanic is currently broken since it only removes the softbody and not the visual
# TODO If the reliability of the softbody position/orientation is not good, use the get_bag_frame() function I made
# TODO decide if the bag_props import can be handled better
# TODO decide if the constants should be moved to class attributes
# TODO decide if we should anchor to the gripper fingers or the arm distal link (currently distal)
# TODO decide if we should set up the class attributes to handle multiple bags loaded in sim
#     (Like the Astrobee class)
# """
import time
from typing import Union, Optional

import pybullet
from pybullet_utils.bullet_client import BulletClient
import numpy as np
import numpy.typing as npt

import pyastrobee.config.bag_properties as bag_props
from pyastrobee.core.astrobee import Astrobee
from pyastrobee.utils.bullet_utils import (
    load_deformable_object,
    create_sphere,
    initialize_pybullet,
)
from pyastrobee.utils.mesh_utils import get_mesh_data, get_closest_mesh_vertex
from pyastrobee.utils.poses import pos_quat_to_tmat, tmat_to_pos_quat
from pyastrobee.utils.python_utils import print_green, flatten
from pyastrobee.utils.quaternions import quats_to_angular_velocities
from pyastrobee.utils.transformations import invert_transform_mat, make_transform_mat
from pyastrobee.utils.rotations import quat_to_rmat, Ry, rmat_to_quat
from pyastrobee.utils.dynamics import box_inertia
from pyastrobee.utils.debug_visualizer import visualize_points

# Constants. TODO Move these to class attributes?
# MESH_DIR = "pyastrobee/assets/meshes/bags/"
SINGLE_HANDLE_BAGS = ["front_handle", "right_handle", "top_handle"]
DUAL_HANDLE_BAGS = ["front_back_handle", "right_left_handle", "top_bottom_handle"]
BAG_NAMES = SINGLE_HANDLE_BAGS + DUAL_HANDLE_BAGS
# _objs = [MESH_DIR + name + ".obj" for name in BAG_NAMES]
# _vtks = [MESH_DIR + name + ".vtk" for name in BAG_NAMES]
# OBJS = dict(zip(BAG_NAMES, _objs))
# VTKS = dict(zip(BAG_NAMES, _vtks))
HANDLE_TRANSFORMS = {
    "front": bag_props.FRONT_HANDLE_TRANSFORM,
    "back": bag_props.BACK_HANDLE_TRANSFORM,
    "left": bag_props.LEFT_HANDLE_TRANSFORM,
    "right": bag_props.RIGHT_HANDLE_TRANSFORM,
    "top": bag_props.TOP_HANDLE_TRANSFORM,
    "bottom": bag_props.BOTTOM_HANDLE_TRANSFORM,
}
# BAG_CORNERS = dict(
#     zip(
#         BAG_NAMES,
#         [
#             bag_props.FRONT_HANDLE_BAG_CORNERS,
#             bag_props.RIGHT_HANDLE_BAG_CORNERS,
#             bag_props.TOP_HANDLE_BAG_CORNERS,
#             bag_props.FRONT_BACK_HANDLE_BAG_CORNERS,
#             bag_props.RIGHT_LEFT_HANDLE_BAG_CORNERS,
#             bag_props.TOP_BOTTOM_HANDLE_BAG_CORNERS,
#         ],
#     )
# )


class RigidCargoBag:
    # """Class for loading and managing properties associated with the cargo bags

    # Args:
    #     bag_name (str): Type of cargo bag to load. Single handle: "front_handle", "right_handle", "top_handle".
    #         Dual handle: "front_back_handle", "right_left_handle", "top_bottom_handle"
    #     pos (npt.ArrayLike, optional): Initial XYZ position to load the bag. Defaults to (0, 0, 0)
    #     orn (npt.ArrayLike, optional): Initial XYZW quaternion to load the bag. Defaults to (0, 0, 0, 1)
    #     client (BulletClient, optional): If connecting to multiple physics servers, include the client
    #         (the class instance, not just the ID) here. Defaults to None (use default connected client)
    # """

    URDF = "pyastrobee/assets/urdf/rigid_bag.urdf"
    LINKS_PER_HANDLE = 4  # 3 dummy links for roll/pitch/yaw, plus the handle itself
    LENGTH = 0.50  # meters
    WIDTH = 0.25  # meters
    HEIGHT = 0.42  # meters

    def __init__(
        self,
        bag_name: str,
        pos: npt.ArrayLike = (0, 0, 0),
        orn: npt.ArrayLike = (0, 0, 0, 1),
        client: Optional[BulletClient] = None,
    ):
        self.client: pybullet = pybullet if client is None else client
        if not self.client.isConnected():
            raise ConnectionError("Need to connect to pybullet before loading a bag")
        if bag_name not in BAG_NAMES:
            raise ValueError(
                f"Invalid bag name: {bag_name}. Must be one of {BAG_NAMES}"
            )

        self.id = self.client.loadURDF(RigidCargoBag.URDF, pos, orn)
        # TODO UPDATE THIS!!! ("unlock" the joints)
        # pybullet.setJointMotorControlArray(
        #     self.id, [0, 1, 2], pybullet.VELOCITY_CONTROL, forces=[0.1, 0.1, 0.1]
        # )
        # This will cause the handle to spring back into its natural position
        pybullet.setJointMotorControlArray(
            self.id,
            [0, 1, 2],
            pybullet.POSITION_CONTROL,
            [0, 0, 0],
            forces=[0.1, 0.1, 0.1],
        )

        self._dt = self.client.getPhysicsEngineParameters()["fixedTimeStep"]
        self.mass = 5  # kg
        # This inertia is slightly approximate because some mass is in the handle and dummy links
        self.inertia = box_inertia(self.mass, self.LENGTH, self.WIDTH, self.HEIGHT)
        self._name = bag_name
        # Initializations
        self._constraints = {}
        self._anchor_objects = {}
        # self._mesh_vertices = None
        # self._num_mesh_vertices = None
        self._attached = []
        self._id = None
        print_green("Bag is ready")

    @property
    def anchors(self) -> tuple[list[int], list[int]]:
        """Anchor IDs and IDs of their associated visual geometries"""
        # Unpack the list of lists in the anchor dictionaries
        return flatten(self._constraints.values()), flatten(
            self._anchor_objects.values()
        )

    @property
    def attached(self) -> list[int]:
        """ID(s) of the robot (or robots) grasping the bag. Empty if no robots are attached"""
        return self._attached

    @property
    def name(self) -> str:
        """Type of cargo bag"""
        return self._name

    @property
    def grasp_transforms(self) -> list[np.ndarray]:
        """Transformation matrices "handle to bag" representing the grasp locations on the handles to the bag COM

        In the case of a single-handled bag, this list will only have one entry
        """
        if self._name == "front_handle":
            return [HANDLE_TRANSFORMS["front"]]
        elif self._name == "right_handle":
            return [HANDLE_TRANSFORMS["right"]]
        elif self._name == "top_handle":
            return [HANDLE_TRANSFORMS["top"]]
        elif self._name == "front_back_handle":
            return [HANDLE_TRANSFORMS["front"], HANDLE_TRANSFORMS["back"]]
        elif self._name == "right_left_handle":
            return [HANDLE_TRANSFORMS["right"], HANDLE_TRANSFORMS["left"]]
        elif self._name == "top_bottom_handle":
            return [HANDLE_TRANSFORMS["top"], HANDLE_TRANSFORMS["bottom"]]
        else:
            raise NotImplementedError(
                f"Grasp transform(s) not available for bag: {self._name}"
            )

    @property
    def pose(self) -> np.ndarray:
        """Current position + XYZW quaternion pose of the bag's COM frame"""
        return np.concatenate(self.client.getBasePositionAndOrientation(self.id))

    @property
    def position(self) -> np.ndarray:
        """Current XYZ position of the origin (COM frame) of the cargo bag"""
        return np.array(self.client.getBasePositionAndOrientation(self.id)[0])

    @property
    def orientation(self) -> np.ndarray:
        """Current XYZW quaternion orientation of the cargo bag's COM frame"""
        return np.array(self.client.getBasePositionAndOrientation(self.id)[1])

    @property
    def velocity(self) -> np.ndarray:
        """Current [vx, vy, vz] velocity of the cargo bag's COM frame

        - If both velocity and angular velocity are desired, use the dynamics_state property instead
        """
        return np.array(self.client.getBaseVelocity(self.id)[0])

    @property
    def angular_velocity(self) -> np.ndarray:
        """Current [wx, wy, wz] angular velocity of the cargo bag's COM frame

        - If both velocity and angular velocity are desired, use the dynamics_state property instead
        """
        return np.array(self.client.getBaseVelocity(self.id)[1])

    @property
    def dynamics_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Current state of the bag dynamics: Position, orientation, linear vel, and angular vel

        Returns:
            Tuple of:
                np.ndarray: Position, shape (3,)
                np.ndarray: XYZW quaternion orientation, shape (4,)
                np.ndarray: Linear velocity, shape (3,)
                np.ndarray: Angular velocity, shape (3,)
        """
        pos, orn = self.client.getBasePositionAndOrientation(self.id)
        lin_vel, ang_vel = self.client.getBaseVelocity(self.id)
        return (
            np.array(pos),
            np.array(orn),
            np.array(lin_vel),
            np.array(ang_vel),
        )

    @property
    def num_handles(self) -> int:
        """Number of handles on the cargo bag"""
        if self._name in SINGLE_HANDLE_BAGS:
            return 1
        elif self._name in DUAL_HANDLE_BAGS:
            return 2
        else:
            return 0  # This may have an application in the future

    def attach_to(
        self,
        robot_or_robots: Union[Astrobee, list[Astrobee], tuple[Astrobee]],
        object_to_move: str = "robot",
    ) -> None:
        """Attaches a robot (or multiple robots) to the handle(s) of the bag

        Args:
            robot_or_robots (Union[Astrobee, list[Astrobee], tuple[Astrobee]]): Robot(s) to attach to the bag
            object_to_move (str, optional): Either "robot" or "bag". This dictates what object will get its position
                reset in order to make the grasp connection. In general, it makes more sense to move the robot to the
                bag (default behavior)

        Raises:
            ValueError: For invalid inputs, or if the bag does not have enough handles for each robot
            NotImplementedError: Multi-robot case with >2 robots
        """
        # Handle inputs
        if isinstance(robot_or_robots, Astrobee):  # Single robot
            num_robots = 1
        elif isinstance(robot_or_robots, (list, tuple)):  # Multi-robot
            if not all(isinstance(r, Astrobee) for r in robot_or_robots):
                raise ValueError("Non-Astrobee input detected")
            num_robots = len(robot_or_robots)
            if self.num_handles < num_robots:
                raise ValueError(
                    f"Bag does not have enough handles to support {num_robots} robots"
                )
            if num_robots == 1:  # Edge case: Unpack the list if only one robot
                robot_or_robots = robot_or_robots[0]
        else:
            raise ValueError(
                "Invalid input: Must provide either an Astrobee or a list of multiple Astrobees"
            )
        if object_to_move not in {"robot", "bag"}:
            raise ValueError("Invalid object to move: Must be either 'robot' or 'bag'.")

        bag_to_world = pos_quat_to_tmat(self.pose)
        if num_robots == 1:
            robot = robot_or_robots  # Unpack list
            if object_to_move == "robot":
                # Reset the position of the robot to interface with the handle
                handle_to_bag = self.grasp_transforms[0]
                handle_to_world = bag_to_world @ handle_to_bag
                handle_pose = tmat_to_pos_quat(handle_to_world)
                robot.reset_to_ee_pose(handle_pose)
            else:  # Move the bag to the robot
                self.reset_to_handle_pose(robot.ee_pose)
            self._attach(robot)
        elif num_robots == 2:
            robot_1, robot_2 = robot_or_robots  # Unpack list
            if object_to_move == "robot":
                # Reset the position of each robot to interface with the two handles
                handle_1_to_bag = self.grasp_transforms[0]
                handle_2_to_bag = self.grasp_transforms[1]
                handle_1_to_world = bag_to_world @ handle_1_to_bag
                handle_2_to_world = bag_to_world @ handle_2_to_bag
                robot_1.reset_to_ee_pose(tmat_to_pos_quat(handle_1_to_world))
                robot_2.reset_to_ee_pose(tmat_to_pos_quat(handle_2_to_world))
                self._attach(robot_1)
                self._attach(robot_2)
            else:  # Move the bag while leaving the robots static
                raise NotImplementedError(
                    "Attaching the bag to multiple robots requires moving at least 1 robot"
                )
                # This generally leads to undesired behavior unless the 2 robots are carefully positioned before
                # attachment, so we'll throw an error instead. Uncomment the lines below if this is desired later on
                # self.reset_to_handle_pose(robot_1.ee_pose, 0)
                # self._attach(robot_1)
                # self.reset_to_handle_pose(robot_2.ee_pose, 1)
                # self._attach(robot_2)
        else:
            raise NotImplementedError(
                "The multi-robot case is only implemented for 2 Astrobees"
            )

    def _attach(self, robot: Astrobee) -> None:
        """Helper function: Connects a single robot to a handle at a specified pose

        This function assumes that the robot and the bag are already correctly positioned for a grasp, which is why
        it should not be called directly

        Args:
            robot (Astrobee): Robot to attach
        """
        # Generate the constraints between the bag and the robot: First, find the points on the mesh on either side
        # of the handle (Using the left/right gripper link frames as reference points), then create the anchors
        # pos_1 = robot.get_link_transform(robot.Links.GRIPPER_LEFT_DISTAL)[:3, 3]
        # pos_2 = robot.get_link_transform(robot.Links.GRIPPER_RIGHT_DISTAL)[:3, 3]
        # v1_pos, v1_id = get_closest_mesh_vertex(pos_1, self.mesh_vertices)
        # v2_pos, v2_id = get_closest_mesh_vertex(pos_2, self.mesh_vertices)
        # # handle_pos = self.get_handle_transform(0)[:3, 3] # in world

        # return  # REMOVE THIS

        HANDLE_LINK_IDX = 3  # TODO FIX THIS
        cid = self.client.createConstraint(
            robot.id,
            robot.Links.ARM_DISTAL.value,
            self.id,
            HANDLE_LINK_IDX,
            pybullet.JOINT_FIXED,  # Point to point
            [0, 0, 1],
            robot.TRANSFORMS.GRIPPER_TO_ARM_DISTAL[:3, 3],
            [0, 0, 0],
            [0, 0, 0, 1],
            rmat_to_quat(Ry(-np.pi / 2)),
        )

        # # Create a collision-less sphere to visualize the anchor position
        # geom_pos = robot.ee_pose
        # geom_id = create_sphere(
        #     geom_pos, 0.01, 0.01, False, [0, 1, 0, 0.5], self.client
        # )
        # # Then create a secondary anchor to make sure this sphere stays in the right place
        # geom_constraint_id = self.client.createConstraint(
        #     robot.id,
        #     robot.Links.ARM_DISTAL.value,
        #     geom_id,
        #     -1,
        #     pybullet.JOINT_FIXED,
        #     [0, 0, 0],
        #     robot.TRANSFORMS.ARM_DISTAL_TO_GRIPPER[:3, 3],
        #     [0, 0, 0],
        # )

        # TODO get the geom in here
        self._constraints.update({robot.id: cid})
        # self._anchors.update({robot.id: [anchor1_id, anchor2_id]})
        # self._anchor_objects.update({robot.id: [geom1_id, geom2_id]})
        # self._attached.append(robot.id)

    def get_handle_transform(self, handle_index: int = 0) -> np.ndarray:
        # """Calculates the transformation matrix (w.r.t the world) for a specified link

        # Args:
        #     link_index (int): Index of the link on the robot

        # Returns:
        #     np.ndarray: Transformation matrix (link to world). Shape = (4,4)
        # """

        if handle_index == 0:
            # Three dummy links for roll, pitch, yaw (0, 1, and 2)
            # The handle is the 4th link (index 3)
            link_index = RigidCargoBag.LINKS_PER_HANDLE - 1

        link_state = self.client.getLinkState(
            self.id, link_index, computeForwardKinematics=True
        )
        # First two link state values are linkWorldPosition, linkWorldOrientation
        # There are other state positions and orientations, but they're confusing. (TODO check on these)
        pos, quat = link_state[:2]
        return make_transform_mat(quat_to_rmat(quat), pos)

    def detach(self) -> None:
        """Remove all anchors (and visuals) from the bag"""

        anchors, anchor_objects = self.anchors
        for cid in anchors:
            self.client.removeConstraint(cid)
        for obj in anchor_objects:
            self.client.removeBody(obj)
        self._constraints = {}
        self._anchor_objects = {}
        self._attached = []

    def detach_robot(self, robot_id: int) -> None:
        """Detaches a specific robot from the bag by removing its associated anchors / visuals

        Args:
            robot_id (int): Pybullet ID of the robot to detach
        """
        if robot_id not in self.attached:
            raise ValueError("Cannot detach robot: ID unknown")
        for cid in self._constraints[robot_id]:
            self.client.removeConstraint(cid)
        for obj in self._anchor_objects[robot_id]:
            self.client.removeBody(obj)
        self._constraints.pop(robot_id)
        self._anchor_objects.pop(robot_id)
        self._attached.remove(robot_id)

    def unload(self) -> None:
        """Removes the cargo bag from the simulation

        TODO this is kinda broken since it only removes the simulated body rather than the visual
        (if we update how we load the texture for the softbody this might be fixed)
        """
        self.client.removeBody(self.id)
        self.id = None

    def reset_to_handle_pose(
        self, handle_pose: npt.ArrayLike, handle_index: int = 0
    ) -> None:
        """Resets the position of the bag so that the handle is positioned at a desired pose

        Args:
            handle_pose (npt.ArrayLike): Desired pose of the handle ("handle-to-world"), shape (7,)
            handle_index (int, optional): Index of the handle to align to the desired pose. Defaults to 0.
        """
        handle_to_world = pos_quat_to_tmat(handle_pose)
        bag_to_handle = invert_transform_mat(self.grasp_transforms[handle_index])
        bag_to_world = handle_to_world @ bag_to_handle
        bag_pose = tmat_to_pos_quat(bag_to_world)
        self.client.resetBasePositionAndOrientation(self.id, bag_pose[:3], bag_pose[3:])


def _main():
    # Very simple example of loading the bag and attaching a robot
    client = initialize_pybullet(bg_color=(1, 0.8, 0.8))
    robot = Astrobee()
    bag = RigidCargoBag("top_handle")
    bag.attach_to(robot)
    input("Press Enter to run the simulation")
    while True:
        client.stepSimulation()
        time.sleep(1 / 120)


if __name__ == "__main__":
    _main()
