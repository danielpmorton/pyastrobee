"""Class for handling loading the cargo bag and managing attributes such as physical properties

TODO !! Fix the softbody velocity issue
TODO ! the unloading mechanic is currently broken since it only removes the softbody and not the visual
TODO If the reliability of the softbody position/orientation is not good, use the get_bag_frame() function I made
TODO Maybe add an anchor to the center of the bag to help with dynamics? But this won't give angular vel I think
TODO decide if the bag_props import can be handled better
TODO decide if the constants should be moved to class attributes
TODO decide if there is a better way to handle the "attached vs freefloating" initialization
TODO decide if we should anchor to the gripper fingers or the arm distal link (currently distal)
TODO decide if we should set up the class attributes to handle multiple bags loaded in sim
    (Like the Astrobee class)
"""
import time
from typing import Optional

import pybullet
import numpy as np
import numpy.typing as npt

import pyastrobee.config.bag_properties as bag_props
from pyastrobee.elements.astrobee import Astrobee
from pyastrobee.utils.bullet_utils import (
    load_deformable_object,
    create_anchor,
    initialize_pybullet,
)
from pyastrobee.utils.mesh_utils import get_mesh_data, get_closest_mesh_vertex
from pyastrobee.utils.poses import pos_quat_to_tmat
from pyastrobee.utils.rotations import rmat_to_quat
from pyastrobee.utils.python_utils import print_green
from pyastrobee.utils.quaternions import quats_to_angular_velocities

# Constants. TODO Move these to class attributes?
MESH_DIR = "pyastrobee/assets/meshes/bags/"
BAG_NAMES = ["front_handle_bag", "side_handle_bag", "top_handle_bag"]
_objs = [MESH_DIR + name + ".obj" for name in BAG_NAMES]
_vtks = [MESH_DIR + name + ".vtk" for name in BAG_NAMES]
OBJS = dict(zip(BAG_NAMES, _objs))
VTKS = dict(zip(BAG_NAMES, _vtks))
GRASP_TRANSFORMS = dict(
    zip(
        BAG_NAMES,
        [
            bag_props.FRONT_BAG_GRASP_TRANSFORM,
            bag_props.SIDE_BAG_GRASP_TRANSFORM,
            bag_props.TOP_BAG_GRASP_TRANSFORM,
        ],
    )
)
BAG_CORNERS = dict(
    zip(
        BAG_NAMES,
        [
            bag_props.FRONT_BAG_CORNER_VERTS,
            bag_props.SIDE_BAG_CORNER_VERTS,
            bag_props.TOP_BAG_CORNER_VERTS,
        ],
    )
)


class CargoBag:
    """Class for loading and managing properties associated with the cargo bags

    Args:
        bag_name (str): Type of cargo bag to load. One of "front_handle_bag", "side_handle_bag", "top_handle_bag"
        attached_robot (Optional[Astrobee], optional): If initially attaching the cargo bag to a robot, include
            the robot here. Defaults to None, in which case pos/orn must be specified
        pos (Optional[npt.ArrayLike], optional): If not attaching to a robot, this is the XYZ position to load
            the bag. Defaults to None, in which case the robot must be speficied
        orn (Optional[npt.ArrayLike], optional): If not attaching to a robot, this is the XYZW quaternion to load
            the bag. Defaults to None, in which case the robot must be speficied
    """

    def __init__(
        self,
        bag_name: str,
        attached_robot: Optional[Astrobee] = None,
        pos: Optional[npt.ArrayLike] = None,
        orn: Optional[npt.ArrayLike] = None,
    ):
        if not pybullet.isConnected():
            raise ConnectionError("Need to connect to pybullet before loading a bag")
        # Validate inputs
        if bag_name not in BAG_NAMES:
            raise ValueError(
                f"Invalid bag name: {bag_name}. Must be one of {BAG_NAMES}"
            )
        if attached_robot is not None:
            if not isinstance(attached_robot, Astrobee):
                raise ValueError("Invalid robot input. Must be an Astrobee() instance")
        else:
            if pos is None or orn is None:
                raise ValueError(
                    "Must provide a position and orientation to load the bag if not attaching to a robot"
                )
        self._name = bag_name
        # Initializations
        self._anchors = []
        self._anchor_objects = []
        self._mesh_vertices = None
        self._num_mesh_vertices = None
        self._attached_robot = None
        self._id = None
        # Get the simulator timestep (used for calculating the velocity of the bag)
        self._dt = pybullet.getPhysicsEngineParameters()["fixedTimeStep"]
        # Load the bag depending on the specified method
        if attached_robot is not None:
            self._load_attached(attached_robot)
        else:
            self._load(pos, orn)
        print_green("Bag is ready")

    # Read-only physical properties defined at initialization of the bag
    @property
    def num_mesh_vertices(self) -> int:
        """Number of vertices in the bag's mesh"""
        if self.id is None:
            raise AttributeError("Mesh has not been loaded")
        if self._num_mesh_vertices is None:
            self._num_mesh_vertices, self._mesh_vertices = get_mesh_data(self.id)
        return self._num_mesh_vertices

    @property
    def mesh_vertices(self) -> np.ndarray:
        """Positions of the mesh vertices, shape (n, 3)"""
        if self.id is None:
            raise AttributeError("Mesh has not been loaded")
        self._num_mesh_vertices, self._mesh_vertices = get_mesh_data(self.id)
        return self._mesh_vertices

    @property
    def mass(self) -> float:
        """Softbody mass"""
        return bag_props.MASS

    @property
    def bending_stiffness(self) -> float:
        """Softbody bending stiffness parameter"""
        return bag_props.BENDING_STIFFNESS

    @property
    def damping_stiffness(self) -> float:
        """Softbody damping stiffness parameter"""
        return bag_props.DAMPING_STIFFNESS

    @property
    def elastic_stiffness(self) -> float:
        """Softbody elastic stiffness parameter"""
        return bag_props.ELASTIC_STIFFNESS

    @property
    def friction_coeff(self) -> float:
        """Softbody friction coefficient"""
        return bag_props.FRICTION_COEFF

    @property
    def anchors(self) -> tuple[list[int], list[int]]:
        """Anchor IDs and IDs of their associated visual geometries"""
        return (self._anchors, self._anchor_objects)

    @property
    def attached_robot(self) -> int:
        """ID of the robot grasping the bag"""
        return self._attached_robot

    @property
    def name(self) -> str:
        """Type of cargo bag"""
        return self._name

    @property
    def obj_file(self) -> str:
        """Path to the .OBJ triangular mesh file"""
        return OBJS[self._name]

    @property
    def vtk_file(self) -> str:
        """Path to the .VTK tetrahedral mesh file"""
        return VTKS[self._name]

    @property
    def grasp_transform(self) -> np.ndarray:
        """Transformation matrix between the bag origin to grasp frame at the handle"""
        return GRASP_TRANSFORMS[self._name]

    @property
    def position(self) -> np.ndarray:
        """Current XYZ position of the origin (COM frame) of the cargo bag"""
        return np.array(pybullet.getBasePositionAndOrientation(self.id)[0])

    @property
    def orientation(self) -> np.ndarray:
        """Current XYZW quaternion orientation of the cargo bag's COM frame"""
        return np.array(pybullet.getBasePositionAndOrientation(self.id)[1])

    @property
    def velocity(self) -> np.ndarray:
        """Current [vx, vy, vz] velocity of the cargo bag's COM frame

        - If both velocity and angular velocity are desired, use the dynamics_state property instead
        """
        return self.dynamics_state[2]

    @property
    def angular_velocity(self) -> np.ndarray:
        """Current [wx, wy, wz] angular velocity of the cargo bag's COM frame

        - If both velocity and angular velocity are desired, use the dynamics_state property instead
        """
        return self.dynamics_state[3]

    @property
    def dynamics_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Current state of the bag dynamics: Position, orientation, linear vel, and angular vel

        - NOTE this moves the simulation forward by 1 step
        - Bullet's velocity definition for softbodies is incorrect or not implemented. I've tried
          implementing this myself in the C++, but it is also not reliable. So instead, we'll need to step
          the sim in this call to do our own calculations

        Returns:
            Tuple of:
                np.ndarray: Position, shape (3,)
                np.ndarray: XYZW quaternion orientation, shape (4,)
                np.ndarray: Linear velocity, shape (3,)
                np.ndarray: Angular velocity, shape (3,)
        """
        old_pos, old_orn = pybullet.getBasePositionAndOrientation(self.id)
        # Step the sim to get a second reference frame we can use to determine velocity
        # This is not ideal, but it's the best way to do this until Pybullet's getBaseVelocity for softbodies works
        pybullet.stepSimulation()
        new_pos, new_orn = pybullet.getBasePositionAndOrientation(self.id)
        lin_vel = np.subtract(new_pos, old_pos) / self._dt
        ang_vel = quats_to_angular_velocities(
            np.row_stack([old_orn, new_orn]), self._dt
        )
        if ang_vel.ndim > 1:
            ang_vel = ang_vel[0, :]
        # Return the stepped-ahead position since it's the most recent state we know about
        return (
            np.array(new_pos),
            np.array(new_orn),
            np.array(lin_vel),
            np.array(ang_vel),
        )

    def _load(self, pos: npt.ArrayLike, orn: npt.ArrayLike) -> None:
        """Loads a cargo bag at the specified position/orientation

        Args:
            pos (npt.ArrayLike): XYZ position, shape (3,)
            orn (npt.ArrayLike): XYZW quaternion, shape (4,)
        """
        texture = None
        scale = 1
        self_collision = False
        self.id = load_deformable_object(
            self.obj_file,
            texture,
            scale,
            pos,
            orn,
            self.mass,
            self.bending_stiffness,
            self.damping_stiffness,
            self.elastic_stiffness,
            self.friction_coeff,
            self_collision,
            self.vtk_file,
        )

    def _load_attached(self, robot: Astrobee) -> None:
        """Load the cargo bag attached to the gripper of an Astrobee

        Args:
            robot (Astrobee): The robot to attach the bag to
        """
        # Determine the transformation which dictates where we need to load the bag
        # for the handle to line up with the gripper
        T_ee2world = pos_quat_to_tmat(robot.ee_pose)
        T_bag2ee = self.grasp_transform
        T_bag2world = T_ee2world @ T_bag2ee
        # Load the bag at the position/orientation specified by the transform
        pos = T_bag2world[:3, 3]
        orn = rmat_to_quat(T_bag2world[:3, :3])
        self._load(pos, orn)
        # Generate the constraints between the bag and the robot
        # Find the points on the mesh on either side of the handle
        # (Using the left/right gripper link frames as reference points)
        pos_1 = robot.get_link_transform(robot.Links.GRIPPER_LEFT_DISTAL)[:3, 3]
        pos_2 = robot.get_link_transform(robot.Links.GRIPPER_RIGHT_DISTAL)[:3, 3]
        v1_pos, v1_id = get_closest_mesh_vertex(pos_1, self.mesh_vertices)
        v2_pos, v2_id = get_closest_mesh_vertex(pos_2, self.mesh_vertices)
        # (TODO) decide if it makes more sense to anchor to the gripper fingers themselves
        # or if anchoring to the "palm" (the arm distal link) is ok
        link_to_anchor = robot.Links.ARM_DISTAL.value
        anchor1_id, geom1_id = create_anchor(
            self.id, v1_id, robot.id, link_to_anchor, add_geom=True, geom_pos=v1_pos
        )
        anchor2_id, geom2_id = create_anchor(
            self.id, v2_id, robot.id, link_to_anchor, add_geom=True, geom_pos=v2_pos
        )
        self._anchors = (anchor1_id, anchor2_id)
        self._anchor_objects = (geom1_id, geom2_id)
        self._attached_robot = robot.id

    def detach(self) -> None:
        """Remove any currently attached anchors from the bag (and any associated visual geometry)"""
        for cid in self._anchors:
            pybullet.removeConstraint(cid)
        for obj in self._anchor_objects:
            pybullet.removeBody(obj)
        self._anchors = []
        self._anchor_objects = []
        self._attached_robot = None

    def unload(self) -> None:
        """Removes the cargo bag from the simulation

        TODO this is kinda broken since it only removes the simulated body rather than the visual
        (if we update how we load the texture for the softbody this might be fixed)
        """
        pybullet.removeBody(self.id)
        self.id = None


if __name__ == "__main__":
    initialize_pybullet()
    robo = Astrobee()
    bag = CargoBag("top_handle_bag", robo)
    init_time = time.time()
    detach_time_lim = 10
    dt = pybullet.getPhysicsEngineParameters()["fixedTimeStep"]
    print(f"Provide a disturbance force. Detaching bag in {detach_time_lim} seconds")
    while time.time() - init_time < detach_time_lim:
        pos, orn, lin_vel, ang_vel = bag.dynamics_state
        print("Position: ", pos)
        print("Orientation: ", orn)
        print("Velocity: ", lin_vel)
        print("Angular velocity: ", ang_vel)
        # visualize_frame(pos_quat_to_tmat([*pos, *orn]), lifetime=0.5)
        pybullet.stepSimulation()
        time.sleep(dt)
    bag.detach()
    detach_time = time.time()
    unload_time_lim = 10
    print(f"Detached. Unloading the bag in {unload_time_lim} seconds")
    while time.time() - detach_time < unload_time_lim:
        pybullet.stepSimulation()
        time.sleep(dt)
    bag.unload()  # Currently kinda weird
    print("Unloaded")
    while True:
        pybullet.stepSimulation()
        time.sleep(dt)
