"""Functions associated with the Pybullet Debug Visualizer GUI

NOTE
- For the GUI camera parameters, since the debug visualizer only specifies yaw and pitch (no roll),
  this is an incomplete definition of rotation, so we can never actually fully change the rotation
  of the debug viz camera to match the robot's position
    - This is not a huge deal, it will just mean we will have at best a "video game perspective" of the robot
"""

import time
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import pybullet

from pyastrobee.elements.astrobee import Astrobee
from pyastrobee.utils.rotations import (
    quat_to_fixed_xyz,
    fixed_xyz_to_quat,
    fixed_xyz_to_rmat,
    quat_to_rmat,
)
from pyastrobee.utils.poses import pos_quat_to_tmat, batched_pos_quats_to_tmats
from pyastrobee.utils.transformations import make_transform_mat
from pyastrobee.utils.coordinates import cartesian_to_spherical
from pyastrobee.config.astrobee_transforms import OBSERVATION_CAM
from pyastrobee.control.controller import PoseController


def visualize_traj(traj: npt.ArrayLike) -> list[int]:
    """Visualizes a trajectory's sequence of poses on the Pybullet GUI

    TODO make it possible to take in a Trajectory class object and extract the relevant info

    Args:
        traj (npt.ArrayLike): Sequence of position + quaternion poses, shape (n, 7)

    Returns:
        list[int]: Pybullet IDs for the lines drawn onto the GUI
    """
    tmats = batched_pos_quats_to_tmats(traj)  # This will validate the input as well
    ids = []
    for i in range(tmats.shape[0]):
        ids += visualize_frame(tmats[i, :, :])
    return ids


def visualize_points(
    position: npt.ArrayLike, color: npt.ArrayLike, size: float = 20, lifetime: float = 0
) -> int:
    """Adds square points to the GUI to visualize positions in the sim

    Args:
        position (npt.ArrayLike): 3D point(s) in the simulation to visualize. Shape (n, 3)
        color (npt.ArrayLike): RGB values, each in range [0, 1]. Shape (n, 3)
        size (float): Size of the points on the GUI, in pixels. Defaults to 20
        lifetime (float, optional): Amount of time to keep the points on the GUI, in seconds.
            Defaults to 0 (keep them on-screen permanently until deleted)

    Returns:
        int: Pybullet object ID for the point / point cloud
    """
    # Pybullet will crash if you try to visualize one point without packing it into a 2D array
    position = np.atleast_2d(position)
    color = np.atleast_2d(color)
    if position.shape[-1] != 3:
        raise ValueError(
            f"Invalid shape of the point positions. Expected (n, 3), got: {position.shape}"
        )
    if color.shape[-1] != 3:
        raise ValueError(
            f"Invalid shape of the colors. Expected (n, 3), got: {color.shape}"
        )
    n = position.shape[0]
    if color.shape[0] != n:
        if color.shape[0] == 1:
            # Map the same color to all of the points
            color = color * np.ones_like(position)
        else:
            raise ValueError(
                f"Number of colors ({color.shape[0]}) does not match the number of points ({n})."
            )
    return pybullet.addUserDebugPoints(position, color, size, lifetime)


def visualize_frame(
    tmat: np.ndarray, length: float = 1, width: float = 3, lifetime: float = 0
) -> tuple[int, int, int]:
    """Adds RGB XYZ axes to the Pybullet GUI for a speficied transformation/frame/pose

    Args:
        tmat (np.ndarray): Transformation matrix specifying a pose w.r.t world frame, shape (4, 4)
        length (float, optional): Length of the axis lines. Defaults to 1.
        width (float, optional): Width of the axis lines. Defaults to 3. (units unknown, maybe mm?)
        lifetime (float, optional): Amount of time to keep the lines on the GUI, in seconds.
            Defaults to 0 (keep them on-screen permanently until deleted)

    Returns:
        tuple[int, int, int]: Pybullet IDs of the three axis lines added to the GUI
    """
    x_color = [1, 0, 0]  # R
    y_color = [0, 1, 0]  # G
    z_color = [0, 0, 1]  # B
    origin = tmat[:3, 3]
    x_endpt = origin + tmat[:3, 0] * length
    y_endpt = origin + tmat[:3, 1] * length
    z_endpt = origin + tmat[:3, 2] * length
    x_ax_id = pybullet.addUserDebugLine(origin, x_endpt, x_color, width, lifetime)
    y_ax_id = pybullet.addUserDebugLine(origin, y_endpt, y_color, width, lifetime)
    z_ax_id = pybullet.addUserDebugLine(origin, z_endpt, z_color, width, lifetime)
    return x_ax_id, y_ax_id, z_ax_id


def visualize_link_frame(
    robot_id: int,
    link_id: int,
    length: float = 1,
    width: float = 3,
    lifetime: float = 0,
) -> tuple[int, int, int]:
    """Adds RGB XYZ axes to the Pybullet GUI for a specific link on the Astrobee

    Args:
        robot_id (int): Pybullet ID of the Astrobee
        link_id (int): ID of the link we want to look at
        length (float, optional): Length of the axis lines. Defaults to 1.
        width (float, optional): Width of the axis lines. Defaults to 3. (units unknown, maybe mm?)
        lifetime (float, optional): Amount of time to keep the lines on the GUI, in seconds.
            Defaults to 0 (keep them on-screen permanently until deleted)

    Returns:
        tuple[int, int, int]: Pybullet IDs of the three axis lines added to the GUI
    """
    x_color = [1, 0, 0]  # R
    y_color = [0, 1, 0]  # G
    z_color = [0, 0, 1]  # B
    origin = [0, 0, 0]
    x_endpt = np.array([1, 0, 0]) * length
    y_endpt = np.array([0, 1, 0]) * length
    z_endpt = np.array([0, 0, 1]) * length
    x_ax_id = pybullet.addUserDebugLine(
        origin, x_endpt, x_color, width, lifetime, robot_id, link_id
    )
    y_ax_id = pybullet.addUserDebugLine(
        origin, y_endpt, y_color, width, lifetime, robot_id, link_id
    )
    z_ax_id = pybullet.addUserDebugLine(
        origin, z_endpt, z_color, width, lifetime, robot_id, link_id
    )
    return x_ax_id, y_ax_id, z_ax_id


def visualize_quaternion(
    quat: npt.ArrayLike, length: float = 1, width: float = 3, lifetime: float = 0
) -> tuple[int, int, int]:
    """Wrapper around visualize_frame specifically for debugging quaternions. Shows the rotated frame at the origin

    Args:
        quat (npt.ArrayLike): XYZW quaternion, shape (4,)
        length (float, optional): Length of the axis lines. Defaults to 1.
        width (float, optional): Width of the axis lines. Defaults to 3. (units unknown, maybe mm?)
        lifetime (float, optional): Amount of time to keep the lines on the GUI, in seconds.
            Defaults to 0 (keep them on-screen permanently until deleted)

    Returns:
        tuple[int, int, int]: Pybullet IDs of the three axis lines added to the GUI
    """
    rmat = quat_to_rmat(quat)
    tmat = make_transform_mat(rmat, [0, 0, 0])
    return visualize_frame(tmat, length, width, lifetime)


def remove_debug_objects(ids: Union[int, list[int], np.ndarray[int]]) -> None:
    """Removes user-created line(s)/point(s)/etc. from the Pybullet GUI

    Args:
        ids (int or list/array of ints): ID(s) of the objects loaded into Pybullet
    """
    if np.ndim(ids) == 0:  # Scalar, not iterable
        pybullet.removeUserDebugItem(ids)
        return
    for i in ids:
        pybullet.removeUserDebugItem(i)


def get_viz_camera_params(
    T_R2W: np.ndarray, T_C2R: Optional[np.ndarray] = None
) -> tuple[float, float, float, np.ndarray]:
    """Calculates the debug visualizer camera position for a specified robot-frame camera view

    Args:
        T_R2W (np.ndarray): Robot to World transformation matrix, shape (4,4). This input is
            equivalent to the current robot pose (just expressed in tmat form)
        T_C2R (np.ndarray, optional): Camera to Robot transformation matrix, shape (4, 4). This input dictates
            the perspective from which we observe the robot. Defaults to the OBSERVATION_CAM constant

    Returns:
        tuple of:
            float: Distance: The distance to the camera's focus point
            float: Yaw: The rotation angle about the Z axis
            float: Pitch: The angle of the camera as measured above/below the XY plane
            np.ndarray: Target: The camera focus point in world coords. Shape (3,)
    """
    if T_C2R is None:
        T_C2R = OBSERVATION_CAM
    R_R2W = T_R2W[:3, :3]  # Robot to world rotation matrix
    robot_to_cam_robot_pos = T_C2R[:3, 3]
    robot_to_cam_world_pos = R_R2W @ robot_to_cam_robot_pos
    cam_to_robot_world_pos = -1 * robot_to_cam_world_pos
    # Determine the direction the camera points via the vector from camera -> robot origin in world frame
    dist, elevation, azimuth = cartesian_to_spherical(cam_to_robot_world_pos)
    # Elevation is measured w.r.t +Z, so we need to change this to w.r.t the XY plane
    # Pybullet also works with the camera angle definitions in degrees, not radians
    pitch = np.rad2deg(np.pi / 2 - elevation)
    # There seems to be a fixed -90 degree offset in how pybullet defines yaw (TODO test this)
    yaw = np.rad2deg(azimuth) - 90
    robot_world_pos = T_R2W[:3, 3]
    target = robot_world_pos
    return dist, yaw, pitch, target


if __name__ == "__main__":
    # Quick test if the debug visualizer camera parameters are able to track the robot motion
    pybullet.connect(pybullet.GUI)
    robot = Astrobee()
    controller = PoseController(robot)
    R = fixed_xyz_to_rmat([0, -np.pi / 4, 0])
    C2R = make_transform_mat(R, [-0.7, 0, 0.5])
    pos = np.array([0.0, 0.0, 0.0])
    orn = np.array([0.0, 0.0, 0.0, 1.0])
    while True:
        R2W = pos_quat_to_tmat(robot.pose)
        d, y, p, t = get_viz_camera_params(R2W, C2R)
        # print(f"Dist = {d}")
        # print(f"Yaw = {y}")
        # print(f"Pitch = {p}")
        # print(f"Target = {t}")
        pos += np.array([0.01, 0, 0])
        orn = quat_to_fixed_xyz(orn)
        orn += np.array([0.01, 0.0, 0.0])
        orn = fixed_xyz_to_quat(orn)
        # orn += 0.1 * np.random.rand(4) + orn
        # orn = orn / np.linalg.norm(orn)
        pybullet.resetDebugVisualizerCamera(d, y, p, t)
        pybullet.changeConstraint(controller.constraint_id, pos, orn)
        pybullet.stepSimulation()
        time.sleep(1 / 20)
