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
from pybullet_utils.bullet_client import BulletClient

from pyastrobee.core.astrobee import Astrobee
from pyastrobee.utils.rotations import (
    quat_to_fixed_xyz,
    fixed_xyz_to_quat,
    fixed_xyz_to_rmat,
    quat_to_rmat,
)
from pyastrobee.utils.poses import pos_quat_to_tmat
from pyastrobee.utils.transformations import make_transform_mat
from pyastrobee.utils.coordinates import cartesian_to_spherical
from pyastrobee.config.astrobee_transforms import OBSERVATION_CAM
from pyastrobee.control.controller import PoseController
from pyastrobee.utils.bullet_utils import create_box


def visualize_points(
    position: npt.ArrayLike,
    color: npt.ArrayLike,
    size: float = 20,
    lifetime: float = 0,
    client: Optional[BulletClient] = None,
) -> int:
    """Adds square points to the GUI to visualize positions in the sim

    Args:
        position (npt.ArrayLike): 3D point(s) in the simulation to visualize. Shape (n, 3)
        color (npt.ArrayLike): RGB values, each in range [0, 1]. Shape (3,) if specifying the same color for all points,
            or (n, 3) to individually specify the colors per-point
        size (float): Size of the points on the GUI, in pixels. Defaults to 20
        lifetime (float, optional): Amount of time to keep the points on the GUI, in seconds.
            Defaults to 0 (keep them on-screen permanently until deleted)
        client (BulletClient, optional): If connecting to multiple physics servers, include the client
            (the class instance, not just the ID) here. Defaults to None (use default connected client)

    Returns:
        int: Pybullet object ID for the point / point cloud
    """
    client: pybullet = pybullet if client is None else client
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
    return client.addUserDebugPoints(position, color, size, lifetime)


def visualize_frame(
    tmat: np.ndarray,
    length: float = 1,
    width: float = 3,
    lifetime: float = 0,
    client: Optional[BulletClient] = None,
) -> tuple[int, int, int]:
    """Adds RGB XYZ axes to the Pybullet GUI for a speficied transformation/frame/pose

    Args:
        tmat (np.ndarray): Transformation matrix specifying a pose w.r.t world frame, shape (4, 4)
        length (float, optional): Length of the axis lines. Defaults to 1.
        width (float, optional): Width of the axis lines. Defaults to 3. (units unknown, maybe mm?)
        lifetime (float, optional): Amount of time to keep the lines on the GUI, in seconds.
            Defaults to 0 (keep them on-screen permanently until deleted)
        client (BulletClient, optional): If connecting to multiple physics servers, include the client
            (the class instance, not just the ID) here. Defaults to None (use default connected client)

    Returns:
        tuple[int, int, int]: Pybullet IDs of the three axis lines added to the GUI
    """
    client: pybullet = pybullet if client is None else client
    x_color = [1, 0, 0]  # R
    y_color = [0, 1, 0]  # G
    z_color = [0, 0, 1]  # B
    origin = tmat[:3, 3]
    x_endpt = origin + tmat[:3, 0] * length
    y_endpt = origin + tmat[:3, 1] * length
    z_endpt = origin + tmat[:3, 2] * length
    x_ax_id = client.addUserDebugLine(origin, x_endpt, x_color, width, lifetime)
    y_ax_id = client.addUserDebugLine(origin, y_endpt, y_color, width, lifetime)
    z_ax_id = client.addUserDebugLine(origin, z_endpt, z_color, width, lifetime)
    return x_ax_id, y_ax_id, z_ax_id


def visualize_link_frame(
    robot_id: int,
    link_id: int,
    length: float = 1,
    width: float = 3,
    lifetime: float = 0,
    client: Optional[BulletClient] = None,
) -> tuple[int, int, int]:
    """Adds RGB XYZ axes to the Pybullet GUI for a specific link on the Astrobee

    Args:
        robot_id (int): Pybullet ID of the Astrobee
        link_id (int): ID of the link we want to look at
        length (float, optional): Length of the axis lines. Defaults to 1.
        width (float, optional): Width of the axis lines. Defaults to 3. (units unknown, maybe mm?)
        lifetime (float, optional): Amount of time to keep the lines on the GUI, in seconds.
            Defaults to 0 (keep them on-screen permanently until deleted)
        client (BulletClient, optional): If connecting to multiple physics servers, include the client
            (the class instance, not just the ID) here. Defaults to None (use default connected client)

    Returns:
        tuple[int, int, int]: Pybullet IDs of the three axis lines added to the GUI
    """
    client: pybullet = pybullet if client is None else client
    x_color = [1, 0, 0]  # R
    y_color = [0, 1, 0]  # G
    z_color = [0, 0, 1]  # B
    origin = [0, 0, 0]
    x_endpt = np.array([1, 0, 0]) * length
    y_endpt = np.array([0, 1, 0]) * length
    z_endpt = np.array([0, 0, 1]) * length
    x_ax_id = client.addUserDebugLine(
        origin, x_endpt, x_color, width, lifetime, robot_id, link_id
    )
    y_ax_id = client.addUserDebugLine(
        origin, y_endpt, y_color, width, lifetime, robot_id, link_id
    )
    z_ax_id = client.addUserDebugLine(
        origin, z_endpt, z_color, width, lifetime, robot_id, link_id
    )
    return x_ax_id, y_ax_id, z_ax_id


def visualize_quaternion(
    quat: npt.ArrayLike,
    length: float = 1,
    width: float = 3,
    lifetime: float = 0,
    client: Optional[BulletClient] = None,
) -> tuple[int, int, int]:
    """Wrapper around visualize_frame specifically for debugging quaternions. Shows the rotated frame at the origin

    Args:
        quat (npt.ArrayLike): XYZW quaternion, shape (4,)
        length (float, optional): Length of the axis lines. Defaults to 1.
        width (float, optional): Width of the axis lines. Defaults to 3. (units unknown, maybe mm?)
        lifetime (float, optional): Amount of time to keep the lines on the GUI, in seconds.
            Defaults to 0 (keep them on-screen permanently until deleted)
        client (BulletClient, optional): If connecting to multiple physics servers, include the client
            (the class instance, not just the ID) here. Defaults to None (use default connected client)

    Returns:
        tuple[int, int, int]: Pybullet IDs of the three axis lines added to the GUI
    """
    rmat = quat_to_rmat(quat)
    tmat = make_transform_mat(rmat, [0, 0, 0])
    return visualize_frame(tmat, length, width, lifetime, client)


def visualize_path(
    positions: npt.ArrayLike,
    n: Optional[int] = None,
    color: npt.ArrayLike = (1, 0, 0),
    width: float = 3,
    lifetime: float = 0,
    client: Optional[BulletClient] = None,
) -> list[int]:
    """Visualize a sequence of positions on the Pybullet GUI

    Args:
        positions (npt.ArrayLike): Sequence of positions, shape (n, 3)
        n (Optional[int]): Number of lines to plot, if plotting the lines between all positions is not desired.
            Defaults to None (plot all lines between positions)
        color (npt.ArrayLike, optional): RGB color values. Defaults to (1, 0, 0) (red).
        width (float, optional): Width of the line. Defaults to 3 (pixels)
        lifetime (float, optional): Amount of time to keep the lines on the GUI, in seconds.
            Defaults to 0 (keep them on-screen permanently until deleted)
        client (BulletClient, optional): If connecting to multiple physics servers, include the client
            (the class instance, not just the ID) here. Defaults to None (use default connected client)

    Returns:
        list[int]: Pybullet IDs of the lines added to the GUI
    """
    client: pybullet = pybullet if client is None else client
    positions = np.atleast_2d(positions)
    n_positions, dim = positions.shape
    assert dim == 3
    # If desired, sample frames evenly across the trajectory to plot a subset
    if n is not None and n < n_positions:
        # This indexing ensures that the first and last frames are plotted
        idx = np.round(np.linspace(0, n_positions - 1, n, endpoint=True)).astype(int)
        positions = positions[idx, :]
    ids = []
    for i in range(positions.shape[0] - 1):
        ids.append(
            client.addUserDebugLine(
                positions[i], positions[i + 1], color, width, lifetime
            )
        )
    return ids


def animate_path(
    positions: npt.ArrayLike,
    duration: float,
    n: Optional[int] = None,
    color: npt.ArrayLike = (1, 1, 1),
    size: float = 20,
    client: Optional[BulletClient] = None,
):
    """Animates a point moving along a sequence of positions

    Args:
        positions (npt.ArrayLike): Path to animate, shape (n, 3)
        duration (float): Desired duration of the animation
        n (Optional[int]): Number of points to use in the animation, if using all of the provided positions will be too
            slow. Defaults to None (animate all points)
        color (npt.ArrayLike): RGB values, each in range [0, 1]. Shape (3,) if specifying the same color for all points,
            or (n, 3) to individually specify the colors per-point
        size (float): Size of the points on the GUI, in pixels. Defaults to 20
        client (BulletClient, optional): If connecting to multiple physics servers, include the client
            (the class instance, not just the ID) here. Defaults to None (use default connected client)
    """
    client: pybullet = pybullet if client is None else client
    # Pybullet will crash if you try to visualize one point without packing it into a 2D array
    positions = np.atleast_2d(positions)
    n_positions, dim = positions.shape
    if dim != 3:
        raise ValueError(
            f"Invalid shape of the point positions. Expected (n, 3), got: {positions.shape}"
        )
    color = np.atleast_2d(color)
    if color.shape[-1] != 3:
        raise ValueError(
            f"Invalid shape of the colors. Expected (n, 3), got: {color.shape}"
        )
    if color.shape[0] != n_positions:
        if color.shape[0] == 1:
            # Map the same color to all of the points
            color = color * np.ones_like(positions)
        else:
            raise ValueError(
                f"Number of colors ({color.shape[0]}) does not match the number of points ({n_positions})."
            )
    # Downsample the points if desired
    if n is not None and n < n_positions:
        # This indexing ensures that the first and last frames are plotted
        idx = np.round(np.linspace(0, n_positions - 1, n, endpoint=True)).astype(int)
        positions = positions[idx, :]
        color = color[idx, :]
        n_positions = n
    uid = None
    for i in range(n_positions):
        start_time = time.time()
        if uid is None:
            uid = client.addUserDebugPoints([positions[i]], [color[i]], size, 0)
        else:
            uid = client.addUserDebugPoints(
                [positions[i]], [color[i]], size, 0, replaceItemUniqueId=uid
            )
        client.stepSimulation()
        elapsed_time = time.time() - start_time
        time.sleep(max(0, duration / n_positions - elapsed_time))


def animate_rotation(
    quats: npt.ArrayLike,
    duration: float,
    object_id: Optional[int] = None,
    client: Optional[BulletClient] = None,
):
    """Animates an object rotating via a sequence of quaternions

    Args:
        quats (npt.ArrayLike): Quaternions to animate, shape (n, 4)
        duration (float): Desired duration of the animation
        object_id (Optional[int]): If you would like to animate the rotation using a specific object, pass in its
            Pybullet ID here. Defaults to None (use a cube as the default object).
        client (BulletClient, optional): If connecting to multiple physics servers, include the client
            (the class instance, not just the ID) here. Defaults to None (use default connected client)
    """
    client: pybullet = pybullet if client is None else client
    if object_id is None:
        object_id = create_box(
            (0, 0, 0), (0, 0, 0, 1), 1, (1, 1, 1), True, client=client
        )
    pos = client.getBasePositionAndOrientation(object_id)[0]
    quats = np.atleast_2d(quats)
    n = quats.shape[0]
    for quat in quats:
        start_time = time.time()
        pybullet.resetBasePositionAndOrientation(object_id, pos, quat)
        pybullet.stepSimulation()
        elapsed_time = time.time() - start_time
        time.sleep(max(0, duration / n - elapsed_time))


def remove_debug_objects(
    ids: Union[int, list[int], np.ndarray[int]], client: Optional[BulletClient] = None
) -> None:
    """Removes user-created line(s)/point(s)/etc. from the Pybullet GUI

    Args:
        ids (int or list/array of ints): ID(s) of the objects loaded into Pybullet
        client (BulletClient, optional): If connecting to multiple physics servers, include the client
            (the class instance, not just the ID) here. Defaults to None (use default connected client)
    """
    client: pybullet = pybullet if client is None else client
    if np.ndim(ids) == 0:  # Scalar, not iterable
        client.removeUserDebugItem(ids)
        return
    for i in ids:
        client.removeUserDebugItem(i)


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
        tuple[float, float, float, np.ndarray]:
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


def _main():
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


if __name__ == "__main__":
    _main()
