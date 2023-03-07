"""Functions associated with the Pybullet Debug Visualizer GUI

NOTE
- Since the debug visualizer only specifies yaw and pitch (no roll), this is an incomplete definition of rotation,
  so we can never actually fully change the rotation of the debug viz camera to match the robot's position
    - This is not a huge deal, it will just mean we will have at best a "video game perspective" of the robot
"""

import time
from typing import Optional

import numpy as np
import pybullet

from pyastrobee.control.astrobee import Astrobee
from pyastrobee.utils.rotations import (
    quat_to_fixed_xyz,
    fixed_xyz_to_quat,
    fixed_xyz_to_rmat,
)
from pyastrobee.utils.poses import pos_quat_to_tmat
from pyastrobee.utils.transformations import make_transform_mat
from pyastrobee.utils.coordinates import cartesian_to_spherical
from pyastrobee.config.astrobee_transforms import OBSERVATION_CAM


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
    pybullet.connect(pybullet.GUI)
    robot = Astrobee()
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
        pybullet.changeConstraint(robot.constraint_id, pos, orn)
        pybullet.stepSimulation()
        time.sleep(1 / 20)
