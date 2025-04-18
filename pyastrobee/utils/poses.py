"""Class/functions for managing poses and conversions between different representations

Our default representation of a pose will be position (xyz) + quaternion (xyzw)
"""

import numpy as np
import numpy.typing as npt
import pytransform3d.trajectories as pt

from pyastrobee.utils import rotations as rts
from pyastrobee.utils import transformations as tfs
from pyastrobee.utils import quaternions as qts


def check_pos_euler_xyz(pose: npt.ArrayLike) -> bool:
    """Checks to see if a position + Euler XYZ pose is valid

    Args:
        pose (npt.ArrayLike): Position + Euler XYZ pose, shape = (6,)

    Returns:
        bool: Whether or not the pose is valid
    """
    return len(pose) == 6


def check_pos_quat(pose: npt.ArrayLike) -> bool:
    """Checks to see if a position + XYZW quaternion pose is valid

    Args:
        pose (npt.ArrayLike): Position + XYZW quaternion pose

    Returns:
        bool: Whether or not the pose is valid
    """
    return len(pose) == 7


def pos_euler_xyz_to_tmat(pose: npt.ArrayLike) -> np.ndarray:
    """Converts a position + Euler pose to a transformation matrix

    Args:
        pose (npt.ArrayLike): Position + Euler XYZ pose, shape = (6,)

    Returns:
        np.ndarray: Transformation matrix, shape (4,4)
    """
    if not check_pos_euler_xyz(pose):
        raise ValueError(f"Invalid position + euler pose.\nGot: {pose}")
    pos = pose[:3]
    orn = pose[3:]
    rmat = rts.euler_xyz_to_rmat(orn)
    return tfs.make_transform_mat(rmat, pos)


def pos_euler_xyz_to_pos_quat(pose: npt.ArrayLike) -> np.ndarray:
    """Converts a position + Euler pose to a position + XYZW quaternion pose

    Args:
        pose (npt.ArrayLike): Position + Euler XYZ pose, shape = (6,)

    Returns:
        np.ndarray: Position + XYZW quaternion pose, shape = (7,)
    """
    if not check_pos_euler_xyz(pose):
        raise ValueError(f"Invalid position + euler pose.\nGot: {pose}")
    pos = pose[:3]
    orn = pose[3:]
    quat = rts.euler_xyz_to_quat(orn)
    return np.array([*pos, *quat])


def tmat_to_pos_euler_xyz(tmat: np.ndarray) -> np.ndarray:
    """Converts a transformation matrix to a position + Euler pose

    Args:
        pose (npt.ArrayLike): Transformation matrix, shape (4,4)

    Returns:
        np.ndarray: Position + Euler XYZ pose, shape = (6,)
    """
    if not tfs.check_transform_mat(tmat):
        raise ValueError(f"Invalid transformation matrix.\nGot: {tmat}")
    rmat = tmat[:3, :3]
    pos = tmat[:3, 3]
    orn = rts.rmat_to_euler_xyz(rmat)
    return np.array([*pos, *orn])


def tmat_to_pos_quat(tmat: np.ndarray) -> np.ndarray:
    """Converts a transformation matrix to a position + XYZW quaternion pose

    Args:
        pose (npt.ArrayLike): Transformation matrix, shape (4,4)

    Returns:
        np.ndarray: Position + XYZW quaternion pose, shape = (7,)
    """
    if not tfs.check_transform_mat(tmat):
        raise ValueError(f"Invalid transformation matrix.\nGot: {tmat}")
    rmat = tmat[:3, :3]
    pos = tmat[:3, 3]
    quat = rts.rmat_to_quat(rmat)
    return np.array([*pos, *quat])


def pos_quat_to_tmat(pose: npt.ArrayLike) -> np.ndarray:
    """Converts a position + XYZW quaternion pose to a transformation matrix

    Args:
        pose (npt.ArrayLike): Position + XYZW quaternion pose, shape = (7,)

    Returns:
        np.ndarray: Transformation matrix, shape (4,4)
    """
    if not check_pos_quat(pose):
        raise ValueError(f"Invalid position + quaternion pose.\nGot: {pose}")
    pos = pose[:3]
    quat = pose[3:]
    rmat = rts.quat_to_rmat(quat)
    return tfs.make_transform_mat(rmat, pos)


def batched_pos_quats_to_tmats(poses: npt.ArrayLike) -> np.ndarray:
    """Converts a array of position + quaternion poses to an array of transformation matrices

    Args:
        poses (npt.ArrayLike): Position + quaternion poses, shape (n, 7)

    Returns:
        np.ndarray: Transformation matrices, shape (n, 4, 4)
    """
    # Assume poses is of shape (n, 7). If not, see if we can fix it, or raise an error
    poses = np.atleast_2d(poses)
    n_rows, n_cols = poses.shape
    if n_cols != 7 and n_rows == 7:
        print("Warning: you might have passed in the transpose of the pose array")
        poses = poses.T
    elif n_cols != 7 and n_rows != 7:
        raise ValueError(
            f"Invalid input shape: {poses.shape} Must be an array of position/quaternion poses"
        )
    # Convert XYZW poses to WXYZ for pytransform3d's quaternion convention
    wxyz_pqs = np.zeros_like(poses)
    wxyz_pqs[:, :3] = poses[:, :3]  # x, y, z
    wxyz_pqs[:, 3] = poses[:, -1]  # qw
    wxyz_pqs[:, 4:] = poses[:, 3:-1]  # qx, qy, qz
    # Use the batched conversion from pytransform3d since this is more efficient than a loop
    return pt.transforms_from_pqs(wxyz_pqs)


def pos_quat_to_pos_euler_xyz(pose: npt.ArrayLike) -> np.ndarray:
    """Converts a position + XYZW quaternion pose to a position + Euler pose

    Args:
        pose (npt.ArrayLike): Position + XYZW quaternion pose, shape = (7,)

    Returns:
        np.ndarray: Position + Euler XYZ pose, shape = (6,)
    """
    if not check_pos_quat(pose):
        raise ValueError(f"Invalid position + quaternion pose.\nGot: {pose}")
    pos = pose[:3]
    quat = pose[3:]
    orn = rts.quat_to_euler_xyz(quat)
    return np.array([*pos, *orn])


def add_global_pose_delta(pose: npt.ArrayLike, pose_delta: npt.ArrayLike) -> np.ndarray:
    """Adds a world-frame "delta" to a pose

    Args:
        pose (npt.ArrayLike): Original reference pose (position + quaternion), shape (7,)
        pose_delta (npt.ArrayLike): Delta to add to the pose (position + quaternion), shape (7,)

    Returns:
        np.ndarray: Position + quaternion pose with the delta applied, shape (7,)
    """
    if not check_pos_quat(pose) or not check_pos_quat(pose_delta):
        raise ValueError(
            f"Invalid inputs: Not position/quaternion form.\nGot: {pose}\nAnd: {pose_delta}"
        )
    new_pos = pose[:3] + pose_delta[:3]
    new_orn = qts.combine_quaternions(pose[3:], pose_delta[3:])
    return np.array([*new_pos, *new_orn])


def add_local_pose_delta(pose: npt.ArrayLike, pose_delta: npt.ArrayLike) -> np.ndarray:
    """Adds a local (robot)-frame "delta" to a pose

    Args:
        pose (npt.ArrayLike): Original reference pose (position + quaternion), shape (7,)
        pose_delta (npt.ArrayLike): Delta to add to the pose (position + quaternion), shape (7,)

    Returns:
        np.ndarray: Position + quaternion pose with the delta applied, shape (7,)
    """
    if not check_pos_quat(pose) or not check_pos_quat(pose_delta):
        raise ValueError(
            f"Invalid inputs: Not position/quaternion form.\nGot: {pose}\nAnd: {pose_delta}"
        )
    T_R2W = pos_quat_to_tmat(pose)  # Robot to world
    T_D2R = pos_quat_to_tmat(pose_delta)  # Delta to robot
    T_D2W = T_R2W @ T_D2R  # Delta to world
    return tmat_to_pos_quat(T_D2W)


def pose_derivatives(
    poses: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculates the linear/angular first and second derivatives of a sequence of poses

    Args:
        poses (np.ndarray): Sequence of position + XYZW quaternion poses, shape (n, 7)
        dt (float): Timestep between poses, in seconds

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            np.ndarray: Linear velocities, shape (n, 3)
            np.ndarray: Angular velocities, shape (n, 3)
            np.ndarray: Linear accelerations, shape (n, 3)
            np.ndarray: Angular accelerations, shape (n, 3)
    """
    if poses.shape[-1] != 7:
        raise ValueError(
            f"Invalid pose array: must be shape (n, 7). Got: {poses.shape}"
        )
    positions = poses[:, :3]
    quaternions = poses[:, 3:]
    velocities = np.gradient(positions, dt, axis=0)
    accels = np.gradient(velocities, dt, axis=0)
    omegas = qts.quats_to_angular_velocities(quaternions, dt)
    alphas = np.gradient(omegas, dt, axis=0)
    return velocities, omegas, accels, alphas
