"""Trajectory generation methods for the multi-robot case"""

import numpy as np

from pyastrobee.trajectories.trajectory import Trajectory
from pyastrobee.utils.poses import pos_quat_to_tmat
from pyastrobee.utils.rotations import rmat_to_quat
from pyastrobee.utils.quaternions import quats_to_angular_velocities


def offset_trajectory(
    reference_traj: Trajectory, offset_transform: np.ndarray
) -> Trajectory:
    """Construct a trajectory with a fixed offset to a given reference trajectory

    This can also be thought as "Leader/Follower" where leader has the reference trajectory, and follower has the offset
    trajectory. However, this can generalize to more cases than just leader/follower

    Args:
        reference_traj (Trajectory): Trajectory to use as a reference
        offset_transform (np.ndarray): "Offset to reference" transformation matrix, dictating the positional/angular
            difference from the reference trajectory. Shape (4, 4)

    Returns:
        Trajectory: Trajectory with a fixed offset to the reference trajectory
    """
    # TODO: check that the transformation matrix is valid?

    # Transform each pose according to the offset transform
    offset_positions = np.zeros_like(reference_traj.positions)
    offset_quats = np.zeros_like(reference_traj.quaternions)
    for i in range(reference_traj.num_timesteps):
        pose = reference_traj.poses[i]
        T_R2W = pos_quat_to_tmat(pose)  # Reference to World
        T_O2W = T_R2W @ offset_transform  # Offset to World
        offset_positions[i] = T_O2W[:3, 3]
        offset_quats[i] = rmat_to_quat(T_O2W[:3, :3])

    # Take the gradients of the poses to determine derivative information
    offset_vels = np.gradient(offset_positions, reference_traj.times, axis=0)
    offset_accels = np.gradient(offset_vels, reference_traj.times, axis=0)
    offset_omegas = quats_to_angular_velocities(
        offset_quats, np.gradient(reference_traj.times)
    )
    offset_alphas = np.gradient(offset_omegas, reference_traj.times, axis=0)
    return Trajectory(
        offset_positions,
        offset_quats,
        offset_vels,
        offset_omegas,
        offset_accels,
        offset_alphas,
        reference_traj.times,
    )


def multi_trajectory(
    reference_traj: Trajectory, transforms: list[np.ndarray]
) -> list[Trajectory]:
    """Construct multiple trajectories moving about a reference trajectory with fixed offset transformations

    For instance: This can be used as a slightly better formulation of leader/follower to ensure similar trajectory
    dynamics for two robots moving together about a reference

    Args:
        reference_traj (Trajectory): Trajectory to use as a central reference trajectory
        transforms (list[np.ndarray]): "Offset to reference" transformation matrices for each trajectory

    Returns:
        list[Trajectory]: The trajectories, each offset about the reference
    """
    return [offset_trajectory(reference_traj, T) for T in transforms]
