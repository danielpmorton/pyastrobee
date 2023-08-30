"""Methods for sampling candidate trajectories about a reference state or trajectory"""

# TODO make a method that uses different reward weighting in the trajectory optimization (e.g. different weighting
# between minimizing jerk and minimizing pathlength)

import numpy as np
import numpy.typing as npt

from pyastrobee.utils.math_utils import spherical_vonmises_sampling
from pyastrobee.trajectories.trajectory import Trajectory
from pyastrobee.trajectories.planner import local_planner


# TODO
# - Decide if we should be passing in covariance matrices or arrays instead of scalars
# - Decide if the "orientation stdev" should be replaced by the von Mises kappa parameter
def generate_trajs(
    cur_pos: npt.ArrayLike,
    cur_orn: npt.ArrayLike,
    cur_vel: npt.ArrayLike,
    cur_ang_vel: npt.ArrayLike,
    cur_accel: npt.ArrayLike,  # Optional?
    cur_alpha: npt.ArrayLike,  # Optional?
    nominal_target_pos: npt.ArrayLike,
    nominal_target_orn: npt.ArrayLike,
    nominal_target_vel: npt.ArrayLike,
    nominal_target_ang_vel: npt.ArrayLike,
    nominal_target_accel: npt.ArrayLike,  # Optional?
    nominal_target_alpha: npt.ArrayLike,  # Optional?
    pos_sampling_stdev: float,
    orn_sampling_stdev: float,
    vel_sampling_stdev: float,
    ang_vel_sampling_stdev: float,
    accel_sampling_stdev: float,
    alpha_sampling_stdev: float,
    n_trajs: int,
    duration: float,
    dt: float,
    include_nominal_traj: bool,
) -> list[Trajectory]:
    """Generate a number of trajectories from the current state to a sampled state about a nominal target

    Args:
        cur_pos (npt.ArrayLike): Current position, shape (3,)
        cur_orn (npt.ArrayLike): Current XYZW quaternion orientation, shape (4,)
        cur_vel (npt.ArrayLike): Current linear velocity, shape (3,)
        cur_ang_vel (npt.ArrayLike): Current angular velocity, shape (3,)
        nominal_target_pos (npt.ArrayLike): Nominal desired position to sample about, shape (3,)
        nominal_target_orn (npt.ArrayLike): Nominal desired XYZW quaternion to sample about, shape (4,)
        nominal_target_vel (npt.ArrayLike): Nominal desired linear velocity to sample about, shape (3,)
        nominal_target_ang_vel (npt.ArrayLike): Nominal desired angular velocity to sample about, shape (3,)
        pos_sampling_stdev (float): Standard deviation of the position sampling distribution
        orn_sampling_stdev (float): Standard deviation of the orientation sampling distribution
        vel_sampling_stdev (float): Standard deviation of the velocity sampling distribution
        ang_vel_sampling_stdev (float): Standard deviation of the angular velocity sampling distribution
        n_trajs (int): Number of trajectories to generate
        duration (float): Trajectory duration, in seconds
        dt (float): Timestep
        include_nominal_traj (bool): Whether or not to include the nominal (non-sampled) trajectory in the output

    Returns:
        list[Trajectory]: Sampled trajectories, length n_trajs
    """
    trajs = []
    if include_nominal_traj:
        # Let the first generated trajectory use the mean of all of the distributions
        trajs.append(
            local_planner(
                cur_pos,
                cur_orn,
                cur_vel,
                cur_ang_vel,
                cur_accel,
                cur_alpha,
                nominal_target_pos,
                nominal_target_orn,
                nominal_target_vel,
                nominal_target_ang_vel,
                nominal_target_accel,
                nominal_target_alpha,
                duration,
                dt,
            )
        )
        # Reduce the number of trajectories to sample since we have added this nominal traj
        n_samples = n_trajs - 1
    else:
        # Sample all of the trajectories
        n_samples = n_trajs

    if n_samples == 0:
        return trajs

    # Sample endpoints for the candidate trajectories about the nominal targets
    sampled_positions = np.random.multivariate_normal(
        nominal_target_pos, pos_sampling_stdev**2 * np.eye(3), n_samples
    )
    sampled_quats = spherical_vonmises_sampling(
        nominal_target_orn, 1 / (orn_sampling_stdev**2), n_samples
    )
    sampled_vels = np.random.multivariate_normal(
        nominal_target_vel, vel_sampling_stdev**2 * np.eye(3), n_samples
    )
    sampled_ang_vels = np.random.multivariate_normal(
        nominal_target_ang_vel, ang_vel_sampling_stdev**2 * np.eye(3), n_samples
    )
    sampled_accels = np.random.multivariate_normal(
        nominal_target_accel, accel_sampling_stdev**2 * np.eye(3), n_samples
    )
    sampled_alphas = np.random.multivariate_normal(
        nominal_target_alpha, alpha_sampling_stdev**2 * np.eye(3), n_samples
    )
    for i in range(n_samples):
        trajs.append(
            local_planner(
                cur_pos,
                cur_orn,
                cur_vel,
                cur_ang_vel,
                cur_accel,
                cur_alpha,
                sampled_positions[i],
                sampled_quats[i],
                sampled_vels[i],
                sampled_ang_vels[i],
                sampled_accels[i],
                sampled_alphas[i],
                duration,
                dt,
            )
        )
    return trajs
