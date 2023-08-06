import numpy as np
import numpy.typing as npt

from pyastrobee.trajectories.trajectory import Trajectory
from pyastrobee.utils.rotations import axis_angle_between_two_vectors
from pyastrobee.utils.quaternions import (
    get_closest_heading_quat,
    quaternion_angular_error,
    quats_to_angular_velocities,
)
from pyastrobee.utils.math_utils import normalize
from pyastrobee.trajectories.polynomials import polynomial_slerp
from pyastrobee.trajectories.quaternion_interpolation import (
    quaternion_interpolation_with_bcs,
)
from pyastrobee.config.astrobee_motion import ANGULAR_SPEED_LIMIT


def heading_alignment_orn_traj(
    q0: npt.ArrayLike,
    w0: npt.ArrayLike,
    dw0: npt.ArrayLike,
    heading: npt.ArrayLike,
    dt: float,
):
    qf = get_closest_heading_quat(q0, heading)
    # Get a timing heuristic based on the quaternion error
    # (not guaranteed to satisfy constraints)
    err = quaternion_angular_error(q0, qf)
    w_nom = 0.5 * ANGULAR_SPEED_LIMIT
    alignment_time = np.linalg.norm(err) / w_nom
    num_alignment_timesteps = int(alignment_time // dt)
    return quaternion_interpolation_with_bcs(
        q0,
        qf,
        w0,
        np.zeros(3),
        dw0,
        np.zeros(3),
        alignment_time,
        num_alignment_timesteps,
    )


def face_forward_traj(
    path: np.ndarray,
    q0: npt.ArrayLike,
    w0: npt.ArrayLike,
    dw0: npt.ArrayLike,
    dt: float,
) -> Trajectory:
    # If the initial heading is not aligned with the start of the path,
    # we have to rotate to get to that point
    # TODO handle if the initial heading IS aligned? heading quat will be identity
    init_heading = path[1] - path[0]
    err_tol = 1e-2
    if (
        np.linalg.norm(
            quaternion_angular_error(q0, get_closest_heading_quat(q0, init_heading))
        )
        > err_tol
    ):
        quats_1 = heading_alignment_orn_traj(q0, w0, dw0, init_heading, dt)
    else:
        quats_1 = np.atleast_2d(q0)
    # Find the rotation to change the heading at each point in the trajectory
    n = path.shape[0]
    cur_q = quats_1[-1]
    quats_2 = np.empty((path.shape[0], 4))
    for i in range(n - 1):
        heading = path[i + 1] - path[i]
        quats_2[i] = get_closest_heading_quat(cur_q, heading)
    quats_2[-1] = quats_2[-2]
    all_quats = np.vstack((quats_1, quats_2))
    all_positions = np.vstack((np.tile(path[0], (quats_1.shape[0], 1)), path))
    omega = quats_to_angular_velocities(all_quats, dt)
    alpha = np.gradient(omega, dt, axis=0)
    velocities = np.gradient(all_positions, dt, axis=0)
    accels = np.gradient(velocities, dt, axis=0)
    total_n = all_positions.shape[0]
    times = np.linspace(0, dt * total_n, total_n, endpoint=True)
    return Trajectory(all_positions, all_quats, velocities, omega, accels, alpha, times)
