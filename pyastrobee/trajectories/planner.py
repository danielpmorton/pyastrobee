"""Current trajectory generation method

We have different methods for generating trajectories (position / orientation), but
this will combine the current best position + orientation methods

This specific method (bezier curves for position, and quaternion polynomials for orientation)
is not "optimal" in the orientation component, so if I figure this out, I'll update the function
"""

from typing import Optional

import numpy as np
import numpy.typing as npt

from pyastrobee.trajectories.trajectory import Trajectory
from pyastrobee.trajectories.bezier import bezier_trajectory
from pyastrobee.trajectories.quaternion_interpolation import (
    quaternion_interpolation_with_bcs,
)
from pyastrobee.utils.quaternions import quats_to_angular_velocities


def plan_trajectory(
    p0: npt.ArrayLike,
    q0: npt.ArrayLike,
    v0: Optional[npt.ArrayLike],
    w0: npt.ArrayLike,
    a0: Optional[npt.ArrayLike],
    dw0: npt.ArrayLike,
    pf: npt.ArrayLike,
    qf: npt.ArrayLike,
    vf: Optional[npt.ArrayLike],
    wf: npt.ArrayLike,
    af: Optional[npt.ArrayLike],
    dwf: npt.ArrayLike,
    duration: float,
    dt: float,
) -> Trajectory:
    """Generate an optimal Bezier-curve-based position trajectory with a polynomial orientation component

    Args:
        p0 (npt.ArrayLike): Initial position, shape (3,)
        q0 (npt.ArrayLike): Initial XYZW quaternion, shape (4,)
        v0 (npt.ArrayLike): Initial linear velocity, shape (3,)
        w0 (npt.ArrayLike): Initial angular velocity, shape (3,)
        a0 (npt.ArrayLike): Initial linear acceleration, shape (3,)
        dw0 (npt.ArrayLike): Initial angular acceleration, shape (3,)
        pf (npt.ArrayLike): Final position, shape (3,)
        qf (npt.ArrayLike): Final XYZW quaternion, shape (4,)
        vf (npt.ArrayLike): Final linear velocity, shape (3,)
        wf (npt.ArrayLike): Final angular velocity, shape (3,)
        af (npt.ArrayLike): Final linear acceleration, shape (3,)
        dwf (npt.ArrayLike): Final angular acceleration, shape (3,)
        duration (float): Trajectory duration (seconds)
        dt (float): Sampling period (seconds)

    Returns:
        Trajectory: Trajectory with position, orientation, lin/ang velocity, lin/ang acceleration, and time info
    """
    times = np.arange(0, duration + dt, dt)
    n = len(times)
    t0 = times[0]
    tf = times[-1]
    # Min-jerk position traj
    n_control_pts = 8
    pos, vel, accel, _ = bezier_trajectory(
        p0, pf, t0, tf, n_control_pts, n, v0, vf, a0, af, jerk_weight=1
    )
    quats = quaternion_interpolation_with_bcs(q0, qf, w0, wf, dw0, dwf, duration, n)
    omega = quats_to_angular_velocities(quats, dt)
    alpha = np.gradient(omega, dt, axis=0)
    return Trajectory(
        positions=pos,
        quats=quats,
        lin_vels=vel,
        ang_vels=omega,
        lin_accels=accel,
        ang_accels=alpha,
        times=times,
    )
