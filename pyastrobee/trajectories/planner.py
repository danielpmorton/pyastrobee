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
from pyastrobee.trajectories.timing import (
    spline_duration_heuristic,
    rotation_duration_heuristic,
)
from pyastrobee.config.astrobee_motion import LINEAR_SPEED_LIMIT, LINEAR_ACCEL_LIMIT
from pyastrobee.config.iss_safe_boxes import ALL_BOXES
from pyastrobee.config.iss_paths import GRAPH, PATHS
from pyastrobee.utils.boxes import Box, find_containing_box_name, compute_graph
from pyastrobee.utils.algos import dfs
from pyastrobee.trajectories.splines import spline_trajectory_with_retiming


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


# WORK IN PROGRESS
# TODO handle the timing better...
def global_planner(
    p0: npt.ArrayLike,
    q0: npt.ArrayLike,
    pf: npt.ArrayLike,
    qf: npt.ArrayLike,
    dt: float,
    all_boxes: dict[str, Box] = ALL_BOXES,
    graph: Optional[dict[str, list[str]]] = GRAPH,
):
    # Dynamics parameters: Assume start and end from rest, satisfy operating limits
    t0 = 0
    v0 = np.zeros(3)
    vf = np.zeros(3)
    a0 = np.zeros(3)
    af = np.zeros(3)
    w0 = np.zeros(3)
    wf = np.zeros(3)
    dw0 = np.zeros(3)
    dwf = np.zeros(3)
    v_max = LINEAR_SPEED_LIMIT
    a_max = LINEAR_ACCEL_LIMIT
    # TODO add some way to know that we enforce angular speed/accel limits
    # Retiming parameters (These are currently the same as the default)
    kappa_min = 0.01
    omega = 3
    max_retiming_iters = 10
    # Curve parameters
    pts_per_curve = 7

    # Determine the path through the safe-space graph
    start = find_containing_box_name(p0, all_boxes)
    end = find_containing_box_name(pf, all_boxes)
    if graph is None:
        graph = compute_graph(all_boxes)
    path = dfs(graph, start, end)
    box_path = [all_boxes[p] for p in path]

    if len(box_path) == 1:
        # Construct from a single bezier curve
        # Can probably use the local planner here
        # Use iterative method for retiming?? or make another function
        # like bezier_trajector_with_retiming
        pass
    init_angular_duration = rotation_duration_heuristic(q0, qf)
    init_pos_duration, init_timing_fractions = spline_duration_heuristic(
        p0, pf, box_path
    )
    duration = max(init_pos_duration, init_angular_duration)
    init_curve_durations = duration * init_timing_fractions

    # times_ = np.arange(0, duration + dt, dt)  # DO WE NEED THIS?
    # n_timesteps = len(times_)
    n_timesteps = int(np.ceil((duration + dt) / dt))

    pos, vel, accel, times = spline_trajectory_with_retiming(
        p0,
        pf,
        t0,
        duration,
        n_timesteps,
        pts_per_curve,
        box_path,
        init_curve_durations,
        v0,
        vf,
        a0,
        af,
        v_max,
        a_max,
        kappa_min,
        omega,
        max_retiming_iters,
    )
    quats = quaternion_interpolation_with_bcs(
        q0, qf, w0, wf, dw0, dwf, duration, n_timesteps
    )
    omega = quats_to_angular_velocities(quats, dt)
    alpha = np.gradient(omega, dt, axis=0)
    return Trajectory(pos, quats, vel, omega, accel, alpha, times)


def local_planner():
    pass
