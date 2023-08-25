"""The main trajectory planning structure for Astrobee

We use a global/local hierarchy where: 

- The global planner computes a reference trajectory which enforces all of the required constraints, such as Astrobee's
  speed/acceleration limits, as well as collision avoidance
- The local planner handles very fast trajectory generation between states with arbitrary boundary conditions on 
  dynamics, but does not enforce some of the more computationally intensive optimizations and constraints

This also assumes decoupled dynamics between the position and orientation components

Note: the orientation component of the trajectories technically isn't "optimal", but it works

TODO add the max-angular-velocity-constrained rotation planner into the global planner (would only really be needed if
we have a very short time horizon on the position component, i.e. if we're not moving much but rotating a lot)
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
from pyastrobee.trajectories.curve_utils import traj_from_curve
from pyastrobee.trajectories.variable_time_curves import (
    free_final_time_bezier,
    free_final_time_spline,
)


def local_planner(
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

    Note that this planner prioritizes solve time over constraint enforcement. This will plan with boundary conditions
    in mind, but other than that, it does not enforce things like speed/acceleration limits or collision avoidance. This
    allows us to use this in very fast online applications.

    In general, this should be paired with the global planner, so that we can precompute a reference trajectory which
    does enforce all constraints (and takes a bit longer to solve), and use this local planner to evaluate different
    rollouts between states along this reference trajectory

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
        Trajectory: The optimal local trajectory plan
    """
    # Min-jerk position traj
    # Don't need a ton of control points because we're not enforcing constraints
    n_control_pts = 8
    curve, _ = bezier_trajectory(
        p0, pf, 0, duration, n_control_pts, v0, vf, a0, af, time_weight=0
    )
    pos_traj = traj_from_curve(curve, dt)
    n_timesteps = len(pos_traj.times)
    quats = quaternion_interpolation_with_bcs(
        q0, qf, w0, wf, dw0, dwf, duration, n_timesteps
    )
    omega = quats_to_angular_velocities(quats, dt)
    alpha = np.gradient(omega, dt, axis=0)
    return Trajectory(
        positions=pos_traj.positions,
        quats=quats,
        lin_vels=pos_traj.linear_velocities,
        ang_vels=omega,
        lin_accels=pos_traj.linear_accels,
        ang_accels=alpha,
        times=pos_traj.times,
    )


def global_planner(
    p0: npt.ArrayLike,
    q0: npt.ArrayLike,
    pf: npt.ArrayLike,
    qf: npt.ArrayLike,
    dt: float,
    all_boxes: dict[str, Box] = ALL_BOXES,
    graph: Optional[dict[str, list[str]]] = GRAPH,
) -> Trajectory:
    """Generate an optimal spline-based position trajectory with a polynomial orientation component

    This will enforce all of the Astrobee's constraints (such as maximum velocity/acceleration and collision avoidance).
    It will also be time-optimal in the sense that 1) The total duration of the curve is minimized, and 2) each Bezier
    curve in the spline will have its relative duration (w.r.t. the total duration) optimized to minimize jerk

    Note that this can take on the order of 10 seconds to compute, so it is not something that should be called online

    Args:
        p0 (npt.ArrayLike): Initial position, shape (3,)
        q0 (npt.ArrayLike): Initial XYZW quaternion, shape (4,)
        pf (npt.ArrayLike): Final position, shape (3,)
        qf (npt.ArrayLike): Final XYZW quaternion, shape (4,)
        dt (float): Sampling period (seconds)
        all_boxes (dict[str, Box], optional): Description of the safe set of the environment. Defaults to ALL_BOXES
            (the precomputed safe-set decomposition for the ISS)
        graph (Optional[dict[str, list[str]]], optional): The graph defining the connections between the safe boxes in
            the environment. Defaults to GRAPH (the precomputed safe-set graph for the ISS). Note: if set to None, this
            graph will be recomputed from the all_boxes parameter

    Returns:
        Trajectory: The optimal global trajectory plan
    """
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
    # Parameters associated with the retiming optimization or the free-final-time optimization
    # will be left at the default values for now

    # Determine the path through the safe-space graph
    start = find_containing_box_name(p0, all_boxes)
    end = find_containing_box_name(pf, all_boxes)
    if graph is None:
        graph = compute_graph(all_boxes)
    path = dfs(graph, start, end)
    box_path = [all_boxes[p] for p in path]

    init_angular_duration = rotation_duration_heuristic(q0, qf)
    init_pos_duration, init_timing_fractions = spline_duration_heuristic(
        p0, pf, box_path
    )
    # TODO enforce a minimum final time in the free-final-time optimization
    # to make sure that the rotation plan still has enough time to execute
    duration = max(init_pos_duration, init_angular_duration)
    init_curve_durations = duration * init_timing_fractions

    if len(box_path) == 1:
        # Crank up the control points for a single Bezier curve so that we can be nice and tight
        # against the velocity/accel constraints
        n_control_pts = 20

        # If our start and end positions are contained within the same free box,
        # we can construct the trajectory from a single Bezier curve
        curve = free_final_time_bezier(
            p0,
            pf,
            t0,
            duration,
            n_control_pts,
            v0,
            vf,
            a0,
            af,
            box_path[0],
            v_max,
            a_max,
        )
    else:
        # Our start/end positions are not in the same box, so use a spline between boxes
        # Use less control points for the spline for optimization stability
        n_control_pts = 10
        curve = free_final_time_spline(
            p0,
            pf,
            t0,
            duration,
            n_control_pts,
            box_path,
            init_curve_durations,
            v0,
            vf,
            a0,
            af,
            v_max,
            a_max,
        )
    pos_traj = traj_from_curve(curve, dt)
    n_timesteps = pos_traj.num_timesteps
    quats = quaternion_interpolation_with_bcs(
        q0, qf, w0, wf, dw0, dwf, duration, n_timesteps
    )
    omega = quats_to_angular_velocities(quats, dt)
    alpha = np.gradient(omega, dt, axis=0)
    return Trajectory(
        pos_traj.positions,
        quats,
        pos_traj.linear_velocities,
        omega,
        pos_traj.linear_accels,
        alpha,
        pos_traj.times,
    )
