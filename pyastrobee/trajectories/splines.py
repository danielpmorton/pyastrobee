"""Splines / "composite Bezier curves" composed of multiple chained Bezier curves

Initial boundary conditions are imposed on the start of the first curve, final B.C.s are imposed at the end of
the third curve, and we enforce C2 derivative continuity at the knot points

TODO
- Return Trajectory class, or just the positions/velocites/accels/times....?

Reference:
Fast Path Planning Through Large Collections of Safe Boxes
https://github.com/cvxgrp/fastpathplanning/blob/main/fastpathplanning/smooth.py
https://web.stanford.edu/~boyd/papers/pdf/fpp.pdf
"""

from typing import Optional, Union, Any

import cvxpy as cp
import numpy as np
import numpy.typing as npt
import pybullet

from pyastrobee.trajectories.bezier import BezierCurve
from pyastrobee.utils.boxes import Box, visualize_3D_box
from pyastrobee.utils.debug_visualizer import visualize_path, visualize_points
from pyastrobee.utils.python_utils import print_red
from pyastrobee.trajectories.timing import retiming
from pyastrobee.utils.errors import OptimizationError


class CompositeBezierCurve:
    """Composite Bezier curve class for a continuous chain of connected Bezier curves

    Args:
        beziers (list[BezierCurve]): Consecutive Bezier curves composing the composite curve.
    """

    def __init__(self, beziers: list[BezierCurve]):
        for bez1, bez2 in zip(beziers[:-1], beziers[1:]):
            assert bez1.b == bez2.a
            assert bez1.d == bez2.d

        self.beziers = beziers
        self.N = len(self.beziers)
        self.d = beziers[0].d
        self.a = beziers[0].a
        self.b = beziers[-1].b
        self.duration = self.b - self.a
        self.transition_times = [self.a] + [bez.b for bez in beziers]

    def find_segment(self, t):
        # return min(bisect(self.transition_times, t) - 1, self.N - 1)
        # TODO check to see if this will work on a cp variable (probably not...)
        return np.minimum(
            np.searchsorted(self.transition_times, t, "right") - 1, self.N - 1
        )

    def __call__(self, t: Union[float, np.ndarray]) -> np.ndarray:
        """Evaluates the composite Bezier curve at specified points, by calling each of the child curves

        Args:
            t (Union[float, npt.ArrayLike]): Evaluation points (for instance, trajectory times)

        Returns:
            np.ndarray: Points along the composite curve, shape (n_pts, dimension)
        """
        seg_map = self.find_segment(t)
        evals = []
        for i in range(self.N):
            evals.append(self.beziers[i](t[seg_map == i]))
        return np.row_stack(evals)

    @property
    def start_point(self):
        """Starting point of the composite Bezier curve (the first point of the first curve)"""
        return self.beziers[0].start_point

    @property
    def end_point(self):
        """Ending point of the composite Bezier curve (the last point of the last curve)"""
        return self.beziers[-1].end_point

    @property
    def derivative(self):
        """Derivative of the composite Bezier curve (a composite Bezier curve of derivative curves)"""
        return CompositeBezierCurve([b.derivative for b in self.beziers])

    @property
    def l2_squared_sum(self) -> Union[float, cp.Expression]:
        """Sum of the squared L2 norm for all curves"""
        return sum(bez.l2_squared for bez in self.beziers)


def spline_trajectory_with_retiming(
    p0: npt.ArrayLike,
    pf: npt.ArrayLike,
    t0: float,
    tf: float,
    pts_per_curve: int,
    boxes: list[Box],
    initial_durations: npt.ArrayLike,
    v0: Optional[npt.ArrayLike] = None,
    vf: Optional[npt.ArrayLike] = None,
    a0: Optional[npt.ArrayLike] = None,
    af: Optional[npt.ArrayLike] = None,
    v_max: Optional[float] = None,
    a_max: Optional[float] = None,
    kappa_min: float = 1e-2,
    omega: float = 3,
    max_iters: int = 10,
    time_weight: float = 0,
) -> tuple[CompositeBezierCurve, float]:
    """Generate a min-jerk trajectory based on chained Bezier curves through a set of safe boxes, for a set time
    interval. Use an iterative retiming method to refine the durations for each segment of the trajectory

    We enforce continuity between curves up to the second derivative

    Note: if position is unconstrained (no need for safe boxes), you should instead use a single Bezier curve

    Args:
        p0 (npt.ArrayLike): Initial position, shape (3,)
        pf (npt.ArrayLike): Final position, shape (3,)
        t0 (float): Starting time
        tf (float): Ending time
        pts_per_curve (int): Number of control points per Bezier curve. Generally, should be around 6-10
        boxes (list[Box]): Sequential list of safe box regions pass through
        initial_durations (npt.ArrayLike): Initial estimate of the durations for each segment of the trajectory. These
            will be refined during the retiming process. Shape (num_boxes,)
        v0 (Optional[npt.ArrayLike]): Initial velocity, shape (3,). Defaults to None (unconstrained)
        vf (Optional[npt.ArrayLike]): Final velocity, shape (3,). Defaults to None (unconstrained)
        a0 (Optional[npt.ArrayLike]): Initial acceleration, shape (3,). Defaults to None (unconstrained)
        af (Optional[npt.ArrayLike]): Final acceleration, shape (3,). Defaults to None (unconstrained)
        v_max (Optional[float]): Maximum L2 norm of the velocity. Defaults to None (unconstrained)
        a_max (Optional[float]): Maximum L2 norm of the acceleration. Defaults to None (unconstrained)
        kappa_min (float, optional): Retiming trust region parameter: Defines the maximum change in adjacent scaling
            factors. Defaults to 1e-2.
        omega (float, optional): Retiming parameter: Defines the rate at which kappa decays after each iteration.
            Must be > 1. Small values (~2) work well when transition time estimates are poor, but larger values (~5)
            are more effective otherwise. Defaults to 3.
        max_iters (int, optional): Maximum number of iterations for the retiming process. Defaults to 10.
        time_weight (float, optional): Objective function weight corresponding to a linear penalty on the duration.
            Defaults to 0 (minimize jerk only). Note: this should be > 0 if evaluating the free-final-time case

    Returns:
        Tuple of:
            CompositeBezierCurve: The optimal curve for the position component of the trajectory. Note: derivatives
                can be evaluated using the curve.derivative property
            float: The optimal cost of the objective function
    """
    # Handle inputs
    if tf <= t0:
        raise ValueError(f"Invalid time interval: ({t0}, {tf})")
    if omega <= 1:
        raise ValueError("The retiming parameter omega must be > 1")
    if pts_per_curve < 6:
        print_red(
            "WARNING: Curves with less than 6 control points may lead to infeasible constraints"
        )

    # Store the curve parameters so that we can reuse these in the repeated calls to the spline function
    curve_kwargs = dict(
        p0=p0,
        pf=pf,
        t0=t0,
        tf=tf,
        pts_per_curve=pts_per_curve,
        boxes=boxes,
        v0=v0,
        vf=vf,
        a0=a0,
        af=af,
        v_max=v_max,
        a_max=a_max,
        time_weight=time_weight,
    )
    # Solve a preliminary trajectory for this initial guess at the durations
    durations = initial_durations
    curve_kwargs["durations"] = durations
    best_curve, best_info = _fixed_timing_spline(**curve_kwargs)

    kappa = 1  # Initialize the trust region parameter
    for _ in range(max_iters):
        # Retime.
        try:
            new_durations, kappa_max = retiming(
                kappa,
                best_info["cost_breakdown"],
                durations,
                best_info["retiming_weights"],
            )
        except OptimizationError:
            print_red(
                "Spline trajectory generation terminated due to failure to solve the retiming problem.\n"
                + "This can sometimes be due to a poor initialization of the curve durations, "
                + "or having a time horizon that is too short"
            )
            break

        # Improve Bezier curves based on the retiming
        curve_kwargs["durations"] = new_durations
        new_curve, new_info = _fixed_timing_spline(**curve_kwargs)
        print("Cost: ", new_info["cost"])
        if new_info["cost"] < best_info["cost"]:  # Accept trajectory with new durations
            durations = new_durations
            best_info = new_info
            best_curve = new_curve

        # Update trust region
        if kappa < kappa_min:
            break
        kappa = kappa_max / omega
    else:
        print_red(
            "Spline trajectory generation terminated due to reaching the maximum number of iterations"
        )

    return best_curve, best_info["cost"]


def _fixed_timing_spline(
    p0: npt.ArrayLike,
    pf: npt.ArrayLike,
    t0: float,
    tf: float,
    pts_per_curve: int,
    boxes: list[Box],
    durations: npt.ArrayLike,
    v0: Optional[npt.ArrayLike] = None,
    vf: Optional[npt.ArrayLike] = None,
    a0: Optional[npt.ArrayLike] = None,
    af: Optional[npt.ArrayLike] = None,
    v_max: Optional[float] = None,
    a_max: Optional[float] = None,
    time_weight: float = 0,
) -> tuple[CompositeBezierCurve, dict[str, Any]]:
    """Generate a min-jerk spline based on chained Bezier curves through a set of safe boxes, for a fixed time
    interval and fixed durations within each box

    NOTE: This is NOT globally optimal if we don't use the retiming process to refine the time breakdown, which is
    why this is a helper function that gets called within the trajectory-with-retiming function

    Args:
        p0 (npt.ArrayLike): Initial position, shape (3,)
        pf (npt.ArrayLike): Final position, shape (3,)
        t0 (float): Starting time
        tf (float): Ending time
        pts_per_curve (int): Number of control points per Bezier curve. Generally, should be around 6-10
        boxes (list[Box]): Sequential list of safe box regions pass through
        durations (npt.ArrayLike): Durations for each segment of the trajectory, shape (num_boxes,)
        v0 (Optional[npt.ArrayLike]): Initial velocity, shape (3,). Defaults to None (unconstrained)
        vf (Optional[npt.ArrayLike]): Final velocity, shape (3,). Defaults to None (unconstrained)
        a0 (Optional[npt.ArrayLike]): Initial acceleration, shape (3,). Defaults to None (unconstrained)
        af (Optional[npt.ArrayLike]): Final acceleration, shape (3,). Defaults to None (unconstrained)
        v_max (Optional[float]): Maximum L2 norm of the velocity. Defaults to None (unconstrained)
        a_max (Optional[float]): Maximum L2 norm of the acceleration. Defaults to None (unconstrained)
        time_weight (float, optional): Objective function weight corresponding to a linear penalty on the duration.
            Defaults to 0 (minimize jerk only). Note: this should be > 0 if evaluating the free-final-time case

    Returns:
        Tuple of:
            CompositeBezierCurve: The solved position curve
            dict[str, Any]: Solution info to use for retiming optimization
    """
    dim = len(p0)
    n_curves = len(boxes)
    # Construct our main optimization variable
    all_control_points = cp.Variable((n_curves * pts_per_curve, dim))

    # Determine the knot times between trajectory segments
    # Includes start/end times as knots too, for ease of indexing
    knot_times = np.concatenate([[t0], np.cumsum(durations)])
    knot_times[-1] = tf  # Fix floating pt issue with cumsum

    # Indexing from the main cvxpy variable containing all of the control points
    pos_pt_sets = [
        all_control_points[pts_per_curve * i : pts_per_curve * (i + 1)]
        for i in range(n_curves)
    ]
    pos_curves = [
        BezierCurve(pos_pt_sets[i], knot_times[i], knot_times[i + 1])
        for i in range(n_curves)
    ]
    # Set up derivative curves and the points associated with them
    vel_curves = [pc.derivative for pc in pos_curves]
    accel_curves = [vc.derivative for vc in vel_curves]
    jerk_curves = [ac.derivative for ac in accel_curves]
    vel_pt_sets = [vc.points for vc in vel_curves]
    accel_pt_sets = [ac.points for ac in accel_curves]
    # Set up constraints
    pos_continuity = [
        pos_pt_sets[i][-1] == pos_pt_sets[i + 1][0] for i in range(n_curves - 1)
    ]
    vel_continuity = [
        vel_pt_sets[i][-1] == vel_pt_sets[i + 1][0] for i in range(n_curves - 1)
    ]
    accel_continuity = [
        accel_pt_sets[i][-1] == accel_pt_sets[i + 1][0] for i in range(n_curves - 1)
    ]
    continuity_constraints = pos_continuity + vel_continuity + accel_continuity
    bc_constraints = [pos_pt_sets[0][0] == p0, pos_pt_sets[-1][-1] == pf]
    if v0 is not None:
        bc_constraints.append(vel_pt_sets[0][0] == v0)
    if a0 is not None:
        bc_constraints.append(accel_pt_sets[0][0] == a0)
    if vf is not None:
        bc_constraints.append(vel_pt_sets[-1][-1] == vf)
    if af is not None:
        bc_constraints.append(accel_pt_sets[-1][-1] == af)
    box_constraints = []
    for i, box in enumerate(boxes):
        lower, upper = box
        n = pos_pt_sets[i].shape[0]  # Number of control points in ith curve
        box_constraints.append(pos_pt_sets[i] >= np.tile(lower, (n, 1)))
        box_constraints.append(pos_pt_sets[i] <= np.tile(upper, (n, 1)))
    dyn_constraints = []
    if v_max is not None:
        for i in range(n_curves):
            dyn_constraints.append(cp.norm2(vel_pt_sets[i], axis=1) <= v_max)
    if a_max is not None:
        for i in range(n_curves):
            dyn_constraints.append(cp.norm2(accel_pt_sets[i], axis=1) <= a_max)
    # Merge the lists of constraints together
    constraints = (
        continuity_constraints + bc_constraints + box_constraints + dyn_constraints
    )
    # Complete the problem formulation and solve it
    jerk = sum(jc.l2_squared for jc in jerk_curves)
    objective = cp.Minimize(jerk + time_weight * (tf - t0))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)
    if prob.status != cp.OPTIMAL:
        raise OptimizationError(
            f"Unable to generate the trajectory (solver status: {prob.status}).\n"
            + "Check on the feasibility of the constraints"
        )
    # Form info about the solution to match the methodology from FPP, so we can use this for retiming
    # For more details on this, refer to the FPP source code and paper
    cost_breakdown = {}
    for k in range(n_curves):
        cost_breakdown[k] = {}
        cost_breakdown[k][3] = jerk_curves[k].l2_squared.value
    retiming_weights = {}
    for k in range(n_curves - 1):
        retiming_weights[k] = {}
        vel_primal = vel_curves[k].points[-1].value
        vel_dual = vel_continuity[k].dual_value
        retiming_weights[k][1] = vel_primal.dot(vel_dual)
        accel_primal = accel_curves[k].points[-1].value
        accel_dual = accel_continuity[k].dual_value
        retiming_weights[k][2] = accel_primal.dot(accel_dual)
    info = {
        "cost": prob.value,
        "retiming_weights": retiming_weights,
        "cost_breakdown": cost_breakdown,
    }
    # Construct the Bezier curves from the solved control points,
    solved_pos_curves = [
        BezierCurve(pos_pt_sets[i].value, knot_times[i], knot_times[i + 1])
        for i in range(n_curves)
    ]
    solved_pos_spline = CompositeBezierCurve(solved_pos_curves)
    return solved_pos_spline, info


def _test_spline_trajectory():
    """Construct and visualize an optimal trajectory composed of three curves between safe sets"""
    p0 = [0.1, 0.2, 0.3]
    pf = [1.5, 5, 1.7]
    t0 = 0
    tf = 5
    pts_per_curve = 8
    v0 = np.zeros(3)
    vf = np.zeros(3)
    a0 = np.zeros(3)
    af = np.zeros(3)
    boxes = [
        Box((0, 0, 0), (1, 1, 1)),
        Box((0.5, 0.5, 0.5), (1.5, 5, 1.5)),
        Box((1, 4.5, 1), (2, 5.5, 2)),
    ]
    n_curves = len(boxes)
    durations = np.ones(n_curves) * (tf - t0) / n_curves
    n_timesteps = 50
    times = np.linspace(t0, tf, n_timesteps, endpoint=True)
    pos_curve, *_ = _fixed_timing_spline(
        p0, pf, t0, tf, pts_per_curve, boxes, durations, v0, vf, a0, af
    )
    pos_pts = pos_curve(times)

    pos_curve_retimed, cost = spline_trajectory_with_retiming(
        p0, pf, t0, tf, pts_per_curve, boxes, durations, v0, vf, a0, af
    )
    pos = pos_curve_retimed(np.linspace(t0, tf, n_timesteps))

    pybullet.connect(pybullet.GUI)
    visualize_points(np.vstack([p0, pf]), (0, 0, 1))
    for box in boxes:
        visualize_3D_box(box)
    simple_traj_color = (0, 0, 1)
    retimed_traj_color = (0, 1, 0)
    visualize_path(pos_pts, 10, simple_traj_color)
    visualize_path(pos, 10, retimed_traj_color)
    pybullet.addUserDebugText("Original curve", [0, 0, -0.2], simple_traj_color)
    pybullet.addUserDebugText("Retimed curve", [0, 0, -0.4], retimed_traj_color)
    input()


if __name__ == "__main__":
    # _test_plotting_spline()
    _test_spline_trajectory()
