"""Splines / "composite Bezier curves" composed of multiple chained Bezier curves

Initial boundary conditions are imposed on the start of the first curve, final B.C.s are imposed at the end of
the third curve, and we enforce C2 derivative continuity at the knot points

TODO
- Determine if we can specify the locations of the knot points

Reference:
Fast Path Planning Through Large Collections of Safe Boxes
https://github.com/cvxgrp/fastpathplanning/blob/main/fastpathplanning/smooth.py
https://web.stanford.edu/~boyd/papers/pdf/fpp.pdf
"""

from typing import Optional, Union, Any

import cvxpy as cp
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import pybullet

from pyastrobee.trajectories.bezier import (
    BezierCurve,
    plot_2d_bezier_curve,
    plot_2d_bezier_pts,
    plot_2d_bezier_hull,
)
from pyastrobee.utils.boxes import Box, visualize_3D_box
from pyastrobee.utils.debug_visualizer import visualize_path, visualize_points


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


def timing_estimate(
    start_pt: npt.ArrayLike, end_pt: npt.ArrayLike, boxes: list[Box], total_time: float
) -> np.ndarray:
    """Calculate a preliminary estimate for the time allocated to each curve based on a straight-line (min-length) path
    through the boxes. Each duration will correspond to that path's fraction of the total trajectory length

    Args:
        start_pt (npt.ArrayLike): Starting XYZ position, shape (3,)
        end_pt (npt.ArrayLike): Ending XYZ position, shape (3,)
        boxes (list[Box]): Sequence of safe boxes that will be traveled through (Not the entire free space)
        total_time (float): Total duration of the trajectory

    Returns:
        np.ndarray: Durations for each trajectory segment, shape (num_boxes,)
    """
    n_boxes = len(boxes)
    points = cp.Variable((n_boxes + 1, 3))
    pathlength = cp.sum(cp.norm2(cp.diff(points, axis=0), axis=1))
    objective = cp.Minimize(pathlength)
    constraints = [points[0] == start_pt, points[-1] == end_pt]
    for i, box in enumerate(boxes):
        lower, upper = box
        constraints.append(points[i] >= lower)
        constraints.append(points[i] <= upper)
        constraints.append(points[i + 1] >= lower)
        constraints.append(points[i + 1] <= upper)
    prob = cp.Problem(objective, constraints)
    prob.solve()
    distances = np.linalg.norm(np.diff(points.value, axis=0), axis=1)
    return (distances / np.sum(distances)) * total_time


def optimal_spline_trajectory(
    p0: npt.ArrayLike,
    pf: npt.ArrayLike,
    t0: float,
    tf: float,
    pts_per_curve: Union[int, npt.ArrayLike],
    n_timesteps: int,
    boxes: list[Box],
    v0: Optional[npt.ArrayLike] = None,
    vf: Optional[npt.ArrayLike] = None,
    a0: Optional[npt.ArrayLike] = None,
    af: Optional[npt.ArrayLike] = None,
    v_max: Optional[float] = None,
    a_max: Optional[float] = None,
    kappa_min: float = 1e-2,
    omega: float = 3,
    max_iters: int = 10,
):
    # Handle inputs
    if tf <= t0:
        raise ValueError(f"Invalid time interval: ({t0}, {tf})")
    n_curves = len(boxes)
    # Form an array specifying the number of control points for each curve
    if np.ndim(pts_per_curve) == 0:
        # Integer input, assume same number of control points for each curve
        pts_per_curve = (pts_per_curve * np.ones(n_curves)).astype(int)
    else:
        # ArrayLike input, size should correspond to the number of curves
        pts_per_curve = np.ravel(pts_per_curve).astype(int)
        assert len(pts_per_curve) == n_curves
    if np.any(pts_per_curve < 6):
        print(
            "WARNING: A segment with less thant 6 control points may lead to infeasible constraints"
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
    )
    # Form an intial estimate for the durations
    # TODO MESS AROUND WITH THESE HEURISTICS
    # Right now trying to average the heuristics because going around a corner might result in a short path length
    # but it might need more time allocated because of the time associated with turning and slowing down
    straight_durations = timing_estimate(p0, pf, boxes, tf - t0)
    uniform_durations = ((tf - t0) / n_curves) * np.ones(n_curves)
    durations = (straight_durations + uniform_durations) / 2
    curve_kwargs["durations"] = durations
    # Solve a preliminary trajectory for this initial guess at the durations
    best_curve, best_info = _fixed_duration_spline_trajectory(**curve_kwargs)

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
        except cp.error.SolverError:
            # This seems to sometimes happen if the initial guess for durations is not good
            print("Terminated due to failure to solve the retiming problem")
            break

        # Improve Bezier curves based on the retiming
        curve_kwargs["durations"] = new_durations
        new_curve, new_info = _fixed_duration_spline_trajectory(**curve_kwargs)
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
        print("Terminated due to reaching the maximum number of iterations")

    # Evaluate the optimal curve and its derivatives at each time
    times = np.linspace(t0, tf, n_timesteps, endpoint=True)
    vel_curve = best_curve.derivative
    accel_curve = vel_curve.derivative
    print("COST: ", best_curve.derivative.derivative.derivative.l2_squared_sum)
    return best_curve(times), vel_curve(times), accel_curve(times), times


def retiming(
    kappa: float,
    costs: dict[int, dict[int, float]],
    durations: npt.ArrayLike,
    retiming_weights: dict[int, dict[int, float]],
) -> tuple[np.ndarray, float]:
    """Run the retiming trust-region-based optimization to generate an improved set of curve durations

    This code is essentially straight from Fast Path Planning with minimal modification since the retiming method
    is a bit complex, and we know that this works

    Args:
        kappa (float): Trust region parameter: Defines the maximum change in adjacent scaling factors
        costs (dict[int, dict[int, float]]): Breakdown of costs per curve and per derivative. Costs[i] gives the info
            for curve i, and costs[i][j] gives the cost associated with the jth derivative curve. Since we deal only
            with min-jerk, we only evaluate the j=3 case. The cost is the squared L2 norm of the jerk
        durations (npt.ArrayLike): Current best known value of the curve durations, shape (n_curves,)
        retiming_weights (dict[int, dict[int, float]]): A combination of Lagrangian multipliers and the last solved
            path. Weights[i] gives the weights associated with the ith differentiability (continuity) constraint, and
            weights[i][j] gives the weight associated with the jth derivative curve continuity. We enforce continuity
            up to the second derivative (j = 1 and 2)

    Returns:
        Tuple of:
            np.ndarray: The updated curve durations
            float: The new trust region parameter
    """
    # Decision variables.
    n_boxes = max(costs) + 1
    eta = cp.Variable(n_boxes)
    eta.value = np.ones(n_boxes)
    constr = [durations @ eta == sum(durations)]

    # Scale costs from previous trajectory.
    cost = 0
    for i, ci in costs.items():
        for j, cij in ci.items():
            cost += cij * cp.power(eta[i], 1 - 2 * j)

    # Retiming weights.
    for k in range(n_boxes - 1):
        for i, w in retiming_weights[k].items():
            cost += i * retiming_weights[k][i] * (eta[k + 1] - eta[k])

    # Trust region.
    if not np.isinf(kappa):
        constr.append(eta[1:] - eta[:-1] <= kappa)
        constr.append(eta[:-1] - eta[1:] <= kappa)

    # Solve SOCP and get new durarations.
    prob = cp.Problem(cp.Minimize(cost), constr)
    prob.solve(solver=cp.CLARABEL)
    if prob.status != cp.OPTIMAL:
        print("Clarabel failed to solve the retiming problem. Retrying with MOSEK")
        prob.solve(solver=cp.MOSEK)
    if prob.status != cp.OPTIMAL:
        raise cp.error.SolverError("Unable to solve the retiming problem")
    new_durations = np.multiply(eta.value, durations)

    # New candidate for kappa.
    kappa_max = max(np.abs(eta.value[1:] - eta.value[:-1]))

    return new_durations, kappa_max


def _fixed_duration_spline_trajectory(
    p0: npt.ArrayLike,
    pf: npt.ArrayLike,
    t0: float,
    tf: float,
    pts_per_curve: npt.ArrayLike,
    boxes: list[Box],
    durations: npt.ArrayLike,
    v0: Optional[npt.ArrayLike] = None,
    vf: Optional[npt.ArrayLike] = None,
    a0: Optional[npt.ArrayLike] = None,
    af: Optional[npt.ArrayLike] = None,
    v_max: Optional[float] = None,
    a_max: Optional[float] = None,
) -> tuple[CompositeBezierCurve, dict[str, Any]]:
    """Generate a min-jerk trajectory based on chained Bezier curves through a set of safe boxes

    We enforce continuity between curves up to the second derivative

    Note: if position is unconstrained (no need for safe boxes), you should instead use a single Bezier curve

    Args:
        p0 (npt.ArrayLike): Initial position, shape (3,)
        pf (npt.ArrayLike): Final position, shape (3,)
        t0 (float): Starting time
        tf (float): Ending time
        pts_per_curve (Union[int, npt.ArrayLike]): Number of control points per Bezier curve.
            Int if using the same number of control points for each curve, or a list/tuple/
            array specifying the number for each curve. Generally, should be around 6-10 per curve

        ^^ TODO update pts per curve!

        boxes (list[Box]): Sequential list of safe box regions pass through
        durations (Optional[npt.ArrayLike]): Durations for each segment of the trajectory, shape (num_boxes,)

        ^^ TODO Update durations!

        v0 (Optional[npt.ArrayLike]): Initial velocity, shape (3,). Defaults to None (unconstrained)
        vf (Optional[npt.ArrayLike]): Final velocity, shape (3,). Defaults to None (unconstrained)
        a0 (Optional[npt.ArrayLike]): Initial acceleration, shape (3,). Defaults to None (unconstrained)
        af (Optional[npt.ArrayLike]): Final acceleration, shape (3,). Defaults to None (unconstrained)
        v_max (Optional[float]): Maximum L2 norm of the velocity. Defaults to None (unconstrained)
        a_max (Optional[float]): Maximum L2 norm of the acceleration. Defaults to None (unconstrained)

    Returns:
        Tuple of:
            CompositeBezierCurve: The solved position curve
            dict[str, Any]: Solution info to use for retiming optimization
    """
    dim = len(p0)
    n_curves = len(boxes)
    total_num_pts = sum(pts_per_curve)
    # Construct our main optimization variable
    all_control_points = cp.Variable((total_num_pts, dim))

    # Determine the knot times between trajectory segments
    # Includes start/end times as knots too, for ease of indexing
    knot_times = np.concatenate([[t0], np.cumsum(durations)])
    knot_times[-1] = tf

    # Indexing from the main cvxpy variable containing all of the control points
    end_idxs = np.cumsum(pts_per_curve)
    start_idxs = end_idxs - pts_per_curve[0]
    pos_pt_sets = [all_control_points[a:b] for a, b in zip(start_idxs, end_idxs)]
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
    objective = cp.Minimize(sum(jc.l2_squared for jc in jerk_curves))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)
    if prob.status != cp.OPTIMAL:
        raise cp.error.SolverError(
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


# Below: plotting functions


def plot_2d_composite_bezier_pts(
    curve: CompositeBezierCurve,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    **kwargs,
) -> plt.Axes:
    """Plots the control points of a 2D composite Bezier curve

    Args:
        curve (CompositeBezierCurve): Composite Bezier curve of interest
        ax (Optional[plt.Axes]): Axes for plotting, if re-using an existing plot. Defaults to None.
        show (bool, optional): Whether or not to show the plot. Defaults to True.

    Returns:
        plt.Axes: The plot
    """
    if ax is None:
        ax = plt.gca()
    for bez in curve.beziers:
        ax = plot_2d_bezier_pts(bez, ax, show=False, **kwargs)
    if show:
        plt.show()
    return ax


def plot_2d_composite_bezier_curve(
    curve: CompositeBezierCurve,
    n_pts: int,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    **kwargs,
) -> plt.Axes:
    """Plots a 2D composite Bezier curve

    Args:
        curve (BezierCurve): Composite Bezier curve to plot
        n_pts (int): Number of points to evaluate the curve
        ax (Optional[plt.Axes]): Axes for plotting, if re-using an existing plot. Defaults to None (create new plot).
        show (bool, optional): Whether or not to show the plot. Defaults to True.

    Returns:
        plt.Axes: The plot
    """
    if ax is None:
        ax = plt.gca()
    for bez in curve.beziers:
        # Assign the number of points to plot based on the fraction of the total interval
        # for a single bezier curve within the whole composite curve
        n = round((bez.duration / curve.duration) * n_pts)
        ax = plot_2d_bezier_curve(bez, n, ax, show=False, **kwargs)
    if show:
        plt.show()
    return ax


def plot_2d_composite_bezier_hull(
    curve: CompositeBezierCurve,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    **kwargs,
) -> plt.Axes:
    """Plots the convex hull of each of the composite Bezier curve's child curves

    Args:
        curve (BezierCurve): Composite Bezier curve of interest
        ax (Optional[plt.Axes]): Axes for plotting, if re-using an existing plot. Defaults to None (create new plot).
        show (bool, optional): Whether or not to show the plot. Defaults to True.

    Returns:
        plt.Axes: The plot
    """
    if ax is None:
        ax = plt.gca()
    for bez in curve.beziers:
        ax = plot_2d_bezier_hull(bez, ax, show=False, **kwargs)
    if show:
        plt.show()
    return ax


# Below: examples of using the functions above


def _test_plotting_spline():
    # Create a test composite curve from three children
    # These curves are randomly selected and do not enforce continuity in any derivatives
    pts_a = np.array([[0, 0], [1, 3], [3, 1], [5, 5]])
    pts_b = np.array([[5, 5], [6, 5], [7, 4], [8, 3]])
    pts_c = np.array([[8, 3], [10, 3], [9, 5], [12, 7]])
    curve_a = BezierCurve(pts_a, 0, 2)
    curve_b = BezierCurve(pts_b, 2, 3)
    curve_c = BezierCurve(pts_c, 3, 5)
    curves = [curve_a, curve_b, curve_c]
    spline = CompositeBezierCurve(curves)
    ax = plot_2d_composite_bezier_hull(spline, show=False, fc="lightcoral", alpha=0.5)
    ax = plot_2d_composite_bezier_curve(spline, 30, show=False)
    ax = plot_2d_composite_bezier_pts(spline, ax, show=False)
    # Create a new plot where we manually evaluate the points of the full composite curve
    # The curve should effectively look the same as the one that was just plotted
    times = np.linspace(spline.a, spline.b, endpoint=True)
    curve_evals = spline(times)
    plt.figure()
    plt.plot(curve_evals[:, 0], curve_evals[:, 1])
    # Show both plots at once
    plt.show()


def _test_spline_trajectory():
    """Construct and visualize an optimal trajectory composed of three curves between safe sets"""
    p0 = [0.1, 0.2, 0.3]
    pf = [1.5, 5, 1.7]
    t0 = 0
    tf = 5
    pts_per_curve = 8
    n_timesteps = 50
    v0 = np.zeros(3)
    vf = np.zeros(3)
    a0 = np.zeros(3)
    af = np.zeros(3)
    safe_boxes = [
        Box((0, 0, 0), (1, 1, 1)),
        Box((0.5, 0.5, 0.5), (1.5, 5, 1.5)),
        Box((1, 4.5, 1), (2, 5.5, 2)),
    ]
    pos, *_ = spline_trajectory(
        p0, pf, t0, tf, pts_per_curve, n_timesteps, v0, vf, a0, af, safe_boxes
    )
    pybullet.connect(pybullet.GUI)
    visualize_points(np.vstack([p0, pf]), (0, 0, 1))
    for box in safe_boxes:
        visualize_3D_box(box)
    visualize_path(pos, 10, (0, 0, 1))
    input()


def _test_timing_estimate():
    p0 = [0.1, 0.2, 0.3]
    pf = [1.5, 5, 1.7]
    T = 5
    safe_boxes = [
        Box((0, 0, 0), (1, 1, 1)),
        Box((0.5, 0.5, 0.5), (1.5, 5, 1.5)),
        Box((1, 4.5, 1), (2, 5.5, 2)),
    ]
    durations = timing_estimate(p0, pf, safe_boxes, T)
    print(durations)


def _test_sequential_opt():
    curve_kwargs = dict(
        p0=[0.1, 0.2, 0.3],
        pf=[1.5, 5, 1.7],
        t0=0,
        tf=5,
        pts_per_curve=8,
        n_timesteps=50,
        v0=np.zeros(3),
        vf=np.zeros(3),
        a0=np.zeros(3),
        af=np.zeros(3),
        boxes=[
            Box((0, 0, 0), (1, 1, 1)),
            Box((0.5, 0.5, 0.5), (1.5, 5, 1.5)),
            Box((1, 4.5, 1), (2, 5.5, 2)),
        ],
        durations=((5 - 0) / 3) * np.ones(3),
    )
    traj = sequential_optimization(curve_kwargs)
    pos = traj[0]

    pybullet.connect(pybullet.GUI)
    visualize_points(np.vstack([curve_kwargs["p0"], curve_kwargs["pf"]]), (0, 0, 1))
    for box in curve_kwargs["boxes"]:
        visualize_3D_box(box)
    visualize_path(pos, 10, (0, 0, 1))
    input()


def _test_timing_estimate_with_path():
    start_pt = [0.1, 0.2, 0.3]
    end_pt = [1.5, 5, 1.7]
    total_time = 5
    boxes = [
        Box((0, 0, 0), (1, 1, 1)),
        Box((0.5, 0.5, 0.5), (1.5, 5, 1.5)),
        Box((1, 4.5, 1), (2, 5.5, 2)),
    ]

    n_boxes = len(boxes)
    points = cp.Variable((n_boxes + 1, 3))
    pathlength = cp.sum(cp.norm2(cp.diff(points, axis=0), axis=1))
    objective = cp.Minimize(pathlength)
    constraints = [points[0] == start_pt, points[-1] == end_pt]
    for i, box in enumerate(boxes):
        lower, upper = box
        constraints.append(points[i] >= lower)
        constraints.append(points[i] <= upper)
        constraints.append(points[i + 1] >= lower)
        constraints.append(points[i + 1] <= upper)
    prob = cp.Problem(objective, constraints)
    prob.solve()
    distances = np.linalg.norm(np.diff(points.value, axis=0), axis=1)
    print((distances / np.sum(distances)) * total_time)
    pybullet.connect(pybullet.GUI)
    visualize_path(points.value, color=(0, 0, 1))
    visualize_points(points.value, (1, 1, 1))
    for box in boxes:
        visualize_3D_box(box)
    input("Press Enter to close")


if __name__ == "__main__":
    # _test_plotting_spline()
    # _test_spline_trajectory()
    # _test_timing_estimate()
    # _test_sequential_opt()
    _test_timing_estimate_with_path()
