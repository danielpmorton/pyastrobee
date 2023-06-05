"""Splines / "composite Bezier curves" composed of multiple chained Bezier curves

Initial boundary conditions are imposed on the start of the first curve, final B.C.s are imposed at the end of
the third curve, and we enforce C2 derivative continuity at the knot points

TODO
- Determine if we can specify the locations of the knot points

Reference:
https://github.com/cvxgrp/fastpathplanning/blob/main/fastpathplanning/smooth.py
"""


from typing import Optional, Union
import cvxpy as cp
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from pyastrobee.trajectories.trajectory import Trajectory
from pyastrobee.trajectories.bezier import (
    BezierCurve,
    plot_2d_bezier_curve,
    plot_2d_bezier_pts,
    plot_2d_bezier_hull,
)


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
        seg_map = self.find_segment(t)  # CHECK IF THIS WORKS WITH T AS AN ARRAY
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


# TODO should we be able to specify the knot locations?
# ^^ This might be useful since otherwise the trajectory will probably look similar to if we just had a bezier curve with more control points
# TODO: add these notes to the docstring
# First curve should enforce the starting boundary conditions and continuity to later curves
# Last curve should enforce the ending boundary conditions and continuity to earlier curves
# Middle curves should just enforce continuity
# I decided to make the curves continuous up to 2 derivatives (velocity and acceleration)
# So, each curve should essentially have at least 6 control points
def spline_trajectory(
    p0: npt.ArrayLike,
    pf: npt.ArrayLike,
    t0: float,
    tf: float,
    pts_per_curve: Union[int, npt.ArrayLike],
    n_timesteps: int,
    n_curves: int,  # UPDATE THIS
    v0: Optional[npt.ArrayLike] = None,
    vf: Optional[npt.ArrayLike] = None,
    a0: Optional[npt.ArrayLike] = None,
    af: Optional[npt.ArrayLike] = None,
    jerk_weight: float = 1.0,
    pathlength_weight: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a trajectory based on chained Bezier curves

    We enforce continuity between curves up to the second derivative

    TODO add ability to set the knot point position/velocity/..

    Args:
        p0 (npt.ArrayLike): Initial position, shape (3,)
        pf (npt.ArrayLike): Final position, shape (3,)
        t0 (float): Starting time
        tf (float): Ending time
        pts_per_curve (Union[int, npt.ArrayLike]): Number of control points per Bezier curve.
            Int if using the same number of control points for each curve, or a list/tuple/
            array specifying the number for each curve. Generally, should be around 6-10 per curve
        n_timesteps (int): Number of timesteps to evaluate
        n_curves (int): Number of child Bezier curves within the composite curve
        v0 (Optional[npt.ArrayLike]): Initial velocity, shape (3,). Defaults to None (unconstrained)
        vf (Optional[npt.ArrayLike]): Final velocity, shape (3,). Defaults to None (unconstrained)
        a0 (Optional[npt.ArrayLike]): Initial acceleration, shape (3,). Defaults to None (unconstrained)
        af (Optional[npt.ArrayLike]): Final acceleration, shape (3,). Defaults to None (unconstrained)
        jerk_weight (float, optional): Objective function weight corresponding to jerk. Defaults to 1.0.
        pathlength_weight (float, optional): Objective function weight corresponding to pathlength . Defaults to 0.0.

    Returns:
        Tuple of:
            np.ndarray: Trajectory positions, shape (n_timesteps, 3)
            np.ndarray: Trajectory velocities, shape (n_timesteps, 3)
            np.ndarray: Trajectory accelerations, shape (n_timesteps, 3)
            np.ndarray: Trajectory times, shape (n_timesteps,)"""

    if tf <= t0:
        raise ValueError(f"Invalid time interval: ({t0}, {tf})")
    dim = len(p0)
    times = np.linspace(t0, tf, n_timesteps, endpoint=True)

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

    total_num_pts = sum(pts_per_curve)
    all_control_points = cp.Variable((total_num_pts, dim))
    # Assume that each curve gets the same amount of time
    # Includes start/end times as "knots" too, for ease of indexing
    knot_times = np.linspace(t0, tf, n_curves + 1, endpoint=True)
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
    # Merge the lists of constraints together
    constraints = continuity_constraints + bc_constraints
    # Complete the problem formulation and solve it
    objective = cp.Minimize(
        jerk_weight * sum(jc.l2_squared for jc in jerk_curves)
        + pathlength_weight * sum(pc.control_points_pathlength for pc in pos_curves)
    )
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)
    # Construct the Bezier curves from the solved control points, and return their evaluations at each time
    solved_pos_curves = [
        BezierCurve(pos_pt_sets[i].value, knot_times[i], knot_times[i + 1])
        for i in range(n_curves)
    ]
    solved_pos_spline = CompositeBezierCurve(solved_pos_curves)
    solved_vel_spline = solved_pos_spline.derivative
    solved_accel_spline = solved_vel_spline.derivative
    return (
        solved_pos_spline(times),
        solved_vel_spline(times),
        solved_accel_spline(times),
        times,
    )


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
    p0 = [1, 2, 3]
    pf = [2, 3, 4]
    t0 = 0
    tf = 5
    pts_per_curve = 8
    n_timesteps = 50
    v0 = [-0.1, -0.2, -0.3]
    vf = [-0.2, -0.2, -0.2]
    a0 = [0, 0, 0]
    af = [0.1, 0.1, 0.1]
    n_curves = 3
    pos, vel, accel, times = spline_trajectory(
        p0, pf, t0, tf, pts_per_curve, n_timesteps, n_curves, v0, vf, a0, af
    )
    # Leaving out any rotational info for now
    traj = Trajectory(pos, None, vel, None, accel, None, times)
    traj.plot()


if __name__ == "__main__":
    _test_plotting_spline()
    _test_spline_trajectory()
