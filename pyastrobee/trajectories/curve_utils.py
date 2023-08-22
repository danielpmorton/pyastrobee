"""Curve tools: Plotting, trajectory construction, and more"""

from typing import Union, Optional

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon

from pyastrobee.trajectories.trajectory import Trajectory
from pyastrobee.trajectories.bezier import BezierCurve
from pyastrobee.trajectories.splines import CompositeBezierCurve


def traj_from_curve(
    curve: Union[BezierCurve, CompositeBezierCurve], dt: float
) -> Trajectory:
    """Construct a position-only trajectory from a Bezier curve or spline

    Args:
        curve (Union[BezierCurve, CompositeBezierCurve]): Curve for the position motion
        dt (float): Timestep

    Returns:
        Trajectory: Position (and derivatives) trajectory information
    """
    t0 = curve.a
    tf = curve.b.value if isinstance(curve.b, (cp.Variable, cp.Expression)) else curve.b
    # TODO see if we can refine how this time works... The spacing isn't going to be exactly dt
    times = np.linspace(t0, tf, round((tf - t0) / dt))
    pos = curve(times)
    vel = curve.derivative(times)
    accel = curve.derivative.derivative(times)
    return Trajectory(pos, None, vel, None, accel, None, times)


def plot_1d_bezier_curve(
    curve: BezierCurve,
    n_pts: int = 50,
    plot_pts: bool = True,
    plot_hull: bool = True,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    **kwargs,
) -> plt.Axes:
    """Plots a 1D Bezier curve assuming the control points are evenly spaced in time

    Args:
        curve (BezierCurve): Bezier curve to plot
        n_pts (int): Number of points to evaluate the curve. Defaults to 50.
        plot_pts (bool, optional): Whether or not to display the curve's control points. Defaults to True.
        plot_hull (bool, optional): Whether or not to display the convex hull of the control points. Defaults to True.
        ax (Optional[plt.Axes]): Axes for plotting, if re-using an existing plot. Defaults to None (create new plot).
        show (bool, optional): Whether or not to show the plot. Defaults to True.

    Returns:
        plt.Axes: The plot
    """
    assert curve.d == 1
    points = np.ravel(curve.points)
    # Times to evaluate the curve
    t = np.linspace(curve.a, curve.b, n_pts, endpoint=True)
    # "Times" at which we assign the control points along the x axis
    x = np.linspace(curve.a, curve.b, len(points), endpoint=True)
    if ax is None:
        ax = plt.gca()
    ax.plot(t, curve(t), **kwargs)
    ax.set_xlabel("Time")
    ax.set_ylabel("Variable")
    color = ax.lines[-1].get_color()
    if plot_pts:
        ax.scatter(x, points, c=color, **kwargs)
    if plot_hull:
        hull = ConvexHull(np.column_stack([x, points]))
        ordered_points = hull.points[hull.vertices]
        poly = Polygon(ordered_points, fc=color, alpha=0.5, **kwargs)
        ax.add_patch(poly)
    if show:
        plt.show()
    return ax


def plot_2d_bezier_curve(
    curve: BezierCurve,
    n_pts: int = 50,
    plot_pts: bool = True,
    plot_hull: bool = True,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    **kwargs,
) -> plt.Axes:
    """Plots a 2D Bezier curve

    Args:
        curve (BezierCurve): Bezier curve to plot
        n_pts (int): Number of points to evaluate the curve. Defaults to 50.
        plot_pts (bool, optional): Whether or not to display the curve's control points. Defaults to True.
        plot_hull (bool, optional): Whether or not to display the convex hull of the control points. Defaults to True.
        ax (Optional[plt.Axes]): Axes for plotting, if re-using an existing plot. Defaults to None (create new plot).
        show (bool, optional): Whether or not to show the plot. Defaults to True.

    Returns:
        plt.Axes: The plot
    """
    assert curve.d == 2
    t = np.linspace(curve.a, curve.b, n_pts, endpoint=True)
    if ax is None:
        ax = plt.gca()
    ax.plot(*curve(t).T, **kwargs)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    color = ax.lines[-1].get_color()
    if plot_pts:
        ax.scatter(*curve.points.T, c=color, **kwargs)
    if plot_hull:
        hull = ConvexHull(curve.points)
        ordered_points = hull.points[hull.vertices]
        poly = Polygon(ordered_points, fc=color, alpha=0.5, **kwargs)
        ax.add_patch(poly)
    if show:
        plt.show()
    return ax


def plot_3d_bezier_traj(curve: BezierCurve, n_pts: int = 50):
    """Plots the trajectory components of a Bezier curve, including its first and second derivatives

    Args:
        curve (BezierCurve): Bezier curve used for a position trajectory
        n_pts (int): Number of points to plot. Defaults to 50.
    """
    assert curve.d == 3
    t = np.linspace(curve.a, curve.b, n_pts, endpoint=True)
    # Evaluate the positions, velocities, and accelerations on the curve at specified times
    pos_evals = curve(t)
    vel_curve = curve.derivative
    vel_evals = vel_curve(t)
    accel_curve = vel_curve.derivative
    accel_evals = accel_curve(t)
    # Plot the position, velocity, acceleration components on separate axes
    fig = plt.figure()
    subfigs = fig.subfigures(1, 3)
    left = subfigs[0].subplots(1, 3)
    middle = subfigs[1].subplots(1, 3)
    right = subfigs[2].subplots(1, 3)
    pos_labels = ["x", "y", "z"]
    vel_labels = ["vx", "vy", "vz"]
    accel_labels = ["ax", "ay", "az"]
    for i, ax in enumerate(left):
        ax.plot(pos_evals[:, i])
        ax.set_title(pos_labels[i])
    for i, ax in enumerate(middle):
        ax.plot(vel_evals[:, i])
        ax.set_title(vel_labels[i])
    for i, ax in enumerate(right):
        ax.plot(accel_evals[:, i])
        ax.set_title(accel_labels[i])
    plt.show()


def plot_1d_composite_bezier_curve(
    curve: CompositeBezierCurve,
    n_pts: int = 50,
    plot_pts: bool = True,
    plot_hull: bool = True,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    **kwargs,
) -> plt.Axes:
    """Plots a 1D composite Bezier curve

    Args:
        curve (BezierCurve): Composite Bezier curve to plot
        n_pts (int): Number of points to evaluate the curve. Defaults to 50.
        plot_pts (bool, optional): Whether or not to display the curve's control points. Defaults to True.
        plot_hull (bool, optional): Whether or not to display the convex hull of the control points. Defaults to True.
        ax (Optional[plt.Axes]): Axes for plotting, if re-using an existing plot. Defaults to None (create new plot).
        show (bool, optional): Whether or not to show the plot. Defaults to True.

    Returns:
        plt.Axes: The plot
    """
    assert curve.d == 1
    if ax is None:
        ax = plt.gca()
    for bez in curve.beziers:
        # Assign the number of points to plot based on the fraction of the total interval
        # for a single bezier curve within the whole composite curve
        n = round((bez.duration / curve.duration) * n_pts)
        ax = plot_1d_bezier_curve(bez, n, plot_pts, plot_hull, ax, show=False, **kwargs)
    if show:
        plt.show()
    return ax


def plot_2d_composite_bezier_curve(
    curve: CompositeBezierCurve,
    n_pts: int = 50,
    plot_pts: bool = True,
    plot_hull: bool = True,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    **kwargs,
) -> plt.Axes:
    """Plots a 2D composite Bezier curve

    Args:
        curve (BezierCurve): Composite Bezier curve to plot
        n_pts (int): Number of points to evaluate the curve. Defaults to 50.
        plot_pts (bool, optional): Whether or not to display the curve's control points. Defaults to True.
        plot_hull (bool, optional): Whether or not to display the convex hull of the control points. Defaults to True.
        ax (Optional[plt.Axes]): Axes for plotting, if re-using an existing plot. Defaults to None (create new plot).
        show (bool, optional): Whether or not to show the plot. Defaults to True.

    Returns:
        plt.Axes: The plot
    """
    assert curve.d == 2
    if ax is None:
        ax = plt.gca()
    for bez in curve.beziers:
        # Assign the number of points to plot based on the fraction of the total interval
        # for a single bezier curve within the whole composite curve
        n = round((bez.duration / curve.duration) * n_pts)
        ax = plot_2d_bezier_curve(bez, n, plot_pts, plot_hull, ax, show=False, **kwargs)
    if show:
        plt.show()
    return ax


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
    ax = plot_2d_composite_bezier_curve(spline, 30, True, True, show=False)
    plt.show()


if __name__ == "__main__":
    _test_plotting_spline()
