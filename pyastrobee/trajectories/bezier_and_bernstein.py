"""Generating trajectories using Bezier curves and Bernstein polynomials

When compared with polynomial trajectories, these are:
- More numerically stable during optimization
- Easier to formulate constraints on maximum velocity, acceleration, ...
- Easier to formulate cost functions (such as minimizing jerk)
- Easier to specify motions within a safe convex set

Imposing constraints on the curve and its derivatives:
- This is quite simple. If we have boundary conditions on position, for instance, we can constraint the start and 
  end points of the curve to our desired start/end positions.
- Likewise, the derivative of a Bezier curve is also a Bezier curve (of lower order: M-1), so we can constrain the 
  start/end points of the derivative curve to meet constraints on the derivative (velocity, for instance)
- This can be extended to higher-order derivatives, provided that the original curve is of a high-enough order
  so that the reduced-order derivative curves still have enough control points to meet the constraints.

Cost function:
- The squared L2 norm of a Bezier curve is a natural (convex, quadratic) choice for a cost function. If we are 
  minimizing jerk, for instance, we can use the third-derivative of a position curve with this function

Safe motion:
- If the free space is defined as a convex set, we can enforce that the trajectory remains within the free space by
  constraining the control points to remain in free space. Since the Bezier curve is contained within the convex hull
  of the control points, this ensures that the entire curve is in free space.

Stephen Boyd and Tobia Marcucci recommended using these. 
Refer to "Fast Path Planning Through Large Collections of Safe Boxes" for more info, as well as Tobia's repository
https://github.com/cvxgrp/fastpathplanning/
"""


from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import cvxpy as cp
from scipy.special import binom
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def bernstein(
    h: int, n: int, a: float, b: float, t: Union[float, npt.ArrayLike]
) -> npt.ArrayLike:
    """Evaluate the nth Bernstein polynomial of degree h at a point (points) t

    Args:
        h (int): Degree of the Bernstein basis
        n (int): Index of the Bernstein polynomial
        a (float): Interval minimum value (e.g. starting time of trajectory)
        b (float): Interval maximum value (e.g. ending time of trajectory)
        t (Union[float, npt.ArrayLike]): Evaluation point(s) (for instance, trajectory times)

    Returns:
        npt.ArrayLike: Evaluation(s) of the Bernstein polynomial at point(s) t. Returns a float if t is a float,
            otherwise will return an array of evaluations
    """
    if n > h:
        raise ValueError(
            "Bernstein polynomial index cannot be larger than the degree of the basis"
        )
    if b <= a:
        raise ValueError(f"Invalid interval limits: ({a}, {b})")
    if np.ndim(t) >= 0:
        t = np.asarray(t)
    if not (np.all(a <= t) and np.all(t <= b)):
        raise ValueError("Cannot evaluate at points outside of the specified interval")
    return binom(h, n) * ((t - a) / (b - a)) ** n * ((b - t) / (b - a)) ** (h - n)


class BezierCurve:
    """Bezier curve class for evaluating a curve, the basis polynomials, and its derivative

    To evaluate the curve at points t, call it with curve(t)

    On initialization, for a standard "unit-time" curve, set a = 0 and b = 1

    Args:
        points (np.ndarray): Control points, shape (n_pts, dimension)
        a (float): Lower limit of the curve interval
        b (float): Upper limit of the curve interval
    """

    def __init__(self, points: np.ndarray, a: float, b: float):
        if b < a:
            raise ValueError(f"Invalid interval limits: ({a}, {b})")
        self.points = points
        self.h = points.shape[0] - 1  # Degree of the curve (AKA M in Tobia's paper)
        self.d = points.shape[1]  # Dimension of the space
        self.a = a  # Lower interval limit
        self.b = b  # Upper interval limit
        self.duration = b - a

    def __call__(self, t: Union[float, npt.ArrayLike]) -> np.ndarray:
        """Evaluates the Bezier curve (a sum of Bernstein polynomials) at specified points

        Args:
            t (Union[float, npt.ArrayLike]): Evaluation points (for instance, trajectory times)

        Returns:
            np.ndarray: Points along the curve, shape (n_pts, dimension)
        """
        c = np.array([self._bernstein(t, n) for n in range(self.h + 1)])
        return c.T.dot(self.points)

    def _bernstein(self, t: Union[float, npt.ArrayLike], n: int) -> npt.ArrayLike:
        """Wrapper around the Bernstein polynomial function, using attributes of self@BezierCurve

        Args:
            t (Union[float, npt.ArrayLike]): Evaluation point(s) (for instance, trajectory times)
            n (int): Index of the Bernstein polynomial

        Returns:
            npt.ArrayLike: Evaluation(s) of the bernstein polynomial at point(s) t. Returns a float if t is a float,
                otherwise will return an array of evaluations
        """
        return bernstein(self.h, n, self.a, self.b, t)

    @property
    def start_point(self):
        """Starting control point of the Bezier curve"""
        return self.points[0]

    @property
    def end_point(self):
        """Ending control point of the Bezier curve"""
        return self.points[-1]

    def derivative(self) -> "BezierCurve":
        """Derivative of the Bezier curve (A Bezier curve of degree h-1)"""
        points = (self.points[1:] - self.points[:-1]) * (self.h / self.duration)
        return BezierCurve(points, self.a, self.b)

    def l2_squared(self):
        """Squared L2 norm of the curve"""
        A = np.zeros((self.h + 1, self.h + 1))
        for m in range(self.h + 1):
            for n in range(self.h + 1):
                A[m, n] = binom(self.h, m) * binom(self.h, n) / binom(2 * self.h, m + n)
        A *= self.duration / (2 * self.h + 1)
        A = np.kron(A, np.eye(self.d))
        p = self.points.flatten()
        return p.dot(A.dot(p))


def plot_2d_bezier_pts(
    curve: BezierCurve, ax: Optional[plt.Axes] = None, show: bool = True, **kwargs
) -> plt.Axes:
    """Plots the control points of a 2D Bezier curve

    Args:
        curve (BezierCurve): Bezier curve of interest
        ax (Optional[plt.Axes]): Axes for plotting, if re-using an existing plot. Defaults to None (create new plot).
        show (bool, optional): Whether or not to show the plot. Defaults to True.

    Returns:
        plt.Axes: The plot
    """
    assert curve.d == 2
    options = {"fc": "orange", "ec": "k", "zorder": 3}
    options.update(kwargs)
    if ax is None:
        ax = plt.gca()
    ax.scatter(*curve.points.T, **options)
    if show:
        plt.show()
    return ax


def plot_2d_bezier_curve(
    curve: BezierCurve,
    n_pts: int,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    **kwargs,
) -> plt.Axes:
    """Plots a 2D Bezier curve

    Args:
        curve (BezierCurve): Bezier curve to plot
        n_pts (int): Number of points to evaluate the curve
        ax (Optional[plt.Axes]): Axes for plotting, if re-using an existing plot. Defaults to None (create new plot).
        show (bool, optional): Whether or not to show the plot. Defaults to True.

    Returns:
        plt.Axes: The plot
    """
    assert curve.d == 2
    options = {"c": "b"}
    options.update(kwargs)
    t = np.linspace(curve.a, curve.b, n_pts, endpoint=True)
    if ax is None:
        ax = plt.gca()
    ax.plot(*curve(t).T, **options)
    if show:
        plt.show()
    return ax


def plot_2d_bezier_hull(
    curve: BezierCurve, ax: Optional[plt.Axes] = None, show: bool = True, **kwargs
) -> plt.Axes:
    """Plots the convex hull of the Bezier curve's control points

    Args:
        curve (BezierCurve): Bezier curve of interest
        ax (Optional[plt.Axes]): Axes for plotting, if re-using an existing plot. Defaults to None (create new plot).
        show (bool, optional): Whether or not to show the plot. Defaults to True.

    Returns:
        plt.Axes: The plot
    """
    assert curve.d == 2
    options = {"fc": "lightcoral"}
    options.update(kwargs)
    hull = ConvexHull(curve.points)
    ordered_points = hull.points[hull.vertices]
    poly = Polygon(ordered_points, **options)
    if ax is None:
        ax = plt.gca()
    ax.add_patch(poly)
    if show:
        plt.show()
    return ax


def plot_3d_bezier_traj(curve: BezierCurve, n_pts: int):
    """Plots the trajectory components of a Bezier curve, including its first and second derivatives

    Args:
        curve (BezierCurve): Bezier curve used for a position trajectory
        n_pts (int): Number of points to plot
    """
    assert curve.d == 3
    t = np.linspace(curve.a, curve.b, n_pts, endpoint=True)
    # Evaluate the positions, velocities, and accelerations on the curve at specified times
    pos_evals = curve(t)
    vel_curve = curve.derivative()
    vel_evals = vel_curve(t)
    accel_curve = vel_curve.derivative()
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


def bezier_derivative_points(
    pts: Union[cp.Variable, cp.Expression, np.ndarray], duration: float
) -> Union[cp.Expression, np.ndarray]:
    """Generate a set of control points corresponding to the derivative of a Bezier Curve

    This function also works with a cvxpy Variable/Expression, so it can generate a cp.Expression relating the
    points variable/expression to its derivative(s)

    Args:
        pts (Union[cp.Variable, cp.Expression, np.ndarray]): Control points, or a cvxpy Variable/Expression of the
            control points. Shape (n_pts, dimension)
        duration (float): Bezier curve trajectory duration

    Returns:
        Union[cp.Expression, np.ndarray]: Control points of the derivative curve. If the input was a numpy array, this
            will return an analytical evaluation of these new points. If the input was cvxpy-based, this will return
            an Expression relating to the original cvxpy Variable. Shape (n_pts, dimension)
    """
    h = pts.shape[0] - 1
    return (pts[1:] - pts[:-1]) * (h / duration)


def _test_bezier_traj():
    """Example of generating a Bezier curve trajectory with various boundary conditions"""
    n_pts = 6
    M = n_pts - 1
    a = 0
    b = 5
    duration = b - a
    dim = 3
    # Arbitrary boundary conditions
    p0 = [0, 0, 0]
    v0 = [0.1, 0.2, 0.3]
    a0 = [0, 0, 0]
    pf = [1, 2, 3]
    vf = [0, 0, 0]
    af = [-0.3, -0.2, -0.1]
    pts = cp.Variable((n_pts, dim))
    d_pts = bezier_derivative_points(pts, duration)
    d2_pts = bezier_derivative_points(d_pts, duration)
    objective = cp.Minimize(0)  # TODO update this... just using feasibility for now
    constraints = [
        pts[0] == p0,
        pts[-1] == pf,
        d_pts[0] == v0,
        d_pts[-1] == vf,
        d2_pts[0] == a0,
        d2_pts[-1] == af,
    ]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    curve = BezierCurve(pts.value, a, b)
    plot_3d_bezier_traj(curve, 50)


def _test_plot_bernstein_polys():
    """Example to visualize Bernstein polynomials of various degrees"""
    a = 0
    b = 10
    n = 50
    t = np.linspace(a, b, n, endpoint=True)
    M = 4
    fig = plt.figure()
    for n in range(M + 1):
        evals = bernstein(M, n, a, b, t)
        plt.plot(t, evals, label=str(n))
    plt.legend()
    plt.title("Bernstein Polynomials")
    plt.show()


if __name__ == "__main__":
    _test_plot_bernstein_polys()
    _test_bezier_traj()
