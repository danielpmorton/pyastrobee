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

TODO
- The L2 squared metric seems to be nonconvex if the total duration of the curve is also an optimization variable...
  See if there is a better way to formulate this
"""


from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import cvxpy as cp
from scipy.special import binom
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from pyastrobee.trajectories.trajectory import Trajectory
from pyastrobee.utils.boxes import Box


class BezierCurve:
    """Bezier curve class for evaluating a curve, the basis polynomials, and its derivative

    To evaluate the curve at points t, call it with curve(t)

    On initialization, for a standard "unit-time" curve, set a = 0 and b = 1

    Args:
        points (Union[cp.Variable, cp.Expression, np.ndarray]): Control points, or a cvxpy Variable/Expression of the
            control points. Shape (n_pts, dimension)
        a (float): Lower limit of the curve interval
        b (Union[float, cp.Variable, cp.Expression]): Upper limit of the curve interval. Variable if we are also
            optimizing the duration of the curve
    """

    def __init__(
        self,
        points: Union[cp.Variable, cp.Expression, np.ndarray],
        a: float,
        b: Union[float, cp.Variable, cp.Expression],
    ):
        if not isinstance(b, (cp.Variable, cp.Expression)) and b < a:
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
        return c.T @ self.points

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
    def start_point(self) -> Union[np.ndarray, cp.Variable, cp.Expression]:
        """Starting control point of the Bezier curve"""
        return self.points[0]

    @property
    def end_point(self) -> Union[np.ndarray, cp.Variable, cp.Expression]:
        """Ending control point of the Bezier curve"""
        return self.points[-1]

    @property
    def derivative(self) -> "BezierCurve":
        """Derivative of the Bezier curve (A Bezier curve of degree h-1)"""
        points = (self.points[1:] - self.points[:-1]) * (self.h / self.duration)
        return BezierCurve(points, self.a, self.b)

    @property
    def l2_squared(self) -> Union[float, cp.Expression]:
        """Squared L2 norm of the curve"""
        A = np.zeros((self.h + 1, self.h + 1))
        for m in range(self.h + 1):
            for n in range(self.h + 1):
                A[m, n] = binom(self.h, m) * binom(self.h, n) / binom(2 * self.h, m + n)
        if isinstance(self.duration, (cp.Variable, cp.Expression)):
            A = cp.multiply(A, self.duration / (2 * self.h + 1))
            A = cp.kron(A, np.eye(self.d))
        else:
            A *= self.duration / (2 * self.h + 1)
            A = np.kron(A, np.eye(self.d))
        if isinstance(self.points, (cp.Variable, cp.Expression)):
            # Note: CVXPY flattens matrices by columns rather than rows (opposite of numpy)
            # So, flatten based on the transpose to make the math consistent
            p = self.points.T.flatten()
            return cp.quad_form(p, A)
        elif isinstance(A, (cp.Variable, cp.Expression)):
            p = self.points.flatten()
            return cp.quad_form(p, A)
        else:  # Numpy
            p = self.points.flatten()
            return p.dot(A.dot(p))

    @property
    def control_points_pathlength(self) -> Union[float, cp.Expression]:
        """Sum of the distances between consecutive control points

        This is an upper bound on the length of the curve itself, so we can use this in a cost function
        if we are trying to minimize the pathlength of a trajectory

        Returns:
            Union[float, cp.Expression]: Float if the points are a numpy array, otherwise yields
                a (convex, nonnegative) expression for this length
        """
        if isinstance(self.points, (cp.Variable, cp.Expression)):
            length = 0
            for i in range(self.h):
                length += cp.norm2(self.points[i + 1] - self.points[i])
            return length
        else:
            return np.sum(np.linalg.norm(np.gradient(self.points, axis=0), axis=1))


def bernstein(
    h: int,
    n: int,
    a: float,
    b: Union[float, cp.Variable, cp.Expression],
    t: Union[float, npt.ArrayLike],
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
    if np.ndim(t) >= 0:
        t = np.asarray(t)
    if not isinstance(b, (cp.Variable, cp.Expression)):
        if b <= a:
            raise ValueError(f"Invalid interval limits: ({a}, {b})")
        if not (np.all(a <= t) and np.all(t <= b)):
            raise ValueError(
                "Cannot evaluate at points outside of the specified interval"
            )
    return binom(h, n) * ((t - a) / (b - a)) ** n * ((b - t) / (b - a)) ** (h - n)


def bezier_trajectory(
    p0: npt.ArrayLike,
    pf: npt.ArrayLike,
    t0: float,
    tf: float,
    n_control_pts: int,
    n_timesteps: int,
    v0: Optional[npt.ArrayLike] = None,
    vf: Optional[npt.ArrayLike] = None,
    a0: Optional[npt.ArrayLike] = None,
    af: Optional[npt.ArrayLike] = None,
    box: Optional[Box] = None,
    v_max: Optional[float] = None,
    a_max: Optional[float] = None,
    jerk_weight: float = 1.0,
    pathlength_weight: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate an optimal trajectory based on Bezier curves which meets the specified constraints

    Optimality is currently with respect to two values: jerk and pathlength. We can evaluate a minimum-jerk
    trajectory, or a minimum-length trajectory, or any balance between these parameters by varying the
    relative objective function weights between them

    The default weighting of these objective function parameters is set to minimize jerk only

    Args:
        p0 (npt.ArrayLike): Initial position, shape (3,)
        pf (npt.ArrayLike): Final position, shape (3,)
        t0 (float): Starting time
        tf (float): Ending time
        n_control_pts (int): Number of control points for the Bezier curve. Must be >= to the number of constraints,
            and should not be too large (>15ish) as this can reduce optimization performance. 6-10 is usually good
        n_timesteps (int): Number of timesteps to evaluate
        v0 (Optional[npt.ArrayLike]): Initial velocity, shape (3,). Defaults to None (unconstrained)
        vf (Optional[npt.ArrayLike]): Final velocity, shape (3,). Defaults to None (unconstrained)
        a0 (Optional[npt.ArrayLike]): Initial acceleration, shape (3,). Defaults to None (unconstrained)
        af (Optional[npt.ArrayLike]): Final acceleration, shape (3,). Defaults to None (unconstrained)
        box (Optional[Box]): Box constraint on (lower, upper) position bounds. Defaults to None (unconstrained)
        v_max (Optional[float]): Maximum L2 norm of the velocity. Defaults to None (unconstrained)
        a_max (Optional[float]): Maximum L2 norm of the acceleration. Defaults to None (unconstrained)
        jerk_weight (float, optional): Objective function weight corresponding to jerk. Defaults to 1.0.
        pathlength_weight (float, optional): Objective function weight corresponding to pathlength . Defaults to 0.0.

    Returns:
        Tuple of:
            np.ndarray: Trajectory positions, shape (n_timesteps, 3)
            np.ndarray: Trajectory velocities, shape (n_timesteps, 3)
            np.ndarray: Trajectory accelerations, shape (n_timesteps, 3)
            np.ndarray: Trajectory times, shape (n_timesteps,)
    """
    # Check inputs
    n_constraints = sum(c is not None for c in [p0, pf, v0, vf, a0, af])
    if n_constraints > n_control_pts:
        raise ValueError(
            "Number of control points must be at least the number of constraints"
        )
    if tf <= t0:
        raise ValueError(f"Invalid time interval: ({t0}, {tf})")
    dim = len(p0)
    times = np.linspace(t0, tf, n_timesteps, endpoint=True)
    # Form the main Variable (the control points for the position curve) and get the Expressions
    # for the control points of the derivative curves
    pos_pts = cp.Variable((n_control_pts, dim))
    pos_curve = BezierCurve(pos_pts, t0, tf)
    vel_curve = pos_curve.derivative
    vel_pts = vel_curve.points
    accel_curve = vel_curve.derivative
    accel_pts = accel_curve.points
    jerk_curve = accel_curve.derivative
    # Form the constraint list depending on what was specified in the inputs
    constraints = [pos_pts[0] == p0, pos_pts[-1] == pf]
    if v0 is not None:
        constraints.append(vel_pts[0] == v0)
    if vf is not None:
        constraints.append(vel_pts[-1] == vf)
    if a0 is not None:
        constraints.append(accel_pts[0] == a0)
    if af is not None:
        constraints.append(accel_pts[-1] == af)
    if box is not None:
        lower, upper = box
        constraints.append(pos_pts >= lower)
        constraints.append(pos_pts <= upper)
    if v_max is not None:
        constraints.append(cp.norm2(vel_pts, axis=1) <= v_max)
    if a_max is not None:
        constraints.append(cp.norm2(accel_pts, axis=1) <= a_max)
    # Objective function criteria
    jerk = jerk_curve.l2_squared
    pathlength = pos_curve.control_points_pathlength
    # Form the objective function based on the relative weighting between the criteria
    objective = cp.Minimize(jerk_weight * jerk + pathlength_weight * pathlength)
    # Form the problem and solve it
    # Note: Clarabel is apparently better for quadratic objectives (like our jerk criteria)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)
    if prob.status != cp.OPTIMAL:
        raise cp.error.SolverError(
            f"Unable to generate the trajectory (solver status: {prob.status}).\n"
            + "Check on the feasibility of the constraints"
        )
    # Construct the Bezier curves from the solved control points, and return their evaluations at each time
    solved_pos_curve = BezierCurve(pos_pts.value, t0, tf)
    solved_vel_curve = solved_pos_curve.derivative
    solved_accel_curve = solved_vel_curve.derivative
    return (
        solved_pos_curve(times),
        solved_vel_curve(times),
        solved_accel_curve(times),
        times,
    )


# Below: Plotting functions


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


# Below: Examples of using the functions above to generate polynomials/curves


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


def _test_bezier_traj():
    p0 = [1, 2, 3]
    pf = [2, 3, 4]
    t0 = 0
    tf = 5
    n_control_pts = 10
    n_timesteps = 50
    v0 = [-0.1, -0.2, -0.3]
    vf = [-0.2, -0.2, -0.2]
    a0 = [0, 0, 0]
    af = [0.1, 0.1, 0.1]
    pos, vel, accel, times = bezier_trajectory(
        p0, pf, t0, tf, n_control_pts, n_timesteps, v0, vf, a0, af
    )
    # Leaving out any rotational info for now
    traj = Trajectory(pos, None, vel, None, accel, None, times)
    traj.plot()


if __name__ == "__main__":
    _test_plot_bernstein_polys()
    _test_bezier_traj()
